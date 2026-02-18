# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Mesh splitting algorithms with fallback strategies.

Handles the actual mesh cutting operations: slicing along planes,
manual capping, and geodesic loop splitting.
No Cura dependencies - uses only trimesh, numpy, and optionally scipy.
"""

import numpy
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger("objectsplitter.mesh_splitter")


def _component_containing_face(mesh: "trimesh.Trimesh", face_id: int) -> List[int]:
    """Get the connected component (face indices) that contains the given face."""
    try:
        from trimesh import graph
        components = graph.connected_components(
            edges=mesh.face_adjacency,
            nodes=numpy.arange(len(mesh.faces)),
            min_len=1,
        )
        for comp in components:
            if face_id in comp:
                return list(comp)
    except Exception as e:
        logger.debug("Connected components failed: %s", e)
    return list(range(len(mesh.faces)))


def _merge_other_components(
    upper: "trimesh.Trimesh",
    lower: "trimesh.Trimesh",
    other_components: List["trimesh.Trimesh"],
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh"]:
    """Assign other components to upper or lower based on centroid side of plane."""
    if not other_components:
        return upper, lower
    for comp in other_components:
        dist = numpy.dot(comp.centroid - plane_origin, plane_normal)
        if dist >= 0:
            upper = trimesh.util.concatenate([upper, comp])
        else:
            lower = trimesh.util.concatenate([lower, comp])
    return upper, lower

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SplitResult:
    """Result of a mesh split operation, with debug metadata."""
    upper: Optional["trimesh.Trimesh"] = None
    lower: Optional["trimesh.Trimesh"] = None
    capped: bool = False
    strategy_used: str = "none"
    strategies_attempted: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return (self.upper is not None and self.lower is not None
                and len(self.upper.vertices) > 0 and len(self.lower.vertices) > 0)

    def summary(self) -> str:
        if not self.success:
            return f"FAILED (tried: {', '.join(self.strategies_attempted)}, error: {self.error})"
        return (f"OK via '{self.strategy_used}' | capped={self.capped} | "
                f"upper={len(self.upper.vertices)} verts/{len(self.upper.faces)} faces | "
                f"lower={len(self.lower.vertices)} verts/{len(self.lower.faces)} faces | "
                f"tried: {', '.join(self.strategies_attempted)}")


def slice_mesh_with_fallback(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
    face_id: Optional[int] = None,
) -> SplitResult:
    """
    Slice mesh with multiple fallback strategies for robustness.

    Strategy 1: trimesh slice with cap=True (requires rtree for watertight meshes).
    Strategy 2: trimesh slice without cap, then manual capping via scipy Delaunay.
    Strategy 3: Manual face-based splitting (no capping).

    If face_id is provided, only the connected component containing that face is cut.
    Other components are assigned to upper/lower based on which side of the plane
    their centroid lies on.

    Args:
        mesh: The trimesh object to split.
        plane_origin: A 3D point on the cutting plane.
        plane_normal: The normal vector of the cutting plane.
        face_id: If provided, only cut the component containing this face.

    Returns:
        SplitResult with the two mesh halves and metadata.
    """
    plane_origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    plane_normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    result = SplitResult()

    # Restrict to clicked component if requested
    mesh_to_cut = mesh
    other_components: List["trimesh.Trimesh"] = []
    if face_id is not None:
        comp_faces = _component_containing_face(mesh, face_id)
        if len(comp_faces) < len(mesh.faces):
            mesh_to_cut = mesh.submesh([comp_faces], append=True)
            try:
                from trimesh import graph
                all_components = graph.connected_components(
                    edges=mesh.face_adjacency,
                    nodes=numpy.arange(len(mesh.faces)),
                    min_len=1,
                )
                comp_set = set(comp_faces)
                for c in all_components:
                    if set(c) != comp_set:
                        other_components.append(mesh.submesh([list(c)], append=True))
            except Exception as e:
                logger.debug("Extracting other components failed: %s", e)

    # Strategy 1: Capped slicing
    result.strategies_attempted.append("capped_slice")
    try:
        upper = trimesh.intersections.slice_mesh_plane(
            mesh_to_cut, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
        lower = trimesh.intersections.slice_mesh_plane(
            mesh_to_cut, plane_normal=-plane_normal, plane_origin=plane_origin, cap=True)
        if upper is not None and lower is not None:
            capped = bool(upper.is_watertight and lower.is_watertight)
            result.upper, result.lower = _merge_other_components(
                upper, lower, other_components, plane_origin, plane_normal
            )
            result.capped = capped
            result.strategy_used = "capped_slice"
            logger.debug("Strategy 1 (capped slice) succeeded")
            return result
    except ImportError as e:
        logger.debug("Strategy 1 failed (missing rtree): %s", e)
    except Exception as e:
        logger.debug("Strategy 1 failed: %s", e)

    # Strategy 2: Uncapped slice + manual capping
    result.strategies_attempted.append("manual_cap")
    try:
        upper = trimesh.intersections.slice_mesh_plane(
            mesh_to_cut, plane_normal=plane_normal, plane_origin=plane_origin, cap=False)
        lower = trimesh.intersections.slice_mesh_plane(
            mesh_to_cut, plane_normal=-plane_normal, plane_origin=plane_origin, cap=False)
        if upper is not None and lower is not None:
            upper_capped = _manual_cap_mesh(upper, plane_origin, plane_normal)
            lower_capped = _manual_cap_mesh(lower, plane_origin, -plane_normal)
            if upper_capped is not None and lower_capped is not None:
                capped = bool(upper_capped.is_watertight and lower_capped.is_watertight)
                result.upper, result.lower = _merge_other_components(
                    upper_capped, lower_capped, other_components, plane_origin, plane_normal
                )
                result.capped = capped
                result.strategy_used = "manual_cap"
                logger.debug("Strategy 2 (manual cap) succeeded")
                return result
            else:
                # Use uncapped meshes as partial success
                result.upper, result.lower = _merge_other_components(
                    upper, lower, other_components, plane_origin, plane_normal
                )
                result.capped = False
                result.strategy_used = "uncapped_slice"
                logger.debug("Strategy 2 partial: uncapped slices (manual cap failed)")
                return result
    except Exception as e:
        logger.debug("Strategy 2 failed: %s", e)

    # Strategy 3: Manual face-based split
    result.strategies_attempted.append("manual_split")
    try:
        upper, lower = _manual_mesh_split(mesh_to_cut, plane_origin, plane_normal)
        if upper is not None and lower is not None:
            result.upper, result.lower = _merge_other_components(
                upper, lower, other_components, plane_origin, plane_normal
            )
            result.capped = False
            result.strategy_used = "manual_split"
            logger.debug("Strategy 3 (manual split) succeeded")
            return result
    except Exception as e:
        logger.debug("Strategy 3 failed: %s", e)
        result.error = str(e)

    result.error = result.error or "All strategies failed"
    return result


def local_plane_partition(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
    source_face_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Partition mesh faces into two sets using the cutting plane, separating
    only the local region near the clicked face.

    The algorithm:
    1. Classify each face as "above" or "below" the plane using its centroid.
    2. Build an adjacency graph connecting only faces on the SAME side.
    3. Find ALL connected components on both sides near the click point.
    4. Pick the SMALLEST component nearest to the click as the piece to separate.

    Args:
        mesh: The trimesh object.
        plane_origin: A point on the cutting plane.
        plane_normal: Normal vector of the cutting plane.
        source_face_id: The face the user clicked on.

    Returns:
        (set_a, set_b) where set_a is the piece to separate.
    """
    from collections import defaultdict, deque

    plane_origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    plane_normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    n_faces = len(mesh.faces)

    # Step 1: Classify each face by which side of the plane its centroid is on
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    face_dists = numpy.dot(face_centroids - plane_origin, plane_normal)
    face_above = face_dists >= 0  # boolean mask

    # Step 2: Build adjacency connecting only faces on the same side
    adj_pairs = mesh.face_adjacency
    graph = defaultdict(list)
    for f1, f2 in adj_pairs:
        f1i, f2i = int(f1), int(f2)
        if face_above[f1i] == face_above[f2i]:
            graph[f1i].append(f2i)
            graph[f2i].append(f1i)

    # Helper: BFS to find connected component from a seed face
    def _bfs_component(seed, allowed=None):
        comp = set()
        queue = deque([seed])
        comp.add(seed)
        while queue:
            face = queue.popleft()
            for neighbor in graph.get(face, []):
                if neighbor not in comp and (allowed is None or neighbor in allowed):
                    comp.add(neighbor)
                    queue.append(neighbor)
        return comp

    click_pos = face_centroids[source_face_id]

    # Step 3: Find the source face's component (on its side of the plane)
    source_comp = _bfs_component(source_face_id)

    # Step 4: Find the nearest component on the OTHER side
    source_is_above = face_above[source_face_id]
    other_face_ids = set(
        i for i in range(n_faces) if face_above[i] != source_is_above
    )

    nearest_other_comp = None
    if other_face_ids:
        remaining = set(other_face_ids)
        other_components = []
        while remaining:
            start = next(iter(remaining))
            comp = _bfs_component(start, allowed=remaining)
            remaining -= comp
            other_components.append(comp)

        # Pick the one nearest to the click point
        nearest_other_comp = min(
            other_components,
            key=lambda c: numpy.linalg.norm(
                face_centroids[sorted(c)].mean(axis=0) - click_pos
            )
        )

    logger.debug("local_plane_partition: source_face=%d, source_comp=%d faces, "
                 "nearest_other_comp=%s faces, total=%d",
                 source_face_id, len(source_comp),
                 len(nearest_other_comp) if nearest_other_comp else "N/A",
                 n_faces)

    if nearest_other_comp is None:
        logger.warning("local_plane_partition: no faces on the other side of plane")
        return list(range(n_faces)), []

    # Step 5: Pick the SMALLER of the two candidates.
    if len(source_comp) <= len(nearest_other_comp):
        chosen = source_comp
        logger.debug("local_plane_partition: separating source component "
                     "(%d faces, smaller)", len(chosen))
    else:
        chosen = nearest_other_comp
        logger.debug("local_plane_partition: separating other-side component "
                     "(%d faces, smaller)", len(chosen))

    set_a = sorted(chosen)
    set_b = sorted(set(range(n_faces)) - chosen)
    return set_a, set_b


def split_by_local_plane(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    candidate_normals: List[numpy.ndarray],
    source_face_id: int,
    min_face_fraction: float = 0.005,
) -> SplitResult:
    """
    Split mesh at the click point using a plane-guided local separation.

    Tries candidate plane normals in order (sorted by cross-section area).
    Skips any that produce a partition where either piece has fewer than
    min_face_fraction of total faces (prevents single-triangle cuts from
    surface-grazing planes).

    Args:
        mesh: The trimesh object.
        plane_origin: A point on the cutting plane.
        candidate_normals: List of normal vectors to try, ordered by preference.
        source_face_id: The face the user clicked on.
        min_face_fraction: Minimum fraction of total faces for each piece (default 0.5%).

    Returns:
        SplitResult with the separated piece and the rest.
    """
    result = SplitResult()
    result.strategies_attempted.append("local_plane_partition")

    plane_origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    n_faces = len(mesh.faces)
    min_faces = max(10, int(n_faces * min_face_fraction))

    logger.debug("split_by_local_plane: trying %d candidates, min_faces=%d (%.1f%% of %d)",
                 len(candidate_normals), min_faces, min_face_fraction * 100, n_faces)

    best_rejected = None  # Track best rejected partition for fallback
    best_rejected_min = 0

    for i, normal in enumerate(candidate_normals):
        normal = numpy.asarray(normal, dtype=numpy.float64)

        set_a, set_b = local_plane_partition(
            mesh, plane_origin, normal, source_face_id,
        )

        if len(set_a) == 0 or len(set_b) == 0:
            logger.debug("Candidate %d: empty partition, skipping", i)
            continue

        smaller = min(len(set_a), len(set_b))

        if smaller < min_faces:
            # Track the best rejected candidate (largest min-side)
            # so fallback uses the most balanced partition, not a surface graze
            if smaller > best_rejected_min:
                best_rejected = (set_a, set_b, i)
                best_rejected_min = smaller
            logger.debug("Candidate %d: partition too small (%d/%d faces, "
                         "min=%d), skipping",
                         i, len(set_a), len(set_b), min_faces)
            continue

        logger.info("Candidate %d accepted: %d/%d faces", i, len(set_a), len(set_b))
        return split_by_face_sets(mesh, set_a, set_b,
                                  strategy_name="local_plane_partition")

    # All candidates rejected. Use the best rejected partition (most balanced)
    # rather than falling back to infinite-plane slice which cuts everything.
    if best_rejected is not None:
        set_a, set_b, idx = best_rejected
        logger.warning("All %d local-plane candidates below min_faces=%d; "
                       "using best rejected (candidate %d: %d/%d faces)",
                       len(candidate_normals), min_faces, idx,
                       len(set_a), len(set_b))
        return split_by_face_sets(mesh, set_a, set_b,
                                  strategy_name="local_plane_partition(best_rejected)")

    # Truly no partition found — fall back to infinite-plane slice
    logger.warning("All %d local-plane candidates gave empty partitions; "
                   "falling back to infinite-plane slice",
                   len(candidate_normals))
    if candidate_normals:
        # Use the LAST candidate (largest area, most likely a real through-cut)
        normal = numpy.asarray(candidate_normals[-1], dtype=numpy.float64)
        result = slice_mesh_with_fallback(mesh, plane_origin, normal,
                                          face_id=source_face_id)
        result.strategies_attempted.insert(0, "local_plane_partition(all_empty)")
        return result

    result.error = ("No valid partition found across %d candidate planes "
                    "(min_faces=%d)" % (len(candidate_normals), min_faces))
    logger.warning("split_by_local_plane: %s", result.error)
    return result


def split_by_face_sets(
    mesh: "trimesh.Trimesh",
    face_set_a: list,
    face_set_b: list,
    strategy_name: str = "face_partition",
    attempt_hole_fill: bool = True,
) -> SplitResult:
    """
    Split a mesh into two parts using pre-computed face partitions.
    Shared by both shortest-seam and local-plane modes.

    Args:
        mesh: The trimesh object.
        face_set_a: Face indices for the first (smaller) part.
        face_set_b: Face indices for the second (larger) part.
        strategy_name: Name for logging/debug.
        attempt_hole_fill: If True, try to fill open boundaries to make
            both parts watertight.

    Returns:
        SplitResult with the two submeshes.
    """
    result = SplitResult()
    result.strategies_attempted.append(strategy_name)

    try:
        upper = mesh.submesh([face_set_a], append=True)
        lower = mesh.submesh([face_set_b], append=True)

        # Attempt hole-filling for watertightness
        capped = False
        try:
            if attempt_hole_fill and (not upper.is_watertight or not lower.is_watertight):
                upper_filled = upper.copy()
                lower_filled = lower.copy()
                upper_filled.fill_holes()
                lower_filled.fill_holes()
                if upper_filled.is_watertight and lower_filled.is_watertight:
                    upper = upper_filled
                    lower = lower_filled
                    capped = True
        except Exception as e:
            logger.debug("Hole filling after %s failed: %s", strategy_name, e)

        result.upper = upper
        result.lower = lower
        result.capped = capped
        result.strategy_used = strategy_name
    except Exception as e:
        result.error = str(e)
        logger.error("%s split failed: %s", strategy_name, e)

    return result


def split_by_shortest_seam(
    mesh: "trimesh.Trimesh",
    face_set_a: list,
    face_set_b: list
) -> SplitResult:
    """
    Split a mesh into two parts using pre-computed face partitions
    (from shortest-seam / min-cut algorithm).

    Args:
        mesh: The trimesh object.
        face_set_a: Face indices for the first part.
        face_set_b: Face indices for the second part.

    Returns:
        SplitResult with the two submeshes.
    """
    return split_by_face_sets(mesh, face_set_a, face_set_b,
                              strategy_name="shortest_seam",
                              attempt_hole_fill=False)


def _manual_cap_mesh(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray
) -> Optional["trimesh.Trimesh"]:
    """
    Cap an open mesh at the cut plane by triangulating the cross-section boundary.

    Uses scipy Delaunay if available, otherwise falls back to trimesh triangulation.
    """
    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            logger.debug("No cross-section found for capping")
            return None

        # Compatible with both trimesh 3.x (to_planar) and 4.x (to_2D)
        if hasattr(section, 'to_2D'):
            try:
                path_2d, to_3d = section.to_2D()
            except Exception:
                path_2d, to_3d = section.to_planar()
        elif hasattr(section, 'to_planar'):
            path_2d, to_3d = section.to_planar()
        else:
            return None
        if path_2d is None:
            return None

        vertices_2d = None
        faces_2d = None

        # Try scipy Delaunay first
        if SCIPY_AVAILABLE:
            try:
                all_vertices = []
                for entity in path_2d.entities:
                    points = path_2d.vertices[entity.points]
                    all_vertices.extend(points)

                if len(all_vertices) < 3:
                    return None

                vertices_2d = numpy.unique(numpy.array(all_vertices), axis=0)
                if len(vertices_2d) < 3:
                    return None

                tri = Delaunay(vertices_2d)
                faces_2d = tri.simplices
                logger.debug("Scipy Delaunay: %d vertices, %d faces", len(vertices_2d), len(faces_2d))
            except Exception as e:
                logger.debug("Scipy triangulation failed: %s", e)
                vertices_2d = None
                faces_2d = None

        # Fallback to trimesh triangulation
        if vertices_2d is None or faces_2d is None:
            try:
                vertices_2d, faces_2d = path_2d.triangulate()
            except Exception as e:
                logger.debug("Trimesh triangulation failed: %s", e)
                return None

        if vertices_2d is None or len(vertices_2d) == 0 or faces_2d is None or len(faces_2d) == 0:
            return None

        # Transform 2D vertices back to 3D
        vertices_3d_hom = numpy.column_stack([
            vertices_2d,
            numpy.zeros(len(vertices_2d)),
            numpy.ones(len(vertices_2d))
        ])
        vertices_3d = (to_3d @ vertices_3d_hom.T).T[:, :3]

        cap_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_2d)

        # Ensure cap normal faces the right direction
        if len(cap_mesh.face_normals) > 0:
            cap_normal = cap_mesh.face_normals.mean(axis=0)
            norm = numpy.linalg.norm(cap_normal)
            if norm > 1e-6:
                cap_normal = cap_normal / norm
                if numpy.dot(cap_normal, plane_normal) < 0:
                    cap_mesh.faces = cap_mesh.faces[:, ::-1]

        combined = trimesh.util.concatenate([mesh, cap_mesh])
        # Stitch shared boundary vertices so the cap seals the cut surface.
        combined.merge_vertices()
        logger.debug("Manual cap: added %d cap verts, %d cap faces", len(vertices_3d), len(faces_2d))
        return combined

    except Exception as e:
        logger.debug("Manual capping error: %s", e)
        return None


def _manual_mesh_split(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray
) -> Tuple[Optional["trimesh.Trimesh"], Optional["trimesh.Trimesh"]]:
    """
    Split mesh by classifying each face to one side of the plane based on centroid.
    No capping - results will have open edges at the cut boundary.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    face_centroids = vertices[faces].mean(axis=1)
    face_distances = numpy.dot(face_centroids - plane_origin, plane_normal)

    upper_faces = faces[face_distances >= 0]
    lower_faces = faces[face_distances < 0]

    if len(upper_faces) == 0 or len(lower_faces) == 0:
        logger.debug("Manual split: one side is empty")
        return None, None

    mesh_upper = trimesh.Trimesh(vertices=vertices.copy(), faces=upper_faces)
    mesh_lower = trimesh.Trimesh(vertices=vertices.copy(), faces=lower_faces)
    mesh_upper.remove_unreferenced_vertices()
    mesh_lower.remove_unreferenced_vertices()

    return mesh_upper, mesh_lower
