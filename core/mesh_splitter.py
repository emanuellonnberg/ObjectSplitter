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


def prune_small_components(
    mesh: "trimesh.Trimesh",
    min_faces: int,
) -> Tuple["trimesh.Trimesh", int, int]:
    """
    Remove disconnected mesh components smaller than min_faces.

    Returns:
        (pruned_mesh, removed_component_count, kept_component_count)

    Notes:
    - Always keeps at least the largest component, even if every component is
      below the threshold, so callers never end up with an empty mesh here.
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0, 0

    min_faces = max(0, int(min_faces))
    if min_faces <= 0:
        return mesh, 0, 1

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception as e:
        logger.debug("Component split failed during prune_small_components: %s", e)
        return mesh, 0, 1

    if len(components) <= 1:
        return mesh, 0, len(components)

    components = sorted(components, key=lambda comp: len(comp.faces), reverse=True)
    kept = [comp for comp in components if len(comp.faces) >= min_faces]
    if not kept:
        kept = [components[0]]

    removed_count = len(components) - len(kept)
    kept_count = len(kept)
    if removed_count <= 0:
        return mesh, 0, kept_count

    if len(kept) == 1:
        return kept[0], removed_count, kept_count
    return trimesh.util.concatenate(kept), removed_count, kept_count

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
    cap_faces_upper: Optional[list] = None
    cap_faces_lower: Optional[list] = None
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


def _signed_area_2d(poly: numpy.ndarray) -> float:
    """Return the signed area of a simple 2D polygon."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(numpy.sum(x * numpy.roll(y, -1) - numpy.roll(x, -1) * y))


def _point_in_triangle_2d(p, a, b, c, eps: float = 1e-12) -> bool:
    """Strict point-in-triangle test in 2D (excludes edges)."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) <= eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    w = 1.0 - u - v
    return (u > eps) and (v > eps) and (w > eps)


def _triangulate_polygon_earclip(poly: numpy.ndarray) -> Optional[numpy.ndarray]:
    """
    Triangulate a simple 2D polygon with ear clipping.
    Returns face indices into `poly` or None on failure.
    """
    n = len(poly)
    if n < 3:
        return None

    area = _signed_area_2d(poly)
    if abs(area) < 1e-12:
        return None

    ccw = area > 0.0
    remaining = list(range(n))
    faces = []
    max_iters = n * n
    iters = 0

    while len(remaining) > 3 and iters < max_iters:
        iters += 1
        ear_found = False
        m = len(remaining)
        for i in range(m):
            ia = remaining[(i - 1) % m]
            ib = remaining[i]
            ic = remaining[(i + 1) % m]
            a = poly[ia]
            b = poly[ib]
            c = poly[ic]

            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if ccw:
                if cross <= 1e-12:
                    continue
            else:
                if cross >= -1e-12:
                    continue

            has_inside = False
            for ip in remaining:
                if ip in (ia, ib, ic):
                    continue
                if _point_in_triangle_2d(poly[ip], a, b, c):
                    has_inside = True
                    break
            if has_inside:
                continue

            if ccw:
                faces.append([ia, ib, ic])
            else:
                faces.append([ia, ic, ib])
            del remaining[i]
            ear_found = True
            break

        if not ear_found:
            return None

    if len(remaining) == 3:
        ia, ib, ic = remaining
        if ccw:
            faces.append([ia, ib, ic])
        else:
            faces.append([ia, ic, ib])

    if not faces:
        return None
    return numpy.asarray(faces, dtype=numpy.int32)


def _sanitize_boundary_loop(loop_3d: numpy.ndarray) -> Optional[numpy.ndarray]:
    """Remove closure duplicate and consecutive duplicate points."""
    pts = numpy.asarray(loop_3d, dtype=numpy.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return None

    # Drop repeated closing point.
    if len(pts) >= 2 and numpy.linalg.norm(pts[0] - pts[-1]) <= 1e-8:
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    dedup = [pts[0]]
    for p in pts[1:]:
        if numpy.linalg.norm(p - dedup[-1]) > 1e-8:
            dedup.append(p)
    pts = numpy.asarray(dedup, dtype=numpy.float64)
    if len(pts) < 3:
        return None
    return pts


def _fit_loop_plane(loop_3d: numpy.ndarray):
    """Fit a best-fit plane basis (origin, u, v) to a 3D boundary loop."""
    origin = loop_3d.mean(axis=0)
    centered = loop_3d - origin
    try:
        _, _, vh = numpy.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    if vh.shape[0] < 2:
        return None
    u = vh[0]
    v = vh[1]
    nu = numpy.linalg.norm(u)
    nv = numpy.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return None
    u = u / nu
    v = v / nv
    # Re-orthogonalize.
    n = numpy.cross(u, v)
    nn = numpy.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    v = numpy.cross(n, u)
    vn = numpy.linalg.norm(v)
    if vn < 1e-12:
        return None
    v = v / vn
    return origin, u, v


def _triangulate_boundary_loop_2d(loop_2d: numpy.ndarray) -> Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]:
    """
    Triangulate a 2D closed loop using trimesh Path2D when available,
    with ear-clipping fallback.
    Returns (vertices_2d, faces) or (None, None).
    """
    if len(loop_2d) < 3:
        return None, None

    try:
        entities = [
            trimesh.path.entities.Line([i, (i + 1) % len(loop_2d)])
            for i in range(len(loop_2d))
        ]
        path2d = trimesh.path.Path2D(
            entities=entities,
            vertices=loop_2d.astype(numpy.float64),
        )
        verts_2d, faces = path2d.triangulate()
        if (
            verts_2d is not None and len(verts_2d) >= 3 and
            faces is not None and len(faces) > 0
        ):
            return (
                numpy.asarray(verts_2d, dtype=numpy.float64),
                numpy.asarray(faces, dtype=numpy.int32),
            )
    except Exception as e:
        logger.debug("Path2D triangulation failed, falling back to ear clipping: %s", e)

    faces = _triangulate_polygon_earclip(loop_2d)
    if faces is None or len(faces) == 0:
        return None, None
    return loop_2d.astype(numpy.float64), faces


def _cap_open_boundaries(
    mesh: "trimesh.Trimesh",
) -> Tuple[Optional["trimesh.Trimesh"], Optional[list]]:
    """
    Cap all open boundary loops of a mesh by triangulating each loop.
    Works for non-planar loops by projecting to a best-fit plane.
    """
    try:
        outline = mesh.outline()
    except Exception as e:
        logger.debug("outline() failed while capping boundaries: %s", e)
        return None, None

    if outline is None or not hasattr(outline, "discrete"):
        return None, None

    cap_meshes = []
    loops = outline.discrete
    if loops is None or len(loops) == 0:
        return None, None

    for li, loop in enumerate(loops):
        loop_3d = _sanitize_boundary_loop(loop)
        if loop_3d is None or len(loop_3d) < 3:
            continue

        basis = _fit_loop_plane(loop_3d)
        if basis is None:
            logger.debug("Boundary loop %d: best-fit plane failed", li)
            continue
        origin, u, v = basis
        centered = loop_3d - origin
        loop_2d = numpy.column_stack([
            centered @ u,
            centered @ v,
        ])

        verts_2d, faces_2d = _triangulate_boundary_loop_2d(loop_2d)
        if verts_2d is None or faces_2d is None or len(faces_2d) == 0:
            logger.debug("Boundary loop %d: triangulation failed", li)
            continue

        # Reconstruct cap vertices in 3D. Snap triangulation vertices that
        # coincide with loop boundary points back to the exact original 3D
        # boundary coordinates so merge_vertices can stitch the seam cleanly.
        verts_3d = (
            origin[None, :] +
            verts_2d[:, 0:1] * u[None, :] +
            verts_2d[:, 1:2] * v[None, :]
        )
        if len(loop_2d) > 0:
            snapped = verts_3d.copy()
            for vi in range(len(verts_2d)):
                d2 = numpy.linalg.norm(loop_2d - verts_2d[vi], axis=1)
                bi = int(numpy.argmin(d2))
                if float(d2[bi]) <= 1e-7:
                    snapped[vi] = loop_3d[bi]
            verts_3d = snapped
        cap = trimesh.Trimesh(
            vertices=verts_3d.astype(numpy.float64),
            faces=faces_2d.astype(numpy.int64),
            process=False,
            validate=False,
        )
        cap_meshes.append(cap)

    if not cap_meshes:
        return None, None

    combined = trimesh.util.concatenate([mesh] + cap_meshes)
    base_faces = len(mesh.faces)
    cap_face_ranges = []
    running = base_faces
    for cap in cap_meshes:
        count = len(cap.faces)
        cap_face_ranges.append((running, running + count))
        running += count
    try:
        combined.merge_vertices(digits_vertex=7)
        combined.remove_unreferenced_vertices()
    except Exception as e:
        logger.debug("Post-cap cleanup failed: %s", e)
    cap_faces = []
    for start, end in cap_face_ranges:
        cap_faces.extend(range(start, end))

    # Orient only the freshly added cap faces to match the existing surface.
    # The original faces are already winding-consistent, so running a global
    # fix_normals()/fix_winding() (which walks every face on the whole mesh)
    # just to orient a few cap triangles was the dominant cost of capping on
    # large meshes. Fall back to fix_normals only if local orientation fails.
    try:
        _orient_caps_to_boundary(combined, base_faces, cap_face_ranges)
    except Exception as e:
        logger.debug("Local cap orientation failed, using fix_normals: %s", e)
        try:
            combined.fix_normals()
        except Exception:
            pass
    return combined, cap_faces


def _orient_caps_to_boundary(
    combined: "trimesh.Trimesh",
    base_faces: int,
    cap_face_ranges: list,
) -> None:
    """Flip each cap's winding in place to agree with the original surface.

    Assumes the first ``base_faces`` faces are already winding-consistent.
    Each planar cap is uniformly wound, so one shared boundary edge decides
    the flip for the whole cap: a manifold boundary edge is traversed in
    *opposite* directions by the two faces sharing it, so if a cap face
    traverses a boundary edge in the *same* direction as the original face,
    the cap is reversed.

    This avoids a global fix_winding over the entire mesh.
    """
    faces = numpy.asarray(combined.faces, dtype=numpy.int64)
    if base_faces <= 0 or base_faces >= len(faces):
        return

    # Encode the directed edges of the original (already-consistent) faces as
    # int64 keys, vectorised -- a per-face Python loop here is the dominant
    # cost on large meshes (hundreds of thousands of faces).
    orig = faces[:base_faces]
    key_base = int(faces.max()) + 1
    orig_directed = numpy.concatenate([
        orig[:, 0] * key_base + orig[:, 1],
        orig[:, 1] * key_base + orig[:, 2],
        orig[:, 2] * key_base + orig[:, 0],
    ])
    orig_directed.sort()

    def _is_directed(x: int, y: int) -> bool:
        k = x * key_base + y
        i = numpy.searchsorted(orig_directed, k)
        return i < len(orig_directed) and orig_directed[i] == k

    new_faces = faces.copy()
    flipped_any = False
    for start, end in cap_face_ranges:
        flip = None
        for fi in range(start, end):
            a, b, c = int(faces[fi][0]), int(faces[fi][1]), int(faces[fi][2])
            for x, y in ((a, b), (b, c), (c, a)):
                if _is_directed(x, y):
                    flip = True  # same direction as original -> inconsistent
                    break
                if _is_directed(y, x):
                    flip = False  # opposite direction -> already consistent
                    break
            if flip is not None:
                break
        if flip:
            new_faces[start:end] = new_faces[start:end][:, ::-1]
            flipped_any = True

    if flipped_any:
        combined.faces = new_faces


def _attempt_watertight_repair(
    mesh: "trimesh.Trimesh",
) -> Tuple["trimesh.Trimesh", Optional[list]]:
    """Try progressively stronger repairs to make a mesh watertight."""
    repaired = mesh.copy()
    try:
        repaired.merge_vertices()
        repaired.remove_unreferenced_vertices()
    except Exception:
        pass

    if repaired.is_watertight:
        return repaired, None

    # Fast trivial-hole repair (triangles/quads only).
    try:
        repaired.fill_holes()
    except Exception as e:
        logger.debug("fill_holes failed: %s", e)

    if repaired.is_watertight:
        return repaired, None

    # General boundary-loop capping for large path seams.
    capped, cap_faces = _cap_open_boundaries(repaired)
    if capped is None:
        return repaired, None

    try:
        if not capped.is_watertight:
            capped.fill_holes()
    except Exception:
        pass
    return capped, cap_faces


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
    min_face_fraction: float = 0.02,
) -> SplitResult:
    """
    Split mesh at the click point using a plane-guided local separation.

    Tries candidate plane normals in order (sorted by cross-section area).
    Skips any that produce a partition where either piece has fewer than
    min_face_fraction of total faces (prevents tiny sliver cuts from
    surface-grazing planes).

    Args:
        mesh: The trimesh object.
        plane_origin: A point on the cutting plane.
        candidate_normals: List of normal vectors to try, ordered by preference.
        source_face_id: The face the user clicked on.
        min_face_fraction: Minimum fraction of total faces for each piece
            (default 2%).

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


def _component_nearest_point(components, point):
    """Return the connected-component submesh whose surface is closest to point."""
    point = numpy.asarray(point, dtype=numpy.float64)
    best = None
    best_d = None
    for comp in components:
        d = float(numpy.linalg.norm(comp.vertices - point, axis=1).min())
        if best_d is None or d < best_d:
            best_d = d
            best = comp
    return best


def clean_local_plane_split(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
    source_face_id: int,
    whole_model: bool = False,
) -> SplitResult:
    """
    Split a mesh with a plane, producing a clean (triangle-sliced) cut surface.

    Local (default): separates only the clicked connected feature. The clicked
    component comes from a capped slice (clean and watertight); the other cut
    components are welded back onto the body and the single leftover hole is
    repaired, so the remainder stays watertight with the other features intact.

    whole_model=True: behaves like the global capped slice (cuts everything the
    plane crosses) -- for stacking-split tall prints.

    Falls back to split_by_local_plane (face partition) on any failure so a cut
    never crashes.

    Args:
        mesh: The trimesh object.
        plane_origin: A point on the cutting plane.
        plane_normal: Normal vector of the cutting plane.
        source_face_id: The face the user clicked on.
        whole_model: If True, cut the whole model instead of just the click.

    Returns:
        SplitResult with upper = separated piece, lower = remainder.
    """
    result = SplitResult()
    result.strategies_attempted.append("clean_local_plane_split")

    origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    norm = float(numpy.linalg.norm(normal))
    if norm > 0:
        normal = normal / norm

    if whole_model:
        gr = slice_mesh_with_fallback(mesh, origin, normal, face_id=source_face_id)
        gr.strategies_attempted.insert(0, "clean_local_plane_split(whole_model)")
        return gr

    try:
        src_centroid = mesh.vertices[mesh.faces[source_face_id]].mean(axis=0)

        def _split_feature_on(side_normal):
            """Separate the near-click feature on the +side_normal side.

            Returns (separated, remainder, separated_face_count) or None.
            """
            cap = trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=side_normal, plane_origin=origin, cap=True)
            cap.merge_vertices()
            cap_comps = cap.split(only_watertight=False)
            if not cap_comps:
                return None
            separated = _component_nearest_point(cap_comps, src_centroid).copy()

            # Uncapped slices share plane-loop vertices, so welding the
            # non-clicked feature-side components back onto the body is seamless.
            feat = trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=side_normal, plane_origin=origin, cap=False)
            feat.merge_vertices()
            body = trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=-side_normal, plane_origin=origin, cap=False)
            body.merge_vertices()
            feat_comps = feat.split(only_watertight=False)
            if not feat_comps:
                return None
            clicked = _component_nearest_point(feat_comps, src_centroid)
            others = [c for c in feat_comps if c is not clicked]

            remainder = trimesh.util.concatenate([body] + others)
            remainder.merge_vertices()
            remainder, _cap_faces = _attempt_watertight_repair(remainder)
            if len(separated.vertices) == 0 or len(remainder.vertices) == 0:
                return None
            return separated, remainder, len(separated.faces)

        # The clicked face lies on the plane, so its centroid side is unreliable.
        # Try both orientations and keep the one whose separated piece is smaller
        # -- the local feature (e.g. a tooth tip), not the whole body.
        candidates = [r for r in (_split_feature_on(normal),
                                  _split_feature_on(-normal)) if r is not None]
        if not candidates:
            raise ValueError("no valid clean split on either side")
        separated, remainder, _ = min(candidates, key=lambda r: r[2])

        result.upper = separated
        result.lower = remainder
        result.capped = bool(separated.is_watertight and remainder.is_watertight)
        result.strategy_used = "clean_local_plane_split"
        return result
    except Exception as e:  # noqa: BLE001 - a cut must never crash
        logger.warning(
            "clean_local_plane_split failed (%s); falling back to face partition", e)
        fb = split_by_local_plane(
            mesh, origin, [numpy.asarray(plane_normal, dtype=numpy.float64)],
            source_face_id)
        fb.strategies_attempted.insert(0, "clean_local_plane_split(failed)")
        return fb


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
        cap_faces_upper = None
        cap_faces_lower = None
        try:
            if attempt_hole_fill and (not upper.is_watertight or not lower.is_watertight):
                upper_repaired, cap_faces_upper = _attempt_watertight_repair(upper)
                lower_repaired, cap_faces_lower = _attempt_watertight_repair(lower)
                upper = upper_repaired
                lower = lower_repaired
                capped = bool(upper.is_watertight and lower.is_watertight)
        except Exception as e:
            logger.debug("Hole filling after %s failed: %s", strategy_name, e)

        result.upper = upper
        result.lower = lower
        result.cap_faces_upper = cap_faces_upper
        result.cap_faces_lower = cap_faces_lower
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
