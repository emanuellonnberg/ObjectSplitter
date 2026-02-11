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
from typing import Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("objectsplitter.mesh_splitter")

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
    plane_normal: numpy.ndarray
) -> SplitResult:
    """
    Slice mesh with multiple fallback strategies for robustness.

    Strategy 1: trimesh slice with cap=True (requires rtree for watertight meshes).
    Strategy 2: trimesh slice without cap, then manual capping via scipy Delaunay.
    Strategy 3: Manual face-based splitting (no capping).

    Args:
        mesh: The trimesh object to split.
        plane_origin: A 3D point on the cutting plane.
        plane_normal: The normal vector of the cutting plane.

    Returns:
        SplitResult with the two mesh halves and metadata.
    """
    plane_origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    plane_normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    result = SplitResult()

    # Strategy 1: Capped slicing
    result.strategies_attempted.append("capped_slice")
    try:
        upper = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
        lower = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=-plane_normal, plane_origin=plane_origin, cap=True)
        if upper is not None and lower is not None:
            result.upper = upper
            result.lower = lower
            result.capped = True
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
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=False)
        lower = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=-plane_normal, plane_origin=plane_origin, cap=False)
        if upper is not None and lower is not None:
            upper_capped = _manual_cap_mesh(upper, plane_origin, plane_normal)
            lower_capped = _manual_cap_mesh(lower, plane_origin, -plane_normal)
            if upper_capped is not None and lower_capped is not None:
                result.upper = upper_capped
                result.lower = lower_capped
                result.capped = True
                result.strategy_used = "manual_cap"
                logger.debug("Strategy 2 (manual cap) succeeded")
                return result
            else:
                # Use uncapped meshes as partial success
                result.upper = upper
                result.lower = lower
                result.capped = False
                result.strategy_used = "uncapped_slice"
                logger.debug("Strategy 2 partial: uncapped slices (manual cap failed)")
                return result
    except Exception as e:
        logger.debug("Strategy 2 failed: %s", e)

    # Strategy 3: Manual face-based split
    result.strategies_attempted.append("manual_split")
    try:
        upper, lower = _manual_mesh_split(mesh, plane_origin, plane_normal)
        if upper is not None and lower is not None:
            result.upper = upper
            result.lower = lower
            result.capped = False
            result.strategy_used = "manual_split"
            logger.debug("Strategy 3 (manual split) succeeded")
            return result
    except Exception as e:
        logger.debug("Strategy 3 failed: %s", e)
        result.error = str(e)

    result.error = result.error or "All strategies failed"
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
    result = SplitResult()
    result.strategies_attempted.append("shortest_seam")

    try:
        upper = mesh.submesh([face_set_a], append=True)
        lower = mesh.submesh([face_set_b], append=True)

        # Attempt hole-filling for watertightness
        capped = False
        try:
            if not upper.is_watertight or not lower.is_watertight:
                upper_filled = upper.copy()
                lower_filled = lower.copy()
                upper_filled.fill_holes()
                lower_filled.fill_holes()
                if upper_filled.is_watertight and lower_filled.is_watertight:
                    upper = upper_filled
                    lower = lower_filled
                    capped = True
        except Exception as e:
            logger.debug("Hole filling after shortest seam failed: %s", e)

        result.upper = upper
        result.lower = lower
        result.capped = capped
        result.strategy_used = "shortest_seam"
    except Exception as e:
        result.error = str(e)
        logger.error("Shortest seam split failed: %s", e)

    return result


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

        path_2d, transform = section.to_2D()
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
        transform_inv = numpy.linalg.inv(transform)
        vertices_3d = (transform_inv @ vertices_3d_hom.T).T[:, :3]

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
