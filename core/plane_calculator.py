# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Cut plane calculation algorithms.

Determines where and how to cut a mesh based on the selected mode.
No Cura dependencies - uses only trimesh and numpy.
"""

import numpy
import logging
import json
import os
import time
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import trimesh
except ImportError:
    trimesh = None

from .geometry import plane_normal_from_spherical

logger = logging.getLogger("objectsplitter.plane_calculator")


def _to_jsonable(value):
    """Convert numpy-heavy values to JSON-serializable Python types."""
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.integer,)):
        return int(value)
    if isinstance(value, (numpy.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _write_json_trace(path: Optional[str], payload: dict) -> None:
    """Best-effort JSON trace write; never raises."""
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug("Failed to write trace '%s': %s", path, e)


def _as_anchor_array(anchor_points) -> numpy.ndarray:
    """Normalize optional anchor points to a float64 (N, 3) array."""
    if anchor_points is None:
        return numpy.zeros((0, 3), dtype=numpy.float64)
    arr = numpy.asarray(anchor_points, dtype=numpy.float64)
    if arr.size == 0:
        return numpy.zeros((0, 3), dtype=numpy.float64)
    arr = arr.reshape(-1, 3)
    return arr


def _section_to_2d(section):
    """
    Convert a Path3D cross-section to 2D, compatible with both
    trimesh 3.x (only has to_planar) and 4.x (to_2D preferred, to_planar deprecated).

    Returns:
        (path_2d, to_3D_transform) or raises if neither method exists.
    """
    # Try to_2D first (trimesh >= 4.x, not deprecated)
    if hasattr(section, 'to_2D'):
        try:
            return section.to_2D()
        except Exception:
            pass
    # Fall back to to_planar (trimesh 3.x, or 4.x where to_2D might fail)
    if hasattr(section, 'to_planar'):
        return section.to_planar()
    raise AttributeError(
        "Path3D has neither 'to_2D' nor 'to_planar' — unsupported trimesh version"
    )


def _local_section_area(path_2d, to_3D, click_position_3d):
    """
    Compute the area of the cross-section polygon nearest to the click point,
    rather than the total area of all disconnected polygons.

    This prevents the algorithm from choosing a plane that has a small total
    area because it passes through a thin feature far from the click point.

    Args:
        path_2d: 2D cross-section path (from _section_to_2d).
        to_3D: 4x4 transform matrix mapping 2D plane coords back to 3D.
        click_position_3d: The 3D click point (which lies on the cutting plane).

    Returns:
        Area of the polygon nearest to the click point.
    """
    try:
        polygons = path_2d.polygons_full
        if polygons is None or len(polygons) == 0:
            return abs(path_2d.area)
    except Exception:
        return abs(path_2d.area)

    # Only one polygon — no ambiguity
    if len(polygons) == 1:
        return abs(polygons[0].area)

    # Project click point from 3D onto the 2D plane
    try:
        to_2D = numpy.linalg.inv(to_3D)
        click_h = numpy.append(numpy.asarray(click_position_3d, dtype=numpy.float64), 1.0)
        click_2d = (to_2D @ click_h)[:2]
    except Exception:
        # If projection fails, fall back to total area
        return abs(path_2d.area)

    from shapely.geometry import Point
    click_pt = Point(float(click_2d[0]), float(click_2d[1]))

    # First check: does the click point land inside a polygon?
    for poly in polygons:
        if poly.contains(click_pt):
            return abs(poly.area)

    # Click point is outside all polygons (likely on an edge or between them).
    # Pick the nearest polygon.
    best_dist = float('inf')
    best_area = abs(path_2d.area)  # fallback: total
    for poly in polygons:
        try:
            dist = poly.distance(click_pt)
            if dist < best_dist:
                best_dist = dist
                best_area = abs(poly.area)
        except Exception:
            continue

    return best_area


def snap_point_to_mesh_surface(
    mesh: "trimesh.Trimesh",
    point: numpy.ndarray
) -> Tuple[numpy.ndarray, Optional[int]]:
    """
    Snap a 3D point to the nearest point on the mesh surface.
    Use this to correct pick positions that may be slightly off.

    Returns:
        (point_on_surface, face_id) - face_id is None if proximity query fails.
    """
    point = numpy.asarray(point, dtype=numpy.float64).reshape(1, -1)
    face_id = None
    try:
        from trimesh.proximity import ProximityQuery
        pq = ProximityQuery(mesh)
        closest, distance, face_ids = pq.on_surface(point)
        if closest is not None and len(closest) > 0:
            pt = numpy.array(closest[0], dtype=numpy.float64)
            if face_ids is not None and len(face_ids) > 0:
                face_id = int(face_ids[0])
            return pt, face_id
    except Exception as e:
        logger.debug("Proximity snap failed: %s", e)

    # Fallback: use mesh centroid (point is likely wrong)
    return numpy.array(mesh.centroid, dtype=numpy.float64), None


def _estimate_face_thinness_sdf(
    mesh: "trimesh.Trimesh",
    face_ids: numpy.ndarray,
    max_faces: int = 3000,
) -> numpy.ndarray:
    """
    Experimental Shape Diameter Function (SDF-like) proxy.

    For each sampled face, cast rays from the face center in +/- face-normal
    directions and estimate local thickness as the sum of first-hit distances.
    Convert thickness to a normalized "thinness" score in [0, 1], where:
      1.0 = thin region (valley/throat-like), 0.0 = thick region.

    Returns:
        thinness array of length len(mesh.faces), zeros for unsampled/invalid.
    """
    n_faces = len(mesh.faces)
    thinness = numpy.zeros(n_faces, dtype=numpy.float64)
    if face_ids is None or len(face_ids) == 0:
        return thinness

    face_ids = numpy.asarray(face_ids, dtype=numpy.int64)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < n_faces)]
    if face_ids.size == 0:
        return thinness

    if face_ids.size > max_faces:
        face_ids = face_ids[:max_faces]

    centers = numpy.asarray(mesh.triangles_center, dtype=numpy.float64)[face_ids]
    normals = numpy.asarray(mesh.face_normals, dtype=numpy.float64)[face_ids]
    normal_norms = numpy.linalg.norm(normals, axis=1)
    valid_normals = normal_norms > 1e-10
    if not numpy.any(valid_normals):
        return thinness
    normals[valid_normals] = normals[valid_normals] / normal_norms[valid_normals, numpy.newaxis]

    mesh_diag = float(numpy.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    eps = max(1e-5, mesh_diag * 1e-6)

    origins_fwd = centers + normals * eps
    origins_bwd = centers - normals * eps
    dirs_fwd = normals.copy()
    dirs_bwd = -normals

    origins = numpy.vstack((origins_fwd, origins_bwd))
    directions = numpy.vstack((dirs_fwd, dirs_bwd))
    n_rays = origins.shape[0]
    min_dist = numpy.full(n_rays, numpy.inf, dtype=numpy.float64)

    try:
        locations, index_ray, _ = mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=True,
        )
    except Exception as e:
        logger.debug("SDF proxy ray query failed: %s", e)
        return thinness

    if index_ray is None or len(index_ray) == 0:
        return thinness

    index_ray = numpy.asarray(index_ray, dtype=numpy.int64)
    ray_origins_hit = origins[index_ray]
    hit_dist = numpy.linalg.norm(locations - ray_origins_hit, axis=1)

    # Ignore near-zero self hits.
    valid_hits = hit_dist > (eps * 2.0)
    if not numpy.any(valid_hits):
        return thinness
    numpy.minimum.at(min_dist, index_ray[valid_hits], hit_dist[valid_hits])

    n = len(face_ids)
    d_fwd = min_dist[:n]
    d_bwd = min_dist[n:]
    valid = numpy.isfinite(d_fwd) & numpy.isfinite(d_bwd)
    if not numpy.any(valid):
        return thinness

    thickness = d_fwd[valid] + d_bwd[valid]
    if thickness.size < 8:
        return thinness

    p10 = float(numpy.percentile(thickness, 10.0))
    p90 = float(numpy.percentile(thickness, 90.0))
    if p90 <= p10 + 1e-12:
        return thinness

    # Smaller thickness => larger thinness score.
    scaled = (thickness - p10) / (p90 - p10)
    local_thinness = 1.0 - numpy.clip(scaled, 0.0, 1.0)

    valid_face_ids = face_ids[valid]
    thinness[valid_face_ids] = local_thinness
    return thinness


@dataclass
class CutPlane:
    """Describes a cutting plane in 3D space."""
    origin: numpy.ndarray   # 3D point on the plane
    normal: numpy.ndarray   # Unit normal vector

    def __repr__(self) -> str:
        return (f"CutPlane(origin=[{self.origin[0]:.3f}, {self.origin[1]:.3f}, "
                f"{self.origin[2]:.3f}], normal=[{self.normal[0]:.3f}, "
                f"{self.normal[1]:.3f}, {self.normal[2]:.3f}])")


def horizontal_cut_plane(
    mesh: "trimesh.Trimesh",
    height_percent: float
) -> CutPlane:
    """
    Compute a horizontal cut plane at a given height percentage of the mesh.

    Args:
        mesh: The trimesh object to cut.
        height_percent: Percentage of the mesh height (0-100).

    Returns:
        CutPlane with Y-up normal at the computed height.
    """
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_y = bounds[0][1]
    max_y = bounds[1][1]
    height = max_y - min_y
    cut_y = min_y + (height * height_percent / 100.0)

    origin = numpy.array([0.0, cut_y, 0.0])
    normal = numpy.array([0.0, 1.0, 0.0])

    logger.debug("Horizontal cut: height_percent=%.1f%%, cut_y=%.3f (range %.3f to %.3f)",
                 height_percent, cut_y, min_y, max_y)
    return CutPlane(origin=origin, normal=normal)


def vertical_cut_plane(
    click_position: numpy.ndarray,
    plane_normal: Optional[numpy.ndarray] = None
) -> CutPlane:
    """
    Compute a vertical cut plane through the click position.

    Args:
        click_position: 3D world-space point where the user clicked.
        plane_normal: Optional custom normal. Defaults to X-axis [1, 0, 0].

    Returns:
        CutPlane at the click position.
    """
    origin = numpy.asarray(click_position, dtype=numpy.float64)
    if plane_normal is not None:
        normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    else:
        normal = numpy.array([1.0, 0.0, 0.0])

    logger.debug("Vertical cut: origin=%s, normal=%s", origin, normal)
    return CutPlane(origin=origin, normal=normal)


@dataclass
class SmallestPlaneSearchResult:
    """Result of the smallest cross-section search, including debug data."""
    plane: CutPlane
    area: float
    samples_tested: int
    # All tested orientations for debugging/visualization
    all_samples: Optional[list] = None  # List of (normal, area) tuples
    # Top N candidates sorted by area (for fallback if best produces bad partition)
    top_candidates: Optional[list] = None  # List of (area, normal) tuples


def find_smallest_cut_plane(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    search_resolution: int = 18,
    collect_all_samples: bool = False,
    surface_normal: Optional[numpy.ndarray] = None,
) -> SmallestPlaneSearchResult:
    """
    Find the plane orientation that produces the smallest cross-sectional area,
    biased toward the surface normal at the click point.

    When a surface_normal is provided, the scoring favors planes whose normal is
    aligned with the surface normal. This means: if you click on the side of a
    neck, the cut plane will align with the direction you clicked from, rather
    than some random surface-grazing angle 90° away.

    Scoring:  effective_score = area * (1 + BIAS * (1 - alignment))
    where alignment = |dot(candidate_normal, surface_normal)| in [0, 1].
    A perfectly aligned plane gets score = area (no penalty).
    A perpendicular plane gets score = area * (1 + BIAS).

    Args:
        mesh: The trimesh object to analyze.
        click_position: 3D point to pass the plane through.
        search_resolution: Number of elevation angles to sample (azimuth = 2x this).
        collect_all_samples: If True, store all (normal, area) pairs for debugging.
        surface_normal: Surface normal at the click point. If provided, strongly
            biases the search toward planes aligned with this direction.

    Returns:
        SmallestPlaneSearchResult with the best plane, area, and optional debug data.
    """
    plane_origin = numpy.asarray(click_position, dtype=numpy.float64)
    best_normal = numpy.array([0.0, 1.0, 0.0])
    best_score = float('inf')
    best_area = float('inf')
    samples_tested = 0
    all_samples = [] if collect_all_samples else None
    valid_candidates = []  # collect all valid (score, area, normal) tuples

    # Alignment bias: gently prefer planes aligned with the surface normal.
    # With BIAS=0.5, a perpendicular plane needs ~1.5x smaller area to beat
    # an aligned one. This is mild enough that a genuinely smaller cut still
    # wins (important when clicking the SIDE of a neck, where the surface
    # normal is perpendicular to the ideal cut direction), but strong enough
    # to break ties between similar-area candidates in favor of the click
    # direction.  The heavy lifting for filtering surface-grazing planes is
    # done by min_section_area, not the bias.
    ALIGNMENT_BIAS = 2.0
    use_bias = (surface_normal is not None and
                numpy.linalg.norm(surface_normal) > 0.5)
    if use_bias:
        surface_normal = numpy.asarray(surface_normal, dtype=numpy.float64)
        surface_normal = surface_normal / numpy.linalg.norm(surface_normal)
        logger.debug("Smallest cut search: using surface_normal=%s with bias=%.1f",
                     surface_normal, ALIGNMENT_BIAS)

    # Minimum area threshold: reject cross-sections smaller than this.
    # Surface-grazing planes can skim the mesh from MANY orientations, each
    # producing a tiny area (e.g. 0.12 mm²). A real through-cut has area
    # comparable to several face areas.
    avg_face_area = mesh.area / max(len(mesh.faces), 1)
    min_section_area = avg_face_area * 5.0
    logger.debug("Smallest cut search: avg_face_area=%.4f, min_section_area=%.4f",
                 avg_face_area, min_section_area)

    def _compute_score(area: float, normal: numpy.ndarray) -> float:
        """Score a candidate: lower is better. Combines area with alignment."""
        if not use_bias:
            return area
        alignment = abs(numpy.dot(normal, surface_normal))  # 0..1
        # Penalize planes ALIGNED with the surface normal: those graze the
        # surface (the plane is tangent to it) and produce misleading slivers.
        # Prefer planes perpendicular to the normal -- they cut ACROSS the
        # clicked feature (a neck/wrist), which is what the user wants.
        penalty = 1.0 + ALIGNMENT_BIAS * alignment
        return area * penalty

    n_theta = search_resolution
    n_phi = search_resolution * 2

    for i in range(n_theta):
        theta = numpy.pi * i / n_theta
        for j in range(n_phi):
            phi = 2 * numpy.pi * j / n_phi

            normal = plane_normal_from_spherical(theta, phi)
            samples_tested += 1

            try:
                section = mesh.section(plane_origin=plane_origin, plane_normal=normal)
                if section is not None:
                    path_2d, to_3D = _section_to_2d(section)
                    area = _local_section_area(path_2d, to_3D, plane_origin)

                    if collect_all_samples:
                        all_samples.append((normal.copy(), area))

                    if area >= min_section_area:
                        score = _compute_score(area, normal)
                        valid_candidates.append((score, area, normal.copy()))

                        if score < best_score:
                            best_score = score
                            best_area = area
                            best_normal = normal.copy()
            except Exception as e:
                if collect_all_samples:
                    all_samples.append((normal.copy(), float('nan')))
                logger.debug("Section failed for normal=%s: %s: %s", normal, type(e).__name__, e)
                continue

    # Fallback: if no valid sections found, try axis-aligned normals.
    if not numpy.isfinite(best_score) or best_area <= 0:
        logger.warning(
            "Smallest cut search found no valid sections (tested %d orientations). "
            "Trying axis-aligned fallback.",
            samples_tested,
        )
        for fallback_normal in [
            numpy.array([0.0, 1.0, 0.0]),
            numpy.array([1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        ]:
            try:
                section = mesh.section(plane_origin=plane_origin, plane_normal=fallback_normal)
                if section is not None:
                    path_2d, to_3D = _section_to_2d(section)
                    area = _local_section_area(path_2d, to_3D, plane_origin)
                    if 0 < area < float('inf'):
                        score = _compute_score(area, fallback_normal)
                        valid_candidates.append((score, area, fallback_normal.copy()))
                    if 0 < area < best_area:
                        best_area = area
                        best_normal = fallback_normal.copy()
            except Exception as e:
                logger.debug("Axis fallback failed for %s: %s", fallback_normal, e)

    # Sort candidates by score (area * alignment penalty).
    # Best candidates are aligned with the click AND have small area.
    valid_candidates.sort(key=lambda x: x[0])  # sort by score
    # Extract (area, normal) for downstream use
    top_candidates = [(area, normal) for _, area, normal
                      in valid_candidates[:100]]

    logger.debug("Smallest cut search: best_area=%.2f mm^2 (score=%.2f), normal=%s, "
                 "tested=%d orientations, %d valid candidates",
                 best_area, best_score, best_normal, samples_tested,
                 len(valid_candidates))

    return SmallestPlaneSearchResult(
        plane=CutPlane(origin=plane_origin, normal=best_normal),
        area=best_area,
        samples_tested=samples_tested,
        all_samples=all_samples,
        top_candidates=top_candidates,
    )


def find_plane_along_normal(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    surface_normal: numpy.ndarray,
    n_angles: int = 18,
) -> CutPlane:
    """Find a cut plane that CONTAINS the surface normal (cuts *along* the arrow).

    The plane is constrained to contain ``surface_normal`` (so its own normal is
    perpendicular to it), which geometrically excludes the grazing tangent plane.
    Rotating about the surface normal, the rotation with the smallest local
    cross-section is the natural neck/base of the clicked feature.

    Args:
        mesh: The trimesh object.
        click_position: 3D point the plane passes through.
        surface_normal: Surface normal at the click (the hover arrow direction).
        n_angles: Number of rotations to test about the surface normal.

    Returns:
        CutPlane through the click whose normal is perpendicular to surface_normal.
    """
    click = numpy.asarray(click_position, dtype=numpy.float64)
    axis = numpy.asarray(surface_normal, dtype=numpy.float64)
    axis_norm = numpy.linalg.norm(axis)
    if axis_norm < 1e-9:
        return CutPlane(origin=click, normal=numpy.array([0.0, 1.0, 0.0]))
    axis = axis / axis_norm

    # Orthonormal basis (u, v) spanning the plane perpendicular to the axis.
    u = numpy.array([1.0, 0.0, 0.0])
    if abs(float(numpy.dot(u, axis))) > 0.9:
        u = numpy.array([0.0, 0.0, 1.0])
    u = u - numpy.dot(u, axis) * axis
    u = u / numpy.linalg.norm(u)
    v = numpy.cross(axis, u)

    avg_face_area = mesh.area / max(len(mesh.faces), 1)
    min_area = avg_face_area * 5.0

    best_normal = None
    best_area = float("inf")
    for k in range(n_angles):
        theta = numpy.pi * k / n_angles
        # Candidate plane normal lies in the plane perpendicular to the axis,
        # so the cut plane itself contains the axis (the surface normal/arrow).
        candidate = numpy.cos(theta) * u + numpy.sin(theta) * v
        try:
            section = mesh.section(plane_origin=click, plane_normal=candidate)
            if section is None:
                continue
            path_2d, to_3D = _section_to_2d(section)
            area = _local_section_area(path_2d, to_3D, click)
        except Exception as e:
            logger.debug("along-normal section failed at theta=%.2f: %s", theta, e)
            continue
        if area >= min_area and area < best_area:
            best_area = area
            best_normal = candidate

    if best_normal is None:
        best_normal = u  # no real cross-section found; any plane containing the axis
    logger.debug("find_plane_along_normal: best_area=%.2f normal=%s",
                 best_area, best_normal)
    return CutPlane(origin=click, normal=best_normal)


def find_valley_cut_plane(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    search_resolution: int = 18,
    collect_all_samples: bool = False,
    surface_normal: Optional[numpy.ndarray] = None,
    sweep_fraction: float = 0.15,
    n_sweep_steps: int = 11,
    n_top_candidates: int = 20,
    use_sdf_bias: bool = False,
    anchor_points=None,
    debug_trace_path: Optional[str] = None,
) -> SmallestPlaneSearchResult:
    """
    Find a cut plane that follows a geographic valley or groove near the click.

    Unlike find_smallest_cut_plane (which pins the plane at the click position),
    this sweeps the plane along each candidate axis to find the local minimum
    in cross-section area near the click point.  This allows detecting grooves
    and valleys (like a figurine's neck) even when the click is slightly off
    from the narrowest point.

    The result is always a plane, so the cut line around the mesh is as
    straight (planar) as possible.

    Algorithm:
        Phase 1 (coarse): For each axis orientation, compute the cross-section
            at the click point.  Keep the top candidates by area.
        Phase 2 (sweep): For each top candidate, sweep the plane origin along
            the axis in a local window to find the position with minimum
            cross-section area.  This locates the groove center.

    Args:
        mesh: The trimesh object to analyze.
        click_position: 3D point near the valley/groove.
        search_resolution: Number of elevation angles (azimuth = 2x this).
        collect_all_samples: Store all (normal, area) pairs for debugging.
        surface_normal: Bias toward planes aligned with this direction.
        sweep_fraction: How far to sweep as fraction of mesh diagonal (default 15%).
        n_sweep_steps: Number of positions to test during the sweep phase.
        n_top_candidates: How many coarse candidates to refine in phase 2.
        use_sdf_bias: Experimental SDF-thinness bias (opt-in).
        anchor_points: Optional 1..N anchor points that must remain near the
            selected plane (strong intent anchoring for curved meshes).

    Returns:
        SmallestPlaneSearchResult with the best groove-following plane.
    """
    plane_origin = numpy.asarray(click_position, dtype=numpy.float64)
    trace = {
        "mode": "valley",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "click_position": plane_origin.copy(),
            "search_resolution": int(search_resolution),
            "collect_all_samples": bool(collect_all_samples),
            "sweep_fraction": float(sweep_fraction),
            "n_sweep_steps": int(n_sweep_steps),
            "n_top_candidates": int(n_top_candidates),
            "use_sdf_bias": bool(use_sdf_bias),
            "surface_normal": surface_normal.copy() if surface_normal is not None else None,
            "anchor_count": 0,
        },
        "coarse": {},
        "sdf": {},
        "result": {},
        "errors": [],
    }
    anchors = _as_anchor_array(anchor_points)
    trace["inputs"]["anchor_count"] = int(len(anchors))

    # Alignment bias (same as find_smallest_cut_plane)
    ALIGNMENT_BIAS = 2.0
    use_bias = (surface_normal is not None and
                numpy.linalg.norm(surface_normal) > 0.5)
    if use_bias:
        surface_normal = numpy.asarray(surface_normal, dtype=numpy.float64)
        surface_normal = surface_normal / numpy.linalg.norm(surface_normal)
        logger.debug("Valley cut search: using surface_normal=%s with bias=%.1f",
                     surface_normal, ALIGNMENT_BIAS)

    def _compute_score(area: float, normal: numpy.ndarray) -> float:
        if not use_bias:
            return area
        alignment = abs(numpy.dot(normal, surface_normal))
        # Penalize planes ALIGNED with the surface normal: those graze the
        # surface (the plane is tangent to it) and produce misleading slivers.
        # Prefer planes perpendicular to the normal -- they cut ACROSS the
        # clicked feature (a neck/wrist), which is what the user wants.
        penalty = 1.0 + ALIGNMENT_BIAS * alignment
        return area * penalty

    # Minimum area threshold (same as smallest mode)
    avg_face_area = mesh.area / max(len(mesh.faces), 1)
    min_section_area = avg_face_area * 5.0
    trace["coarse"]["avg_face_area"] = float(avg_face_area)
    trace["coarse"]["min_section_area"] = float(min_section_area)

    # Sweep distance: fraction of mesh diagonal
    extent = mesh.bounds[1] - mesh.bounds[0]
    mesh_diagonal = float(numpy.linalg.norm(extent))
    sweep_distance = mesh_diagonal * sweep_fraction
    trace["coarse"]["mesh_diagonal"] = float(mesh_diagonal)
    trace["coarse"]["sweep_distance"] = float(sweep_distance)

    logger.debug("Valley cut search: sweep_distance=%.2f (%.0f%% of diagonal %.2f), "
                 "%d sweep steps, resolution=%d",
                 sweep_distance, sweep_fraction * 100, mesh_diagonal,
                 n_sweep_steps, search_resolution)

    sdf_face_ids = numpy.array([], dtype=numpy.int64)
    sdf_face_centers = None
    sdf_face_thinness = None
    trace["sdf"] = {
        "enabled": bool(use_sdf_bias),
        "sampled_faces": 0,
        "nonzero_thin_faces": 0,
    }
    if use_sdf_bias and len(mesh.faces) > 0:
        try:
            face_centers = numpy.asarray(mesh.triangles_center, dtype=numpy.float64)
            center_d = numpy.linalg.norm(face_centers - plane_origin, axis=1)
            keep = int(min(len(face_centers), 3000))
            sdf_face_ids = numpy.argsort(center_d)[:keep]
            sdf_face_centers = face_centers[sdf_face_ids]
            sdf_face_thinness = _estimate_face_thinness_sdf(
                mesh,
                sdf_face_ids,
                max_faces=keep,
            )
            logger.debug(
                "Valley cut SDF bias: sampled=%d valid=%d",
                len(sdf_face_ids),
                int(numpy.count_nonzero(sdf_face_thinness[sdf_face_ids] > 0.0)),
            )
            trace["sdf"]["sampled_faces"] = int(len(sdf_face_ids))
            trace["sdf"]["nonzero_thin_faces"] = int(
                numpy.count_nonzero(sdf_face_thinness[sdf_face_ids] > 0.0)
            )
        except Exception as e:
            logger.debug("Valley cut SDF setup failed: %s", e)
            trace["errors"].append(f"sdf_setup: {e}")
            sdf_face_ids = numpy.array([], dtype=numpy.int64)
            sdf_face_centers = None
            sdf_face_thinness = None

    # ---- Phase 1: Coarse search (same as smallest mode) ----
    all_samples = [] if collect_all_samples else None
    coarse_candidates = []  # (score, area, normal) — above min_section_area
    coarse_all = []         # all valid sections (fallback for low-poly meshes)
    samples_tested = 0

    n_theta = search_resolution
    n_phi = search_resolution * 2

    for i in range(n_theta):
        theta = numpy.pi * i / n_theta
        for j in range(n_phi):
            phi = 2 * numpy.pi * j / n_phi
            normal = plane_normal_from_spherical(theta, phi)
            samples_tested += 1

            try:
                section = mesh.section(plane_origin=plane_origin,
                                       plane_normal=normal)
                if section is not None:
                    path_2d, to_3D = _section_to_2d(section)
                    area = _local_section_area(path_2d, to_3D, plane_origin)

                    if collect_all_samples:
                        all_samples.append((normal.copy(), area))

                    if area > 0:
                        score = _compute_score(area, normal)
                        if len(anchors) > 0:
                            coarse_dist = numpy.abs(numpy.dot(anchors - plane_origin, normal))
                            mean_dist = float(numpy.mean(coarse_dist))
                            max_dist = float(numpy.max(coarse_dist))
                            anchor_factor = (
                                1.0 +
                                anchor_weight_mean * (mean_dist / max(mesh_diagonal, 1e-6)) +
                                anchor_weight_max * (max_dist / max(mesh_diagonal, 1e-6))
                            )
                            score *= anchor_factor
                        coarse_all.append((score, area, normal.copy()))
                        if area >= min_section_area:
                            coarse_candidates.append((score, area,
                                                      normal.copy()))
            except Exception as e:
                if collect_all_samples:
                    all_samples.append((normal.copy(), float('nan')))
                logger.debug("Valley phase 1: section failed for normal=%s: %s",
                             normal, e)
                if len(trace["errors"]) < 50:
                    trace["errors"].append(f"phase1_section: {e}")
                continue

    # Sort by score and take top N for refinement.
    # For low-poly meshes (e.g. a box with 12 faces) min_section_area can
    # exceed the actual cross-section area, filtering out everything.
    # If that happens, fall back to all candidates without the threshold.
    coarse_candidates.sort(key=lambda x: x[0])
    top_coarse = coarse_candidates[:n_top_candidates]

    if not top_coarse and coarse_all:
        coarse_all.sort(key=lambda x: x[0])
        top_coarse = coarse_all[:n_top_candidates]
        logger.debug("Valley phase 1: min_section_area filter removed all "
                     "candidates, using %d unfiltered", len(top_coarse))
        trace["coarse"]["used_unfiltered_fallback"] = True
    else:
        trace["coarse"]["used_unfiltered_fallback"] = False

    logger.debug("Valley phase 1: %d valid candidates from %d samples, "
                 "refining top %d",
                 len(coarse_candidates), samples_tested, len(top_coarse))
    trace["coarse"]["samples_tested"] = int(samples_tested)
    trace["coarse"]["valid_candidates"] = int(len(coarse_candidates))
    trace["coarse"]["refined_candidates"] = int(len(top_coarse))
    trace["coarse"]["top_coarse_preview"] = [
        {
            "score": float(score),
            "area": float(area),
            "normal": normal.copy(),
        }
        for score, area, normal in top_coarse[: min(20, len(top_coarse))]
    ]
    anchor_weight_mean = 5.0
    anchor_weight_max = 2.0
    if len(anchors) > 0:
        trace["coarse"]["anchor_weight_mean"] = float(anchor_weight_mean)
        trace["coarse"]["anchor_weight_max"] = float(anchor_weight_max)

    # Anti-graze floor for sweep phase:
    # avoid selecting tiny near-tangent slivers that are much smaller than
    # the local coarse sections near the click.
    if top_coarse:
        coarse_area_median = float(numpy.median([area for _, area, _ in top_coarse]))
    else:
        coarse_area_median = float(min_section_area)
    sweep_min_area = max(min_section_area, 0.25 * coarse_area_median)
    if len(mesh.faces) >= 20000:
        # Large dense meshes are especially prone to grazing minima.
        sweep_min_area = max(sweep_min_area, min_section_area * 2.0)
    trace["coarse"]["coarse_area_median"] = float(coarse_area_median)
    trace["coarse"]["sweep_min_area"] = float(sweep_min_area)

    # ---- Phase 2: Sweep each top candidate to find the groove ----
    best_normal = numpy.array([0.0, 1.0, 0.0])
    best_origin = plane_origin.copy()
    best_score = float('inf')
    best_area = float('inf')
    valid_candidates = []
    refined_preview = []
    graze_fallback_candidates = 0

    for _, coarse_area, normal in top_coarse:
        # Sweep positions along this normal, centered at the click point
        offsets = numpy.linspace(-sweep_distance, sweep_distance, n_sweep_steps)

        sweep_best_area = float('inf')
        sweep_best_offset = 0.0
        sweep_best_any_area = float('inf')
        sweep_best_any_offset = 0.0

        for offset in offsets:
            sweep_origin = plane_origin + offset * normal
            try:
                section = mesh.section(plane_origin=sweep_origin,
                                       plane_normal=normal)
                if section is not None:
                    path_2d, to_3D = _section_to_2d(section)
                    area = _local_section_area(path_2d, to_3D, sweep_origin)

                    if area > 0:
                        if area < sweep_best_any_area:
                            sweep_best_any_area = area
                            sweep_best_any_offset = offset
                        if area >= sweep_min_area and area < sweep_best_area:
                            sweep_best_area = area
                            sweep_best_offset = offset
            except Exception:
                continue

        used_graze_fallback = False
        if not numpy.isfinite(sweep_best_area):
            # If no non-grazing section exists for this normal, keep a fallback
            # candidate but penalize it heavily so true groove sections win.
            if not numpy.isfinite(sweep_best_any_area):
                continue
            sweep_best_area = sweep_best_any_area
            sweep_best_offset = sweep_best_any_offset
            used_graze_fallback = True
            graze_fallback_candidates += 1

        # Score the swept result: area * alignment bias * proximity bias.
        # Strongly discourage sweep-edge picks that can detach from clicked intent.
        offset_ratio = abs(sweep_best_offset) / max(sweep_distance, 1e-6)
        proximity_penalty = 1.0 + 0.25 * offset_ratio + 1.5 * (offset_ratio ** 2)
        if offset_ratio > 0.85:
            proximity_penalty *= 1.4
        score = _compute_score(sweep_best_area, normal) * proximity_penalty
        if used_graze_fallback:
            score *= 2.5
        swept_origin = plane_origin + sweep_best_offset * normal
        if len(anchors) > 0:
            anchor_dist = numpy.abs(numpy.dot(anchors - swept_origin, normal))
            mean_dist = float(numpy.mean(anchor_dist))
            max_dist = float(numpy.max(anchor_dist))
            anchor_factor = (
                1.0 +
                anchor_weight_mean * (mean_dist / max(mesh_diagonal, 1e-6)) +
                anchor_weight_max * (max_dist / max(mesh_diagonal, 1e-6))
            )
            score *= anchor_factor
        else:
            mean_dist = 0.0
            max_dist = 0.0
            anchor_factor = 1.0

        if (
            use_sdf_bias and
            sdf_face_thinness is not None and
            sdf_face_centers is not None and
            len(sdf_face_ids) > 0
        ):
            try:
                local_idx = int(
                    numpy.argmin(
                        numpy.linalg.norm(sdf_face_centers - swept_origin, axis=1)
                    )
                )
                thinness = float(sdf_face_thinness[sdf_face_ids[local_idx]])
                # Thin regions should win against similarly-sized sections.
                sdf_factor = max(0.65, 1.0 - 0.35 * thinness)
                score *= sdf_factor
            except Exception:
                pass

        valid_candidates.append((score, sweep_best_area, normal.copy(),
                                 swept_origin.copy()))
        if len(refined_preview) < 40:
            refined_preview.append({
                "score": float(score),
                "area": float(sweep_best_area),
                "offset": float(sweep_best_offset),
                "offset_ratio": float(offset_ratio),
                "proximity_penalty": float(proximity_penalty),
                "anchor_mean_dist": float(mean_dist),
                "anchor_max_dist": float(max_dist),
                "anchor_factor": float(anchor_factor),
                "used_graze_fallback": bool(used_graze_fallback),
                "normal": normal.copy(),
                "origin": swept_origin.copy(),
            })

        if score < best_score:
            best_score = score
            best_area = sweep_best_area
            best_normal = normal.copy()
            best_origin = swept_origin.copy()

    # Fallback: axis-aligned normals (same as find_smallest_cut_plane)
    if not numpy.isfinite(best_score):
        logger.warning("Valley sweep found no valid sections; trying axis "
                       "fallback")
        for fallback_normal in [
            numpy.array([0.0, 1.0, 0.0]),
            numpy.array([1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        ]:
            for offset in numpy.linspace(-sweep_distance, sweep_distance,
                                         n_sweep_steps):
                sweep_origin = plane_origin + offset * fallback_normal
                try:
                    section = mesh.section(plane_origin=sweep_origin,
                                           plane_normal=fallback_normal)
                    if section is not None:
                        path_2d, to_3D = _section_to_2d(section)
                        area = _local_section_area(path_2d, to_3D,
                                                   sweep_origin)
                        if 0 < area < best_area:
                            best_area = area
                            best_normal = fallback_normal.copy()
                            best_origin = sweep_origin.copy()
                            best_score = _compute_score(area, fallback_normal)
                            valid_candidates.append(
                                (best_score, area, fallback_normal.copy(),
                                 sweep_origin.copy()))
                except Exception:
                    continue

    # Sort candidates and extract top for downstream fallback
    valid_candidates.sort(key=lambda x: x[0])
    top_candidates = [(area, normal) for _, area, normal, _ in
                      valid_candidates[:100]]

    logger.debug("Valley cut search: best_area=%.2f mm^2 (score=%.2f), "
                 "normal=%s, origin_offset=%.2f, tested=%d orientations, "
                 "%d refined candidates",
                 best_area, best_score, best_normal,
                 float(numpy.linalg.norm(best_origin - plane_origin)),
                 samples_tested, len(valid_candidates))
    trace["result"] = {
        "best_area": float(best_area),
        "best_score": float(best_score),
        "best_normal": best_normal.copy(),
        "best_origin": best_origin.copy(),
        "origin_offset": float(numpy.linalg.norm(best_origin - plane_origin)),
        "refined_candidates": int(len(valid_candidates)),
        "graze_fallback_candidates": int(graze_fallback_candidates),
        "top_candidates_preview": [
            {
                "area": float(area),
                "normal": normal.copy(),
            }
            for area, normal in top_candidates[: min(20, len(top_candidates))]
        ],
        "refined_preview": refined_preview,
    }
    _write_json_trace(debug_trace_path, trace)

    return SmallestPlaneSearchResult(
        plane=CutPlane(origin=best_origin, normal=best_normal),
        area=best_area,
        samples_tested=samples_tested,
        all_samples=all_samples,
        top_candidates=top_candidates,
    )


def find_valley_seam_partition(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    source_face_hint: Optional[int] = None,
    target_face_hint: Optional[int] = None,
    surface_normal: Optional[numpy.ndarray] = None,
    use_sdf_bias: bool = False,
    anchor_points=None,
    debug_trace_path: Optional[str] = None,
) -> Tuple[list, list, int, int]:
    """
    Compute a seam-based valley partition that prefers concave surface paths.

    Unlike planar valley mode, this algorithm works directly on the face graph:
    - edge costs are reduced on concave adjacencies,
    - edge costs increase away from the click region,
    - dual-Dijkstra + threshold sweep finds a short seam-like partition.

    The computation runs in a worker thread with a hard timeout, matching the
    behavior of radial/shortest seam search.

    Optional multi-point controls:
      - target_face_hint: explicit sink face hint (e.g., from a second click).
      - anchor_points: 1..N points to bias seam locality toward the clicked
        feature corridor.
    """
    import threading

    HARD_TIMEOUT_SEC = 15.0
    result_holder = [None]
    error_holder = [None]
    cancel_event = threading.Event()

    def _worker():
        try:
            result_holder[0] = _find_valley_seam_partition_impl(
                mesh,
                click_position,
                source_face_hint,
                target_face_hint,
                surface_normal,
                use_sdf_bias,
                anchor_points,
                debug_trace_path,
                cancel_event,
            )
        except Exception as e:
            error_holder[0] = e

    logger.info(
        "find_valley_seam_partition: Starting worker thread (timeout=%.1fs)",
        HARD_TIMEOUT_SEC,
    )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=HARD_TIMEOUT_SEC)

    if t.is_alive():
        cancel_event.set()
        msg = (
            f"Valley seam search timed out after {HARD_TIMEOUT_SEC}s "
            "(thread still running, cancel sent)."
        )
        logger.warning(msg)
        _write_json_trace(debug_trace_path, {
            "mode": "valley_seam",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "timeout",
            "error": msg,
        })
        raise TimeoutError(msg)

    if error_holder[0] is not None:
        _write_json_trace(debug_trace_path, {
            "mode": "valley_seam",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "error": str(error_holder[0]),
        })
        raise error_holder[0]

    if result_holder[0] is None:
        raise RuntimeError("Valley seam search returned no result")

    logger.info("find_valley_seam_partition: Worker thread completed successfully")
    return result_holder[0]


def _find_valley_seam_partition_impl(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    source_face_hint: Optional[int] = None,
    target_face_hint: Optional[int] = None,
    surface_normal: Optional[numpy.ndarray] = None,
    use_sdf_bias: bool = False,
    anchor_points=None,
    debug_trace_path: Optional[str] = None,
    cancel_event=None,
) -> Tuple[list, list, int, int]:
    import heapq
    _log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "valley_seam_debug.log",
    )

    def _log(msg: str) -> None:
        try:
            logger.info(msg)
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except Exception:
            pass

    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("Valley seam search cancelled")

    point = numpy.asarray(click_position, dtype=numpy.float64).reshape(1, -1)
    face_index = source_face_hint
    trace = {
        "mode": "valley_seam",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "click_position": point.reshape(3).copy(),
            "source_face_hint": source_face_hint,
            "target_face_hint": target_face_hint,
            "surface_normal": surface_normal.copy() if surface_normal is not None else None,
            "use_sdf_bias": bool(use_sdf_bias),
            "anchor_count": 0,
        },
        "mesh": {},
        "sdf": {"enabled": bool(use_sdf_bias)},
        "source_sink": {},
        "threshold_sweep": {},
        "result": {},
    }
    anchors = _as_anchor_array(anchor_points)
    trace["inputs"]["anchor_count"] = int(len(anchors))
    _log(
        "valley_seam start: faces=%d source_hint=%s sdf=%s" %
        (len(mesh.faces), str(source_face_hint), str(use_sdf_bias))
    )

    if face_index is None:
        try:
            from trimesh.proximity import ProximityQuery

            pq = ProximityQuery(mesh)
            _, _, face_ids = pq.on_surface(point)
            if face_ids is not None and len(face_ids) > 0:
                face_index = int(face_ids[0])
        except Exception as e:
            logger.debug("valley_seam: proximity query failed: %s", e)

    if face_index is None:
        if mesh.vertices.shape[0] > 0:
            distances = numpy.linalg.norm(mesh.vertices - point, axis=1)
            nearest_idx = int(numpy.argmin(distances))
            faces_with_vertex = numpy.where(mesh.faces == nearest_idx)[0]
            face_index = int(faces_with_vertex[0]) if faces_with_vertex.size > 0 else 0
        else:
            face_index = 0
    trace["source_sink"]["source_face"] = int(face_index)
    _log(f"valley_seam source face: {face_index}")

    faces_count = len(mesh.faces)
    if faces_count < 4:
        raise RuntimeError("valley_seam: mesh too small for partitioning")

    adj_pairs = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    num_adj = len(adj_pairs)
    if num_adj == 0:
        raise RuntimeError("valley_seam: mesh has no face adjacency")
    trace["mesh"]["faces_count"] = int(faces_count)
    trace["mesh"]["adjacency_count"] = int(num_adj)

    f1_arr = adj_pairs[:, 0].astype(numpy.int32)
    f2_arr = adj_pairs[:, 1].astype(numpy.int32)

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    c1 = face_centroids[f1_arr]
    c2 = face_centroids[f2_arr]
    centroid_dists = numpy.linalg.norm(c1 - c2, axis=1)
    base_edge_lengths = numpy.maximum(centroid_dists, 1e-8)

    # Concavity-aware edge weighting.
    concave_strength = numpy.zeros(num_adj, dtype=numpy.float64)
    convex_strength = numpy.zeros(num_adj, dtype=numpy.float64)
    concave_edge_ratio = 0.0
    try:
        angles = numpy.asarray(mesh.face_adjacency_angles, dtype=numpy.float64)
        convex_mask = numpy.asarray(mesh.face_adjacency_convex, dtype=bool)
        if len(angles) == num_adj and len(convex_mask) == num_adj:
            normalized_angle = numpy.clip(angles / (numpy.pi / 2.0), 0.0, 1.0)
            concave_strength[~convex_mask] = normalized_angle[~convex_mask]
            convex_strength[convex_mask] = normalized_angle[convex_mask]
            concave_edge_ratio = float(numpy.count_nonzero(~convex_mask)) / float(num_adj)
    except Exception as e:
        logger.debug("valley_seam: concavity extraction failed: %s", e)
    trace["mesh"]["concave_edge_ratio"] = float(concave_edge_ratio)

    click = point.reshape(3)
    edge_midpoints = 0.5 * (c1 + c2)
    mesh_diag = float(numpy.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    mesh_diag = max(mesh_diag, 1e-6)
    locality_norm = numpy.linalg.norm(edge_midpoints - click, axis=1) / mesh_diag
    if len(anchors) > 0:
        anchor_mid_d = numpy.min(
            numpy.linalg.norm(
                edge_midpoints[:, numpy.newaxis, :] - anchors[numpy.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        ) / mesh_diag
        locality_norm = numpy.minimum(locality_norm, anchor_mid_d)
    edge_thinness = numpy.zeros(num_adj, dtype=numpy.float64)
    trace["mesh"]["mesh_diagonal"] = float(mesh_diag)

    if use_sdf_bias:
        try:
            face_click_d = numpy.linalg.norm(face_centroids - click, axis=1)
            local_radius = mesh_diag * 0.45
            local_face_ids = numpy.where(face_click_d <= local_radius)[0]
            if local_face_ids.size == 0:
                keep = int(min(faces_count, 1500))
                local_face_ids = numpy.argsort(face_click_d)[:keep]
            elif local_face_ids.size > 3000:
                order = numpy.argsort(face_click_d[local_face_ids])
                local_face_ids = local_face_ids[order[:3000]]

            face_thinness = _estimate_face_thinness_sdf(
                mesh,
                local_face_ids,
                max_faces=3000,
            )
            edge_thinness = 0.5 * (face_thinness[f1_arr] + face_thinness[f2_arr])
            logger.debug(
                "valley_seam SDF bias: local_faces=%d, nonzero_faces=%d",
                len(local_face_ids),
                int(numpy.count_nonzero(face_thinness > 0.0)),
            )
            trace["sdf"]["local_radius"] = float(local_radius)
            trace["sdf"]["local_faces"] = int(len(local_face_ids))
            trace["sdf"]["nonzero_faces"] = int(numpy.count_nonzero(face_thinness > 0.0))
        except Exception as e:
            logger.debug("valley_seam: SDF bias setup failed: %s", e)
            edge_thinness = numpy.zeros(num_adj, dtype=numpy.float64)
            trace["sdf"]["error"] = str(e)

    # Favor concave edges and keep seams local to the click neighborhood.
    # Lower weights are preferred by Dijkstra.
    geo_weights_per_adj = base_edge_lengths.copy()
    geo_weights_per_adj *= (1.0 + 0.55 * convex_strength)
    geo_weights_per_adj /= (1.0 + 2.6 * concave_strength)
    if use_sdf_bias:
        # Thin regions should be easier to traverse when tracing groove seams.
        geo_weights_per_adj /= (1.0 + 1.2 * edge_thinness)
    geo_weights_per_adj *= (1.0 + 0.75 * locality_norm)
    geo_weights_per_adj = numpy.maximum(geo_weights_per_adj, base_edge_lengths * 0.05)

    # Build CSR adjacency from weighted undirected face graph.
    edge_count_per_node = numpy.zeros(faces_count, dtype=numpy.int32)
    for i in range(num_adj):
        edge_count_per_node[f1_arr[i]] += 1
        edge_count_per_node[f2_arr[i]] += 1

    offsets = numpy.zeros(faces_count + 1, dtype=numpy.int32)
    offsets[1:] = numpy.cumsum(edge_count_per_node)
    total_geo_edges = int(offsets[faces_count])

    geo_targets = numpy.zeros(total_geo_edges, dtype=numpy.int32)
    geo_weights = numpy.zeros(total_geo_edges, dtype=numpy.float64)
    fill_pos = offsets[:-1].copy()

    for i in range(num_adj):
        u, v = int(f1_arr[i]), int(f2_arr[i])
        w = float(geo_weights_per_adj[i])

        pos_u = fill_pos[u]
        geo_targets[pos_u] = v
        geo_weights[pos_u] = w
        fill_pos[u] += 1

        pos_v = fill_pos[v]
        geo_targets[pos_v] = u
        geo_weights[pos_v] = w
        fill_pos[v] += 1

    def _run_dijkstra(start_face: int) -> numpy.ndarray:
        dist = numpy.full(faces_count, numpy.inf, dtype=numpy.float64)
        dist[start_face] = 0.0
        pq_heap = [(0.0, int(start_face))]
        pops = 0
        while pq_heap:
            d, u = heapq.heappop(pq_heap)
            pops += 1
            if pops % 2000 == 0:
                _check_cancel()
            if d > dist[u]:
                continue
            for e_idx in range(offsets[u], offsets[u + 1]):
                v = int(geo_targets[e_idx])
                nd = d + geo_weights[e_idx]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq_heap, (nd, v))
        return dist

    dist_src = _run_dijkstra(face_index)
    finite_mask = numpy.isfinite(dist_src)
    if not numpy.any(finite_mask):
        raise RuntimeError("valley_seam: source Dijkstra produced no finite distances")

    max_dist = float(numpy.max(dist_src[finite_mask]))
    if max_dist <= 1e-12:
        max_dist = 1.0

    sink_face = None
    if target_face_hint is not None:
        tfi = int(target_face_hint)
        if 0 <= tfi < faces_count and tfi != face_index:
            sink_face = tfi
            trace["source_sink"]["sink_seed"] = "target_face_hint"
    if sink_face is None and len(anchors) >= 2:
        target_point = anchors[-1]
        try:
            from trimesh.proximity import ProximityQuery
            pq = ProximityQuery(mesh)
            _, _, target_face_ids = pq.on_surface(target_point.reshape(1, -1))
            if target_face_ids is not None and len(target_face_ids) > 0:
                tfi = int(target_face_ids[0])
            else:
                tfi = -1
        except Exception:
            d = numpy.linalg.norm(mesh.triangles_center - target_point, axis=1)
            tfi = int(numpy.argmin(d)) if len(d) > 0 else -1
        if 0 <= tfi < faces_count and tfi != face_index:
            sink_face = tfi
            trace["source_sink"]["sink_seed"] = "anchor_last_point"

    if sink_face is None and surface_normal is not None and len(mesh.triangles_center) > 0:
        normal = numpy.asarray(surface_normal, dtype=numpy.float64)
        normal_norm = numpy.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal = normal / normal_norm
            norm_dist = numpy.where(finite_mask, dist_src / max_dist, 0.0)

            vecs = mesh.triangles_center - click
            norms = numpy.linalg.norm(vecs, axis=1)
            euclid_ratio = norms / max(mesh_diag, 1e-6)
            valid = norms > 1e-6
            if numpy.any(valid):
                vecs[valid] = vecs[valid] / norms[valid, numpy.newaxis]
                align_score = numpy.full_like(norm_dist, -1.0)
                align_score[valid] = numpy.dot(vecs[valid], -normal)

                candidate_mask = finite_mask & (norm_dist >= 0.15)
                max_sink_euclid_ratio = 0.24 if faces_count >= 20000 else 0.40
                candidate_mask &= (euclid_ratio <= max_sink_euclid_ratio)
                trace["source_sink"]["max_sink_euclid_ratio"] = float(max_sink_euclid_ratio)
                combined = numpy.full_like(norm_dist, -numpy.inf)
                if numpy.any(candidate_mask):
                    combined[candidate_mask] = (
                        (0.60 * norm_dist[candidate_mask]) +
                        (0.40 * align_score[candidate_mask])
                    )
                else:
                    # If no local candidate survives the Euclidean cap, relax it.
                    relaxed_mask = finite_mask & (norm_dist >= 0.15)
                    if numpy.any(relaxed_mask):
                        combined[relaxed_mask] = (
                            (0.60 * norm_dist[relaxed_mask]) +
                            (0.40 * align_score[relaxed_mask])
                        )
                    else:
                        combined[finite_mask] = norm_dist[finite_mask]
                sink_face = int(numpy.argmax(combined))

    if sink_face is None:
        finite_dist = numpy.where(finite_mask, dist_src, -1.0)
        sink_face = int(numpy.argmax(finite_dist))

    if sink_face == face_index:
        finite_dist = numpy.where(finite_mask, dist_src, -1.0)
        finite_dist[face_index] = -1.0
        sink_face = int(numpy.argmax(finite_dist))
    trace["source_sink"]["sink_face"] = int(sink_face)
    _log(f"valley_seam sink face: {sink_face}")

    dist_sink = _run_dijkstra(sink_face)

    d_src = numpy.where(numpy.isfinite(dist_src), dist_src, max_dist)
    d_snk = numpy.where(numpy.isfinite(dist_sink), dist_sink, max_dist)
    total_d = d_src + d_snk
    safe_total = numpy.where(total_d > 1e-12, total_d, 1.0)
    score = d_src / safe_total

    # Threshold sweep with cost preferring short, concave, local boundaries.
    # On large, smooth meshes (very low explicit concave-edge ratio), tiny local
    # cuts can win purely on perimeter; guard with an adaptive second pass below.
    thresholds = numpy.linspace(0.05, 0.45, 25)

    s1 = score[f1_arr]
    s2 = score[f2_arr]
    score_lo = numpy.minimum(s1, s2)
    score_hi = numpy.maximum(s1, s2)

    boundary_edge_lengths = numpy.linalg.norm(
        mesh.vertices[adj_edges[:, 0]] - mesh.vertices[adj_edges[:, 1]],
        axis=1,
    )
    threshold_samples = []

    if faces_count >= 20000:
        min_ratio_base = 0.05
    elif faces_count >= 8000:
        min_ratio_base = 0.035
    else:
        min_ratio_base = 0.02
    max_ratio_base = 0.40
    target_ratio = 0.12 if faces_count >= 20000 else 0.10

    def _ratio_penalty(ratio: float) -> float:
        # Harder penalty below the minimum useful partition size.
        if ratio < min_ratio_base:
            return 1.0 + 4.0 * ((min_ratio_base - ratio) / max(min_ratio_base, 1e-6)) ** 2
        # Mild penalty above broad upper bound to avoid half-object splits.
        if ratio > max_ratio_base:
            return 1.0 + 1.5 * ((ratio - max_ratio_base) / max(1.0 - max_ratio_base, 1e-6)) ** 2
        # Soft pull toward a feature-sized target partition.
        return 1.0 + 0.6 * (abs(ratio - target_ratio) / max(target_ratio, 1e-6))

    def _select_threshold(min_ratio: float, max_ratio: Optional[float] = None):
        min_faces_threshold = max(10, int(faces_count * min_ratio))
        best_thresh_local = None
        best_cost_local = float("inf")
        best_small_local = 0

        for t in thresholds:
            cross_mask = (score_lo < t) & (score_hi >= t)
            len_a = int(numpy.sum(score < t))
            len_b = faces_count - len_a
            if len_a < min_faces_threshold or len_b < min_faces_threshold:
                continue
            if not numpy.any(cross_mask):
                continue

            small_faces = min(len_a, len_b)
            ratio = float(small_faces) / float(faces_count)
            if max_ratio is not None and ratio > max_ratio:
                continue

            perimeter = float(numpy.sum(boundary_edge_lengths[cross_mask]))
            mean_concavity = float(numpy.mean(concave_strength[cross_mask]))
            mean_locality = float(numpy.mean(locality_norm[cross_mask]))
            mean_thinness = float(numpy.mean(edge_thinness[cross_mask])) if use_sdf_bias else 0.0

            concavity_factor = max(0.2, 1.0 - 0.65 * mean_concavity)
            locality_factor = 1.0 + 0.35 * mean_locality
            sdf_factor = max(0.65, 1.0 - 0.35 * mean_thinness) if use_sdf_bias else 1.0
            ratio_factor = _ratio_penalty(ratio)
            cost = perimeter * concavity_factor * locality_factor * sdf_factor * ratio_factor
            if len(threshold_samples) < 120:
                threshold_samples.append({
                    "threshold": float(t),
                    "len_a": int(len_a),
                    "len_b": int(len_b),
                    "small_ratio": float(ratio),
                    "perimeter": float(perimeter),
                    "mean_concavity": float(mean_concavity),
                    "mean_locality": float(mean_locality),
                    "mean_thinness": float(mean_thinness),
                    "ratio_factor": float(ratio_factor),
                    "cost": float(cost),
                })

            if cost < best_cost_local:
                best_cost_local = cost
                best_thresh_local = float(t)
                best_small_local = small_faces

        return best_thresh_local, best_cost_local, best_small_local

    best_thresh, best_cost, best_small = _select_threshold(
        min_ratio=min_ratio_base,
        max_ratio=max_ratio_base,
    )

    # Smooth, high-face meshes often have near-zero explicit concave edges.
    # In that regime, tiny local partitions can dominate on perimeter alone.
    # If that happens, re-run with a slightly larger minimum partition and
    # exclude near-half splits to keep the cut focused.
    smooth_guard_used = False
    if best_thresh is not None:
        best_ratio = float(best_small) / float(faces_count)
        use_smooth_guard = (
            faces_count >= 8000 and
            concave_edge_ratio < 0.002 and
            best_ratio < 0.05
        )
        if use_smooth_guard:
            alt_thresh, alt_cost, alt_small = _select_threshold(
                min_ratio=0.05,
                max_ratio=0.35
            )
            if alt_thresh is not None:
                logger.debug(
                    "valley_seam: smooth-mesh guard adjusted threshold "
                    "(concave_ratio=%.4f, ratio %.3f -> %.3f)",
                    concave_edge_ratio,
                    best_ratio,
                    float(alt_small) / float(faces_count),
                )
                best_thresh = alt_thresh
                best_cost = alt_cost
                best_small = alt_small
                smooth_guard_used = True

    if best_thresh is None:
        best_thresh = 0.25
        logger.debug("valley_seam: no valid threshold found, fallback to score=0.25")
        _log("valley_seam threshold fallback -> 0.25")

    mask_a = score < best_thresh
    mask_a[face_index] = True
    set_a = numpy.where(mask_a)[0].tolist()
    set_b = numpy.where(~mask_a)[0].tolist()

    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    # Keep refinement for smaller meshes where it meaningfully improves the seam.
    refined = False
    if faces_count <= 4000:
        set_a, set_b = refine_partition_with_mincut(mesh, set_a, set_b, score, best_thresh)
        refined = True
    final_ratio = float(len(set_a)) / float(faces_count) if faces_count > 0 else 0.0

    trace["threshold_sweep"] = {
        "chosen_threshold": float(best_thresh),
        "best_cost": float(best_cost) if numpy.isfinite(best_cost) else None,
        "best_small_faces": int(best_small),
        "min_ratio_base": float(min_ratio_base),
        "max_ratio_base": float(max_ratio_base),
        "target_ratio": float(target_ratio),
        "smooth_guard_used": bool(smooth_guard_used),
        "samples": threshold_samples,
    }
    trace["result"] = {
        "set_a_size": int(len(set_a)),
        "set_b_size": int(len(set_b)),
        "set_a_ratio": float(final_ratio),
        "source_face": int(face_index),
        "sink_face": int(sink_face),
        "refined_with_mincut": bool(refined),
    }
    _write_json_trace(debug_trace_path, trace)
    _log(
        "valley_seam done: threshold=%.4f ratio=%.3f A=%d B=%d src=%d sink=%d" %
        (float(best_thresh), float(final_ratio), len(set_a), len(set_b), int(face_index), int(sink_face))
    )

    return set_a, set_b, face_index, sink_face


def find_shortest_seam_partition(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    source_face_hint: Optional[int] = None,
    surface_normal: Optional[numpy.ndarray] = None,
) -> Tuple[list, list, int, int]:
    """
    Compute a geodesic shortest-seam cut. Wraps the heavy computation in a
    daemon thread with a hard 10-second timeout to prevent UI freezes.

    If the computation doesn't finish in time, raises TimeoutError.
    The caller in ObjectSplitter.py catches Exception and falls back to
    a plane cut.
    """
    import threading

    HARD_TIMEOUT_SEC = 10.0
    result_holder = [None]   # mutable container to capture thread result
    error_holder = [None]    # mutable container to capture thread exception
    cancel_event = threading.Event()  # signal worker to stop on timeout

    def _worker():
        try:
            result_holder[0] = _find_shortest_seam_partition_impl(
                mesh, click_position, source_face_hint, surface_normal,
                cancel_event
            )
        except Exception as e:
            error_holder[0] = e

    logger.info("find_shortest_seam_partition: Starting worker thread (timeout=%.1fs)", HARD_TIMEOUT_SEC)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=HARD_TIMEOUT_SEC)

    if t.is_alive():
        # Signal the worker to stop, then raise timeout
        cancel_event.set()
        msg = f"Radial search timed out after {HARD_TIMEOUT_SEC}s (thread still running, cancel sent). Falling back."
        logger.warning(msg)
        raise TimeoutError(msg)

    if error_holder[0] is not None:
        raise error_holder[0]

    if result_holder[0] is None:
        raise RuntimeError("Radial search returned no result")

    logger.info("find_shortest_seam_partition: Worker thread completed successfully")
    return result_holder[0]


def _find_shortest_seam_partition_impl(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    source_face_hint: Optional[int] = None,
    surface_normal: Optional[numpy.ndarray] = None,
    cancel_event=None,
) -> Tuple[list, list, int, int]:
    """Internal implementation of the radial search. Runs in a worker thread.
    
    Optimized for large meshes (20k+ faces) using:
    - Numpy arrays for flow graph edges (instead of Python dicts)
    - Numpy array for Dijkstra distances
    - Iterative DFS (instead of recursive)
    """
    import heapq
    import time
    import os
    
    # File-based logging — neither print() nor logger output appears from
    # daemon threads in Cura's embedded Python environment.
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radial_debug.log")
    def _log(msg):
        try:
            logger.info(msg)
            with open(_log_path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except Exception:
            pass
    
    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("Cancelled by main thread")
    
    t0 = time.perf_counter()
    _log(f"ENTERED. faces={len(mesh.faces)}")

    # 1. Identify Source Face
    face_index = source_face_hint
    point = numpy.asarray(click_position, dtype=numpy.float64).reshape(1, -1)
    
    if face_index is None:
        try:
            from trimesh.proximity import ProximityQuery
            pq = ProximityQuery(mesh)
            _, _, face_ids = pq.on_surface(point)
            if face_ids is not None and len(face_ids) > 0:
                face_index = int(face_ids[0])
        except Exception as e:
            logger.debug("Proximity query failed: %s", e)

    if face_index is None:
        if mesh.vertices.shape[0] > 0:
            distances = numpy.linalg.norm(mesh.vertices - point, axis=1)
            nearest_idx = int(numpy.argmin(distances))
            faces_with_vertex = numpy.where(mesh.faces == nearest_idx)[0]
            face_index = int(faces_with_vertex[0]) if faces_with_vertex.size > 0 else 0
        else:
            face_index = 0

    faces_count = len(mesh.faces)
    _log(f"  Source face found: {face_index}. faces_count={faces_count}. Accessing adjacency...")
    t1 = time.perf_counter()
    adj_pairs = mesh.face_adjacency          # shape (M, 2)  -- may trigger trimesh lazy computation!
    adj_edges = mesh.face_adjacency_edges    # shape (M, 2)
    num_adj = len(adj_pairs)
    _log(f"  Adjacency access took {time.perf_counter()-t1:.3f}s. {num_adj} adj pairs.")

    # 2. Geometry: centroids and edge lengths (vectorized numpy)
    t2 = time.perf_counter()
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    edge_lengths = numpy.linalg.norm(
        mesh.vertices[adj_edges[:, 0]] - mesh.vertices[adj_edges[:, 1]], axis=1
    )
    c1 = face_centroids[adj_pairs[:, 0]]
    c2 = face_centroids[adj_pairs[:, 1]]
    centroid_dists = numpy.linalg.norm(c1 - c2, axis=1)
    _log(f"  Geometry computed in {time.perf_counter()-t2:.3f}s. Proceeding to Dijkstra...")

    # 3. Dijkstra — numpy array for dist, Python heapq for priority queue
    # Build adjacency list using numpy arrays for neighbor storage
    # head[u] = index of first edge from u, next_edge[e] = index of next edge from same source
    # This avoids creating 22k Python lists of tuples
    
    # Count edges per node (each adj pair creates 2 directed edges)
    edge_count_per_node = numpy.zeros(faces_count, dtype=numpy.int32)
    f1_arr = adj_pairs[:, 0].astype(numpy.int32)
    f2_arr = adj_pairs[:, 1].astype(numpy.int32)
    for i in range(num_adj):
        edge_count_per_node[f1_arr[i]] += 1
        edge_count_per_node[f2_arr[i]] += 1
    
    # Build CSR-like adjacency: offsets array + targets + weights
    offsets = numpy.zeros(faces_count + 1, dtype=numpy.int32)
    offsets[1:] = numpy.cumsum(edge_count_per_node)
    total_geo_edges = int(offsets[faces_count])
    
    geo_targets = numpy.zeros(total_geo_edges, dtype=numpy.int32)
    geo_weights = numpy.zeros(total_geo_edges, dtype=numpy.float64)
    fill_pos = offsets[:-1].copy()
    
    for i in range(num_adj):
        u, v = int(f1_arr[i]), int(f2_arr[i])
        w = float(centroid_dists[i])
        pos_u = fill_pos[u]
        geo_targets[pos_u] = v
        geo_weights[pos_u] = w
        fill_pos[u] += 1
        
        pos_v = fill_pos[v]
        geo_targets[pos_v] = u
        geo_weights[pos_v] = w
        fill_pos[v] += 1

    # Dijkstra with numpy dist array
    t3 = time.perf_counter()
    dist = numpy.full(faces_count, numpy.inf, dtype=numpy.float64)
    dist[face_index] = 0.0
    pq_heap = [(0.0, face_index)]
    
    dijkstra_pops = 0
    while pq_heap:
        d, u = heapq.heappop(pq_heap)
        dijkstra_pops += 1
        if dijkstra_pops % 2000 == 0:
            _check_cancel()
        if d > dist[u]:
            continue
        start_e = offsets[u]
        end_e = offsets[u + 1]
        for e_idx in range(start_e, end_e):
            v = int(geo_targets[e_idx])
            nd = d + geo_weights[e_idx]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq_heap, (nd, v))

    max_dist = float(numpy.max(dist[numpy.isfinite(dist)])) if numpy.any(numpy.isfinite(dist)) else 1.0
    _log(f"  Dijkstra complete in {time.perf_counter()-t3:.3f}s. max_dist={max_dist:.2f}")

    # 4. Smart Sink Selection (fully vectorized)
    # Goal: find a face that is perfectly on the opposite side of the clicked volume.
    t4 = time.perf_counter()
    sink_face = None
    finite_mask = numpy.isfinite(dist)
    
    # 3. Sink selection (Heuristic: opposite side preferred, or just farthest)
    # The Dual-Dijkstra sweep functions best when Source and Sink are opposite "Poles" of the mesh.
    # This allows the distance field `d_src / (d_src + d_snk)` to flow along the primary axis,
    # forming parallel horizontal cross-sections that the cost-evaluator sweeps for minimum perimeters.
    sink_face = None
    dist_sink = None
    max_d = numpy.max(dist[finite_mask])
    
    # Base sink on geodesic distance and (if provided) click-direction anti-alignment.
    # Opposite surface normals should steer sink selection to different regions.
    if surface_normal is not None and len(mesh.triangles_center) > 0:
        normal = numpy.asarray(surface_normal, dtype=numpy.float64)
        normal_norm = numpy.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal = normal / normal_norm
        else:
            normal = None

        face_centers = mesh.triangles_center
        norm_dist = numpy.where(finite_mask, dist / max(max_d, 1e-12), 0.0)

        if normal is not None:
            # Vector from click/source region to each face center
            vecs = face_centers - point
            norms = numpy.linalg.norm(vecs, axis=1)
            valid = norms > 1e-6

            if numpy.any(valid):
                vecs[valid] = vecs[valid] / norms[valid, numpy.newaxis]

                # Signed anti-alignment: +1 means directly opposite click normal,
                # -1 means same direction as click normal.
                align_score = numpy.full_like(norm_dist, 0.0)
                align_score[valid] = numpy.dot(vecs[valid], -normal)

                # Bias toward geodesic poles while letting opposite click normals
                # steer the destination to different sides on complex meshes.
                combined = numpy.full_like(norm_dist, -numpy.inf)
                combined[finite_mask] = (
                    (0.65 * norm_dist[finite_mask]) +
                    (0.35 * align_score[finite_mask])
                )
                sink_face = int(numpy.argmax(combined))
            
    if sink_face is None:
        sink_face = int(numpy.argmax(dist))
            
    _log(f"  Sink node selected: face {sink_face}")
    
    if sink_face == face_index:
        finite_dist = numpy.where(finite_mask, dist, -1.0)
        sink_face = int(numpy.argmax(finite_dist))
    
    _log(f"  Sink selected in {time.perf_counter()-t4:.3f}s: face {sink_face} "
         f"(source={face_index}, geodist={dist[sink_face]:.1f})")

    # ---------------------------------------------------------------
    # 5. Dual-Dijkstra directional partition (replaces Dinic's max-flow)
    #
    # Run Dijkstra from BOTH source and sink, then partition faces by
    # relative distance:  score = d_source / (d_source + d_sink).
    # Faces with score < threshold → set_a (near source), rest → set_b.
    # The boundary is perpendicular to the source→sink axis — mimicking
    # what max-flow would have done, in O(N) numpy operations.
    # ---------------------------------------------------------------
    t5 = time.perf_counter()
    _check_cancel()

    # 5a. Second Dijkstra from sink (reuses CSR adjacency structure)
    dist_sink = numpy.full(faces_count, numpy.inf, dtype=numpy.float64)
    dist_sink[sink_face] = 0.0
    pq_heap2 = [(0.0, sink_face)]

    dijkstra_pops2 = 0
    while pq_heap2:
        d, u = heapq.heappop(pq_heap2)
        dijkstra_pops2 += 1
        if dijkstra_pops2 % 2000 == 0:
            _check_cancel()
        if d > dist_sink[u]:
            continue
        start_e = offsets[u]
        end_e = offsets[u + 1]
        for e_idx in range(start_e, end_e):
            v = int(geo_targets[e_idx])
            nd = d + geo_weights[e_idx]
            if nd < dist_sink[v]:
                dist_sink[v] = nd
                heapq.heappush(pq_heap2, (nd, v))

    _log(f"  Dijkstra from sink in {time.perf_counter()-t5:.3f}s")

    # 5b. Compute relative distance score for each face
    #     score = d_source / (d_source + d_sink)
    #     0 → at source,  1 → at sink,  0.5 → equidistant
    d_src = numpy.where(numpy.isfinite(dist), dist, max_dist)
    d_snk = numpy.where(numpy.isfinite(dist_sink), dist_sink, max_dist)
    total_d = d_src + d_snk
    # Avoid division by zero
    safe_total = numpy.where(total_d > 1e-12, total_d, 1.0)
    score = d_src / safe_total

    # 5c. Sweep thresholds on score to find minimum-cost boundary
    min_faces_threshold = max(10, int(faces_count * 0.02))

    # Edge scores for cross-boundary detection
    s1 = score[f1_arr]
    s2 = score[f2_arr]
    score_lo = numpy.minimum(s1, s2)
    score_hi = numpy.maximum(s1, s2)

    # 5c. Score Threshold Search: 0.01 to 0.45 (focused on source side)
    t5c = time.perf_counter()
    
    # Test N possible thresholds
    N_THRESH_STEPS = 20
    # Because the distance field relies on sweeping to a distant Sink Pole,
    # we enforce a constraint of < 0.5 to guarantee the algorithm finds the bottleneck
    # physically located on the same side of the model as the user's click.
    thresholds = numpy.linspace(0.01, 0.45, N_THRESH_STEPS)

    best_thresh = None
    best_cost = float('inf')
    best_len_a = 0

    for i in range(len(thresholds)):
        t = float(thresholds[i])
        cross_mask = (score_lo < t) & (score_hi >= t)
        cross = int(numpy.sum(cross_mask))
        len_a = int(numpy.sum(score < t))
        len_b = faces_count - len_a
        if len_a < min_faces_threshold or len_b < min_faces_threshold:
            continue
        # face_adjacency_edges is (N, 2) pair of VERTEX indices for each shared edge
        cross_edges_vertex_indices = mesh.face_adjacency_edges[cross_mask]
        
        if cross_edges_vertex_indices.size == 0:
            continue
            
        # Calculate base perimeter length
        v0_coords = mesh.vertices[cross_edges_vertex_indices[:, 0]]
        v1_coords = mesh.vertices[cross_edges_vertex_indices[:, 1]]
        edge_lengths = numpy.linalg.norm(v0_coords - v1_coords, axis=1)
        
        # Penalize non-planar loops using PCA
        unique_pts = numpy.unique(cross_edges_vertex_indices.reshape(-1, 1), axis=0)
        # Get coordinates of unique vertices
        unique_coords = mesh.vertices[unique_pts.flatten()]

        pca_penalty = 1.0
        if len(unique_coords) >= 3:
            centroid = numpy.mean(unique_coords, axis=0)
            cov = numpy.cov((unique_coords - centroid).T)
            try:
                eigenvalues, _ = numpy.linalg.eigh(cov)
                eigenvalues = numpy.sort(numpy.abs(eigenvalues))[::-1]
                if eigenvalues[0] > 1e-6:
                    planarity = eigenvalues[2] / eigenvalues[0]
                    pca_penalty = 1.0 + (planarity * 5.0)  # Moderate penalty for non-flat cuts
            except numpy.linalg.LinAlgError:
                # Handle cases where covariance matrix might be singular (e.g., all points collinear)
                pca_penalty = 1.0 # No penalty
                
        # Prefer straight geodesic cuts across small perimeters
        perim_len = numpy.sum(edge_lengths)
        cost = perim_len * pca_penalty
        logger.debug(
            "threshold sweep: t=%.3f cost=%.2f perim_len=%.2f pca=%.2f len_a=%d",
            t,
            cost,
            perim_len,
            pca_penalty,
            len_a,
        )

        if cost < best_cost:
            best_cost = cost
            best_thresh = t
            best_len_a = len_a
    
    if best_thresh is None:
        # Fallback: split at score = 0.5 (equidistant boundary)
        _log("DEBUG BEST THRESHOLD: none valid, fallback to score=0.5")
        best_thresh = 0.5
        _log(f"  No valid threshold found, fallback to score=0.5")
    else:
        _log(f"DEBUG BEST THRESHOLD: t={best_thresh:.3f}, best_cost={best_cost:.2f}")

    _log(f"  Best score_thresh={best_thresh:.3f}, cut_cost={best_cost:.1f}, set_a~={best_len_a}")

    mask_a = score < best_thresh
    # Ensure source is always in set_a
    mask_a[face_index] = True
    set_a = numpy.where(mask_a)[0].tolist()
    set_b = numpy.where(~mask_a)[0].tolist()

    # Ensure smaller set is set_a
    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    _log(f"  Partition computed in {time.perf_counter()-t5:.3f}s")

    # Min-cut refinement is expensive on larger meshes and can exceed the UI timeout.
    # Keep it for small/medium meshes where it improves seam quality, but skip it on
    # large captures so shortest-seam remains responsive.
    REFINE_MAX_FACES = 4000
    if faces_count <= REFINE_MAX_FACES:
        _log(f"  Refining partition with min-cut bottleneck search...")
        set_a, set_b = refine_partition_with_mincut(mesh, set_a, set_b, score, best_thresh)
    else:
        _log(f"  Skipping min-cut refinement for large mesh ({faces_count} faces)")

    total_time = time.perf_counter() - t0
    _log(f"  DONE. set_a={len(set_a)}, set_b={len(set_b)} faces. Total time: {total_time:.3f}s")
    return set_a, set_b, face_index, sink_face


def refine_partition_with_mincut(
    mesh: "trimesh.Trimesh",
    initial_set_a: list,
    initial_set_b: list,
    score: Optional[numpy.ndarray] = None,
    best_thresh: Optional[float] = None,
    min_face_fraction: float = 0.005,
) -> Tuple[list, list]:
    from collections import deque
    faces_count = len(mesh.faces)
    min_faces = max(10, int(faces_count * min_face_fraction))
    
    if len(initial_set_a) < 2 or len(initial_set_b) < 2:
        logger.debug("refine_partition: partitions too small to refine (%d/%d)",
                     len(initial_set_a), len(initial_set_b))
        return initial_set_a, initial_set_b

    if score is None or best_thresh is None:
        logger.debug(
            "refine_partition: score/best_thresh not provided, "
            "falling back to boundary smoothing only"
        )
        return smooth_partition_boundary(mesh, initial_set_a, initial_set_b)

    if len(score) != faces_count:
        logger.warning(
            "refine_partition: invalid score length (%d != %d), "
            "falling back to boundary smoothing only",
            len(score),
            faces_count,
        )
        return smooth_partition_boundary(mesh, initial_set_a, initial_set_b)

    # We will build a flow network with a Super-Source and Super-Sink.
    # Source Region: faces very close to the click point (score near 0)
    # Sink Region: faces far away (score > best_thresh)
    source_thresh = min(best_thresh * 0.5, 0.15) # Ensure source is a solid region
    sink_thresh = min(best_thresh * 1.5, 0.8)    # Ensure sink is the rest of the object

    source_region = set(numpy.where(score <= source_thresh)[0])
    sink_region = set(numpy.where(score >= sink_thresh)[0])
    
    if not source_region or not sink_region:
        logger.warning("refine_partition: degenerate score regions, returning original")
        return initial_set_a, initial_set_b

    # Step 3: Build flow network and run Dinic's max-flow
    adj_edges = mesh.face_adjacency_edges
    adj_pairs = mesh.face_adjacency
    
    super_source = faces_count
    super_sink = faces_count + 1
    num_nodes = faces_count + 2
    
    graph = [[] for _ in range(num_nodes)]

    def _add_edge(u, v, cap):
        graph[u].append({"v": v, "cap": cap, "rev": len(graph[v])})
        graph[v].append({"v": u, "cap": 0, "rev": len(graph[u]) - 1})

    # Connect super_source to source_region, super_sink to sink_region
    for f in source_region:
        _add_edge(super_source, f, 1e9)
    for f in sink_region:
        _add_edge(f, super_sink, 1e9)

    # Pre-calculate edge lengths for capacity
    for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
        f1 = int(f1)
        f2 = int(f2)
        weight = float(numpy.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
        weight = max(weight, 1e-10)
        
        _add_edge(f1, f2, weight)
        _add_edge(f2, f1, weight)

    # Dinic's max-flow
    def _bfs_level():
        level = [-1] * num_nodes
        queue = deque([super_source])
        level[super_source] = 0
        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                if level[edge["v"]] < 0 and edge["cap"] > 1e-7:
                    level[edge["v"]] = level[u] + 1
                    queue.append(edge["v"])
        return level

    def _dfs_flow(u, sink, f, level, it):
        if u == sink: return f
        for i in range(it[u], len(graph[u])):
            it[u] = i
            edge = graph[u][i]
            if edge["cap"] <= 1e-7 or level[edge["v"]] != level[u] + 1:
                continue
            pushed = _dfs_flow(edge["v"], sink, min(f, edge["cap"]), level, it)
            if pushed > 0:
                edge["cap"] -= pushed
                graph[edge["v"]][edge["rev"]]["cap"] += pushed
                return pushed
        return 0

    flow = 0
    while True:
        level = _bfs_level()
        if level[super_sink] < 0:
            break
        it = [0] * num_nodes
        while True:
            pushed = _dfs_flow(super_source, super_sink, 1e9, level, it)
            if pushed <= 0:
                break
            flow += pushed

    # After max-flow, BFS from super-source to find all nodes in the source component
    refined_a = set()
    queue = deque([super_source])
    visited = [False] * num_nodes
    visited[super_source] = True
    while queue:
        u = queue.popleft()
        if u < faces_count: # Only add actual mesh faces to refined_a
            refined_a.add(u)
        for edge in graph[u]:
            v = edge["v"]
            if not visited[v] and edge["cap"] > 1e-7:
                visited[v] = True
                queue.append(v)

    refined_a = sorted(list(refined_a))
    refined_b = sorted([i for i in range(faces_count) if i not in refined_a])

    # Ensure set_a is the smaller partition
    if len(refined_a) > len(refined_b):
        refined_a, refined_b = refined_b, refined_a

    logger.debug("refine_partition: min-cut result: %d/%d faces (from %d/%d)",
                 len(refined_a), len(refined_b),
                 len(initial_set_a), len(initial_set_b))

    # Validate: refined partition should have both sides >= min_faces
    if len(refined_a) < min_faces or len(refined_b) < min_faces:
        logger.warning("refine_partition: min-cut gave trivial result (%d/%d), "
                       "keeping original plane partition (%d/%d)",
                       len(refined_a), len(refined_b),
                       len(initial_set_a), len(initial_set_b))
        return smooth_partition_boundary(mesh, initial_set_a, initial_set_b)

    return smooth_partition_boundary(mesh, refined_a, refined_b)


def smooth_partition_boundary(
    mesh: "trimesh.Trimesh",
    set_a: list,
    set_b: list,
    iterations: int = 5,
) -> Tuple[list, list]:
    """
    Smooth a partition boundary by greedy boundary-length minimization.

    For each face on the boundary, compute the cut edge lengths if it stays
    vs if it moves to the other side. Only move it if moving SHORTENS the
    total boundary length. This ensures every move improves the seam.

    A face in set_a has:
    - "cut edges" to neighbors in set_b (these contribute to boundary length)
    - "interior edges" to neighbors in set_a (these don't)

    If we move the face to set_b:
    - Old cut edges become interior (saved length)
    - Old interior edges become cut edges (new length)

    Move if: new_cut_length < old_cut_length  (boundary gets shorter)

    Args:
        mesh: The trimesh object.
        set_a: Face indices for the first partition.
        set_b: Face indices for the second partition.
        iterations: Number of smoothing passes (default 5).

    Returns:
        (smoothed_set_a, smoothed_set_b) with shorter boundary.
    """
    members_a = set(set_a)
    members_b = set(set_b)

    # Build neighbor lookup with shared edge lengths
    adj_pairs = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    # face_neighbors[f] = list of (neighbor_face, shared_edge_length)
    face_neighbors = [[] for _ in range(len(mesh.faces))]
    for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
        edge_len = float(numpy.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
        face_neighbors[int(f1)].append((int(f2), edge_len))
        face_neighbors[int(f2)].append((int(f1), edge_len))

    for iteration in range(iterations):
        # Find boundary faces
        boundary_faces = []
        for f in members_a:
            for nb, _ in face_neighbors[f]:
                if nb in members_b:
                    boundary_faces.append(f)
                    break
        for f in members_b:
            for nb, _ in face_neighbors[f]:
                if nb in members_a:
                    boundary_faces.append(f)
                    break

        moves = 0
        for f in boundary_faces:
            is_in_a = (f in members_a)
            my_set = members_a if is_in_a else members_b
            other_set = members_b if is_in_a else members_a

            # Current cut length: sum of edges to neighbors on the other side
            current_cut = sum(el for nb, el in face_neighbors[f] if nb in other_set)
            # Cut length if we move f: edges to neighbors on our CURRENT side become cuts
            new_cut = sum(el for nb, el in face_neighbors[f] if nb in my_set)

            if new_cut < current_cut:
                # Moving shortens the boundary
                my_set.discard(f)
                other_set.add(f)
                moves += 1

        if moves == 0:
            logger.debug("smooth_partition: converged after %d iterations", iteration + 1)
            break
        logger.debug("smooth_partition: iteration %d, moved %d faces",
                     iteration + 1, moves)

    # Ensure set_a is still the smaller partition
    result_a = sorted(members_a)
    result_b = sorted(members_b)
    if len(result_a) > len(result_b):
        result_a, result_b = result_b, result_a

    return result_a, result_b
