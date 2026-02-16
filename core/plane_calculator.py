# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Cut plane calculation algorithms.

Determines where and how to cut a mesh based on the selected mode.
No Cura dependencies - uses only trimesh and numpy.
"""

import numpy
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import trimesh
except ImportError:
    trimesh = None

from .geometry import plane_normal_from_spherical

logger = logging.getLogger("objectsplitter.plane_calculator")


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
    ALIGNMENT_BIAS = 0.5
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
        penalty = 1.0 + ALIGNMENT_BIAS * (1.0 - alignment)
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


def find_valley_cut_plane(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    search_resolution: int = 18,
    collect_all_samples: bool = False,
    surface_normal: Optional[numpy.ndarray] = None,
    sweep_fraction: float = 0.15,
    n_sweep_steps: int = 11,
    n_top_candidates: int = 20,
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

    Returns:
        SmallestPlaneSearchResult with the best groove-following plane.
    """
    plane_origin = numpy.asarray(click_position, dtype=numpy.float64)

    # Alignment bias (same as find_smallest_cut_plane)
    ALIGNMENT_BIAS = 0.5
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
        penalty = 1.0 + ALIGNMENT_BIAS * (1.0 - alignment)
        return area * penalty

    # Minimum area threshold (same as smallest mode)
    avg_face_area = mesh.area / max(len(mesh.faces), 1)
    min_section_area = avg_face_area * 5.0

    # Sweep distance: fraction of mesh diagonal
    extent = mesh.bounds[1] - mesh.bounds[0]
    mesh_diagonal = float(numpy.linalg.norm(extent))
    sweep_distance = mesh_diagonal * sweep_fraction

    logger.debug("Valley cut search: sweep_distance=%.2f (%.0f%% of diagonal %.2f), "
                 "%d sweep steps, resolution=%d",
                 sweep_distance, sweep_fraction * 100, mesh_diagonal,
                 n_sweep_steps, search_resolution)

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
                        coarse_all.append((score, area, normal.copy()))
                        if area >= min_section_area:
                            coarse_candidates.append((score, area,
                                                      normal.copy()))
            except Exception as e:
                if collect_all_samples:
                    all_samples.append((normal.copy(), float('nan')))
                logger.debug("Valley phase 1: section failed for normal=%s: %s",
                             normal, e)
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

    logger.debug("Valley phase 1: %d valid candidates from %d samples, "
                 "refining top %d",
                 len(coarse_candidates), samples_tested, len(top_coarse))

    # ---- Phase 2: Sweep each top candidate to find the groove ----
    best_normal = numpy.array([0.0, 1.0, 0.0])
    best_origin = plane_origin.copy()
    best_score = float('inf')
    best_area = float('inf')
    valid_candidates = []

    for _, coarse_area, normal in top_coarse:
        # Sweep positions along this normal, centered at the click point
        offsets = numpy.linspace(-sweep_distance, sweep_distance, n_sweep_steps)

        sweep_best_area = float('inf')
        sweep_best_offset = 0.0

        for offset in offsets:
            sweep_origin = plane_origin + offset * normal
            try:
                section = mesh.section(plane_origin=sweep_origin,
                                       plane_normal=normal)
                if section is not None:
                    path_2d, to_3D = _section_to_2d(section)
                    area = _local_section_area(path_2d, to_3D, sweep_origin)

                    if area > 0 and area < sweep_best_area:
                        sweep_best_area = area
                        sweep_best_offset = offset
            except Exception:
                continue

        if not numpy.isfinite(sweep_best_area):
            continue

        # Score the swept result: area * alignment bias * proximity bias
        # Prefer planes closer to the click point (mild penalty for offset)
        proximity_penalty = 1.0 + 0.3 * (abs(sweep_best_offset) / max(sweep_distance, 1e-6))
        score = _compute_score(sweep_best_area, normal) * proximity_penalty

        swept_origin = plane_origin + sweep_best_offset * normal
        valid_candidates.append((score, sweep_best_area, normal.copy(),
                                 swept_origin.copy()))

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

    return SmallestPlaneSearchResult(
        plane=CutPlane(origin=best_origin, normal=best_normal),
        area=best_area,
        samples_tested=samples_tested,
        all_samples=all_samples,
        top_candidates=top_candidates,
    )


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
    # Goal: find a face that is both FAR from source AND aligned with -surface_normal.
    # Use normalized scores so distance and alignment contribute equally.
    t4 = time.perf_counter()
    min_radius = max(5.0, max_dist * 0.4)  # Sink must be at least 40% of max_dist away
    
    finite_mask = numpy.isfinite(dist)
    candidate_mask = finite_mask & (dist >= min_radius)
    
    if not numpy.any(candidate_mask):
        # Fallback: relax to 10% of max_dist
        candidate_mask = finite_mask & (dist >= max_dist * 0.1)
    if not numpy.any(candidate_mask):
        # Fallback: pick farthest face
        finite_dist = numpy.where(finite_mask, dist, -1.0)
        sink_face = int(numpy.argmax(finite_dist))
        _log(f"  Sink fallback to max dist: face {sink_face}")
    else:
        # Normalized distance score: 0 = at source, 1 = at max_dist
        norm_dist = numpy.where(candidate_mask, dist / max_dist, 0.0)
        
        if surface_normal is not None:
            sn_len = numpy.linalg.norm(surface_normal)
            if sn_len > 1e-6:
                target_dir = -surface_normal / sn_len
                face_centers = mesh.triangles_center
                click_pt = face_centers[face_index]
                vecs = face_centers - click_pt  # (F, 3)
                v_lens = numpy.linalg.norm(vecs, axis=1)
                safe_lens = numpy.where(v_lens > 1e-6, v_lens, 1.0)
                normed = vecs / safe_lens[:, None]
                alignment = normed @ target_dir  # -1 to 1
                # Remap to 0-1: alignment_score = (alignment + 1) / 2
                align_01 = (alignment + 1.0) / 2.0
                # Final score = distance * alignment (both 0-1)
                scores = numpy.where(candidate_mask & (v_lens > 1e-6),
                                     norm_dist * align_01, -numpy.inf)
            else:
                scores = numpy.where(candidate_mask, norm_dist, -numpy.inf)
        else:
            # No surface normal — just use distance
            scores = numpy.where(candidate_mask, norm_dist, -numpy.inf)
        
        sink_face = int(numpy.argmax(scores))
        _log(f"  Sink score={scores[sink_face]:.3f}, dist={dist[sink_face]:.1f}/{max_dist:.1f}")
    
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

    n_thresh = 100
    thresholds = numpy.linspace(0.1, 0.9, n_thresh)

    best_thresh = None
    best_cost = float('inf')
    best_len_a = 0

    for i in range(n_thresh):
        t = float(thresholds[i])
        cross_mask = (score_lo < t) & (score_hi >= t)
        cross = int(numpy.sum(cross_mask))
        len_a = int(numpy.sum(score < t))
        len_b = faces_count - len_a

        # Skip if partition too small on either side
        if len_a < min_faces_threshold or len_b < min_faces_threshold:
            continue

        # Cost = total length of cross-boundary edges (shorter seam preferred)
        cost = float(numpy.sum(edge_lengths[cross_mask])) if cross > 0 else 0.0

        if cost < best_cost:
            best_cost = cost
            best_thresh = t
            best_len_a = len_a

    if best_thresh is None:
        # Fallback: split at score = 0.5 (equidistant boundary)
        best_thresh = 0.5
        _log(f"  No valid threshold found, fallback to score=0.5")

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

    total_time = time.perf_counter() - t0
    _log(f"  DONE. set_a={len(set_a)}, set_b={len(set_b)} faces. Total time: {total_time:.3f}s")
    return set_a, set_b, face_index, sink_face


def refine_partition_with_mincut(
    mesh: "trimesh.Trimesh",
    initial_set_a: list,
    initial_set_b: list,
    min_face_fraction: float = 0.005,
) -> Tuple[list, list]:
    """
    Refine a plane-based partition by finding the shortest seam (min-cut)
    between the two regions.

    The initial partition (from local_plane_partition) gives us the right
    REGION to cut, but the boundary is a straight plane. Min-cut can find
    a shorter path along the mesh surface that follows the actual geometry.

    The trick: place source and sink DEEP inside each partition (farthest
    from the boundary). This prevents the min-cut from trivially isolating
    a single triangle — it's forced to go through the actual neck between
    the two regions.

    Args:
        mesh: The trimesh object.
        initial_set_a: Face indices for the smaller partition (piece to cut off).
        initial_set_b: Face indices for the larger partition (body).
        min_face_fraction: Minimum fraction of faces for the result to be valid.

    Returns:
        (refined_set_a, refined_set_b) — the refined partition if min-cut
        succeeds, or the original partition if it fails.
    """
    import heapq
    from collections import deque

    faces_count = len(mesh.faces)
    min_faces = max(10, int(faces_count * min_face_fraction))

    if len(initial_set_a) < 2 or len(initial_set_b) < 2:
        logger.debug("refine_partition: partitions too small to refine (%d/%d)",
                     len(initial_set_a), len(initial_set_b))
        return initial_set_a, initial_set_b

    # Step 1: Find boundary faces (faces in set_a adjacent to set_b and vice versa)
    set_a_ids = set(initial_set_a)
    set_b_ids = set(initial_set_b)
    adj_pairs = mesh.face_adjacency

    boundary_a = set()  # faces in set_a that touch set_b
    boundary_b = set()  # faces in set_b that touch set_a
    for f1, f2 in adj_pairs:
        f1i, f2i = int(f1), int(f2)
        if f1i in set_a_ids and f2i in set_b_ids:
            boundary_a.add(f1i)
            boundary_b.add(f2i)
        elif f2i in set_a_ids and f1i in set_b_ids:
            boundary_a.add(f2i)
            boundary_b.add(f1i)

    logger.debug("refine_partition: boundary faces: %d in set_a, %d in set_b",
                 len(boundary_a), len(boundary_b))

    # Step 2: BFS from boundary inward to find deepest face in each partition
    # (face farthest from the boundary = safest source/sink for min-cut)
    face_adj = [[] for _ in range(faces_count)]
    for f1, f2 in adj_pairs:
        face_adj[int(f1)].append(int(f2))
        face_adj[int(f2)].append(int(f1))

    def _find_deepest(partition_set, boundary_faces):
        """BFS from boundary inward; return face with max depth."""
        depth = {}
        queue = deque()
        for bf in boundary_faces:
            if bf in partition_set:
                depth[bf] = 0
                queue.append(bf)
        # If no boundary faces found, pick any face
        if not queue:
            first = next(iter(partition_set))
            return first
        deepest_face = queue[0]
        max_depth = 0
        while queue:
            f = queue.popleft()
            for nb in face_adj[f]:
                if nb in partition_set and nb not in depth:
                    depth[nb] = depth[f] + 1
                    queue.append(nb)
                    if depth[nb] > max_depth:
                        max_depth = depth[nb]
                        deepest_face = nb
        return deepest_face

    source_face = _find_deepest(set_a_ids, boundary_a)
    sink_face = _find_deepest(set_b_ids, boundary_b)

    logger.debug("refine_partition: source=%d (deep in set_a, %d faces), "
                 "sink=%d (deep in set_b, %d faces)",
                 source_face, len(initial_set_a),
                 sink_face, len(initial_set_b))

    if source_face == sink_face:
        logger.warning("refine_partition: source == sink, returning original")
        return initial_set_a, initial_set_b

    # Step 3: Build flow network and run Dinic's max-flow
    adj_edges = mesh.face_adjacency_edges
    graph = [[] for _ in range(faces_count)]

    def _add_edge(u, v, cap):
        graph[u].append({"v": v, "cap": cap, "rev": len(graph[v])})
        graph[v].append({"v": u, "cap": 0, "rev": len(graph[u]) - 1})

    # Use PURE edge lengths as weights — no per-edge constant.
    # The constant (used in find_shortest_seam_partition) prevents single-face
    # isolation, but here source/sink are deep inside each partition, so
    # trivial cuts are already impossible. Without the constant, the min-cut
    # truly minimizes total boundary LENGTH: it always takes the shortest
    # edge through a triangle rather than going the long way around via
    # the two longer edges.
    # UPDATE: User reports jagged edges (stair-stepping) on diagonals.
    # We MUST adding a small base cost (edge count regularization) to prefer straight cuts.
    
    # Calculate average edge length for scaling
    total_len = 0.0
    count_len = 0
    edge_weights = []
    
    for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
        w = float(numpy.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
        total_len += w
        count_len += 1
        edge_weights.append(w)
        
    avg_len = total_len / count_len if count_len > 0 else 1.0
    BASE_COST = avg_len * 0.1
    
    for i, ((f1, f2), _) in enumerate(zip(adj_pairs, adj_edges)):
        weight = edge_weights[i]
        weight = max(weight, 1e-10)
        
        # Add regularization to penalize edge count (prefer fewer edges aka straighter path)
        final_capacity = weight + BASE_COST
        
        _add_edge(int(f1), int(f2), final_capacity)
        _add_edge(int(f2), int(f1), final_capacity)

    # Dinic's max-flow
    def _bfs_level():
        level = [-1] * faces_count
        queue = [source_face]
        level[source_face] = 0
        for u in queue:
            for edge in graph[u]:
                if level[edge["v"]] < 0 and edge["cap"] > 0:
                    level[edge["v"]] = level[u] + 1
                    queue.append(edge["v"])
        return level

    def _dfs_flow(u, sink, f, level, it):
        if u == sink:
            return f
        for i in range(it[u], len(graph[u])):
            it[u] = i
            edge = graph[u][i]
            if edge["cap"] <= 0 or level[edge["v"]] != level[u] + 1:
                continue
            ret = _dfs_flow(edge["v"], sink, min(f, edge["cap"]), level, it)
            if ret > 0:
                edge["cap"] -= ret
                graph[edge["v"]][edge["rev"]]["cap"] += ret
                return ret
        return 0

    while True:
        level = _bfs_level()
        if level[sink_face] < 0:
            break
        it = [0] * faces_count
        while True:
            pushed = _dfs_flow(source_face, sink_face, float("inf"), level, it)
            if pushed <= 1e-9:
                break

    # Extract reachable set
    reachable = [False] * faces_count
    stack = [source_face]
    reachable[source_face] = True
    while stack:
        u = stack.pop()
        for edge in graph[u]:
            if edge["cap"] > 0 and not reachable[edge["v"]]:
                reachable[edge["v"]] = True
                stack.append(edge["v"])

    refined_a = [i for i, r in enumerate(reachable) if r]
    refined_b = [i for i, r in enumerate(reachable) if not r]

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
