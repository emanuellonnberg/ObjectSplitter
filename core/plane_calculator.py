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


def find_smallest_cut_plane(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    search_resolution: int = 18,
    collect_all_samples: bool = False
) -> SmallestPlaneSearchResult:
    """
    Find the plane orientation that produces the smallest cross-sectional area.

    Samples orientations in spherical coordinates and measures the cross-section
    area at each orientation through the click position.

    Args:
        mesh: The trimesh object to analyze.
        click_position: 3D point to pass the plane through.
        search_resolution: Number of elevation angles to sample (azimuth = 2x this).
        collect_all_samples: If True, store all (normal, area) pairs for debugging.

    Returns:
        SmallestPlaneSearchResult with the best plane, area, and optional debug data.
    """
    plane_origin = numpy.asarray(click_position, dtype=numpy.float64)
    best_normal = numpy.array([0.0, 1.0, 0.0])
    best_area = float('inf')
    samples_tested = 0
    all_samples = [] if collect_all_samples else None

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

                    if 0 < area < best_area:
                        best_area = area
                        best_normal = normal.copy()
            except Exception as e:
                if collect_all_samples:
                    all_samples.append((normal.copy(), float('nan')))
                logger.debug("Section failed for normal=%s: %s: %s", normal, type(e).__name__, e)
                continue

    # Fallback: if no valid sections found (e.g. mesh.section fails for real Cura meshes),
    # try axis-aligned normals and pick the best. Avoids always returning horizontal.
    if not numpy.isfinite(best_area) or best_area <= 0:
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
                    if 0 < area < best_area:
                        best_area = area
                        best_normal = fallback_normal.copy()
            except Exception as e:
                logger.debug("Axis fallback failed for %s: %s", fallback_normal, e)

    logger.debug("Smallest cut search: best_area=%.2f mm^2, normal=%s, tested=%d orientations",
                 best_area, best_normal, samples_tested)

    return SmallestPlaneSearchResult(
        plane=CutPlane(origin=plane_origin, normal=best_normal),
        area=best_area,
        samples_tested=samples_tested,
        all_samples=all_samples
    )


def find_shortest_seam_partition(
    mesh: "trimesh.Trimesh",
    click_position: numpy.ndarray,
    source_face_hint: Optional[int] = None
) -> Tuple[list, list, int, int]:
    """
    Compute a geodesic shortest-seam cut by finding a minimum cut in the
    face adjacency graph (Dinic's max-flow algorithm).

    This separates the mesh into two face sets that minimize the total
    length of shared edges between them.

    Args:
        mesh: The trimesh object to partition.
        click_position: 3D point identifying the source face (used if source_face_hint is None).
        source_face_hint: If provided, use this face as the source (from snap_point_to_mesh_surface).

    Returns:
        (set_a_faces, set_b_faces, source_face, sink_face) where
        set_a is the smaller partition and set_b is the larger.
    """
    import heapq

    # Use provided face if available (from snapped point), else find via proximity
    face_index = source_face_hint
    if face_index is None:
        point = numpy.asarray(click_position, dtype=numpy.float64).reshape(1, -1)
        try:
            from trimesh.proximity import ProximityQuery
            pq = ProximityQuery(mesh)
            _, _, face_ids = pq.on_surface(point)
            face_index = int(face_ids[0]) if face_ids is not None else None
        except Exception as e:
            logger.debug("Proximity query failed: %s", e)

    if face_index is None:
        if mesh.vertices.shape[0] > 0:
            distances = numpy.linalg.norm(mesh.vertices - point, axis=1)
            nearest_idx = int(numpy.argmin(distances))
        else:
            nearest_idx = 0
        faces_with_vertex = numpy.where(mesh.faces == nearest_idx)[0]
        face_index = int(faces_with_vertex[0]) if faces_with_vertex.size > 0 else 0

    faces_count = len(mesh.faces)
    adj_pairs = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    # Build flow network
    graph = [[] for _ in range(faces_count)]

    def _add_edge(u, v, cap):
        graph[u].append({"v": v, "cap": cap, "rev": len(graph[v])})
        graph[v].append({"v": u, "cap": 0, "rev": len(graph[u]) - 1})

    for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
        edge_length = float(numpy.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
        _add_edge(int(f1), int(f2), edge_length)
        _add_edge(int(f2), int(f1), edge_length)

    # Dijkstra to find farthest face (= sink)
    dist = [float("inf")] * faces_count
    dist[face_index] = 0.0
    pq_heap = [(0.0, face_index)]
    while pq_heap:
        d, f = heapq.heappop(pq_heap)
        if d > dist[f]:
            continue
        for edge in graph[f]:
            if edge["cap"] > 0:
                nd = d + edge["cap"]
                if nd < dist[edge["v"]]:
                    dist[edge["v"]] = nd
                    heapq.heappush(pq_heap, (nd, edge["v"]))

    sink_face = int(numpy.argmax(dist)) if faces_count > 0 else face_index

    # Dinic's max-flow
    def _bfs_level():
        level = [-1] * faces_count
        queue = [face_index]
        level[face_index] = 0
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
            pushed = _dfs_flow(face_index, sink_face, float("inf"), level, it)
            if pushed <= 1e-9:
                break

    # Extract reachable set from residual graph
    reachable = [False] * faces_count
    stack = [face_index]
    reachable[face_index] = True
    while stack:
        u = stack.pop()
        for edge in graph[u]:
            if edge["cap"] > 0 and not reachable[edge["v"]]:
                reachable[edge["v"]] = True
                stack.append(edge["v"])

    set_a = [i for i, r in enumerate(reachable) if r]
    set_b = [i for i, r in enumerate(reachable) if not r]

    # Ensure set_a is the smaller partition
    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    logger.debug("Shortest seam: source=%d, sink=%d, set_a=%d faces, set_b=%d faces",
                 face_index, sink_face, len(set_a), len(set_b))

    return set_a, set_b, face_index, sink_face
