"""
Unit tests for core.path_cutter module.

Tests geodesic path finding, waypoint chaining, and face partitioning.
"""
import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trimesh
from core.path_cutter import (
    find_geodesic_path,
    chain_paths,
    partition_faces_by_path,
    isolate_region_by_loops,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid_mesh(nx: int = 5, ny: int = 5) -> trimesh.Trimesh:
    """
    Create a flat grid mesh on the XY plane with nx*ny vertices.
    Vertices are laid out in a regular grid [0..nx-1] x [0..ny-1].
    """
    vertices = []
    for j in range(ny):
        for i in range(nx):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int32)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def make_cylinder_mesh(radius: float = 1.0, height: float = 2.0,
                       sections: int = 16) -> trimesh.Trimesh:
    """Create a cylinder mesh for testing on curved surfaces."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height,
                                      sections=sections)
    mesh.merge_vertices()
    return mesh


def make_mask_mesh(mask: np.ndarray) -> trimesh.Trimesh:
    """Create a planar triangle mesh from a boolean cell mask."""
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape

    vertices = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.asarray(vertices, dtype=np.float64)

    faces = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            v0 = j * stride + i
            v1 = v0 + 1
            v2 = v0 + stride
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int32), process=False)


def make_double_bridge_surface_mesh() -> trimesh.Trimesh:
    """
    Create a closed solid with a lower body, an upper island, and two narrow
    bridges between them. This matches the intended multi-loop isolate use case
    better than an open planar sheet.
    """
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        pytest.skip("shapely not available")

    poly = unary_union([
        box(-30.0, -20.0, 30.0, 12.0),
        box(-18.0, 24.0, 18.0, 40.0),
        box(-14.0, 12.0, -6.0, 24.0),
        box(6.0, 12.0, 14.0, 24.0),
    ]).buffer(0)
    mesh = trimesh.creation.extrude_polygon(poly, height=6.0)
    vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def nearest_face_to_point(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    dists = np.linalg.norm(centroids - np.asarray(point, dtype=np.float64), axis=1)
    return int(np.argmin(dists))


# ===========================================================================
# Tests for find_geodesic_path
# ===========================================================================

class TestFindGeodesicPath:
    """Tests for vertex-based Dijkstra shortest path."""

    def test_adjacent_vertices(self):
        """Path between adjacent vertices should be just those two vertices."""
        mesh = make_grid_mesh(5, 5)
        path = find_geodesic_path(mesh, 0, 1)
        assert path[0] == 0
        assert path[-1] == 1
        assert len(path) == 2

    def test_same_vertex(self):
        """Path from vertex to itself should be just that vertex."""
        mesh = make_grid_mesh(5, 5)
        path = find_geodesic_path(mesh, 7, 7)
        assert path == [7]

    def test_path_exists(self):
        """Path between distant vertices on connected mesh should exist."""
        mesh = make_grid_mesh(5, 5)
        # Corner to opposite corner
        path = find_geodesic_path(mesh, 0, 24)
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 24

    def test_path_vertices_are_connected(self):
        """Each consecutive pair in the path should share an edge."""
        mesh = make_grid_mesh(5, 5)
        path = find_geodesic_path(mesh, 0, 24)

        edges_set = set()
        for edge in mesh.edges_unique:
            edges_set.add((edge[0], edge[1]))
            edges_set.add((edge[1], edge[0]))

        for i in range(len(path) - 1):
            assert (path[i], path[i + 1]) in edges_set, \
                f"Vertices {path[i]} and {path[i+1]} are not connected by an edge"

    def test_path_is_shortest(self):
        """On a grid, diagonal path should exist and be reasonably short."""
        mesh = make_grid_mesh(5, 5)
        path = find_geodesic_path(mesh, 0, 24)

        # Compute path length
        path_length = 0.0
        verts = mesh.vertices
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(verts[path[i + 1]] - verts[path[i]])

        # L-shaped path length = 4 + 4 = 8
        # Euclidean distance = 4*sqrt(2) ≈ 5.66
        # The grid has diagonal edges (triangle hypotenuse = sqrt(2))
        # So the path should be <= 8.0 (it might be exactly 8 if no diagonals 
        # align, or shorter if diagonals do)
        assert path_length <= 8.0 + 1e-6, \
            f"Path length {path_length} exceeds L-path upper bound"
        # Path must be at least the Euclidean distance
        euclidean = np.linalg.norm(verts[24] - verts[0])
        assert path_length >= euclidean - 1e-6, \
            f"Path length {path_length} is less than Euclidean distance {euclidean}"

    def test_disconnected_raises(self):
        """Path between disconnected components should raise ValueError."""
        # Create two separate triangles
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],  # Triangle A
            [5, 5, 0], [6, 5, 0], [5, 6, 0],  # Triangle B
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        with pytest.raises(ValueError, match="[Nn]o path"):
            find_geodesic_path(mesh, 0, 3)

    def test_on_cylinder(self):
        """Path on curved surface (cylinder) should work."""
        mesh = make_cylinder_mesh()
        # Pick two vertices on opposite sides
        v0 = 0
        v1 = len(mesh.vertices) // 2
        path = find_geodesic_path(mesh, v0, v1)
        assert len(path) >= 2
        assert path[0] == v0
        assert path[-1] == v1


# ===========================================================================
# Tests for chain_paths
# ===========================================================================

class TestChainPaths:
    """Tests for waypoint chaining."""

    def test_two_waypoints(self):
        """Chaining two waypoints should produce a valid path."""
        mesh = make_grid_mesh(5, 5)
        wp1 = np.array([0.1, 0.1, 0.0])  # Near vertex 0
        wp2 = np.array([3.9, 3.9, 0.0])  # Near vertex 24
        path = chain_paths(mesh, [wp1, wp2])
        assert len(path) >= 2

    def test_three_waypoints(self):
        """Chaining three waypoints should visit vertices near each."""
        mesh = make_grid_mesh(5, 5)
        wp1 = np.array([0.0, 0.0, 0.0])  # vertex 0
        wp2 = np.array([4.0, 0.0, 0.0])  # vertex 4
        wp3 = np.array([4.0, 4.0, 0.0])  # vertex 24
        path = chain_paths(mesh, [wp1, wp2, wp3])
        assert len(path) >= 3
        # Path should start near wp1 and end near wp3
        assert path[0] == 0  # Nearest to [0,0,0]
        assert path[-1] == 24  # Nearest to [4,4,0]

    def test_single_waypoint_raises(self):
        """Chain with <2 waypoints should raise ValueError."""
        mesh = make_grid_mesh(5, 5)
        with pytest.raises(ValueError):
            chain_paths(mesh, [np.array([0.0, 0.0, 0.0])])

    def test_no_duplicate_vertices(self):
        """Chained path should not have consecutive duplicate vertices."""
        mesh = make_grid_mesh(5, 5)
        wp1 = np.array([0.0, 0.0, 0.0])
        wp2 = np.array([2.0, 2.0, 0.0])
        wp3 = np.array([4.0, 4.0, 0.0])
        path = chain_paths(mesh, [wp1, wp2, wp3])
        for i in range(len(path) - 1):
            assert path[i] != path[i + 1], \
                f"Consecutive duplicate at index {i}: vertex {path[i]}"


# ===========================================================================
# Tests for partition_faces_by_path
# ===========================================================================

class TestPartitionFacesByPath:
    """Tests for face partitioning along a path."""

    def test_basic_partition(self):
        """Partitioning should produce two non-empty sets covering all faces."""
        mesh = make_grid_mesh(5, 5)
        # Create a path across the middle row
        path = find_geodesic_path(mesh, 2, 22)  # Column 2, rows 0-4
        face_a, face_b = partition_faces_by_path(mesh, path)

        assert len(face_a) > 0, "Partition A is empty"
        assert len(face_b) > 0, "Partition B is empty"

        # Together should cover all faces (some boundary faces may be in either)
        all_faces = set(face_a) | set(face_b)
        assert len(all_faces) == len(mesh.faces), \
            f"Partitions cover {len(all_faces)} faces but mesh has {len(mesh.faces)}"

    def test_no_overlap(self):
        """Partitions should not overlap."""
        mesh = make_grid_mesh(5, 5)
        path = find_geodesic_path(mesh, 2, 22)
        face_a, face_b = partition_faces_by_path(mesh, path)
        overlap = set(face_a) & set(face_b)
        assert len(overlap) == 0, f"Partitions overlap on {len(overlap)} faces"

    def test_partition_on_cylinder(self):
        """Partition on cylinder should produce two non-empty halves."""
        mesh = make_cylinder_mesh(sections=16)
        # Find two vertices roughly on opposite ends
        verts = mesh.vertices
        # Top vertex (max z) and bottom vertex (min z)
        top_v = int(np.argmax(verts[:, 2]))
        bot_v = int(np.argmin(verts[:, 2]))
        path = find_geodesic_path(mesh, top_v, bot_v)
        face_a, face_b = partition_faces_by_path(mesh, path)

        total = len(mesh.faces)
        assert len(face_a) > 0, "Partition A is empty"
        assert len(face_b) > 0, "Partition B is empty"
        assert len(face_a) + len(face_b) == total, \
            f"Partitions don't cover all faces: {len(face_a)} + {len(face_b)} != {total}"

    def test_short_path_partition(self):
        """Even a short path (2 vertices) should partition."""
        mesh = make_grid_mesh(5, 5)
        path = [6, 7]
        face_a, face_b = partition_faces_by_path(mesh, path)
        assert len(face_a) + len(face_b) == len(mesh.faces)

    def test_path_cut_on_stl(self):
        """Test path cut on captured STL fixture if available."""
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        stl_path = os.path.join(fixture_dir, "captured_model_small.stl")
        if not os.path.exists(stl_path):
            pytest.skip("STL fixture not available")

        mesh = trimesh.load(stl_path, force="mesh")
        mesh.merge_vertices()

        # Pick two distant vertices
        verts = mesh.vertices
        distances = np.linalg.norm(verts - verts[0], axis=1)
        far_vertex = int(np.argmax(distances))

        path = find_geodesic_path(mesh, 0, far_vertex)
        assert len(path) >= 2

        face_a, face_b = partition_faces_by_path(mesh, path)
        assert len(face_a) > 0
        assert len(face_b) > 0
        assert len(face_a) + len(face_b) == len(mesh.faces)


class TestChainPathsCrossComponent:
    """Tests for chain_paths handling disconnected mesh components."""

    def test_resnap_to_target_component(self):
        """Waypoints on other components should be re-snapped to the first waypoint's component."""
        # Create two disconnected triangles
        vertices = np.array([
            # Triangle A: around origin
            [0, 0, 0], [2, 0, 0], [1, 2, 0],
            [0, 0, 0.1], [2, 0, 0.1], [1, 2, 0.1],
            # Triangle B: far away
            [100, 100, 0], [102, 100, 0], [101, 102, 0],
            [100, 100, 0.1], [102, 100, 0.1], [101, 102, 0.1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [3, 4, 5], [0, 1, 3], [1, 3, 4],
            [1, 2, 4], [2, 4, 5], [0, 2, 3], [2, 3, 5],
            [6, 7, 8], [9, 10, 11], [6, 7, 9], [7, 9, 10],
            [7, 8, 10], [8, 10, 11], [6, 8, 9], [8, 9, 11],
        ], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # First waypoint on component A, second on component B
        wp1 = np.array([0.0, 0.0, 0.0])     # On component A
        wp2 = np.array([100.0, 100.0, 0.0])  # On component B → should re-snap to A

        path = chain_paths(mesh, [wp1, wp2])
        # Path should exist entirely on component A's vertices (indices 0-5)
        for v in path:
            assert v < 6, f"Vertex {v} is not on target component A"

    def test_all_waypoints_on_same_component(self):
        """When all waypoints are on the same component, no re-snapping needed."""
        mesh = make_grid_mesh(5, 5)
        wp1 = np.array([0.0, 0.0, 0.0])
        wp2 = np.array([4.0, 4.0, 0.0])
        path = chain_paths(mesh, [wp1, wp2])
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 24


class TestPartitionFacesByPathOnCurved:
    """Tests for partition_faces_by_path on non-trivial geometries."""

    def test_sphere_partition(self):
        """Partitioning a sphere by a geodesic path should work."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        mesh.merge_vertices()
        verts = mesh.vertices

        # Pick two distant vertices
        distances = np.linalg.norm(verts - verts[0], axis=1)
        far_v = int(np.argmax(distances))

        path = find_geodesic_path(mesh, 0, far_v)
        face_a, face_b = partition_faces_by_path(mesh, path)
        assert len(face_a) > 0
        assert len(face_b) > 0
        assert len(face_a) + len(face_b) == len(mesh.faces)
        assert len(set(face_a) & set(face_b)) == 0
        ratio = len(face_a) / len(mesh.faces)
        assert 0.30 <= ratio <= 0.70, (
            f"Open sphere path should produce a meaningful split, got {len(face_a)}/{len(mesh.faces)} "
            f"faces ({ratio:.1%})."
        )

    def test_partition_set_a_is_smaller(self):
        """set_a should always be the smaller partition."""
        mesh = make_grid_mesh(8, 8)
        path = find_geodesic_path(mesh, 4, 60)  # Roughly middle column
        face_a, face_b = partition_faces_by_path(mesh, path)
        assert len(face_a) <= len(face_b)

    def test_closed_loop_uses_topological_components(self):
        """Closed loops should partition by seam-separated components when possible."""
        mesh = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=32)
        mesh.merge_vertices()
        waypoints = [
            np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0], dtype=np.float64)
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ]
        waypoints.append(waypoints[0].copy())

        vertex_path = chain_paths(mesh, waypoints)
        face_a, face_b = partition_faces_by_path(mesh, vertex_path)

        assert face_a and face_b
        ratio = len(face_a) / len(mesh.faces)
        assert 0.15 <= ratio <= 0.40, (
            f"Closed-loop cylinder cut should isolate a meaningful smaller component, "
            f"got {len(face_a)}/{len(mesh.faces)} faces ({ratio:.1%})."
        )

    def test_partition_then_split(self):
        """Path partition should produce splittable face sets."""
        from core.mesh_splitter import split_by_face_sets

        mesh = trimesh.creation.cylinder(radius=5, height=10, sections=16)
        mesh.merge_vertices()
        verts = mesh.vertices

        top_v = int(np.argmax(verts[:, 2]))
        bot_v = int(np.argmin(verts[:, 2]))
        path = find_geodesic_path(mesh, top_v, bot_v)
        face_a, face_b = partition_faces_by_path(mesh, path)

        result = split_by_face_sets(mesh, face_a, face_b)
        assert result.success
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0


class TestIsolateRegionByLoops:
    """Tests for isolating a target region using one or more closed loops."""

    def test_single_closed_loop_isolates_target_component(self):
        mesh = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=32)
        mesh.merge_vertices()
        waypoints = [
            np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0], dtype=np.float64)
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ]
        waypoints.append(waypoints[0].copy())
        loop = chain_paths(mesh, waypoints)
        target_face = nearest_face_to_point(mesh, np.array([0.0, 0.0, 4.5]))

        extracted, remainder = isolate_region_by_loops(mesh, [loop], target_face)

        assert len(extracted) > 0
        assert len(remainder) > 0
        assert set(extracted).isdisjoint(remainder)

    def test_two_loops_isolate_multi_bridge_region(self):
        mesh = make_double_bridge_surface_mesh()
        left_loop = chain_paths(mesh, [
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
        ])
        right_loop = chain_paths(mesh, [
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
        ])
        target_face = nearest_face_to_point(mesh, np.array([0.0, 32.0, 3.0]))

        extracted, remainder = isolate_region_by_loops(mesh, [left_loop, right_loop], target_face)

        assert extracted
        assert remainder
        assert len(set(extracted) | set(remainder)) == len(mesh.faces)
        ratio = len(extracted) / len(mesh.faces)
        assert 0.40 <= ratio <= 0.60

    def test_target_pick_changes_extracted_component(self):
        mesh = make_double_bridge_surface_mesh()
        left_loop = chain_paths(mesh, [
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
        ])
        right_loop = chain_paths(mesh, [
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
        ])
        target_face = nearest_face_to_point(mesh, np.array([0.0, 0.0, 3.0]))

        extracted, remainder = isolate_region_by_loops(mesh, [left_loop, right_loop], target_face)

        assert target_face in extracted
        assert len(extracted) < len(remainder)

    def test_invalid_unclosed_loop_raises(self):
        mesh = make_grid_mesh(5, 5)
        target_face = nearest_face_to_point(mesh, np.array([1.5, 1.5, 0.0]))

        with pytest.raises(ValueError, match="not closed"):
            isolate_region_by_loops(mesh, [[6, 7, 12, 11]], target_face)

    def test_non_separating_loop_set_raises(self):
        mesh = trimesh.creation.torus(major_radius=20.0, minor_radius=5.0)
        mesh.merge_vertices()

        waypoints = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            waypoints.append(np.array([20.0 * np.cos(angle), 20.0 * np.sin(angle), 0.0], dtype=np.float64))
        waypoints.append(waypoints[0].copy())

        loop = chain_paths(mesh, waypoints)
        target_face = nearest_face_to_point(mesh, np.array([20.0, 0.0, 5.0], dtype=np.float64))

        with pytest.raises(ValueError, match="does not separate"):
            isolate_region_by_loops(mesh, [loop], target_face)

    def test_isolated_region_can_be_split(self):
        from core.mesh_splitter import split_by_face_sets

        mesh = make_double_bridge_surface_mesh()
        left_loop = chain_paths(mesh, [
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 6.0], dtype=np.float64),
            np.array([-5.5, 18.0, 0.0], dtype=np.float64),
            np.array([-14.5, 18.0, 0.0], dtype=np.float64),
        ])
        right_loop = chain_paths(mesh, [
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 6.0], dtype=np.float64),
            np.array([14.5, 18.0, 0.0], dtype=np.float64),
            np.array([5.5, 18.0, 0.0], dtype=np.float64),
        ])
        target_face = nearest_face_to_point(mesh, np.array([0.0, 32.0, 3.0]))

        extracted, remainder = isolate_region_by_loops(mesh, [left_loop, right_loop], target_face)
        result = split_by_face_sets(
            mesh,
            extracted,
            remainder,
            strategy_name="path_isolate",
            attempt_hole_fill=False,
        )

        assert result.success
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
