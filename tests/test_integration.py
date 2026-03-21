# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Integration tests simulating the full Cura workflow.

These tests exercise the complete pipeline: mesh -> plane calculation ->
split -> connectors, mimicking what happens when a user clicks in Cura.
They also demonstrate the capture/replay and visualization systems.
"""

import os
import tempfile
import numpy
import pytest
import trimesh

from core.plane_calculator import (
    horizontal_cut_plane,
    vertical_cut_plane,
    find_smallest_cut_plane,
    find_valley_cut_plane,
    find_valley_seam_partition,
    find_shortest_seam_partition,
)
from core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
    split_by_local_plane,
    split_by_face_sets,
)
from core.connectors import add_connectors, ConnectorConfig
from core.path_cutter import isolate_region_by_loops
from core.debug_capture import (
    capture_operation,
    load_captured_operation,
    replay_operation,
    save_result_meshes,
)


def _make_mask_mesh(mask: numpy.ndarray) -> trimesh.Trimesh:
    mask = numpy.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    vertices = numpy.array(
        [[float(i), float(j), 0.0] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=numpy.float64,
    )
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
    return trimesh.Trimesh(vertices=vertices, faces=numpy.asarray(faces, dtype=numpy.int32), process=False)


def _make_double_bridge_surface_mesh() -> trimesh.Trimesh:
    from shapely.geometry import box
    from shapely.ops import unary_union

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


def _nearest_face_to_point(mesh: trimesh.Trimesh, point: numpy.ndarray) -> int:
    centroids = numpy.asarray(mesh.triangles_center, dtype=numpy.float64)
    dists = numpy.linalg.norm(centroids - numpy.asarray(point, dtype=numpy.float64), axis=1)
    return int(numpy.argmin(dists))


class TestFullHorizontalWorkflow:
    """End-to-end horizontal cut tests."""

    def test_cube_horizontal_50pct(self, cube_mesh):
        """Standard use case: horizontal cut at 50% of a cube."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success
        # Both halves should have reasonable face counts
        assert len(result.upper.faces) > 2
        assert len(result.lower.faces) > 2

    def test_cube_horizontal_with_connectors(self, cube_mesh):
        """Horizontal cut + connectors on a cube."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success
        if result.capped:
            config = ConnectorConfig(diameter=3.0, height=2.0, clearance=0.15)
            cr = add_connectors(result.upper, result.lower,
                                plane.origin, plane.normal, config)
            assert isinstance(cr.upper, trimesh.Trimesh)
            assert isinstance(cr.lower, trimesh.Trimesh)

    def test_tall_box_various_heights(self, tall_box_mesh):
        """Cut a tall box at multiple heights."""
        for pct in [10, 25, 50, 75, 90]:
            plane = horizontal_cut_plane(tall_box_mesh, pct)
            result = slice_mesh_with_fallback(tall_box_mesh, plane.origin, plane.normal)
            assert result.success, f"Failed at {pct}%"


class TestFullVerticalWorkflow:
    """End-to-end vertical cut tests."""

    def test_cube_vertical_center(self, cube_mesh):
        """Vertical cut through the center of a cube."""
        click = cube_mesh.centroid
        plane = vertical_cut_plane(click)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

    def test_flat_box_vertical(self, flat_box_mesh):
        """Vertical cut on a flat box."""
        click = flat_box_mesh.centroid
        plane = vertical_cut_plane(click)
        result = slice_mesh_with_fallback(flat_box_mesh, plane.origin, plane.normal)
        assert result.success


class TestFullSmallestWorkflow:
    """End-to-end smallest cross-section tests."""

    def test_cylinder_finds_circle(self, cylinder_mesh):
        """For a cylinder, smallest section should be the circular cross-section."""
        center = cylinder_mesh.centroid
        search = find_smallest_cut_plane(cylinder_mesh, center, search_resolution=6)
        result = slice_mesh_with_fallback(
            cylinder_mesh, search.plane.origin, search.plane.normal)
        assert result.success


class TestFullShortestSeamWorkflow:
    """End-to-end shortest seam tests."""

    def test_cube_shortest_seam(self, cube_mesh):
        """Shortest seam partition and split on a cube."""
        click = numpy.array([10.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(cube_mesh, click)
        result = split_by_shortest_seam(cube_mesh, set_a, set_b)
        assert result.success

    def test_sphere_shortest_seam(self, sphere_mesh):
        """Shortest seam on a sphere."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        result = split_by_shortest_seam(sphere_mesh, set_a, set_b)
        assert result.success


class TestFullValleyWorkflow:
    """End-to-end valley/groove detection tests."""

    def test_cylinder_valley_cut(self, cylinder_mesh):
        """Valley mode should find and cut through a cylinder's narrowest section."""
        # Click slightly off-center — valley sweep should still find the groove
        center = cylinder_mesh.centroid + numpy.array([0.0, 3.0, 0.0])
        search = find_valley_cut_plane(cylinder_mesh, center, search_resolution=6)
        result = slice_mesh_with_fallback(
            cylinder_mesh, search.plane.origin, search.plane.normal)
        assert result.success

    def test_sphere_valley_cut(self, sphere_mesh):
        """Valley mode on a sphere should produce a valid split."""
        center = sphere_mesh.centroid + numpy.array([2.0, 0.0, 0.0])
        search = find_valley_cut_plane(sphere_mesh, center, search_resolution=6)
        # Use local plane partition (same path as ObjectSplitter._performCut)
        from core.plane_calculator import snap_point_to_mesh_surface
        snap_point, face_id = snap_point_to_mesh_surface(sphere_mesh, center)
        if face_id is None:
            face_id = 0  # fallback for proximity query failures
        candidate_normals = ([n for _, n in search.top_candidates]
                             if search.top_candidates else [search.plane.normal])
        result = split_by_local_plane(
            sphere_mesh, search.plane.origin, candidate_normals, face_id)
        assert result.success

    def test_sphere_valley_cut_with_sdf_bias(self, sphere_mesh):
        """Valley mode with SDF bias should still produce a valid split."""
        center = sphere_mesh.centroid + numpy.array([2.0, 0.0, 0.0])
        search = find_valley_cut_plane(
            sphere_mesh,
            center,
            search_resolution=6,
            use_sdf_bias=True,
        )
        from core.plane_calculator import snap_point_to_mesh_surface
        snap_point, face_id = snap_point_to_mesh_surface(sphere_mesh, center)
        if face_id is None:
            face_id = 0
        candidate_normals = ([n for _, n in search.top_candidates]
                             if search.top_candidates else [search.plane.normal])
        result = split_by_local_plane(
            sphere_mesh, search.plane.origin, candidate_normals, face_id)
        assert result.success


class TestFullValleySeamWorkflow:
    """End-to-end seam-based valley tests."""

    def test_sphere_valley_seam(self, sphere_mesh):
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_valley_seam_partition(
            sphere_mesh, click, surface_normal=numpy.array([1.0, 0.0, 0.0])
        )
        result = split_by_face_sets(
            sphere_mesh,
            set_a,
            set_b,
            strategy_name="valley_seam",
            attempt_hole_fill=False,
        )
        assert result.success
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0

    def test_sphere_valley_seam_with_sdf_bias(self, sphere_mesh):
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_valley_seam_partition(
            sphere_mesh,
            click,
            surface_normal=numpy.array([1.0, 0.0, 0.0]),
            use_sdf_bias=True,
        )
        result = split_by_face_sets(
            sphere_mesh,
            set_a,
            set_b,
            strategy_name="valley_seam",
            attempt_hole_fill=False,
        )
        assert result.success
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0


class TestCaptureAndReplay:
    """Tests for the debug capture/replay system."""

    def test_capture_roundtrip(self, cube_mesh):
        """Capture an operation and verify it can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = numpy.array([0.0, 0.0, 0.0])
            path = capture_operation(
                mesh=cube_mesh,
                cut_mode="horizontal",
                click_position=click,
                height_percent=50.0,
                name="test_roundtrip",
                capture_dir=tmpdir,
            )
            mesh, params = load_captured_operation(path)
            assert len(mesh.vertices) == len(cube_mesh.vertices)
            assert params.cut_mode == "horizontal"
            assert params.height_percent == 50.0

    def test_replay_produces_results(self, cube_mesh):
        """Replay a captured operation and verify it produces split results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = numpy.array([0.0, 0.0, 0.0])
            path = capture_operation(
                mesh=cube_mesh,
                cut_mode="horizontal",
                click_position=click,
                height_percent=50.0,
                connector_enabled=False,
                name="test_replay",
                capture_dir=tmpdir,
            )
            result = replay_operation(path)
            assert result['split_result'].success
            assert result['plane'] is not None

    def test_save_result_meshes(self, cube_mesh):
        """Save result meshes to disk and verify they exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = numpy.array([0.0, 0.0, 0.0])
            path = capture_operation(
                mesh=cube_mesh,
                cut_mode="horizontal",
                click_position=click,
                height_percent=50.0,
                connector_enabled=False,
                name="test_save",
                capture_dir=tmpdir,
            )
            result = replay_operation(path)
            output_dir = os.path.join(tmpdir, "output")
            saved = save_result_meshes(result, output_dir)
            assert len(saved) == 2


class TestPathIsolateWorkflow:
    """Integration tests for multi-loop path isolation."""

    def test_isolate_region_split_produces_two_meshes(self):
        mesh = _make_double_bridge_surface_mesh()
        loops = [
            [
                numpy.array([-14.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([-14.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([-5.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([-5.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([-14.5, 18.0, 0.0], dtype=numpy.float64),
            ],
            [
                numpy.array([5.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([5.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([14.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([14.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([5.5, 18.0, 0.0], dtype=numpy.float64),
            ],
        ]
        from core.path_cutter import chain_paths
        loop_paths = [chain_paths(mesh, loop) for loop in loops]
        target_face = _nearest_face_to_point(mesh, numpy.array([0.0, 32.0, 3.0]))

        extracted, remainder = isolate_region_by_loops(mesh, loop_paths, target_face)
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
        assert len(result.upper.faces) + len(result.lower.faces) == len(mesh.faces)

    def test_isolate_mode_returns_plain_two_mesh_split(self):
        mesh = _make_double_bridge_surface_mesh()
        loops = [
            [
                numpy.array([-14.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([-14.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([-5.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([-5.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([-14.5, 18.0, 0.0], dtype=numpy.float64),
            ],
            [
                numpy.array([5.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([5.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([14.5, 18.0, 6.0], dtype=numpy.float64),
                numpy.array([14.5, 18.0, 0.0], dtype=numpy.float64),
                numpy.array([5.5, 18.0, 0.0], dtype=numpy.float64),
            ],
        ]
        from core.path_cutter import chain_paths
        loop_paths = [chain_paths(mesh, loop) for loop in loops]
        target_face = _nearest_face_to_point(mesh, numpy.array([0.0, 32.0, 3.0]))

        extracted, remainder = isolate_region_by_loops(mesh, loop_paths, target_face)
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

    def test_capture_with_plane_override(self, cube_mesh):
        """Capture with explicit plane should replay using that plane."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = numpy.array([0.0, 0.0, 0.0])
            path = capture_operation(
                mesh=cube_mesh,
                cut_mode="horizontal",
                click_position=click,
                height_percent=50.0,
                connector_enabled=False,
                plane_origin=numpy.array([0.0, 3.0, 0.0]),
                plane_normal=numpy.array([0.0, 1.0, 0.0]),
                name="test_override",
                capture_dir=tmpdir,
            )
            result = replay_operation(path)
            assert result['split_result'].success
            # Plane origin should be the overridden value
            assert result['plane'].origin[1] == pytest.approx(3.0)

    def test_replay_valley_mode(self, sphere_mesh):
        """Captured valley mode should replay through valley search + local partition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = sphere_mesh.centroid + numpy.array([2.0, 0.0, 0.0])
            path = capture_operation(
                mesh=sphere_mesh,
                cut_mode="valley",
                click_position=click,
                search_resolution=6,
                valley_sdf_bias_enabled=True,
                connector_enabled=False,
                name="test_valley_replay",
                capture_dir=tmpdir,
            )
            result = replay_operation(path)
            assert result["split_result"].success
            assert result["plane"] is not None
            assert os.path.exists(os.path.join(path, "valley_trace_replay.json"))

    def test_replay_valley_seam_with_anchor_points(self, sphere_mesh):
        """Captured valley seam with point anchors should replay successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            click = sphere_mesh.centroid + numpy.array([15.0, 0.0, 0.0])
            anchors = [
                click.tolist(),
                (sphere_mesh.centroid + numpy.array([-15.0, 0.0, 0.0])).tolist(),
            ]
            path = capture_operation(
                mesh=sphere_mesh,
                cut_mode="valley_seam",
                click_position=click,
                search_resolution=6,
                valley_sdf_bias_enabled=True,
                anchor_points=anchors,
                connector_enabled=False,
                name="test_valley_seam_anchor_replay",
                capture_dir=tmpdir,
            )
            result = replay_operation(path)
            assert result["split_result"].success
            assert os.path.exists(os.path.join(path, "valley_seam_trace_replay.json"))


class TestVisualization:
    """Tests for visualization output generation."""

    def test_html_viewer_generation(self, cube_mesh):
        """Generate an HTML viewer and verify the file exists."""
        from viz.visualizer import generate_html_viewer

        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "viewer.html")
            ok = generate_html_viewer(
                result.upper, result.lower, html_path,
                title="Test Cube Split",
                metadata={"mode": "horizontal", "height": "50%"},
                plane_origin=plane.origin,
                plane_normal=plane.normal,
            )
            assert ok
            assert os.path.exists(html_path)
            with open(html_path) as f:
                content = f.read()
            assert "three" in content.lower()
            assert "Test Cube Split" in content

    def test_report_generation(self, cube_mesh, sphere_mesh):
        """Generate a test report from multiple results."""
        from viz.visualizer import generate_report

        results = []
        for mesh, name in [(cube_mesh, "cube"), (sphere_mesh, "sphere")]:
            plane = horizontal_cut_plane(mesh, 50.0)
            split = slice_mesh_with_fallback(mesh, plane.origin, plane.normal)
            results.append({
                'name': f"{name}_horizontal_50",
                'status': 'pass' if split.success else 'fail',
                'split_result': split,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.html")
            ok = generate_report(results, report_path, title="Test Report")
            assert ok
            assert os.path.exists(report_path)
            with open(report_path) as f:
                content = f.read()
            assert "2 passed" in content

    def test_image_render(self, cube_mesh):
        """Generate a PNG image of a split result (if matplotlib available)."""
        from viz.visualizer import render_split_image

        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "split.png")
            ok = render_split_image(
                result.upper, result.lower, img_path,
                title="Test Split",
                plane_normal=plane.normal,
            )
            # ok is False if matplotlib not installed - that's acceptable
            if ok:
                assert os.path.exists(img_path)
