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
    find_shortest_seam_partition,
)
from core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
)
from core.connectors import add_connectors, ConnectorConfig
from core.debug_capture import (
    capture_operation,
    load_captured_operation,
    replay_operation,
    save_result_meshes,
)


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
            for f in saved:
                assert os.path.exists(f)

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
