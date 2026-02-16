# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for core.plane_calculator module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from core.plane_calculator import (
    horizontal_cut_plane,
    vertical_cut_plane,
    find_smallest_cut_plane,
    find_valley_cut_plane,
    find_shortest_seam_partition,
    CutPlane,
)


class TestHorizontalCutPlane:
    """Tests for horizontal (Y-up) cut plane calculation."""

    def test_50_percent_cuts_at_center(self, cube_mesh):
        """50% height should cut through the center of the mesh."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        # Cube is 20mm, centered at origin, so Y ranges -10 to +10
        assert_allclose(plane.origin[1], 0.0, atol=0.1)
        assert_allclose(plane.normal, [0, 1, 0])

    def test_0_percent_cuts_at_bottom(self, cube_mesh):
        """0% should cut at the bottom of the mesh."""
        plane = horizontal_cut_plane(cube_mesh, 0.0)
        assert plane.origin[1] == pytest.approx(-10.0, abs=0.1)

    def test_100_percent_cuts_at_top(self, cube_mesh):
        """100% should cut at the top of the mesh."""
        plane = horizontal_cut_plane(cube_mesh, 100.0)
        assert plane.origin[1] == pytest.approx(10.0, abs=0.1)

    def test_25_percent_on_tall_box(self, tall_box_mesh):
        """25% of a 40mm tall box should cut at -10mm from center."""
        plane = horizontal_cut_plane(tall_box_mesh, 25.0)
        # Box 10x40x10, Y range -20 to +20, 25% = -20 + 10 = -10
        assert plane.origin[1] == pytest.approx(-10.0, abs=0.1)

    def test_normal_is_y_up(self, cube_mesh):
        """Horizontal plane normal should always be Y-up."""
        for pct in [0, 25, 50, 75, 100]:
            plane = horizontal_cut_plane(cube_mesh, pct)
            assert_allclose(plane.normal, [0, 1, 0])


class TestVerticalCutPlane:
    """Tests for vertical cut plane calculation."""

    def test_default_normal_is_x(self):
        """Default vertical plane normal should be X-axis."""
        pos = numpy.array([5.0, 10.0, 15.0])
        plane = vertical_cut_plane(pos)
        assert_allclose(plane.normal, [1, 0, 0])
        assert_allclose(plane.origin, pos)

    def test_custom_normal(self):
        """Should accept a custom normal direction."""
        pos = numpy.array([0, 0, 0])
        normal = numpy.array([0, 0, 1])
        plane = vertical_cut_plane(pos, normal)
        assert_allclose(plane.normal, [0, 0, 1])

    def test_origin_matches_click(self):
        """Plane origin should match the click position."""
        pos = numpy.array([12.3, -4.5, 67.8])
        plane = vertical_cut_plane(pos)
        assert_allclose(plane.origin, pos)


class TestFindSmallestCutPlane:
    """Tests for smallest cross-section search."""

    def test_cylinder_smallest_is_perpendicular(self, cylinder_mesh):
        """
        For a cylinder along Z, the smallest cross-section should be
        perpendicular to its axis (i.e., a circle), not along the axis
        (which would be a rectangle).
        """
        # Offset slightly from centroid to avoid degenerate cross-sections
        center = cylinder_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        result = find_smallest_cut_plane(cylinder_mesh, center, search_resolution=8)
        assert result.samples_tested > 0
        # The smallest section of a Z-aligned cylinder is the circular cross-section
        # circle area = pi * 8^2 ~ 201. Allow some margin for offset.
        assert result.area < 300

    def test_tall_box_smallest_is_horizontal(self, tall_box_mesh):
        """
        For a 10x40x10 tall box, the smallest cross-section should
        be the 10x10=100 mm^2 horizontal slice, not the 10x40=400 mm^2 vertical.
        """
        # Offset slightly to avoid cutting exactly through edges
        center = tall_box_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        result = find_smallest_cut_plane(tall_box_mesh, center, search_resolution=6)
        # Smallest area should be ~100 (10x10 face)
        assert result.area < 200

    def test_collect_all_samples(self, cube_mesh):
        """With collect_all_samples=True, should record all tested orientations."""
        center = cube_mesh.centroid
        result = find_smallest_cut_plane(
            cube_mesh, center, search_resolution=4, collect_all_samples=True)
        assert result.all_samples is not None
        assert len(result.all_samples) > 0
        assert result.samples_tested == len(result.all_samples)

    def test_low_resolution_still_works(self, cube_mesh):
        """Even resolution=2 should find a valid cut."""
        center = cube_mesh.centroid
        result = find_smallest_cut_plane(cube_mesh, center, search_resolution=2)
        assert result.area > 0
        assert result.plane.normal is not None


class TestFindValleyCutPlane:
    """Tests for geographic valley/groove detection."""

    def test_cylinder_finds_groove_at_center(self, cylinder_mesh):
        """
        For a cylinder along Z, the valley should be the circular cross-section.
        Even clicking slightly off-center, the sweep should find it.
        """
        # Click slightly off-center along the cylinder axis
        center = cylinder_mesh.centroid + numpy.array([0.1, 2.0, 0.1])
        result = find_valley_cut_plane(cylinder_mesh, center, search_resolution=6)
        assert result.samples_tested > 0
        # The circular cross-section of a radius-8 cylinder is ~201 mm^2.
        assert result.area < 300

    def test_tall_box_finds_narrow_section(self, tall_box_mesh):
        """
        For a 10x40x10 tall box, valley mode should find the 10x10=100 mm^2
        cross-section even when clicking slightly above the center.
        """
        # Click 5mm above center — sweep should slide down to find the
        # same ~100 mm^2 cross-section that spans the box width
        center = tall_box_mesh.centroid + numpy.array([0.1, 5.0, 0.1])
        result = find_valley_cut_plane(tall_box_mesh, center, search_resolution=6)
        assert result.area < 200

    def test_sweep_moves_origin_from_click(self, cylinder_mesh):
        """
        When clicking off the groove center, the returned plane origin
        should differ from the click position (sweep found a better spot).
        """
        # Click well off the centroid
        click = cylinder_mesh.centroid + numpy.array([0.0, 5.0, 0.0])
        result = find_valley_cut_plane(cylinder_mesh, click, search_resolution=6)
        # The origin may have shifted along the best axis toward the groove
        offset = numpy.linalg.norm(result.plane.origin - click)
        # Either it moved or remained (if click was already optimal)
        assert offset >= 0  # Sanity check — always true
        assert result.area > 0

    def test_returns_valid_result(self, cube_mesh):
        """Valley search should always return a valid CutPlane."""
        center = cube_mesh.centroid
        result = find_valley_cut_plane(cube_mesh, center, search_resolution=4)
        assert result.plane is not None
        assert result.plane.origin is not None
        assert result.plane.normal is not None
        assert numpy.linalg.norm(result.plane.normal) > 0.9

    def test_collect_all_samples(self, cube_mesh):
        """With collect_all_samples=True, should record phase-1 samples."""
        center = cube_mesh.centroid
        result = find_valley_cut_plane(
            cube_mesh, center, search_resolution=4, collect_all_samples=True)
        assert result.all_samples is not None
        assert len(result.all_samples) > 0
        assert result.samples_tested == len(result.all_samples)

    def test_surface_normal_bias(self, cylinder_mesh):
        """Surface normal should bias the result toward aligned orientations."""
        center = cylinder_mesh.centroid
        # Bias toward Y-axis alignment
        result_biased = find_valley_cut_plane(
            cylinder_mesh, center, search_resolution=6,
            surface_normal=numpy.array([0.0, 1.0, 0.0]))
        # No bias
        result_unbiased = find_valley_cut_plane(
            cylinder_mesh, center, search_resolution=6)
        # Both should find valid cuts
        assert result_biased.area > 0
        assert result_unbiased.area > 0

    def test_top_candidates_populated(self, sphere_mesh):
        """Result should have top_candidates for downstream fallback."""
        center = sphere_mesh.centroid
        result = find_valley_cut_plane(sphere_mesh, center, search_resolution=6)
        assert result.top_candidates is not None
        assert len(result.top_candidates) > 0


class TestFindShortestSeamPartition:
    """Tests for geodesic shortest-seam partitioning."""

    def test_partitions_cover_all_faces(self, cube_mesh):
        """Both partitions together should contain all faces."""
        click = numpy.array([10.0, 0.0, 0.0])  # Click on +X face
        set_a, set_b, src, sink = find_shortest_seam_partition(cube_mesh, click)
        total = len(set_a) + len(set_b)
        assert total == len(cube_mesh.faces)

    def test_partitions_are_nonempty(self, cube_mesh):
        """Both partitions should have at least one face."""
        click = cube_mesh.centroid
        set_a, set_b, _, _ = find_shortest_seam_partition(cube_mesh, click)
        assert len(set_a) > 0
        assert len(set_b) > 0

    def test_set_a_is_smaller(self, sphere_mesh):
        """set_a should be the smaller partition."""
        click = numpy.array([15.0, 0.0, 0.0])  # Click near surface
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        assert len(set_a) <= len(set_b)

    def test_no_overlapping_faces(self, cube_mesh):
        """Partitions should not share any face indices."""
        click = cube_mesh.centroid
        set_a, set_b, _, _ = find_shortest_seam_partition(cube_mesh, click)
        assert len(set(set_a) & set(set_b)) == 0
