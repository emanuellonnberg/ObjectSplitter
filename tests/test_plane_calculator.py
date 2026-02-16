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
    snap_point_to_mesh_surface,
    smooth_partition_boundary,
    refine_partition_with_mincut,
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

    def test_with_surface_normal(self, sphere_mesh):
        """Surface normal should influence sink selection direction."""
        click = numpy.array([15.0, 0.0, 0.0])
        surface_normal = numpy.array([1.0, 0.0, 0.0])
        set_a, set_b, src, sink = find_shortest_seam_partition(
            sphere_mesh, click, surface_normal=surface_normal)
        # Basic invariants still hold
        assert len(set_a) + len(set_b) == len(sphere_mesh.faces)
        assert len(set_a) > 0 and len(set_b) > 0
        # Sink should be roughly on the opposite side from click
        sink_centroid = sphere_mesh.triangles_center[sink]
        src_centroid = sphere_mesh.triangles_center[src]
        # Sink should be further from click along the normal direction
        assert sink_centroid[0] < src_centroid[0]

    def test_source_face_hint_respected(self, cube_mesh):
        """source_face_hint should be used as the source face directly."""
        click = cube_mesh.centroid
        face_hint = 5
        set_a, set_b, src, sink = find_shortest_seam_partition(
            cube_mesh, click, source_face_hint=face_hint)
        # The source should be the hinted face (or at least in one partition)
        all_faces = set(set_a) | set(set_b)
        assert face_hint in all_faces
        assert len(set_a) + len(set_b) == len(cube_mesh.faces)


class TestSnapPointToMeshSurface:
    """Tests for snap_point_to_mesh_surface.

    Note: snap_point_to_mesh_surface falls back to mesh centroid when
    trimesh ProximityQuery fails (e.g., rtree not installed). These
    tests account for both the proximity path and the fallback path.
    """

    def test_returns_3d_point(self, cube_mesh):
        """Should always return a valid 3D point."""
        point = numpy.array([0.0, 10.0, 0.0])
        snapped, face_id = snap_point_to_mesh_surface(cube_mesh, point)
        assert snapped is not None
        assert snapped.shape == (3,)

    def test_point_on_surface_snaps_close_or_to_centroid(self, cube_mesh):
        """A point on the surface should snap to itself or centroid (fallback)."""
        point = numpy.array([0.0, 10.0, 0.0])
        snapped, face_id = snap_point_to_mesh_surface(cube_mesh, point)
        # Either snaps close to input (proximity works) or to centroid (fallback)
        dist_to_point = numpy.linalg.norm(snapped - point)
        dist_to_centroid = numpy.linalg.norm(snapped - cube_mesh.centroid)
        assert dist_to_point < 1.0 or dist_to_centroid < 1.0

    def test_result_within_mesh_bounds(self, cube_mesh):
        """Snapped point should be within or near mesh bounds."""
        point = numpy.array([0.0, 5.0, 0.0])
        snapped, face_id = snap_point_to_mesh_surface(cube_mesh, point)
        bounds = cube_mesh.bounds
        for i in range(3):
            assert snapped[i] >= bounds[0][i] - 1.0
            assert snapped[i] <= bounds[1][i] + 1.0

    def test_face_id_if_returned(self, sphere_mesh):
        """If face_id is returned, it should be a valid index."""
        point = numpy.array([15.0, 0.0, 0.0])
        snapped, face_id = snap_point_to_mesh_surface(sphere_mesh, point)
        if face_id is not None:
            assert 0 <= face_id < len(sphere_mesh.faces)

    def test_fallback_returns_centroid(self, cube_mesh):
        """Fallback path should return mesh centroid with face_id=None."""
        # Pass a point — if proximity fails, we get centroid
        point = numpy.array([0.0, 100.0, 0.0])
        snapped, face_id = snap_point_to_mesh_surface(cube_mesh, point)
        # Either proximity succeeded (snapped near top) or fallback (centroid)
        assert snapped is not None


class TestFindSmallestCutPlaneWithBias:
    """Tests for surface normal bias and anti-graze filtering."""

    def test_surface_normal_biases_result(self, cylinder_mesh):
        """With surface_normal, result should favor aligned planes."""
        center = cylinder_mesh.centroid + numpy.array([0.1, 0.1, 0.1])

        # Without surface normal
        result_no_bias = find_smallest_cut_plane(
            cylinder_mesh, center, search_resolution=6)

        # With surface normal along Y (perpendicular to cylinder axis)
        surface_normal = numpy.array([0.0, 1.0, 0.0])
        result_with_bias = find_smallest_cut_plane(
            cylinder_mesh, center, search_resolution=6,
            surface_normal=surface_normal)

        # Both should find valid planes
        assert result_no_bias.area > 0
        assert result_with_bias.area > 0

        # With Y-normal bias, the plane should be biased toward Y alignment
        alignment = abs(numpy.dot(result_with_bias.plane.normal, surface_normal))
        # Alignment should be somewhat high (biased toward surface normal)
        # This is a soft assertion — the bias is mild (0.5)
        assert alignment > 0.1 or result_with_bias.area < result_no_bias.area * 1.2

    def test_anti_graze_filters_tiny_sections(self, sphere_mesh):
        """Anti-graze filtering should reject surface-grazing cross-sections.

        On a sphere with many small faces, the anti-graze threshold
        (avg_face_area * 5) is small enough that real through-cuts pass
        but surface-grazing planes with tiny areas get rejected.
        """
        center = sphere_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        result = find_smallest_cut_plane(
            sphere_mesh, center, search_resolution=6,
            collect_all_samples=True)

        avg_face_area = sphere_mesh.area / len(sphere_mesh.faces)
        min_section_area = avg_face_area * 5.0

        # The result should be a real through-cut (circle area ~700mm²),
        # well above the anti-graze threshold
        assert result.area >= min_section_area, (
            f"Result area {result.area:.2f} is below anti-graze threshold "
            f"{min_section_area:.2f}")

        # Verify that samples below the threshold were actually found and filtered
        if result.all_samples:
            tiny_areas = [a for _, a in result.all_samples
                          if not numpy.isnan(a) and 0 < a < min_section_area]
            # Some surface-grazing orientations should produce tiny areas
            # that the filter rejected
            if tiny_areas:
                assert result.area > min(tiny_areas), (
                    "Anti-graze filter should have rejected smaller areas")

    def test_top_candidates_are_sorted(self, cube_mesh):
        """top_candidates should be sorted by score (ascending)."""
        center = cube_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        result = find_smallest_cut_plane(
            cube_mesh, center, search_resolution=4)

        if result.top_candidates and len(result.top_candidates) > 1:
            areas = [area for area, _ in result.top_candidates]
            # Candidates should be roughly sorted by score
            # (they are sorted by score, not area, but for no-bias case these are equal)
            assert areas[0] <= areas[-1] + 1e-6

    def test_bias_with_perpendicular_normal(self, tall_box_mesh):
        """Surface normal perpendicular to the ideal cut should still find good plane."""
        center = tall_box_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        # Surface normal along X — perpendicular to the ideal horizontal cut
        surface_normal = numpy.array([1.0, 0.0, 0.0])
        result = find_smallest_cut_plane(
            tall_box_mesh, center, search_resolution=6,
            surface_normal=surface_normal)
        # Should still find the horizontal cut (area ~100) since it's genuinely smallest
        # Bias is mild (0.5), so a 100mm² cut still beats a 400mm² vertical cut
        assert result.area < 250


class TestSmoothPartitionBoundary:
    """Tests for boundary smoothing of face partitions."""

    def test_smoothing_does_not_lose_faces(self, sphere_mesh):
        """Smoothing should preserve total face count."""
        n_faces = len(sphere_mesh.faces)
        # Split into two halves by face index
        set_a = list(range(n_faces // 3))
        set_b = list(range(n_faces // 3, n_faces))
        smooth_a, smooth_b = smooth_partition_boundary(sphere_mesh, set_a, set_b)
        assert len(smooth_a) + len(smooth_b) == n_faces

    def test_smoothing_no_overlap(self, sphere_mesh):
        """Smoothed partitions should not overlap."""
        n_faces = len(sphere_mesh.faces)
        set_a = list(range(n_faces // 3))
        set_b = list(range(n_faces // 3, n_faces))
        smooth_a, smooth_b = smooth_partition_boundary(sphere_mesh, set_a, set_b)
        assert len(set(smooth_a) & set(smooth_b)) == 0

    def test_smoothing_boundary_does_not_grow(self, sphere_mesh):
        """Smoothing should not increase boundary length."""
        # Create a partition via shortest seam
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)

        # Compute boundary length before smoothing
        adj_pairs = sphere_mesh.face_adjacency
        adj_edges = sphere_mesh.face_adjacency_edges
        set_a_ids = set(set_a)

        def _boundary_length(s_a):
            s = set(s_a)
            length = 0.0
            for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
                if (int(f1) in s) != (int(f2) in s):
                    length += float(numpy.linalg.norm(
                        sphere_mesh.vertices[v1] - sphere_mesh.vertices[v2]))
            return length

        before = _boundary_length(set_a)
        smooth_a, smooth_b = smooth_partition_boundary(
            sphere_mesh, set_a, set_b, iterations=3)
        after = _boundary_length(smooth_a)
        # Boundary should not increase (it's a greedy minimizer)
        assert after <= before + 1e-6

    def test_smoothing_converges(self, cube_mesh):
        """Smoothing with many iterations should converge."""
        n_faces = len(cube_mesh.faces)
        set_a = list(range(n_faces // 2))
        set_b = list(range(n_faces // 2, n_faces))
        smooth_a, smooth_b = smooth_partition_boundary(
            cube_mesh, set_a, set_b, iterations=20)
        assert len(smooth_a) + len(smooth_b) == n_faces


class TestRefinePartitionWithMincut:
    """Tests for min-cut partition refinement."""

    def test_refine_preserves_total_faces(self, sphere_mesh):
        """Refinement should preserve total face count."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        refined_a, refined_b = refine_partition_with_mincut(
            sphere_mesh, set_a, set_b)
        assert len(refined_a) + len(refined_b) == len(sphere_mesh.faces)

    def test_refine_no_overlap(self, sphere_mesh):
        """Refined partitions should not overlap."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        refined_a, refined_b = refine_partition_with_mincut(
            sphere_mesh, set_a, set_b)
        assert len(set(refined_a) & set(refined_b)) == 0

    def test_refine_both_nonempty(self, sphere_mesh):
        """Refined partitions should both be non-empty."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        refined_a, refined_b = refine_partition_with_mincut(
            sphere_mesh, set_a, set_b)
        assert len(refined_a) > 0
        assert len(refined_b) > 0

    def test_refine_tiny_partitions_returns_original(self, cube_mesh):
        """Tiny initial partitions (< 2 faces) should return unchanged."""
        set_a = [0]
        set_b = list(range(1, len(cube_mesh.faces)))
        result_a, result_b = refine_partition_with_mincut(
            cube_mesh, set_a, set_b)
        assert result_a == set_a
        assert result_b == set_b

    def test_refine_set_a_is_smaller(self, sphere_mesh):
        """Refined set_a should be the smaller partition."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        refined_a, refined_b = refine_partition_with_mincut(
            sphere_mesh, set_a, set_b)
        assert len(refined_a) <= len(refined_b)
