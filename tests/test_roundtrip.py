# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Geometric verification and roundtrip tests for mesh splitting.

These tests go beyond "did the split succeed?" to verify measurable
geometric properties of the output meshes: volume conservation,
bounding-box correctness, vertex-side consistency, and merge-then-split
roundtrip identity.
"""

import numpy
import pytest
import trimesh

from core.plane_calculator import horizontal_cut_plane, vertical_cut_plane
from core.mesh_splitter import slice_mesh_with_fallback


# ---------------------------------------------------------------------------
# Volume conservation
# ---------------------------------------------------------------------------

class TestVolumeConservation:
    """Splitting should preserve total volume (within tolerance)."""

    def test_cube_horizontal_50pct(self, cube_mesh):
        """Cube split at 50% -> two halves with equal volume."""
        original_vol = cube_mesh.volume
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success
        if result.capped:
            combined_vol = abs(result.upper.volume) + abs(result.lower.volume)
            assert combined_vol == pytest.approx(original_vol, rel=0.05)

    def test_cube_horizontal_25pct(self, cube_mesh):
        """Cube split at 25% -> 1:3 volume ratio."""
        original_vol = cube_mesh.volume
        plane = horizontal_cut_plane(cube_mesh, 25.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success
        if result.capped:
            combined_vol = abs(result.upper.volume) + abs(result.lower.volume)
            assert combined_vol == pytest.approx(original_vol, rel=0.05)
            # Lower should be roughly 25% of total, upper 75%
            lower_ratio = abs(result.lower.volume) / original_vol
            assert lower_ratio == pytest.approx(0.25, abs=0.05)

    def test_sphere_horizontal_50pct(self, sphere_mesh):
        """Sphere split at 50% -> two roughly equal hemispheres."""
        original_vol = sphere_mesh.volume
        plane = horizontal_cut_plane(sphere_mesh, 50.0)
        result = slice_mesh_with_fallback(sphere_mesh, plane.origin, plane.normal)
        assert result.success
        if result.capped:
            combined_vol = abs(result.upper.volume) + abs(result.lower.volume)
            # Sphere discretization is less precise; allow wider tolerance
            assert combined_vol == pytest.approx(original_vol, rel=0.10)

    def test_tall_box_multiple_heights(self, tall_box_mesh):
        """Volume conservation across several cut heights."""
        original_vol = tall_box_mesh.volume
        for pct in [10, 30, 50, 70, 90]:
            plane = horizontal_cut_plane(tall_box_mesh, pct)
            result = slice_mesh_with_fallback(
                tall_box_mesh, plane.origin, plane.normal)
            assert result.success, f"Split failed at {pct}%"
            if result.capped:
                combined_vol = abs(result.upper.volume) + abs(result.lower.volume)
                assert combined_vol == pytest.approx(original_vol, rel=0.05), (
                    f"Volume mismatch at {pct}%: {combined_vol:.2f} vs {original_vol:.2f}")

    def test_cube_vertical_center(self, cube_mesh):
        """Vertical split should also conserve volume."""
        original_vol = cube_mesh.volume
        plane = vertical_cut_plane(cube_mesh.centroid)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success
        if result.capped:
            combined_vol = abs(result.upper.volume) + abs(result.lower.volume)
            assert combined_vol == pytest.approx(original_vol, rel=0.05)


# ---------------------------------------------------------------------------
# Bounding-box correctness
# ---------------------------------------------------------------------------

class TestBoundingBoxCorrectness:
    """Each half should fit inside the original and respect the cut plane."""

    def test_cube_horizontal_halves_within_original(self, cube_mesh):
        """Both halves of a horizontal cut must fit inside the original bbox."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        orig_min, orig_max = cube_mesh.bounds
        for half in [result.upper, result.lower]:
            h_min, h_max = half.bounds
            # Each half's bounds must not exceed the original (with tolerance)
            numpy.testing.assert_array_less(orig_min - 0.5, h_min)
            numpy.testing.assert_array_less(h_max, orig_max + 0.5)

    def test_cube_horizontal_centroids_correct_side(self, cube_mesh):
        """Mesh centroid of each half should be on the correct side of cut."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        cut_y = plane.origin[1]
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        assert result.upper.centroid[1] > cut_y
        assert result.lower.centroid[1] < cut_y

    def test_tall_box_30pct_volume_ratio(self, tall_box_mesh):
        """30% cut on a 40mm tall box -> lower ~30% volume, upper ~70%."""
        plane = horizontal_cut_plane(tall_box_mesh, 30.0)
        result = slice_mesh_with_fallback(
            tall_box_mesh, plane.origin, plane.normal)
        assert result.success

        if result.capped:
            original_vol = tall_box_mesh.volume
            lower_ratio = abs(result.lower.volume) / original_vol
            upper_ratio = abs(result.upper.volume) / original_vol
            assert lower_ratio == pytest.approx(0.30, abs=0.05)
            assert upper_ratio == pytest.approx(0.70, abs=0.05)

    def test_vertical_split_x_extents(self, cube_mesh):
        """Vertical split through center -> each half ~10mm wide in X."""
        plane = vertical_cut_plane(cube_mesh.centroid)  # normal=[1,0,0]
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        for half in [result.upper, result.lower]:
            x_extent = half.bounds[1][0] - half.bounds[0][0]
            assert x_extent == pytest.approx(10.0, abs=1.0)


# ---------------------------------------------------------------------------
# Vertex-side consistency
# ---------------------------------------------------------------------------

class TestFaceAreaDistribution:
    """The vast majority of face area should be on the correct side.

    The capping process (Delaunay triangulation of the cross-section
    boundary) can introduce cap faces whose centroids project to the
    wrong side of the plane.  Without manifold3d / mapbox-earcut the
    Delaunay cap is coarser, so we use a conservative threshold of 80%.
    With better cap engines installed the fractions are typically >99%.
    """

    @staticmethod
    def _correct_side_area_fraction(mesh, plane_origin, plane_normal):
        """Fraction of total face area whose centroid is on the correct side."""
        centroids = mesh.triangles_center
        dists = numpy.dot(centroids - plane_origin, plane_normal)
        areas = mesh.area_faces
        correct_area = areas[dists >= -1e-6].sum()
        return correct_area / areas.sum()

    def test_cube_horizontal_area_distribution(self, cube_mesh):
        """>=80% of upper face area should be above the cut plane."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        upper_frac = self._correct_side_area_fraction(
            result.upper, plane.origin, plane.normal)
        lower_frac = self._correct_side_area_fraction(
            result.lower, plane.origin, -plane.normal)

        assert upper_frac >= 0.80, f"Only {upper_frac:.1%} of upper area is above plane"
        assert lower_frac >= 0.80, f"Only {lower_frac:.1%} of lower area is below plane"

    def test_sphere_horizontal_area_distribution(self, sphere_mesh):
        """Same check on a sphere (curved geometry)."""
        plane = horizontal_cut_plane(sphere_mesh, 50.0)
        result = slice_mesh_with_fallback(sphere_mesh, plane.origin, plane.normal)
        assert result.success

        upper_frac = self._correct_side_area_fraction(
            result.upper, plane.origin, plane.normal)
        lower_frac = self._correct_side_area_fraction(
            result.lower, plane.origin, -plane.normal)

        assert upper_frac >= 0.80, f"Only {upper_frac:.1%} of upper area is above plane"
        assert lower_frac >= 0.80, f"Only {lower_frac:.1%} of lower area is below plane"

    def test_vertical_split_area_distribution(self, cube_mesh):
        """Vertical split: check X-axis separation via face area."""
        plane = vertical_cut_plane(cube_mesh.centroid)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        upper_frac = self._correct_side_area_fraction(
            result.upper, plane.origin, plane.normal)
        lower_frac = self._correct_side_area_fraction(
            result.lower, plane.origin, -plane.normal)

        assert upper_frac >= 0.80, f"Only {upper_frac:.1%} of upper area is above plane"
        assert lower_frac >= 0.80, f"Only {lower_frac:.1%} of lower area is below plane"


# ---------------------------------------------------------------------------
# Merge-then-split roundtrip
# ---------------------------------------------------------------------------

class TestMergeThenSplit:
    """Stack two known objects, cut at the seam, verify we recover them."""

    @staticmethod
    def _boxes_stacked_on_y(height_a: float, height_b: float):
        """Create two boxes stacked along Y and the combined mesh."""
        width, depth = 20.0, 20.0
        box_a = trimesh.creation.box(extents=[width, height_a, depth])
        box_b = trimesh.creation.box(extents=[width, height_b, depth])

        # Place A below, B above, seam at y=0
        box_a.apply_translation([0, -height_a / 2, 0])
        box_b.apply_translation([0, height_b / 2, 0])

        combined = trimesh.util.concatenate([box_a, box_b])
        return box_a, box_b, combined

    def test_equal_boxes_roundtrip(self):
        """Two 20mm cubes stacked -> split at seam -> recover each cube."""
        box_a, box_b, combined = self._boxes_stacked_on_y(20.0, 20.0)

        origin = numpy.array([0.0, 0.0, 0.0])
        normal = numpy.array([0.0, 1.0, 0.0])
        result = slice_mesh_with_fallback(combined, origin, normal)
        assert result.success

        # Volume check: each half ≈ one original box
        if result.capped:
            assert abs(result.upper.volume) == pytest.approx(
                abs(box_b.volume), rel=0.05)
            assert abs(result.lower.volume) == pytest.approx(
                abs(box_a.volume), rel=0.05)

        # Bounding-box extents should match originals
        upper_extents = result.upper.bounds[1] - result.upper.bounds[0]
        lower_extents = result.lower.bounds[1] - result.lower.bounds[0]
        numpy.testing.assert_allclose(upper_extents, [20, 20, 20], atol=1.0)
        numpy.testing.assert_allclose(lower_extents, [20, 20, 20], atol=1.0)

    def test_unequal_boxes_roundtrip(self):
        """10mm + 30mm boxes stacked -> split -> recover original proportions."""
        box_a, box_b, combined = self._boxes_stacked_on_y(10.0, 30.0)

        origin = numpy.array([0.0, 0.0, 0.0])
        normal = numpy.array([0.0, 1.0, 0.0])
        result = slice_mesh_with_fallback(combined, origin, normal)
        assert result.success

        # Heights should match original boxes
        lower_height = result.lower.bounds[1][1] - result.lower.bounds[0][1]
        upper_height = result.upper.bounds[1][1] - result.upper.bounds[0][1]
        assert lower_height == pytest.approx(10.0, abs=1.0)
        assert upper_height == pytest.approx(30.0, abs=1.0)

        # Volume ratios
        if result.capped:
            assert abs(result.lower.volume) == pytest.approx(
                abs(box_a.volume), rel=0.05)
            assert abs(result.upper.volume) == pytest.approx(
                abs(box_b.volume), rel=0.05)

    def test_three_equal_boxes_two_cuts(self):
        """Three stacked boxes -> two sequential cuts -> three pieces."""
        h = 10.0
        width, depth = 20.0, 20.0
        box_a = trimesh.creation.box(extents=[width, h, depth])
        box_b = trimesh.creation.box(extents=[width, h, depth])
        box_c = trimesh.creation.box(extents=[width, h, depth])

        box_a.apply_translation([0, -h, 0])     # y: -15 to -5
        # box_b stays centered                   # y:  -5 to  5
        box_c.apply_translation([0, h, 0])       # y:   5 to 15

        combined = trimesh.util.concatenate([box_a, box_b, box_c])
        original_vol = combined.volume

        # First cut at y=5 -> lower=A+B, upper=C
        r1 = slice_mesh_with_fallback(
            combined,
            numpy.array([0.0, 5.0, 0.0]),
            numpy.array([0.0, 1.0, 0.0]),
        )
        assert r1.success

        upper_c = r1.upper
        upper_c_height = upper_c.bounds[1][1] - upper_c.bounds[0][1]
        assert upper_c_height == pytest.approx(h, abs=1.0)

        # Second cut at y=-5 on the lower piece -> lower=A, upper=B
        r2 = slice_mesh_with_fallback(
            r1.lower,
            numpy.array([0.0, -5.0, 0.0]),
            numpy.array([0.0, 1.0, 0.0]),
        )
        assert r2.success

        # Check heights of all three pieces
        for piece, expected_h in [(r2.lower, h), (r2.upper, h), (upper_c, h)]:
            piece_height = piece.bounds[1][1] - piece.bounds[0][1]
            assert piece_height == pytest.approx(expected_h, abs=1.0)

        # Volume conservation across all three pieces
        if r1.capped and r2.capped:
            total = (abs(r2.lower.volume)
                     + abs(r2.upper.volume)
                     + abs(upper_c.volume))
            assert total == pytest.approx(original_vol, rel=0.05)

    def test_side_by_side_vertical_roundtrip(self):
        """Two boxes side-by-side along X -> vertical split recovers them."""
        width = 15.0
        box_left = trimesh.creation.box(extents=[width, 20, 20])
        box_right = trimesh.creation.box(extents=[width, 20, 20])
        box_left.apply_translation([-width / 2, 0, 0])
        box_right.apply_translation([width / 2, 0, 0])

        combined = trimesh.util.concatenate([box_left, box_right])

        origin = numpy.array([0.0, 0.0, 0.0])
        normal = numpy.array([1.0, 0.0, 0.0])
        result = slice_mesh_with_fallback(combined, origin, normal)
        assert result.success

        # Each half should be ~15mm wide in X
        for half in [result.upper, result.lower]:
            x_extent = half.bounds[1][0] - half.bounds[0][0]
            assert x_extent == pytest.approx(width, abs=1.0)

        if result.capped:
            assert abs(result.upper.volume) == pytest.approx(
                abs(box_right.volume), rel=0.05)
            assert abs(result.lower.volume) == pytest.approx(
                abs(box_left.volume), rel=0.05)


# ---------------------------------------------------------------------------
# Centroid position verification
# ---------------------------------------------------------------------------

class TestCentroidPosition:
    """The centroid of each half should be on the correct side of the plane."""

    def test_cube_horizontal_centroids(self, cube_mesh):
        """Upper centroid above cut, lower centroid below."""
        plane = horizontal_cut_plane(cube_mesh, 50.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        cut_y = plane.origin[1]
        assert result.upper.centroid[1] > cut_y - 0.1
        assert result.lower.centroid[1] < cut_y + 0.1

    def test_cube_horizontal_25pct_centroids(self, cube_mesh):
        """25% cut: lower centroid below cut, upper centroid above cut."""
        plane = horizontal_cut_plane(cube_mesh, 25.0)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        cut_y = plane.origin[1]
        # Lower centroid must be below the cut plane
        assert result.lower.centroid[1] < cut_y
        # Upper centroid must be above the cut plane
        assert result.upper.centroid[1] > cut_y

        # Upper should have a higher centroid than lower
        assert result.upper.centroid[1] > result.lower.centroid[1]

    def test_vertical_split_centroids(self, cube_mesh):
        """Vertical X-axis split: centroids should be on opposite sides."""
        plane = vertical_cut_plane(cube_mesh.centroid)
        result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
        assert result.success

        cut_x = plane.origin[0]
        assert result.upper.centroid[0] > cut_x - 0.1
        assert result.lower.centroid[0] < cut_x + 0.1
