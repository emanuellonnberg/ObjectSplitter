# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for core.mesh_splitter module."""

import numpy
import pytest

from core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
    SplitResult,
)
from core.plane_calculator import find_shortest_seam_partition


class TestSliceMeshWithFallback:
    """Tests for the multi-strategy mesh slicing."""

    def test_horizontal_cube_split(self, cube_mesh):
        """Splitting a cube horizontally should produce two valid halves."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        assert result.success
        assert len(result.upper.vertices) > 0
        assert len(result.lower.vertices) > 0

    def test_vertical_cube_split(self, cube_mesh):
        """Splitting a cube vertically should produce two valid halves."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([1, 0, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        assert result.success

    def test_diagonal_cube_split(self, cube_mesh):
        """Splitting a cube diagonally should work."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([1, 1, 0])
        normal = normal / numpy.linalg.norm(normal)
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        assert result.success

    def test_sphere_split(self, sphere_mesh):
        """Splitting a sphere should produce two valid halves."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(sphere_mesh, origin, normal)
        assert result.success
        # Sphere split in half should give roughly equal parts
        ratio = len(result.upper.faces) / max(1, len(result.lower.faces))
        assert 0.3 < ratio < 3.0

    def test_off_center_split(self, cube_mesh):
        """Splitting off-center should produce two valid non-empty halves."""
        origin = numpy.array([0, 5, 0])  # Near top of 20mm cube
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        assert result.success
        # Both halves must be non-empty and the split actually occurred
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0
        # Upper centroid should be above cut plane, lower below
        assert result.upper.centroid[1] > origin[1] - 1.0
        assert result.lower.centroid[1] < origin[1] + 1.0

    def test_plane_outside_mesh_returns_empty(self, cube_mesh):
        """A cut plane entirely outside the mesh should fail gracefully."""
        origin = numpy.array([0, 100, 0])  # Way above the cube
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        # Should fail since one side will be empty
        assert not result.success

    def test_strategy_tracking(self, cube_mesh):
        """Result should record which strategies were attempted."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        assert len(result.strategies_attempted) > 0
        assert result.strategy_used != "none"

    def test_summary_string(self, cube_mesh):
        """summary() should return a descriptive string."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "OK" in summary or "FAILED" in summary

    def test_cylinder_horizontal_split(self, cylinder_mesh):
        """Splitting a cylinder horizontally should work."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cylinder_mesh, origin, normal)
        assert result.success

    def test_translated_mesh_split(self, translated_cube_mesh):
        """Splitting a translated mesh should work with correct origin."""
        center = translated_cube_mesh.centroid
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(translated_cube_mesh, center, normal)
        assert result.success

    def test_only_cuts_clicked_component(self):
        """When face_id is provided, only the connected component containing that face is cut."""
        import trimesh
        # Two disconnected cubes: A at x=-15..-5, B at x=5..15
        cube_a = trimesh.creation.box(extents=[10, 10, 10])
        cube_a.apply_translation([-10, 0, 0])
        cube_b = trimesh.creation.box(extents=[10, 10, 10])
        cube_b.apply_translation([10, 0, 0])
        combined = trimesh.util.concatenate([cube_a, cube_b])
        # Plane at x=0 would cut through both. Face 0 is in cube A.
        plane_origin = numpy.array([0, 0, 0])
        plane_normal = numpy.array([1, 0, 0])
        result = slice_mesh_with_fallback(
            combined, plane_origin, plane_normal, face_id=0
        )
        assert result.success
        # Cube A has 12 faces. We cut it -> upper + lower. Cube B (12 faces) stays intact.
        # Upper: right half of A + entire B. Lower: left half of A.
        total_faces = len(result.upper.faces) + len(result.lower.faces)
        assert total_faces == len(combined.faces)
        # Cube B should be entirely in one result (upper, since centroid x=10 > 0)
        # So upper should have more faces (half of A + all of B)
        assert len(result.upper.faces) >= 12  # At least cube B
        assert len(result.lower.faces) >= 1   # At least part of cube A


class TestSplitByShortestSeam:
    """Tests for face-partition-based splitting."""

    def test_cube_split(self, cube_mesh):
        """Should split a cube into two non-empty submeshes."""
        click = numpy.array([10.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(cube_mesh, click)
        result = split_by_shortest_seam(cube_mesh, set_a, set_b)
        assert result.success
        assert result.strategy_used == "shortest_seam"

    def test_sphere_split(self, sphere_mesh):
        """Should split a sphere into two non-empty submeshes."""
        click = numpy.array([15.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(sphere_mesh, click)
        result = split_by_shortest_seam(sphere_mesh, set_a, set_b)
        assert result.success

    def test_face_counts_preserved(self, cube_mesh):
        """Total faces across both parts should equal original face count."""
        click = numpy.array([10.0, 0.0, 0.0])
        set_a, set_b, _, _ = find_shortest_seam_partition(cube_mesh, click)
        result = split_by_shortest_seam(cube_mesh, set_a, set_b)
        assert result.success
        total_faces = len(result.upper.faces) + len(result.lower.faces)
        assert total_faces == len(cube_mesh.faces)
