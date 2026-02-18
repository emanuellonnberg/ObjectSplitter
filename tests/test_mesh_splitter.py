# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for core.mesh_splitter module."""

import numpy
import pytest

import core.mesh_splitter as mesh_splitter_module
from core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
    split_by_local_plane,
    split_by_face_sets,
    local_plane_partition,
    _component_containing_face,
    SplitResult,
)
from core.plane_calculator import find_shortest_seam_partition, find_smallest_cut_plane


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


class TestComponentContainingFace:
    """Tests for _component_containing_face helper."""

    def test_single_component_returns_all(self, cube_mesh):
        """For a single-component mesh, should return all faces."""
        faces = _component_containing_face(cube_mesh, 0)
        assert len(faces) == len(cube_mesh.faces)

    def test_disconnected_returns_correct_component(self):
        """For disconnected mesh, should return only the clicked component's faces."""
        import trimesh
        cube_a = trimesh.creation.box(extents=[10, 10, 10])
        cube_a.apply_translation([-20, 0, 0])
        cube_b = trimesh.creation.box(extents=[10, 10, 10])
        cube_b.apply_translation([20, 0, 0])
        combined = trimesh.util.concatenate([cube_a, cube_b])

        faces_a = _component_containing_face(combined, 0)
        # cube_a has 12 faces, cube_b has 12 faces
        assert len(faces_a) == 12

        faces_b = _component_containing_face(combined, len(cube_a.faces))
        assert len(faces_b) == 12

    def test_three_components(self):
        """Should correctly identify component in a 3-component mesh."""
        import trimesh
        boxes = []
        for i in range(3):
            box = trimesh.creation.box(extents=[5, 5, 5])
            box.apply_translation([i * 20, 0, 0])
            boxes.append(box)
        combined = trimesh.util.concatenate(boxes)

        # Face 0 is in first box (12 faces each)
        faces_0 = _component_containing_face(combined, 0)
        assert len(faces_0) == 12

        # Face 12 is in second box
        faces_12 = _component_containing_face(combined, 12)
        assert len(faces_12) == 12

        # Face 24 is in third box
        faces_24 = _component_containing_face(combined, 24)
        assert len(faces_24) == 12


class TestLocalPlanePartition:
    """Tests for local_plane_partition function."""

    def test_basic_partition(self, cube_mesh):
        """Should partition cube into two non-empty sets."""
        origin = numpy.array([0.0, 0.0, 0.0])
        normal = numpy.array([0.0, 1.0, 0.0])
        set_a, set_b = local_plane_partition(cube_mesh, origin, normal, 0)
        assert len(set_a) > 0
        assert len(set_b) > 0
        assert len(set_a) + len(set_b) == len(cube_mesh.faces)

    def test_smaller_set_is_first(self, cube_mesh):
        """set_a should be the smaller partition."""
        origin = numpy.array([0.0, 5.0, 0.0])  # Near top
        normal = numpy.array([0.0, 1.0, 0.0])
        set_a, set_b = local_plane_partition(cube_mesh, origin, normal, 0)
        assert len(set_a) <= len(set_b)

    def test_no_overlap(self, cube_mesh):
        """Partitions should not overlap."""
        origin = numpy.array([0.0, 0.0, 0.0])
        normal = numpy.array([1.0, 0.0, 0.0])
        set_a, set_b = local_plane_partition(cube_mesh, origin, normal, 0)
        assert len(set(set_a) & set(set_b)) == 0

    def test_sphere_local_partition(self, sphere_mesh):
        """Should work on curved geometry."""
        origin = sphere_mesh.centroid
        normal = numpy.array([1.0, 0.0, 0.0])
        set_a, set_b = local_plane_partition(sphere_mesh, origin, normal, 0)
        assert len(set_a) > 0
        assert len(set_b) > 0
        assert len(set_a) + len(set_b) == len(sphere_mesh.faces)


class TestSplitByLocalPlane:
    """Tests for split_by_local_plane function."""

    def test_basic_split(self, sphere_mesh):
        """Should split a sphere using local plane partition."""
        center = sphere_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        normals = [
            numpy.array([0.0, 1.0, 0.0]),
            numpy.array([1.0, 0.0, 0.0]),
            numpy.array([0.0, 0.0, 1.0]),
        ]
        result = split_by_local_plane(sphere_mesh, center, normals, 0)
        assert result.success
        assert "local_plane_partition" in result.strategies_attempted

    def test_split_preserves_faces(self, cube_mesh):
        """Total faces should be preserved."""
        normals = [numpy.array([0.0, 1.0, 0.0])]
        result = split_by_local_plane(cube_mesh, cube_mesh.centroid, normals, 0)
        assert result.success
        total = len(result.upper.faces) + len(result.lower.faces)
        assert total == len(cube_mesh.faces)

    def test_empty_candidates_fails(self, cube_mesh):
        """No candidate normals should fail gracefully."""
        result = split_by_local_plane(cube_mesh, cube_mesh.centroid, [], 0)
        assert not result.success

    def test_with_real_smallest_candidates(self, cylinder_mesh):
        """Use actual candidates from find_smallest_cut_plane."""
        center = cylinder_mesh.centroid + numpy.array([0.1, 0.1, 0.1])
        search = find_smallest_cut_plane(
            cylinder_mesh, center, search_resolution=6)
        if search.top_candidates:
            normals = [n for _, n in search.top_candidates[:5]]
            result = split_by_local_plane(
                cylinder_mesh, center, normals, source_face_id=0)
            assert result.success


class TestMultiComponentSlicing:
    """Tests for multi-component mesh slicing with face_id."""

    def test_two_components_only_cuts_target(self):
        """With face_id, only the target component should be cut."""
        import trimesh
        cube_a = trimesh.creation.box(extents=[10, 10, 10])
        cube_a.apply_translation([-15, 0, 0])
        cube_b = trimesh.creation.box(extents=[10, 10, 10])
        cube_b.apply_translation([15, 0, 0])
        combined = trimesh.util.concatenate([cube_a, cube_b])

        # Cut at x=0, targeting cube_a (face 0)
        result = slice_mesh_with_fallback(
            combined,
            numpy.array([0, 0, 0]),
            numpy.array([1, 0, 0]),
            face_id=0
        )
        assert result.success
        total = len(result.upper.faces) + len(result.lower.faces)
        assert total == len(combined.faces)

    def test_three_components_preserves_others(self):
        """Three cubes: cut only one, others assigned by centroid side."""
        import trimesh
        cubes = []
        for x in [-30, 0, 30]:
            box = trimesh.creation.box(extents=[10, 10, 10])
            box.apply_translation([x, 0, 0])
            cubes.append(box)
        combined = trimesh.util.concatenate(cubes)

        # Cut the middle cube (faces 12-23), plane along Y
        result = slice_mesh_with_fallback(
            combined,
            numpy.array([0, 0, 0]),
            numpy.array([0, 1, 0]),
            face_id=12  # A face in the middle cube
        )
        assert result.success
        # Both halves should be non-empty
        assert len(result.upper.faces) > 0
        assert len(result.lower.faces) > 0
        # Total faces may exceed original due to capping faces
        total = len(result.upper.faces) + len(result.lower.faces)
        assert total >= len(combined.faces)

    def test_without_face_id_cuts_everything(self):
        """Without face_id, the plane should cut through all components."""
        import trimesh
        cube_a = trimesh.creation.box(extents=[10, 10, 10])
        cube_a.apply_translation([-10, 0, 0])
        cube_b = trimesh.creation.box(extents=[10, 10, 10])
        cube_b.apply_translation([10, 0, 0])
        combined = trimesh.util.concatenate([cube_a, cube_b])

        # Horizontal cut with no face_id — should cut both
        result = slice_mesh_with_fallback(
            combined,
            numpy.array([0, 0, 0]),
            numpy.array([0, 1, 0])
        )
        assert result.success
        # Both halves should have faces from both cubes
        assert len(result.upper.faces) > 12
        assert len(result.lower.faces) > 12


class TestFallbackChain:
    """Tests for the strategy fallback chain in slice_mesh_with_fallback."""

    def test_strategy_is_recorded(self, cube_mesh):
        """Successful split should record the strategy used."""
        result = slice_mesh_with_fallback(
            cube_mesh,
            numpy.array([0, 0, 0]),
            numpy.array([0, 1, 0])
        )
        assert result.success
        assert result.strategy_used in [
            "capped_slice", "manual_cap", "uncapped_slice", "manual_split"
        ]

    def test_failure_records_strategies(self, cube_mesh):
        """Failed split should record at least one strategy attempted."""
        # Plane far outside mesh — should fail
        result = slice_mesh_with_fallback(
            cube_mesh,
            numpy.array([0, 1000, 0]),
            numpy.array([0, 1, 0])
        )
        assert not result.success
        # Strategy 1 returns partial results (upper empty, lower full) and
        # may short-circuit. At minimum 1 strategy should be recorded.
        assert len(result.strategies_attempted) >= 1

    def test_torus_split_uses_fallback(self, torus_mesh):
        """Torus geometry may require fallback strategies."""
        result = slice_mesh_with_fallback(
            torus_mesh,
            torus_mesh.centroid,
            numpy.array([0, 1, 0])
        )
        # Whether it succeeds depends on the strategy chain
        assert len(result.strategies_attempted) >= 1
        if result.success:
            assert result.strategy_used != "none"

    def test_manual_cap_path_is_watertight_on_cube(self, cube_mesh, monkeypatch):
        """Force strategy-2 and verify manual capping really seals both halves."""
        original_slice = mesh_splitter_module.trimesh.intersections.slice_mesh_plane

        def _force_manual_cap(mesh, plane_normal, plane_origin, cap=False):
            if cap:
                raise ImportError("forced test fallback to manual cap")
            return original_slice(
                mesh,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                cap=False,
            )

        monkeypatch.setattr(
            mesh_splitter_module.trimesh.intersections,
            "slice_mesh_plane",
            _force_manual_cap,
        )

        result = slice_mesh_with_fallback(
            cube_mesh,
            numpy.array([0, 0, 0]),
            numpy.array([0, 1, 0]),
        )
        assert result.success
        assert result.strategy_used == "manual_cap"
        assert result.capped
        assert result.upper.is_watertight
        assert result.lower.is_watertight

    def test_manual_cap_failure_falls_back_to_uncapped(self, cube_mesh, monkeypatch):
        """If manual cap generation fails, uncapped slices should still be returned."""
        original_slice = mesh_splitter_module.trimesh.intersections.slice_mesh_plane

        def _force_manual_cap(mesh, plane_normal, plane_origin, cap=False):
            if cap:
                raise ImportError("forced test fallback to manual cap")
            return original_slice(
                mesh,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                cap=False,
            )

        monkeypatch.setattr(
            mesh_splitter_module.trimesh.intersections,
            "slice_mesh_plane",
            _force_manual_cap,
        )
        monkeypatch.setattr(mesh_splitter_module, "_manual_cap_mesh", lambda *_args, **_kwargs: None)

        result = slice_mesh_with_fallback(
            cube_mesh,
            numpy.array([0, 0, 0]),
            numpy.array([0, 1, 0]),
        )
        assert result.success
        assert result.strategy_used == "uncapped_slice"
        assert not result.capped
