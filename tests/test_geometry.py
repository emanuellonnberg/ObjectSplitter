# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for core.geometry module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from core.geometry import (
    rotation_matrix_from_vectors,
    world_to_local_transform,
    transform_point_to_local,
    plane_normal_from_spherical,
    create_plane_mesh_data,
    create_marker_mesh_data,
    create_pin_mesh_data,
)


class TestRotationMatrixFromVectors:
    """Tests for Rodrigues' rotation formula implementation."""

    def test_identity_same_direction(self):
        """Same direction vectors produce identity matrix."""
        v = numpy.array([0, 1, 0])
        R = rotation_matrix_from_vectors(v, v)
        assert_allclose(R, numpy.eye(3), atol=1e-10)

    def test_opposite_direction(self):
        """Opposite direction produces 180-degree rotation."""
        v1 = numpy.array([0, 0, 1])
        v2 = numpy.array([0, 0, -1])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        assert_allclose(result, v2, atol=1e-10)

    def test_90_degree_rotation(self):
        """Rotation from X to Y axis."""
        v1 = numpy.array([1, 0, 0])
        v2 = numpy.array([0, 1, 0])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        assert_allclose(result, v2, atol=1e-10)

    def test_arbitrary_rotation(self):
        """Rotation between arbitrary unit vectors."""
        v1 = numpy.array([1, 0, 0])
        v2 = numpy.array([0.5, 0.5, numpy.sqrt(0.5)])
        v2 = v2 / numpy.linalg.norm(v2)
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        assert_allclose(result, v2, atol=1e-10)

    def test_result_is_rotation_matrix(self):
        """Result should be orthogonal with determinant +1."""
        v1 = numpy.array([0.3, 0.7, 0.1])
        v2 = numpy.array([-0.5, 0.2, 0.8])
        R = rotation_matrix_from_vectors(v1, v2)
        # Orthogonality: R^T @ R = I
        assert_allclose(R.T @ R, numpy.eye(3), atol=1e-10)
        # Determinant = 1 (proper rotation, not reflection)
        assert_allclose(numpy.linalg.det(R), 1.0, atol=1e-10)

    def test_unnormalized_inputs(self):
        """Should handle non-unit input vectors."""
        v1 = numpy.array([3, 0, 0])
        v2 = numpy.array([0, 5, 0])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ (v1 / numpy.linalg.norm(v1))
        expected = v2 / numpy.linalg.norm(v2)
        assert_allclose(result, expected, atol=1e-10)


class TestWorldToLocalTransform:
    """Tests for coordinate system estimation from vertex pairs."""

    def test_identity_transform(self):
        """Same local and world vertices => identity transform."""
        verts = numpy.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
        ], dtype=numpy.float64)
        result = world_to_local_transform(verts, verts)
        assert result is not None
        R, T, scale = result
        assert_allclose(R, numpy.eye(3), atol=1e-6)
        assert_allclose(T, [0, 0, 0], atol=1e-6)
        assert_allclose(scale, 1.0, atol=1e-6)

    def test_pure_translation(self):
        """Translated vertices should recover the translation."""
        local_verts = numpy.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
        ], dtype=numpy.float64)
        offset = numpy.array([100, 200, 300])
        world_verts = local_verts + offset
        result = world_to_local_transform(local_verts, world_verts)
        assert result is not None
        R, T, scale = result
        assert_allclose(T, offset, atol=1e-3)
        assert_allclose(scale, 1.0, atol=1e-3)

    def test_uniform_scale(self):
        """Scaled vertices should recover the scale factor."""
        local_verts = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=numpy.float64)
        world_verts = local_verts * 2.5
        result = world_to_local_transform(local_verts, world_verts)
        assert result is not None
        _, _, scale = result
        assert_allclose(scale, 2.5, atol=1e-3)

    def test_too_few_vertices(self):
        """Should return None with < 3 vertices."""
        verts = numpy.array([[0, 0, 0], [1, 0, 0]])
        assert world_to_local_transform(verts, verts) is None

    def test_none_input(self):
        """Should return None for None input."""
        assert world_to_local_transform(None, numpy.zeros((3, 3))) is None


class TestTransformPointToLocal:
    """Tests for point transformation from world to local space."""

    def test_identity_transform(self):
        """Identity rotation, zero translation, unit scale."""
        point = numpy.array([5.0, 10.0, 15.0])
        R = numpy.eye(3)
        T = numpy.zeros(3)
        result = transform_point_to_local(point, R, T, 1.0)
        assert_allclose(result, point, atol=1e-10)

    def test_translation_only(self):
        """Pure translation should subtract T."""
        point = numpy.array([15.0, 25.0, 35.0])
        R = numpy.eye(3)
        T = numpy.array([10.0, 20.0, 30.0])
        result = transform_point_to_local(point, R, T, 1.0)
        assert_allclose(result, [5.0, 5.0, 5.0], atol=1e-10)

    def test_scale_only(self):
        """Uniform scale should divide coordinates."""
        point = numpy.array([10.0, 20.0, 30.0])
        R = numpy.eye(3)
        T = numpy.zeros(3)
        result = transform_point_to_local(point, R, T, 2.0)
        assert_allclose(result, [5.0, 10.0, 15.0], atol=1e-10)


class TestPlaneNormalFromSpherical:
    """Tests for spherical coordinate to normal conversion."""

    def test_north_pole(self):
        """theta=0 should give +Y direction."""
        normal = plane_normal_from_spherical(0, 0)
        assert_allclose(normal, [0, 1, 0], atol=1e-10)

    def test_south_pole(self):
        """theta=pi should give -Y direction."""
        normal = plane_normal_from_spherical(numpy.pi, 0)
        assert_allclose(normal, [0, -1, 0], atol=1e-10)

    def test_equator_x(self):
        """theta=pi/2, phi=0 should give +X direction."""
        normal = plane_normal_from_spherical(numpy.pi / 2, 0)
        assert_allclose(normal, [1, 0, 0], atol=1e-10)

    def test_equator_z(self):
        """theta=pi/2, phi=pi/2 should give +Z direction."""
        normal = plane_normal_from_spherical(numpy.pi / 2, numpy.pi / 2)
        assert_allclose(normal, [0, 0, 1], atol=1e-10)

    def test_unit_length(self):
        """Result should always be a unit vector."""
        for theta in numpy.linspace(0, numpy.pi, 10):
            for phi in numpy.linspace(0, 2 * numpy.pi, 10):
                normal = plane_normal_from_spherical(theta, phi)
                assert_allclose(numpy.linalg.norm(normal), 1.0, atol=1e-10)


class TestCreatePlaneMeshData:
    """Tests for preview plane mesh generation."""

    def test_basic_output_shape(self):
        """Should return 4 vertices and 4 triangle faces."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        vertices, indices = create_plane_mesh_data(origin, normal, 10.0)
        assert vertices.shape == (4, 3)
        assert indices.shape == (4, 3)

    def test_vertices_centered_on_origin(self):
        """Vertices should be centered around the specified origin."""
        origin = numpy.array([10, 20, 30])
        normal = numpy.array([0, 1, 0])
        vertices, _ = create_plane_mesh_data(origin, normal, 10.0)
        centroid = vertices.mean(axis=0)
        assert_allclose(centroid, origin, atol=1e-5)

    def test_vertices_on_plane(self):
        """All vertices should lie on the specified plane."""
        origin = numpy.array([5, 5, 5])
        normal = numpy.array([0, 1, 0])
        vertices, _ = create_plane_mesh_data(origin, normal, 20.0)
        # Distance to plane = dot(v - origin, normal) should be ~0
        distances = numpy.dot(vertices - origin, normal)
        assert_allclose(distances, 0.0, atol=1e-5)

    def test_plane_size(self):
        """Vertices should span approximately the requested size."""
        origin = numpy.zeros(3)
        normal = numpy.array([0, 1, 0])
        size = 50.0
        vertices, _ = create_plane_mesh_data(origin, normal, size)
        span = vertices.max(axis=0) - vertices.min(axis=0)
        # The plane lies in XZ (normal=Y), so X and Z spans should be ~size
        assert span[0] == pytest.approx(size, rel=0.01) or span[2] == pytest.approx(size, rel=0.01)


class TestCreateMarkerMeshData:
    """Tests for arrow marker mesh."""

    def test_arrow_at_origin(self):
        """Arrow marker should produce valid vertices and faces."""
        vertices, indices = create_marker_mesh_data(numpy.array([0, 0, 0]), 1.0)
        assert vertices.shape[1] == 3
        assert indices.shape[1] == 3
        assert len(vertices) > 0
        assert len(indices) > 0

    def test_arrow_tip_at_center(self):
        """Arrow tip (first vertex) should be at the given center point."""
        center = numpy.array([10, 20, 30])
        vertices, _ = create_marker_mesh_data(center, 2.0)
        assert_allclose(vertices[0], center, atol=1e-5)

    def test_arrow_default_direction(self):
        """Default arrow direction should extend along Y+ from center."""
        center = numpy.array([0, 0, 0])
        vertices, _ = create_marker_mesh_data(center, 1.0)
        # All non-tip vertices should have Y >= 0
        assert numpy.all(vertices[1:, 1] >= -0.01)
        assert vertices[:, 1].max() > 2.0

    def test_arrow_custom_direction(self):
        """Arrow with custom normal should point in that direction."""
        center = numpy.array([0, 0, 0])
        # Arrow along +X: body should extend in +X direction
        vertices, _ = create_marker_mesh_data(center, 1.0, direction=numpy.array([1, 0, 0]))
        assert vertices[:, 0].max() > 2.0  # shaft extends in +X
        assert_allclose(vertices[0], center, atol=1e-5)  # tip at center

    def test_arrow_negative_direction(self):
        """Arrow pointing in -Z should extend in -Z direction."""
        center = numpy.array([5, 5, 5])
        vertices, _ = create_marker_mesh_data(center, 1.0, direction=numpy.array([0, 0, -1]))
        assert_allclose(vertices[0], center, atol=1e-5)
        # Shaft should extend in -Z from center
        assert vertices[:, 2].min() < 5.0 - 2.0

    def test_arrow_zero_direction_falls_back(self):
        """Zero-length direction should fall back to default Y-up."""
        center = numpy.array([0, 0, 0])
        vertices, _ = create_marker_mesh_data(center, 1.0, direction=numpy.array([0, 0, 0]))
        # Should behave like default: shaft extends in +Y
        assert vertices[:, 1].max() > 2.0
        assert_allclose(vertices[0], center, atol=1e-5)

    def test_arrow_face_indices_valid(self):
        """All face indices should reference existing vertices."""
        center = numpy.array([0, 0, 0])
        vertices, indices = create_marker_mesh_data(center, 2.0,
                                                     direction=numpy.array([1, 1, 0]))
        assert indices.min() >= 0
        assert indices.max() < len(vertices)


class TestCreatePinMeshData:
    """Tests for pin/dot marker mesh generation."""

    def test_pin_basic_shape(self):
        """Pin should produce valid vertices and triangular faces."""
        center = numpy.array([0, 0, 0])
        vertices, indices = create_pin_mesh_data(center, 1.0)
        assert vertices.shape[1] == 3
        assert indices.shape[1] == 3
        assert len(vertices) > 0
        assert len(indices) > 0

    def test_pin_face_indices_valid(self):
        """All face indices should reference existing vertices."""
        center = numpy.array([5, 10, 15])
        vertices, indices = create_pin_mesh_data(center, 2.0)
        assert indices.min() >= 0
        assert indices.max() < len(vertices)

    def test_pin_default_direction_extends_along_y(self):
        """Default direction (Y-up) should have geometry above and below center."""
        center = numpy.array([0, 0, 0])
        vertices, _ = create_pin_mesh_data(center, 1.0)
        # Diamond extends both above and below center
        assert vertices[:, 1].max() > 0.0
        assert vertices[:, 1].min() < 0.0

    def test_pin_custom_direction(self):
        """Pin with custom direction should orient along that direction."""
        center = numpy.array([0, 0, 0])
        direction = numpy.array([0, 0, 1])
        vertices, _ = create_pin_mesh_data(center, 1.0, direction=direction)
        # Pin extends in +Z (above) and -Z (spike below)
        assert vertices[:, 2].max() > 0.0
        assert vertices[:, 2].min() < 0.0

    def test_pin_at_position(self):
        """Pin should be centered near the specified position."""
        center = numpy.array([10, 20, 30])
        vertices, _ = create_pin_mesh_data(center, 1.0)
        centroid = vertices.mean(axis=0)
        # Centroid should be near center (may be offset slightly due to asymmetry)
        assert numpy.linalg.norm(centroid - center) < 2.0

    def test_pin_scale_factor(self):
        """Larger scale should produce larger geometry."""
        center = numpy.array([0, 0, 0])
        verts_small, _ = create_pin_mesh_data(center, 0.5)
        verts_large, _ = create_pin_mesh_data(center, 2.0)
        extent_small = verts_small.max(axis=0) - verts_small.min(axis=0)
        extent_large = verts_large.max(axis=0) - verts_large.min(axis=0)
        assert numpy.all(extent_large > extent_small)

    def test_pin_zero_direction_falls_back(self):
        """Zero direction should fall back to Y-up."""
        center = numpy.array([0, 0, 0])
        vertices, _ = create_pin_mesh_data(center, 1.0, direction=numpy.array([0, 0, 0]))
        # Should still produce valid geometry along Y axis
        assert vertices[:, 1].max() > 0.0

    def test_pin_vertex_count_matches_hexagonal(self):
        """Pin uses 6-sided cross-section: 6 ring + 1 top + 1 bottom = 8 vertices."""
        center = numpy.array([0, 0, 0])
        vertices, indices = create_pin_mesh_data(center, 1.0)
        assert len(vertices) == 8  # 6 ring + top + bottom
        assert len(indices) == 12  # 6 upper cone + 6 lower cone
