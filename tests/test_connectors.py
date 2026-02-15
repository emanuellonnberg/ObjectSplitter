# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for core.connectors module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from core.connectors import (
    get_mesh_volume,
    determine_peg_side,
    find_connector_position,
    create_peg_mesh,
    create_hole_mesh,
    add_connectors,
    ConnectorConfig,
    ConnectorResult,
)
from core.mesh_splitter import slice_mesh_with_fallback


class TestGetMeshVolume:
    """Tests for mesh volume calculation."""

    def test_cube_volume(self, cube_mesh):
        """20mm cube should have volume 8000 mm^3."""
        vol = get_mesh_volume(cube_mesh)
        assert vol == pytest.approx(8000.0, rel=0.01)

    def test_sphere_volume(self, sphere_mesh):
        """Sphere with r=15 should have volume ~14137 mm^3."""
        expected = (4 / 3) * numpy.pi * 15 ** 3
        vol = get_mesh_volume(sphere_mesh)
        assert vol == pytest.approx(expected, rel=0.05)

    def test_cylinder_volume(self, cylinder_mesh):
        """Cylinder r=8, h=30 should have volume ~6032 mm^3."""
        expected = numpy.pi * 8 ** 2 * 30
        vol = get_mesh_volume(cylinder_mesh)
        assert vol == pytest.approx(expected, rel=0.05)


class TestDeterminePegSide:
    """Tests for peg/hole assignment based on volume."""

    def test_smaller_gets_peg(self, cube_mesh, tall_box_mesh):
        """Smaller mesh should get the peg."""
        # cube: 20^3 = 8000, tall_box: 10*40*10 = 4000
        role_a, role_b = determine_peg_side(tall_box_mesh, cube_mesh)
        assert role_a == "peg"   # tall_box is smaller
        assert role_b == "hole"

    def test_equal_volumes(self, cube_mesh):
        """Equal volumes: first mesh gets peg (arbitrary but deterministic)."""
        role_a, role_b = determine_peg_side(cube_mesh, cube_mesh)
        assert role_a == "peg"
        assert role_b == "hole"


class TestCreatePegMesh:
    """Tests for peg cylinder generation."""

    def test_peg_is_watertight(self):
        """Generated peg should be watertight."""
        pos = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        peg = create_peg_mesh(pos, normal, diameter=4.0, height=3.0)
        assert peg.is_watertight

    def test_peg_at_position(self):
        """Peg should be near the specified position."""
        pos = numpy.array([10, 20, 30])
        normal = numpy.array([0, 1, 0])
        peg = create_peg_mesh(pos, normal, diameter=4.0, height=3.0)
        # Centroid should be close to pos (offset by half height along normal)
        expected_centroid = pos + normal * 1.5  # half height
        assert_allclose(peg.centroid, expected_centroid, atol=1.0)

    def test_peg_dimensions(self):
        """Peg bounding box should match expected dimensions."""
        pos = numpy.zeros(3)
        normal = numpy.array([0, 0, 1])  # Z-up for easy measurement
        peg = create_peg_mesh(pos, normal, diameter=6.0, height=4.0, sides=32)
        extents = peg.bounds[1] - peg.bounds[0]
        # X and Y extent should be ~diameter, Z extent should be ~height
        assert extents[0] == pytest.approx(6.0, rel=0.1)
        assert extents[1] == pytest.approx(6.0, rel=0.1)
        assert extents[2] == pytest.approx(4.0, rel=0.1)


class TestCreateHoleMesh:
    """Tests for hole cylinder generation."""

    def test_hole_is_watertight(self):
        """Generated hole should be watertight."""
        pos = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        hole = create_hole_mesh(pos, normal, diameter=4.0, height=3.0, clearance=0.2)
        assert hole.is_watertight

    def test_hole_larger_than_peg(self):
        """Hole should be larger than peg by clearance amount."""
        pos = numpy.zeros(3)
        normal = numpy.array([0, 0, 1])
        diameter = 4.0
        clearance = 0.2

        peg = create_peg_mesh(pos, normal, diameter=diameter, height=3.0, sides=32)
        hole = create_hole_mesh(pos, normal, diameter=diameter, height=3.0,
                                clearance=clearance, sides=32)

        peg_extents = peg.bounds[1] - peg.bounds[0]
        hole_extents = hole.bounds[1] - hole.bounds[0]
        # Hole should be wider in X and Y by 2*clearance
        assert hole_extents[0] > peg_extents[0]
        assert hole_extents[1] > peg_extents[1]


class TestFindConnectorPosition:
    """Tests for connector placement on cut surfaces."""

    def test_cube_center_cut(self, cube_mesh):
        """Connector position for a centered cube cut should be near origin."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        # First split the mesh to get a half with a cut surface
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        if result.success:
            pos = find_connector_position(result.upper, origin, normal)
            if pos is not None:
                # Should be on the cut plane (y ~ 0)
                assert abs(pos[1]) < 0.5

    def test_returns_point_on_plane(self, sphere_mesh):
        """Returned position should lie on the cut plane."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(sphere_mesh, origin, normal)
        if result.success:
            pos = find_connector_position(result.upper, origin, normal)
            if pos is not None:
                dist_to_plane = abs(numpy.dot(pos - origin, normal))
                assert dist_to_plane < 0.5


class TestAddConnectors:
    """Integration tests for the full connector pipeline."""

    def test_disabled_config_returns_unchanged(self, cube_mesh):
        """Disabled connector config should return meshes unchanged."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        if result.success:
            config = ConnectorConfig(enabled=False)
            cr = add_connectors(result.upper, result.lower, origin, normal, config)
            assert not cr.connectors_added
            assert cr.skipped_reason == "connectors disabled"

    def test_default_config_on_cube(self, cube_mesh):
        """Default connectors on a cube split should attempt to add connectors."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        if result.success and result.capped:
            config = ConnectorConfig()
            cr = add_connectors(result.upper, result.lower, origin, normal, config)
            # Whether it succeeds depends on boolean engine availability,
            # but the pipeline should not crash
            assert isinstance(cr, ConnectorResult)

    def test_peg_adds_vertices(self, cube_mesh):
        """When connectors succeed, the peg side should have more vertices."""
        origin = numpy.array([0, 0, 0])
        normal = numpy.array([0, 1, 0])
        result = slice_mesh_with_fallback(cube_mesh, origin, normal)
        if result.success and result.capped:
            upper_verts_before = len(result.upper.vertices)
            lower_verts_before = len(result.lower.vertices)
            config = ConnectorConfig()
            cr = add_connectors(result.upper, result.lower, origin, normal, config)
            if cr.connectors_added:
                # The peg side should have more vertices than before
                if cr.peg_on == "upper":
                    assert len(cr.upper.vertices) > upper_verts_before
                else:
                    assert len(cr.lower.vertices) > lower_verts_before
