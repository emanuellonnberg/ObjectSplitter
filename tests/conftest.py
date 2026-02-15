# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Pytest fixtures providing reusable test meshes and configurations.

These fixtures create simple geometric shapes that mimic the kinds of
meshes Cura would provide, allowing tests to run entirely outside of Cura.
"""

import sys
import os

# Ensure the project root (ObjectSplitter/) is on sys.path so that
# `core` and `viz` packages can be imported directly without triggering
# the Cura-dependent __init__.py of the parent package.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy
import pytest
import trimesh


# ---------------------------------------------------------------------------
# Simple primitive meshes
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_mesh():
    """A 20x20x20 mm cube centered at origin (watertight)."""
    return trimesh.creation.box(extents=[20, 20, 20])


@pytest.fixture
def tall_box_mesh():
    """A 10x40x10 mm tall box - good for testing horizontal cuts."""
    return trimesh.creation.box(extents=[10, 40, 10])


@pytest.fixture
def flat_box_mesh():
    """A 40x5x40 mm flat box - good for testing vertical cuts."""
    return trimesh.creation.box(extents=[40, 5, 40])


@pytest.fixture
def sphere_mesh():
    """A sphere with radius 15 mm (watertight, ~5000 faces)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=15.0)


@pytest.fixture
def cylinder_mesh():
    """A cylinder, radius=8, height=30, 32 segments."""
    return trimesh.creation.cylinder(radius=8, height=30, sections=32)


@pytest.fixture
def torus_mesh():
    """A torus (donut shape) - tests cuts on genus-1 surfaces."""
    # Major radius 15, minor radius 5
    torus = trimesh.creation.torus(major_radius=15.0, minor_radius=5.0)
    return torus


# ---------------------------------------------------------------------------
# Composite / irregular meshes for stress testing
# ---------------------------------------------------------------------------

@pytest.fixture
def l_shaped_mesh():
    """
    An L-shaped mesh made by combining two boxes.
    Tests non-convex geometry splitting.
    """
    box1 = trimesh.creation.box(extents=[20, 40, 10])
    box2 = trimesh.creation.box(extents=[20, 10, 10])
    box2.apply_translation([20, -15, 0])
    return trimesh.util.concatenate([box1, box2])


@pytest.fixture
def translated_cube_mesh():
    """A cube translated away from origin - tests coordinate handling."""
    mesh = trimesh.creation.box(extents=[20, 20, 20])
    mesh.apply_translation([50, 30, -10])
    return mesh


@pytest.fixture
def scaled_cube_mesh():
    """A non-uniformly scaled cube."""
    mesh = trimesh.creation.box(extents=[10, 30, 5])
    return mesh


# ---------------------------------------------------------------------------
# Click positions simulating Cura user interactions
# ---------------------------------------------------------------------------

@pytest.fixture
def center_click():
    """Click at the origin."""
    return numpy.array([0.0, 0.0, 0.0])


@pytest.fixture
def offset_click():
    """Click offset from origin."""
    return numpy.array([5.0, 10.0, 3.0])


@pytest.fixture
def surface_click_cube():
    """Click on the surface of a 20x20x20 cube at the top face center."""
    return numpy.array([0.0, 10.0, 0.0])


# ---------------------------------------------------------------------------
# Connector configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def default_connector_config():
    """Default connector configuration."""
    from core.connectors import ConnectorConfig
    return ConnectorConfig()


@pytest.fixture
def small_connector_config():
    """Small connector for fine detail work."""
    from core.connectors import ConnectorConfig
    return ConnectorConfig(diameter=2.0, height=1.5, clearance=0.1, sides=12)


@pytest.fixture
def large_connector_config():
    """Large connector for structural parts."""
    from core.connectors import ConnectorConfig
    return ConnectorConfig(diameter=8.0, height=6.0, clearance=0.3, sides=24)
