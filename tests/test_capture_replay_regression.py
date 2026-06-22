# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Replay-based regression tests using real captured operations."""

from pathlib import Path

import pytest

from core.debug_capture import load_captured_operation, replay_operation


CAPTURE_ROOT = Path(__file__).resolve().parent.parent / "captures"


def _capture_path(name: str) -> Path:
    """Resolve a named capture path or skip when local captures are absent."""
    if not CAPTURE_ROOT.exists():
        pytest.skip("captures/ directory not available in this environment")
    path = CAPTURE_ROOT / name
    if not path.exists():
        pytest.skip(f"capture not available: {name}")
    return path


@pytest.mark.parametrize(
    "capture_name",
    [
        "horizontal_20260212_215339",
        "vertical_20260212_210257",
        "smallest_20260212_205855",
        "shortest_20260212_205919",
    ],
)
def test_replay_capture_basic_split_invariants(capture_name):
    """
    Replay selected real captures and verify split invariants:
    - split succeeds
    - both output meshes are non-empty
    - if marked capped, outputs are actually watertight
    """
    capture_path = _capture_path(capture_name)
    result = replay_operation(str(capture_path))
    split = result["split_result"]

    assert split.success
    assert len(split.upper.faces) > 0
    assert len(split.lower.faces) > 0

    if split.capped:
        assert split.upper.is_watertight
        assert split.lower.is_watertight


@pytest.mark.parametrize(
    "capture_name",
    [
        "vertical_20260212_210257",
        "smallest_20260212_205855",
    ],
)
def test_replay_capture_expected_to_cap_is_watertight(capture_name):
    """Known real captures that should produce capped, watertight halves."""
    capture_path = _capture_path(capture_name)
    result = replay_operation(str(capture_path))
    split = result["split_result"]

    assert split.success
    assert split.capped
    assert split.upper.is_watertight
    assert split.lower.is_watertight


def test_replay_shortest_capture_preserves_face_partition():
    """
    Shortest-seam replay should preserve input face count because
    face-set splitting does not add capping faces.
    """
    capture_path = _capture_path("shortest_20260212_205919")
    mesh, params = load_captured_operation(str(capture_path))
    assert params.cut_mode == "shortest"

    result = replay_operation(str(capture_path))
    split = result["split_result"]
    assert split.success

    total_faces = len(split.upper.faces) + len(split.lower.faces)
    assert total_faces == len(mesh.faces)


def test_path_capture_replay_roundtrip(tmp_path):
    """
    Capture a path (multi-point) operation synthetically and replay it.

    This is the capture format the path/multi-point mode uses for offline
    profiling; verify the roundtrip works end to end without a pre-existing
    fixture.
    """
    import numpy
    import trimesh
    from core.debug_capture import capture_operation

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)

    # A closed loop of waypoints around the equator (what multi-point produces
    # with "close loop" enabled).
    thetas = numpy.linspace(0, 2 * numpy.pi, 6, endpoint=False)
    waypoints = [
        numpy.array([10.0 * numpy.cos(t), 10.0 * numpy.sin(t), 0.0])
        for t in thetas
    ]
    waypoints.append(waypoints[0].copy())  # close the loop

    capture_dir = capture_operation(
        mesh=mesh,
        cut_mode="path",
        click_position=waypoints[0],
        waypoints=waypoints,
        path_close_loop=True,
        path_cap_ends=True,
        connector_enabled=False,
        connector_diameter=4.0,
        connector_height=3.0,
        connector_clearance=0.2,
        capture_dir=str(tmp_path),
    )

    mesh_loaded, params = load_captured_operation(capture_dir)
    assert params.cut_mode == "path"
    assert params.waypoints is not None
    assert len(params.waypoints) == len(waypoints)

    result = replay_operation(capture_dir)
    split = result["split_result"]
    assert split.success
    assert len(split.upper.faces) > 0
    assert len(split.lower.faces) > 0


def test_path_capture_replay_includes_connectors(tmp_path):
    """A path capture with connectors enabled exercises connector placement
    during offline replay (not just the split)."""
    import numpy
    import trimesh
    from core.debug_capture import capture_operation

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=15.0)
    thetas = numpy.linspace(0, 2 * numpy.pi, 6, endpoint=False)
    waypoints = [
        numpy.array([15.0 * numpy.cos(t), 15.0 * numpy.sin(t), 0.0])
        for t in thetas
    ]
    waypoints.append(waypoints[0].copy())

    capture_dir = capture_operation(
        mesh=mesh,
        cut_mode="path",
        click_position=waypoints[0],
        waypoints=waypoints,
        path_close_loop=True,
        path_cap_ends=True,
        connector_enabled=True,
        connector_diameter=3.0,
        connector_height=2.0,
        connector_clearance=0.15,
        capture_dir=str(tmp_path),
    )

    result = replay_operation(capture_dir)
    assert result["split_result"].success
    # Connector path is now exercised by replay (was always None before).
    cr = result["connector_result"]
    assert cr is not None
    assert cr.connectors_added
    assert cr.peg_on != cr.hole_on
