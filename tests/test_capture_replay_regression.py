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
