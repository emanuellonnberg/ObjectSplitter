# Copyright (c) 2026 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for the clean local plane split (clean cut surface + localization)."""

import numpy as np
import trimesh

from scripts.visual_verification import create_fork
from core.mesh_splitter import clean_local_plane_split
from core.plane_calculator import snap_point_to_mesh_surface


def _fork_middle_tooth():
    """Fork mesh + the face id under a click on the middle tooth."""
    mesh = create_fork()
    click = np.array([0.0, 35.0, 2.5])
    _, face_id = snap_point_to_mesh_surface(mesh, click)
    return mesh, face_id


def test_separated_piece_is_watertight_and_localized():
    mesh, face_id = _fork_middle_tooth()
    r = clean_local_plane_split(
        mesh, np.array([0.0, 35.0, 0.0]), np.array([0.0, 1.0, 0.0]), face_id)
    assert r.success, r.summary()
    sep = r.upper
    assert sep.is_watertight
    # Only the middle tooth (X within [-6, 6]); other teeth not included.
    assert sep.bounds[0][0] >= -6.0 and sep.bounds[1][0] <= 6.0


def test_remainder_keeps_other_teeth_and_is_watertight():
    mesh, face_id = _fork_middle_tooth()
    r = clean_local_plane_split(
        mesh, np.array([0.0, 35.0, 0.0]), np.array([0.0, 1.0, 0.0]), face_id)
    assert r.success, r.summary()
    rem = r.lower
    assert rem.is_watertight
    v = rem.vertices
    left_kept = bool(np.any((v[:, 0] <= -11) & (v[:, 1] >= 37)))
    right_kept = bool(np.any((v[:, 0] >= 11) & (v[:, 1] >= 37)))
    middle_gone = not bool(np.any((v[:, 0] > -2) & (v[:, 0] < 2) & (v[:, 1] >= 37)))
    assert left_kept and right_kept and middle_gone


def test_cut_face_is_flat_on_the_plane():
    mesh, face_id = _fork_middle_tooth()
    r = clean_local_plane_split(
        mesh, np.array([0.0, 35.0, 0.0]), np.array([0.0, 1.0, 0.0]), face_id)
    assert r.success, r.summary()
    # The cut face is the bottom of the tooth tip; it must lie on the plane Y=35.
    assert abs(r.upper.bounds[0][1] - 35.0) < 1e-4


def test_single_feature_gives_two_watertight_halves():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    n = np.array([0.3, 1.0, 0.2]); n = n / np.linalg.norm(n)
    _, face_id = snap_point_to_mesh_surface(mesh, n * 20.0)
    r = clean_local_plane_split(mesh, np.array([0.0, 0.0, 0.0]), n, face_id)
    assert r.success, r.summary()
    assert r.upper.is_watertight and r.lower.is_watertight


def test_whole_model_cuts_every_feature():
    mesh, face_id = _fork_middle_tooth()
    r = clean_local_plane_split(
        mesh, np.array([0.0, 35.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        face_id, whole_model=True)
    assert r.success, r.summary()
    # A whole-model horizontal cut at Y=35 takes ALL three tooth tips into the
    # upper part, so it spans the full fork width.
    assert (r.upper.bounds[1][0] - r.upper.bounds[0][0]) > 20.0


def test_degenerate_normal_does_not_crash():
    mesh, face_id = _fork_middle_tooth()
    # Zero normal must not raise; it returns a SplitResult (via fallback).
    r = clean_local_plane_split(
        mesh, np.array([0.0, 35.0, 0.0]), np.array([0.0, 0.0, 0.0]), face_id)
    assert "clean_local_plane_split" in r.strategies_attempted[0]
