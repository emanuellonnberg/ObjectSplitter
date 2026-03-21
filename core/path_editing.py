# Copyright (c) 2024 Emanuel Lonnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Helpers for editing ordered waypoint lists used by path mode.

These helpers stay Cura-independent so waypoint ordering behavior can be
tested without UI or scene dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy


def _copy_waypoints(waypoints: Sequence[numpy.ndarray]) -> List[numpy.ndarray]:
    return [numpy.asarray(point, dtype=numpy.float64).copy() for point in waypoints]


def append_waypoint(
    waypoints: Sequence[numpy.ndarray],
    point: numpy.ndarray,
) -> Tuple[List[numpy.ndarray], int]:
    updated = _copy_waypoints(waypoints)
    updated.append(numpy.asarray(point, dtype=numpy.float64).copy())
    return updated, len(updated) - 1


def nearest_segment_insert_index(
    waypoints: Sequence[numpy.ndarray],
    point: numpy.ndarray,
) -> int:
    if len(waypoints) < 2:
        return len(waypoints)

    target = numpy.asarray(point, dtype=numpy.float64)
    best_distance = None
    best_index = len(waypoints)

    for start_index in range(len(waypoints) - 1):
        a = numpy.asarray(waypoints[start_index], dtype=numpy.float64)
        b = numpy.asarray(waypoints[start_index + 1], dtype=numpy.float64)
        ab = b - a
        denom = float(numpy.dot(ab, ab))
        if denom <= 1e-12:
            closest = a
        else:
            t = float(numpy.dot(target - a, ab) / denom)
            t = min(1.0, max(0.0, t))
            closest = a + (ab * t)
        distance = float(numpy.linalg.norm(target - closest))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = start_index + 1

    return best_index


def insert_waypoint_nearest_segment(
    waypoints: Sequence[numpy.ndarray],
    point: numpy.ndarray,
) -> Tuple[List[numpy.ndarray], int]:
    updated = _copy_waypoints(waypoints)
    insert_index = nearest_segment_insert_index(updated, point)
    updated.insert(insert_index, numpy.asarray(point, dtype=numpy.float64).copy())
    return updated, insert_index


def remove_selected_waypoint(
    waypoints: Sequence[numpy.ndarray],
    selected_index: Optional[int],
) -> Tuple[List[numpy.ndarray], Optional[int]]:
    updated = _copy_waypoints(waypoints)
    if selected_index is None or selected_index < 0 or selected_index >= len(updated):
        return updated, selected_index

    del updated[selected_index]
    if not updated:
        return updated, None
    return updated, min(selected_index, len(updated) - 1)
