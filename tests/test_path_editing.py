# Copyright (c) 2024 Emanuel Lonnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Tests for path waypoint editing helpers."""

import numpy

from core.path_editing import (
    append_waypoint,
    insert_waypoint_nearest_segment,
    nearest_segment_insert_index,
    remove_selected_waypoint,
)


class TestPathEditingHelpers:
    def test_append_waypoint_adds_to_end(self):
        waypoints = [
            numpy.array([0.0, 0.0, 0.0]),
            numpy.array([10.0, 0.0, 0.0]),
        ]

        updated, index = append_waypoint(waypoints, numpy.array([20.0, 0.0, 0.0]))

        assert index == 2
        assert len(updated) == 3
        assert numpy.allclose(updated[-1], [20.0, 0.0, 0.0])
        assert numpy.allclose(waypoints[-1], [10.0, 0.0, 0.0])

    def test_insert_uses_nearest_segment_not_nearest_point(self):
        waypoints = [
            numpy.array([0.0, 0.0, 0.0]),
            numpy.array([10.0, 0.0, 0.0]),
            numpy.array([10.0, 10.0, 0.0]),
        ]
        point = numpy.array([10.1, 5.0, 0.0])

        insert_index = nearest_segment_insert_index(waypoints, point)
        updated, selected_index = insert_waypoint_nearest_segment(waypoints, point)

        assert insert_index == 2
        assert selected_index == 2
        assert numpy.allclose(updated[2], point)
        assert numpy.allclose(updated[1], [10.0, 0.0, 0.0])
        assert numpy.allclose(updated[3], [10.0, 10.0, 0.0])

    def test_insert_falls_back_to_append_for_single_point(self):
        waypoints = [numpy.array([0.0, 0.0, 0.0])]

        updated, selected_index = insert_waypoint_nearest_segment(
            waypoints,
            numpy.array([5.0, 0.0, 0.0]),
        )

        assert selected_index == 1
        assert len(updated) == 2
        assert numpy.allclose(updated[-1], [5.0, 0.0, 0.0])

    def test_remove_selected_waypoint_selects_neighbor(self):
        waypoints = [
            numpy.array([0.0, 0.0, 0.0]),
            numpy.array([10.0, 0.0, 0.0]),
            numpy.array([20.0, 0.0, 0.0]),
        ]

        updated, selected_index = remove_selected_waypoint(waypoints, 1)

        assert len(updated) == 2
        assert selected_index == 1
        assert numpy.allclose(updated[0], [0.0, 0.0, 0.0])
        assert numpy.allclose(updated[1], [20.0, 0.0, 0.0])

    def test_remove_last_remaining_point_clears_selection(self):
        updated, selected_index = remove_selected_waypoint(
            [numpy.array([0.0, 0.0, 0.0])],
            0,
        )

        assert updated == []
        assert selected_index is None
