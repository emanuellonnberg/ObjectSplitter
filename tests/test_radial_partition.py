# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Tests for the radial (dual-Dijkstra) partition algorithm using captured models.

These tests use real meshes captured from Cura sessions to verify that
find_shortest_seam_partition produces valid, balanced partitions on
realistic geometry.
"""

import json
import os
import time

import numpy
import pytest
import trimesh

from core.plane_calculator import find_shortest_seam_partition


# ---------------------------------------------------------------------------
# Fixtures: load captured models
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def captured_model_small():
    """Load the smaller captured model (~250KB, ~3000 faces)."""
    path = os.path.join(FIXTURES_DIR, "captured_model_small.stl")
    if not os.path.exists(path):
        pytest.skip("captured_model_small.stl not found in fixtures/")
    mesh = trimesh.load(path)
    return mesh


@pytest.fixture
def captured_model_main():
    """Load the main captured model (~1MB, ~22000 faces)."""
    path = os.path.join(FIXTURES_DIR, "captured_model_main.stl")
    if not os.path.exists(path):
        pytest.skip("captured_model_main.stl not found in fixtures/")
    mesh = trimesh.load(path)
    return mesh


# Captured click positions from params.json
CLICK_SMALL = numpy.array([0.72427530132979, 11.668981600344182, 9.799031630456454])
CLICK_MAIN = numpy.array([-1.0433432066440531, 13.149210913062092, 3.9393035045564133])


# ---------------------------------------------------------------------------
# Basic partition invariants
# ---------------------------------------------------------------------------

class TestRadialPartitionInvariants:
    """Verify that the radial partition satisfies basic correctness properties."""

    def test_partitions_cover_all_faces_small(self, captured_model_small):
        """Both partitions together should contain all faces."""
        set_a, set_b, src, sink = find_shortest_seam_partition(
            captured_model_small, CLICK_SMALL)
        assert len(set_a) + len(set_b) == len(captured_model_small.faces)

    def test_partitions_cover_all_faces_main(self, captured_model_main):
        """Both partitions together should contain all faces."""
        set_a, set_b, src, sink = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN)
        assert len(set_a) + len(set_b) == len(captured_model_main.faces)

    def test_no_overlapping_faces_small(self, captured_model_small):
        """Partitions should not share any face indices."""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_small, CLICK_SMALL)
        overlap = set(set_a) & set(set_b)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping faces"

    def test_no_overlapping_faces_main(self, captured_model_main):
        """Partitions should not share any face indices."""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN)
        overlap = set(set_a) & set(set_b)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping faces"

    def test_both_partitions_nonempty_small(self, captured_model_small):
        """Both partitions should have at least one face."""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_small, CLICK_SMALL)
        assert len(set_a) > 0, "set_a is empty"
        assert len(set_b) > 0, "set_b is empty"

    def test_both_partitions_nonempty_main(self, captured_model_main):
        """Both partitions should have at least one face."""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN)
        assert len(set_a) > 0, "set_a is empty"
        assert len(set_b) > 0, "set_b is empty"


# ---------------------------------------------------------------------------
# Partition quality
# ---------------------------------------------------------------------------

class TestRadialPartitionQuality:
    """Verify a reasonable cut quality — not just a sliver."""

    def test_minimum_partition_size_small(self, captured_model_small):
        """The smaller partition should be at least 1% of total faces.
        (Small models may produce less balanced cuts due to geometry.)"""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_small, CLICK_SMALL)
        total = len(captured_model_small.faces)
        min_size = len(set_a) if len(set_a) <= len(set_b) else len(set_b)
        ratio = min_size / total
        assert ratio >= 0.01, (
            f"Smaller partition is only {ratio:.1%} of total "
            f"({min_size}/{total} faces) — cut is too unbalanced"
        )

    def test_minimum_partition_size_main(self, captured_model_main):
        """The smaller partition should be at least 5% of total faces."""
        set_a, set_b, _, _ = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN)
        total = len(captured_model_main.faces)
        min_size = len(set_a) if len(set_a) <= len(set_b) else len(set_b)
        ratio = min_size / total
        assert ratio >= 0.05, (
            f"Smaller partition is only {ratio:.1%} of total "
            f"({min_size}/{total} faces) — cut is too unbalanced"
        )

    def test_sink_is_far_from_source_main(self, captured_model_main):
        """
        The sink face should be geodesically far from the source face,
        not a nearby face. Check via Euclidean distance of face centers
        as a proxy (sink should be at least 30% of mesh diameter away).
        """
        set_a, set_b, src, sink = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN)
        centers = captured_model_main.triangles_center
        src_center = centers[src]
        sink_center = centers[sink]

        euclidean_dist = numpy.linalg.norm(sink_center - src_center)
        bbox_diag = numpy.linalg.norm(
            captured_model_main.bounds[1] - captured_model_main.bounds[0])

        ratio = euclidean_dist / bbox_diag
        assert ratio >= 0.3, (
            f"Sink is only {ratio:.1%} of bbox diagonal from source "
            f"({euclidean_dist:.1f}mm / {bbox_diag:.1f}mm) — sink too close"
        )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestRadialPartitionPerformance:
    """Verify the partition completes within reasonable time."""

    def test_completes_within_timeout_small(self, captured_model_small):
        """Should complete well under the 10s timeout."""
        t0 = time.perf_counter()
        find_shortest_seam_partition(captured_model_small, CLICK_SMALL)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Took {elapsed:.2f}s — too slow for small model"

    def test_completes_within_timeout_main(self, captured_model_main):
        """Should complete well under the 10s timeout."""
        t0 = time.perf_counter()
        find_shortest_seam_partition(captured_model_main, CLICK_MAIN)
        elapsed = time.perf_counter() - t0
        assert elapsed < 8.0, f"Took {elapsed:.2f}s — too slow for main model"


# ---------------------------------------------------------------------------
# Surface normal influence
# ---------------------------------------------------------------------------

class TestRadialPartitionWithNormal:
    """Test that surface_normal parameter influences the partition direction."""

    def test_opposite_normals_give_different_sinks(self, captured_model_main):
        """
        Clicking the same point but with opposite surface normals should
        produce different sink selections (and thus different cuts).
        """
        normal_up = numpy.array([0.0, 1.0, 0.0])
        normal_down = numpy.array([0.0, -1.0, 0.0])

        _, _, _, sink_up = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN, surface_normal=normal_up)
        _, _, _, sink_down = find_shortest_seam_partition(
            captured_model_main, CLICK_MAIN, surface_normal=normal_down)

        # Different normals should typically pick different sinks
        # (on a complex mesh, opposite normals point to opposite sides)
        assert sink_up != sink_down, (
            f"Both normals picked the same sink face {sink_up} — "
            "surface_normal has no effect on directionality"
        )
