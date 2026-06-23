# Clean Local Plane Split (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `clean_local_plane_split` core function that produces a clean, watertight, flat cut surface localized to the clicked feature, and route the Smallest and Valley modes through it so their cuts stop being a triangle-edge sawtooth.

**Architecture:** A planar cut is done with trimesh's capped slice (which slices triangles along the plane), then the connected component containing the click is taken as the separated piece. The other cut components are welded back onto the body via shared plane-loop vertices and the single leftover hole is repaired, keeping the remainder watertight with other features intact. On any failure it falls back to the existing face-partition split so a cut never crashes.

**Tech Stack:** Python, trimesh, numpy. Tests via pytest.

## Global Constraints

- Pure logic in `core/`, no Cura imports (verified by `tests/conftest.py` adding the project root to `sys.path`).
- Copyright header (LGPLv3) on every new file.
- A cut must never crash: wrap the new strategy in try/except and fall back to `split_by_local_plane`.
- This phase is **planar modes only** (Smallest, Valley). Seam modes are untouched.
- `pytest` must stay green. Run it before committing each task.
- Test interpreter: `.venv/Scripts/python.exe` (py3.14, has trimesh/rtree/triangle/mapbox-earcut); CI uses py3.10-3.12.

---

### Task 1: `clean_local_plane_split` core function

**Files:**
- Modify: `core/mesh_splitter.py` (add `_component_nearest_point` and `clean_local_plane_split` after `split_by_local_plane`, which ends near line 848)
- Test: `tests/test_clean_local_split.py` (create)

**Interfaces:**
- Consumes (already in `mesh_splitter.py`): `SplitResult`, `slice_mesh_with_fallback(mesh, origin, normal, face_id=...) -> SplitResult`, `split_by_local_plane(mesh, origin, candidate_normals, source_face_id) -> SplitResult`, `_attempt_watertight_repair(mesh) -> (mesh, cap_faces)`, module-level `trimesh`, `numpy`, `logger`.
- Produces: `clean_local_plane_split(mesh, plane_origin, plane_normal, source_face_id, whole_model=False) -> SplitResult` where `result.upper` is the separated (clicked) piece and `result.lower` is the remainder.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clean_local_split.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clean_local_split.py -q`
Expected: FAIL with `ImportError: cannot import name 'clean_local_plane_split'`.

- [ ] **Step 3: Implement `clean_local_plane_split`**

In `core/mesh_splitter.py`, immediately after the end of `split_by_local_plane` (near line 848), add:

```python
def _component_nearest_point(components, point):
    """Return the connected-component submesh whose surface is closest to point."""
    point = numpy.asarray(point, dtype=numpy.float64)
    best = None
    best_d = None
    for comp in components:
        d = float(numpy.linalg.norm(comp.vertices - point, axis=1).min())
        if best_d is None or d < best_d:
            best_d = d
            best = comp
    return best


def clean_local_plane_split(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
    source_face_id: int,
    whole_model: bool = False,
) -> SplitResult:
    """
    Split a mesh with a plane, producing a clean (triangle-sliced) cut surface.

    Local (default): separates only the clicked connected feature. The clicked
    component comes from a capped slice (clean and watertight); the other cut
    components are welded back onto the body and the single leftover hole is
    repaired, so the remainder stays watertight with the other features intact.

    whole_model=True: behaves like the global capped slice (cuts everything the
    plane crosses) -- for stacking-split tall prints.

    Falls back to split_by_local_plane (face partition) on any failure so a cut
    never crashes.

    Args:
        mesh: The trimesh object.
        plane_origin: A point on the cutting plane.
        plane_normal: Normal vector of the cutting plane.
        source_face_id: The face the user clicked on.
        whole_model: If True, cut the whole model instead of just the click.

    Returns:
        SplitResult with upper = separated piece, lower = remainder.
    """
    result = SplitResult()
    result.strategies_attempted.append("clean_local_plane_split")

    origin = numpy.asarray(plane_origin, dtype=numpy.float64)
    normal = numpy.asarray(plane_normal, dtype=numpy.float64)
    norm = float(numpy.linalg.norm(normal))
    if norm > 0:
        normal = normal / norm

    if whole_model:
        gr = slice_mesh_with_fallback(mesh, origin, normal, face_id=source_face_id)
        gr.strategies_attempted.insert(0, "clean_local_plane_split(whole_model)")
        return gr

    try:
        # Orient the normal so the clicked feature is on the +normal (separated) side.
        src_centroid = mesh.vertices[mesh.faces[source_face_id]].mean(axis=0)
        if float(numpy.dot(src_centroid - origin, normal)) < 0:
            normal = -normal

        # Separated piece: capped slice, then the component nearest the click.
        up_cap = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=normal, plane_origin=origin, cap=True)
        up_cap.merge_vertices()
        up_cap_comps = up_cap.split(only_watertight=False)
        if not up_cap_comps:
            raise ValueError("capped slice produced no geometry")
        separated = _component_nearest_point(up_cap_comps, src_centroid).copy()

        # Remainder: uncapped slices share plane-loop vertices, so welding the
        # non-clicked components back onto the body is seamless.
        upper = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=normal, plane_origin=origin, cap=False)
        upper.merge_vertices()
        lower = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=-normal, plane_origin=origin, cap=False)
        lower.merge_vertices()
        upper_comps = upper.split(only_watertight=False)
        if not upper_comps:
            raise ValueError("uncapped slice produced no geometry")
        clicked = _component_nearest_point(upper_comps, src_centroid)
        others = [c for c in upper_comps if c is not clicked]

        remainder = trimesh.util.concatenate([lower] + others)
        remainder.merge_vertices()
        remainder, cap_faces = _attempt_watertight_repair(remainder)

        if len(separated.vertices) == 0 or len(remainder.vertices) == 0:
            raise ValueError("empty separated or remainder piece")

        result.upper = separated
        result.lower = remainder
        result.cap_faces_lower = cap_faces
        result.capped = bool(separated.is_watertight and remainder.is_watertight)
        result.strategy_used = "clean_local_plane_split"
        return result
    except Exception as e:  # noqa: BLE001 - a cut must never crash
        logger.warning(
            "clean_local_plane_split failed (%s); falling back to face partition", e)
        fb = split_by_local_plane(
            mesh, origin, [numpy.asarray(plane_normal, dtype=numpy.float64)],
            source_face_id)
        fb.strategies_attempted.insert(0, "clean_local_plane_split(failed)")
        return fb
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clean_local_split.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass (previous count + 6).

- [ ] **Step 6: Commit**

```bash
git add core/mesh_splitter.py tests/test_clean_local_split.py
git commit -m "feat: add clean_local_plane_split for clean, local, watertight cuts"
```

---

### Task 2: Route Smallest and Valley through the clean split

**Files:**
- Modify: `ObjectSplitter.py` (import list near line 99-101; `_performCut` Smallest/Valley branch near line 2194-2206)

**Interfaces:**
- Consumes: `clean_local_plane_split` from `core.mesh_splitter` (Task 1); `plane` (a `CutPlane` with `.origin`, `.normal`) and `click_face_id`, both already computed earlier in `_performCut`.
- Produces: no new public interface; changes the split strategy used by the `smallest` and `valley` modes.

- [ ] **Step 1: Add the import**

In `ObjectSplitter.py`, find the import block that pulls from `.core.mesh_splitter` (around line 99-101, where `slice_mesh_with_fallback` and `split_by_local_plane` are imported) and add `clean_local_plane_split`:

```python
    slice_mesh_with_fallback,
    split_by_face_sets,
    split_by_local_plane,
    clean_local_plane_split,
```

(Match the existing names already in that import; only add the `clean_local_plane_split` line.)

- [ ] **Step 2: Reroute the Smallest/Valley split**

In `_performCut`, replace the Smallest/Valley split branch (near line 2194-2206), which currently reads:

```python
            elif self._cut_mode in (self.CUT_MODE_SMALLEST, self.CUT_MODE_VALLEY):
                # Use graph-based local separation with candidate fallback.
                # If the best plane only grazes the surface (single-triangle cut),
                # try the next-best candidates until we get a meaningful partition.
                candidate_normals = []
                if search_result and search_result.top_candidates:
                    candidate_normals = [n for _, n in search_result.top_candidates]
                if not candidate_normals:
                    candidate_normals = [plane.normal]

                split_result = split_by_local_plane(
                    tm, plane.origin, candidate_normals, click_face_id
                )
```

with:

```python
            elif self._cut_mode in (self.CUT_MODE_SMALLEST, self.CUT_MODE_VALLEY):
                # Clean local split: capped slice + clicked-component selection
                # gives a flat watertight cut instead of a triangle-edge sawtooth.
                # It falls back to the face partition internally if the slice fails.
                split_result = clean_local_plane_split(
                    tm, plane.origin, plane.normal, click_face_id
                )
```

- [ ] **Step 3: Verify Python still compiles**

Run: `.venv/Scripts/python.exe -m py_compile ObjectSplitter.py`
Expected: no output (exit 0).

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass. (The orthographic signature tests call `split_by_local_plane` directly and are unaffected by this adapter reroute.)

- [ ] **Step 5: Commit**

```bash
git add ObjectSplitter.py
git commit -m "feat: route Smallest and Valley cuts through clean_local_plane_split"
```

- [ ] **Step 6: Manual Cura smoke (record result, do not skip)**

In Cura on the dev junction, run a **Smallest** cut on a real model (the one that showed the sawtooth) and confirm the cut surface is now clean/flat and the piece is watertight. Note the outcome. This is the real acceptance check; the unit tests cover correctness on the fork/icosphere but cannot judge a real STL.

---

## Self-Review

- **Spec coverage:** The spec's "Cut strategy: clean local plane split" maps to Task 1; "Apply the clean split to Smallest and Valley too" maps to Task 2. The Plane-mode UI, `plane_for_orientation`, migration, and combobox changes are explicitly **Phase 2** and not in this plan.
- **Placeholder scan:** none — all steps contain full code/commands.
- **Type consistency:** `clean_local_plane_split(mesh, plane_origin, plane_normal, source_face_id, whole_model=False) -> SplitResult` is defined in Task 1 and consumed with that exact signature in Task 2. `result.upper`/`result.lower`/`result.success`/`result.strategies_attempted` match the `SplitResult` dataclass.
- **Known follow-up (not this phase):** the orthographic signature tests still exercise `split_by_local_plane`; a later task may point them at `clean_local_plane_split` and refresh baselines once the clean cut is the canonical path.
