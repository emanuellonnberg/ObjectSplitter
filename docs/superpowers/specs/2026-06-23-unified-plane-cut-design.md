# Unified "Plane" Cut — Design

Date: 2026-06-23

## Goal

Replace the separate **Horizontal** and **Vertical** cut modes with a single
**Plane** mode, and make a plane cut act **locally on the clicked feature**
(e.g. cut one tooth of a fork) instead of slicing the whole model at that
plane. Keep the whole-model slice available for height-based stacking splits.

Motivation: horizontal/vertical "don't work well" because they use the global
infinite-plane slice (`slice_mesh_with_fallback`), which cuts every feature the
plane passes through. Smallest/Valley already cut locally via
`split_by_local_plane`; Plane mode should use the same machinery.

## Behaviour

### Local by default, whole-model opt-in

- **Local (default):** the cut separates only the connected feature at the
  click, via `split_by_local_plane(mesh, origin, [normal...], click_face_id)`.
- **Whole model (checkbox "Cut whole model"):** the current global slice
  (`slice_mesh_with_fallback`) at the plane, for splitting a tall print into
  stackable parts.

### Orientation selector — Auto / Horizontal / Vertical

- **Horizontal** — plane normal = up (Y axis); parallel to the build plate.
- **Vertical** — plane through the click, aligned to the camera view direction
  (today's vertical behaviour); falls back to the X axis when no view normal is
  available.
- **Auto** — orient automatically by running the smallest-cross-section search
  (`find_smallest_cut_plane`) at the click and using its best normal (and its
  ranked `top_candidates` as fallbacks). This is the same engine the Smallest
  mode uses; Auto is that search surfaced inside Plane mode. (Folding the
  standalone Smallest mode into Plane/Auto is explicitly out of scope here.)

### Position

- Local cut (any orientation): plane origin = the click point.
- Whole-model + Horizontal: plane origin from the existing **Height %** slider.
- Whole-model + Vertical/Auto: plane origin = the click point.

## Components

### Backend (`ObjectSplitter.py`)

- Add `CUT_MODE_PLANE = "plane"`. Keep `CUT_MODE_HORIZONTAL`/`_VERTICAL`
  constants only as migration aliases (see Migration); remove them from the UI
  combobox.
- New exposed properties:
  - `PlaneOrientation` (`"auto" | "horizontal" | "vertical"`, default `"auto"`).
  - `CutWholeModel` (bool, default `false`).
- `_performCut` for `plane` mode:
  - Build candidate normals + origin from orientation:
    - horizontal → `horizontal_cut_plane` normal/origin (origin = click for
      local; height-% for whole-model).
    - vertical → `vertical_cut_plane(click, view_normal?)`.
    - auto → `find_smallest_cut_plane(...)` at the click; candidates = its
      `top_candidates`.
  - Local: `split_by_local_plane(tm, origin, candidate_normals, click_face_id)`.
  - Whole model: `slice_mesh_with_fallback(tm, origin, normal, face_id=...)`.

### Plane construction helper (`core/plane_calculator.py`)

- Small helper `plane_for_orientation(mesh, orientation, click_point,
  click_face_id, view_normal, height_percent, whole_model)` returning
  `(origin, candidate_normals)`, so `_performCut` stays thin and the logic is
  unit-testable without Cura. Auto delegates to `find_smallest_cut_plane`.

### UI (`qml/` and `qt6/`, kept in sync)

- Combobox: replace the two entries with one **"Plane"**.
- New controls, visible only in Plane mode:
  - Orientation selector (Auto / Horizontal / Vertical) — a small ComboBox or
    segmented row.
  - "Cut whole model" checkbox.
  - Existing **Height %** slider shown only when Horizontal + whole-model.
- Mode description + `getModeHelp("plane")` entry updated; remove the
  `horizontal`/`vertical` help entries (or keep, harmless).

### Migration

- On load, map a saved `cut_mode` of `"horizontal"`/`"vertical"` to
  `plane` with `PlaneOrientation` set accordingly and `CutWholeModel = true`
  (old behaviour was whole-model), so existing preferences keep working.

## Testing

- `core` unit tests for `plane_for_orientation`: correct normals/origins for
  each orientation; Auto returns the smallest-search candidates; height-% vs
  click origin selection.
- Split behaviour: local plane cut on the fork separates only the clicked tooth
  (assert the other teeth stay with the body); whole-model cut splits across.
- Integration test exercising `plane` mode end-to-end through the existing
  capture/replay or direct core calls.
- `pytest` green; QML paren/brace balance; manual Cura smoke (local + whole
  model, all three orientations).

## Validation (pre-implementation spike)

Ran the core machinery directly on the fork test mesh (no Cura) to confirm the
foundation before building on it, because the user reported the "advanced"
modes being unreliable:

- **Local horizontal** cut through a middle-tooth click → separated piece was
  exactly that tooth tip (X[-3,3], Y[35,41], 286 faces); the other teeth stayed
  with the body. Correct.
- **Local vertical** through a tooth cut *along* it (split the whole fork
  left/right) — geometrically expected, since the tooth extends upward; matches
  the user's "across, not along" caveat. Vertical is the wrong tool for a tooth.
- **Auto** (`find_smallest_cut_plane`) localized correctly for both a middle-
  tooth click (132 faces, X[-3,3]) and a left-tooth side click (639 faces,
  X[-17,-11], cut across the tooth).

Conclusion: `split_by_local_plane` and the Smallest engine are sound for this
mode. The unreliable "advanced" modes are the **geodesic seam** family
(shortest / radial / valley_seam), which Plane mode does **not** use. Real-world
STLs still warrant a Cura smoke test (tessellation / non-watertight meshes).

## Out of scope

- Folding the standalone Smallest mode into Plane/Auto (possible later).
- Changing Smallest/Valley/seam/path modes.
- New connector behaviour.

## Branch

Feature, not packaging — its own branch + PR off `main`, after PR #12
(packaging-hardening) merges. Not added to the packaging branch.
