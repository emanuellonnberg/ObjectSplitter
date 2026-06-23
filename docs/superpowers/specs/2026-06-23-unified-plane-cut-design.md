# Unified "Plane" Cut + Clean Local Split — Design

Date: 2026-06-23 (revised after spikes)

## Goal

Two linked goals:

1. **Fix the sawtooth.** The local/face-partition cut modes
   (smallest, valley, and the seam modes) produce a jagged cut surface that
   follows triangle edges. This is the root cause of the "advanced modes don't
   work" experience. Make planar local cuts produce a **clean, flat, watertight**
   cut surface.
2. **Unify + localize plane cuts.** Replace the separate Horizontal and
   Vertical modes with one **Plane** mode whose cut acts on the **clicked
   feature** (e.g. one fork tooth), with a whole-model option for stacking
   splits.

## Findings from spikes (measured, not assumed)

- `split_by_face_sets` builds each side with `mesh.submesh(face_set)` — whole
  triangles, never sliced along the cut. On an irregular mesh (icosphere,
  tilted plane) the open boundary spans **3.34 mm** off the plane: the sawtooth.
  Every face-partition mode (smallest, valley, shortest, radial, valley_seam,
  path_isolate) shares this.
- trimesh's **capped slice** (`slice_mesh_plane`) slices triangles along the
  plane: the cut vertices land exactly on the plane (spread ~0) — clean.
- Picking the **connected component containing the click** from a capped slice
  yields a clean, **watertight** separated piece (fork middle-tooth tip: 286
  faces, watertight, X[-3,3]).
- The **remainder** stays watertight with other features intact by slicing
  uncapped, welding the non-clicked components back to the body
  (`merge_vertices` welds the shared plane loops), then repairing the single
  leftover hole with the existing `_attempt_watertight_repair` (fork remainder:
  3750 faces, watertight, both other teeth kept).

These confirm the strategy below works on a connected multi-feature mesh.

## Cut strategy: clean local plane split

New core function in `core/mesh_splitter.py`:

```
clean_local_plane_split(mesh, plane_origin, plane_normal, source_face_id,
                        whole_model=False) -> SplitResult
```

- **whole_model=True:** current behaviour — `slice_mesh_with_fallback`
  (global capped slice). For stacking-split tall prints.
- **whole_model=False (default, local):**
  1. `up_cap = slice_mesh_plane(mesh, +n, o, cap=True)`; `merge_vertices`.
  2. `separated` = the connected component of `up_cap` containing the clicked
     face's centroid side (nearest the click). Clean, watertight.
  3. `up = slice(+n, cap=False)`, `lo = slice(-n, cap=False)`, both
     `merge_vertices`. `others` = `up` components except the clicked one.
  4. `remainder = concatenate([lo] + others)`, `merge_vertices`,
     `_attempt_watertight_repair` to cap the single removed-feature hole.
  5. Return `SplitResult(upper=separated, lower=remainder, capped=...)`.
- **Robustness:** wrap in try/except; on any failure fall back to the current
  `split_by_local_plane` (face partition) so a cut never crashes. Record the
  strategy used in `SplitResult.strategies_attempted`.

Scope: **planar modes only** (there is a real plane to slice along). The
**seam modes** (shortest/radial/valley_seam/path_isolate) have no single plane;
de-sawtoothing them is a separate, harder problem and is **out of scope here**.

## Plane mode (UI), built on the clean split

Replace Horizontal + Vertical with one **Plane** mode.

- `CUT_MODE_PLANE = "plane"`. Keep `CUT_MODE_HORIZONTAL`/`_VERTICAL` only as
  migration aliases; remove from the combobox.
- Properties:
  - `PlaneOrientation` (`"auto" | "horizontal" | "vertical"`, default `"auto"`).
  - `CutWholeModel` (bool, default `false`).
- Orientation → (origin, normal):
  - **Horizontal** — normal = up (Y); origin = click (local) or Height-% point
    (whole model).
  - **Vertical** — normal = view-aligned (else X) through the click.
  - **Auto** — `find_smallest_cut_plane` at the click; use its best normal.
    (Auto reuses the Smallest engine; folding the standalone Smallest mode into
    Plane is out of scope.)
- `_performCut` calls `clean_local_plane_split(tm, origin, normal,
  click_face_id, whole_model=self._cut_whole_model)`.

### Apply the clean split to Smallest and Valley too

Smallest and Valley are planar; route them through `clean_local_plane_split`
(with their searched normal) instead of `split_by_local_plane`, so they also get
clean cuts. This is the highest-value part — it fixes the modes the user already
relies on the search for.

### UI (`qml/` and `qt6/`, kept in sync)

- Combobox: replace the two entries with one **"Plane"**.
- Plane-mode controls: orientation selector (Auto/Horizontal/Vertical),
  "Cut whole model" checkbox, Height-% slider shown only for
  Horizontal + whole-model.
- `getModeDescription` + `getModeHelp("plane")` entry; drop `horizontal`/
  `vertical` entries.

### Migration

On load, map a saved `cut_mode` of `"horizontal"`/`"vertical"` to `plane` with
`PlaneOrientation` set and `CutWholeModel = true` (old behaviour was global).

## Testing

- `core` unit tests for `clean_local_plane_split`:
  - separated piece is watertight and localized to the clicked feature
    (fork middle tooth: X within [-6,6]);
  - remainder is watertight and keeps the other features (left/right teeth
    present);
  - cut-face flatness: separated piece vertices on the cut all lie on the plane
    (distance spread < 1e-4);
  - on a single-feature mesh (icosphere) it still produces two watertight halves;
  - failure path falls back to `split_by_local_plane`.
- `plane_for_orientation` helper: correct normals/origins per orientation; Auto
  delegates to `find_smallest_cut_plane`.
- Smallest/Valley still pass existing tests after rerouting (or update the
  orthographic baselines if the cleaner cut changes them).
- `pytest` green; QML balance; manual Cura smoke (Plane local + whole-model,
  three orientations; re-test Smallest for the sawtooth being gone).

## Out of scope

- De-sawtoothing the seam modes (shortest/radial/valley_seam/path_isolate).
- Folding the standalone Smallest mode into Plane/Auto.
- Connector changes.

## Branch

`unified-plane-cut`, off `packaging-hardening` (inherits the UI infra). Rebase
onto `main` after PR #12 merges.
