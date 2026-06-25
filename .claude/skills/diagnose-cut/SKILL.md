---
name: diagnose-cut
description: Diagnose a wrong or failing cut in ObjectSplitter by replaying its debug capture offline and comparing to the Cura log. Use when a cut "doesn't work", grazes, cuts the wrong feature, or falls back unexpectedly.
---

# Diagnose a cut

Real-mesh cuts behave differently from synthetic test meshes. Never guess from
the test meshes -- reproduce on the actual capture and read the host log.

## 1. Get the capture

Captures live in `captures/<mode>_<timestamp>/` (each has `input_mesh.stl` +
`params.json`). The user enables **Debug capture** in the UI and reproduces the
cut. Find the newest:

```bash
ls -dt captures/*/ | head -3
cat captures/<name>/params.json   # mode, click_position, anchor_points, ...
```

## 2. Read the Cura log first

The log records what the cut actually did and why it fell back:

```bash
grep -E "Splitting object|Split result" "$APPDATA/cura/<ver>/cura.log" | tail
```

The `Split result: OK via '<strategy>' | capped=<bool> | ... | tried: <chain>`
line is the key. Watch for `clean_local_plane_split(failed: <Error>)` -- that
means the clean split threw and fell back to the sawtooth partition. (Common
cause: *"No available triangulation engine!"* -- core must not use trimesh
`cap=True`; it ships none in Cura.)

## 3. Replay offline on the captured mesh

Reproduce the exact pipeline `_performCut` runs (use `.venv`):

```python
import numpy as np, trimesh, json
from core.plane_calculator import snap_point_to_mesh_surface, find_plane_along_normal
from core.mesh_splitter import clean_local_plane_split
p = json.load(open(r"captures/<name>/params.json"))
m = trimesh.load(r"captures/<name>/input_mesh.stl")
click = np.array(p["click_position"]); snap, fid = snap_point_to_mesh_surface(m, click)
arrow = np.array(m.face_normals[fid], float)
plane = find_plane_along_normal(m, snap, arrow)        # Plane "along surface"
r = clean_local_plane_split(m, plane.origin, plane.normal, fid)
print("normal", np.round(plane.normal,2), "piece", len(r.upper.faces),
      "watertight", r.upper.is_watertight, "strategy", r.strategy_used)
```

Classify the result: `thick = (r.upper.bounds[1]-r.upper.bounds[0]).min()` --
`< 1.5mm` is a **grazing sliver**; a piece spanning the whole model bound is a
**body-half** (wrong feature / planar limit on interconnected geometry).

## 4. Reproduce the environment, not just the code

If the replay succeeds but Cura fails, the difference is usually the **runtime
env**: Cura uses bundled `lib/` trimesh + its own numpy/scipy, and ships **no
triangulation engine**. To mimic it, run the replay in a venv with
`trimesh` but `mapbox_earcut`/`triangle` uninstalled.

## 5. Then fix at the root

Plane-mode cuts are deterministic and view-independent (`find_plane_along_normal`
rotates about the surface normal). If a *search* mode (Smallest/Valley/seam)
misbehaves, that is expected -- they are experimental; prefer the user-defined
modes. See "Critical gotchas" in CLAUDE.md.
