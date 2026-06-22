# Packaging & Dependency Hardening — Design

Date: 2026-06-22

## Goal

Make ObjectSplitter robust on a clean Cura install and prepare it for an
eventual UltiMaker Marketplace submission. **Windows first**; structure the
work so Mac/Linux is a later add-on, not a rewrite.

This is a packaging/dependency project — it does not change cut behavior.

## Findings (verified on this machine)

- Every installed Cura **5.3 → 5.12 ships `trimesh 3.9.36`** and (5.12)
  Python **3.12**. UltiMaker has **not** moved to trimesh 4.x, so the plugin
  must supply 4.x itself or port to 3.9.
- Cura ships, ABI-correct for its own Python/platform: **numpy, scipy,
  shapely, trimesh 3.9.36, numpy-stl**.
- The plugin bundles in `lib/`: trimesh 4.11 (pure-Python), networkx
  (pure-Python), **numpy / scipy / shapely (compiled, `cp311-win_amd64`
  only — 131 `.pyd` files)**, and rtree (native `libspatialindex`).
- The bundled compiled deps are **cp311**, but Cura 5.12 is **py312** — they
  cannot load there. The plugin only works today because Cura imports its own
  numpy/scipy/shapely first and those get cached, so `lib/` is shadowed. This
  is fragile and is the root of both the Windows-only and ABI problems.

## The core insight

The cross-platform + ABI problem is **entirely** from bundling
numpy/scipy/shapely. Cura already ships all three. Removing them from `lib/`:

- deletes all 131 platform-locked `.pyd` files,
- removes the cp311-vs-py312 fragility,
- and works on every platform/Python that Cura itself runs,

with **no per-platform build matrix**. This is the highest-leverage change and
applies even to the Windows-first milestone.

## Decisions

- **Platforms:** Windows first; defer Mac/Linux verification. Still do the
  dependency slimming now (it helps Windows robustness and is the prerequisite
  for cross-platform later).
- **numpy / scipy / shapely:** stop bundling; use Cura's. (Must verify Cura's
  versions expose the APIs the code uses — see Risks.)
- **trimesh:** Cura is stuck on 3.9.36, so 4.x must be supplied by the plugin.
  Handling is the one **open decision** (see below).
- **rtree:** drop it. It is only used by mesh_splitter's preferred capped-slice
  strategy, which already has manual-cap and face-split fallbacks (now solid).
  Removing it eliminates a native dependency. Verify the fallbacks produce
  acceptable caps without it.
- **manifold3d:** remains optional and unbundled; cap-native connectors work
  without it. Document the degraded boolean path.
- **networkx:** keep bundling (pure-Python, platform-independent).

## Open decision: trimesh 3.9 vs bundled 4.x

The plugin needs 4.x APIs (`section()` without rtree, `to_2D`). Three options,
to be resolved before implementation:

1. **Vendor 4.x privately (recommended for Marketplace):** load bundled 4.x
   only for this plugin, without replacing Cura's global `trimesh` 3.9. Cleanest
   for coexistence; medium effort (handle trimesh's internal absolute imports
   under a scoped path).
2. **Keep guarded global replace (lowest effort):** continue swapping Cura's
   3.9 for bundled 4.x in `sys.modules`, but only when the loaded version is
   < 4 and restore on unload. Works today; UltiMaker review may object to
   replacing a shipped library, and it affects other plugins.
3. **Port down to 3.9 (cleanest deps, most risk):** drop bundled trimesh
   entirely, replace the 4.x-only calls with our own helpers + shapely. Biggest
   change to the cut core and the most regression risk.

Recommendation: **#2 for the Windows-first milestone** (fast, unblocks the dep
slim), with **#1 tracked as the pre-Marketplace step**. Final call is the
maintainer's.

## Plan (Windows-first milestone)

1. **Slim the bundle.** Remove numpy, scipy, shapely (and rtree) from `lib/`.
   Keep trimesh 4.x + networkx. Update `__init__.py` so it adds `lib/` to
   `sys.path` for the pure-Python packages only and no longer expects bundled
   numpy/scipy.
2. **trimesh loading.** Implement the chosen option (default #2: guarded,
   version-checked, reversible global replace).
3. **rtree removal.** Confirm mesh_splitter falls back cleanly; adjust the
   strategy chain/messaging so a missing rtree is normal, not a warning.
4. **Refresh `scripts/bundle_deps.py`** to produce the slim, pure-Python bundle
   (and document which deps are expected from Cura).
5. **CI.** Add a GitHub Actions workflow running `pytest` on push/PR.
6. **Packaging.** Add a script to build a `.curapackage`; bump version to a
   real release number; ensure `plugin.json` SDK range matches tested Cura
   versions.
7. **Clean-install verification (Windows).** Install the built package into a
   clean Cura 5.12, run the per-mode smoke; confirm no reliance on a dev
   Python/site-packages.

## Out of scope (this milestone)

- Mac/Linux binaries and verification (future milestone; enabled by the slim).
- Bundling manifold3d (boolean connectors stay optional).
- Cut-algorithm changes.
- Actual Marketplace submission paperwork.

## Risks

- **Cura's numpy/scipy API level.** The perf work uses
  `scipy.sparse.csgraph.dijkstra(min_only=True)` (scipy ≥ 1.8) and
  `scipy.spatial.cKDTree`, plus numpy 2.x-ish behavior. Cura's bundled scipy
  must be new enough. **Partially verified:** Cura 5.12 ships a `cp312` scipy
  whose `csgraph` shortest-path extension contains the `min_only` symbol and
  includes `_ckdtree.cp312`, so the needed APIs are present there. Still
  confirm at runtime inside Cura (and check the minimum supported Cura in the
  SDK range, whose scipy may be older — fall back or keep a documented minimal
  bundle if so).
- **rtree removal** changes the capping path; verify cap quality/watertightness
  on representative meshes via the replay harness.
- **trimesh option #2** is a stopgap; carries the coexistence risk until #1.
- **No clean-room CI for QML/Cura**; CI covers `pytest` only. The clean-install
  smoke stays manual.

## Verification

- `pytest` (279) green throughout; it runs against whatever numpy/scipy/trimesh
  resolve, so also run it once in an environment matching Cura's dep versions.
- Replay harness (`scripts/profile_capture.py`, capture replay tests) to
  confirm cut/cap results unchanged after rtree removal and dep swap.
- Manual clean-install smoke on Windows Cura 5.12.
