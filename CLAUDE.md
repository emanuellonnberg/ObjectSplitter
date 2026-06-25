# ObjectSplitter - Claude Code Project Guide

## What is this?

A Cura plugin that splits 3D meshes along planes for 3D printing. All
computational logic lives in `core/` (pure Python, no Cura deps).
`ObjectSplitter.py` is a thin adapter that wires Cura UI events to the core
pipeline. See [docs/architecture.md](docs/architecture.md) for the full
architecture breakdown.

## Architecture

```
core/                  Pure logic - testable outside Cura
  geometry.py          Rotation matrices, coordinate transforms
  plane_calculator.py  Cut plane algorithms (7 modes)
  mesh_splitter.py     Mesh slicing with 3 fallback strategies
  connectors.py        Peg & hole creation via boolean ops
  path_cutter.py       Geodesic shortest-path cutting
  debug_capture.py     Serialize operations for offline replay

ObjectSplitter.py      Cura tool adapter (events, scene nodes, undo, QML)
__init__.py            Plugin registration (conditional Cura imports)
qml/                   Qt5 / Cura 4.x UI
qt6/                   Qt6 / Cura 5.x UI (keep both in sync)
viz/                   HTML/PNG visualization for debugging
docs/                  Architecture and algorithm documentation
```

## Running tests

```bash
pytest                          # ~286 tests, ~20s
pytest tests/test_geometry.py   # single module
pytest -v                       # verbose
pytest -k smallest              # filter by keyword
```

Config is in `pytest.ini`. Test deps are in `requirements-test.txt` (note: it
must include `rtree` for ray-cast projection tests, plus `triangle` and
`mapbox-earcut` for trimesh mesh-builder triangulation). Fork tessellation
tests skip when `triangle` has no wheel for the running Python.
CI runs the suite on py3.10-3.12 (Linux) + py3.12 (Windows) via
`.github/workflows/tests.yml`. **Always run `pytest` before pushing.**

## Critical gotchas (hard-won)

- **No triangulation engine in Cura.** Cura's bundled Python ships neither
  `mapbox_earcut` nor `triangle`, so `trimesh`'s `cap=True` slice and
  `triangulate_polygon` raise *"No available triangulation engine!"* at runtime.
  Core code must never depend on one. `clean_local_plane_split` slices uncapped
  and caps via the scipy-based `_attempt_watertight_repair`. (Tests pass because
  test deps include the engines; a cut that works in tests can still fall back
  in Cura -- check the host log.)
- **Cut-mode reliability.** The default modes (Multi-point, Plane, Isolate) are
  user-defined and predictable. The single-click *search* modes (Smallest,
  Shortest, Radial, Valley, Valley-seam) pick grazing/unintended planes on real
  meshes (the min-cross-section objective favors grazing slivers; no fixed bias
  fully fixes it) -- they are kept behind the experimental toggle. Prefer
  user-defined cuts; do not build new features on the search.
- **Plane mode** is the axis/surface hub: orientation `surface` (deterministic
  `find_plane_along_normal`, **not** camera-dependent), `horizontal`, `points`
  (3-point). The standalone horizontal/vertical modes were folded into it.
- **Diagnose cuts with capture + replay, not guesswork.** Enable Debug capture,
  reproduce in Cura, then `replay_operation("captures/<name>")` offline and
  compare to the host `cura.log` "Split result" line (it records the strategy
  used and any `(failed: ...)` reason). Real-mesh behavior differs from
  synthetic test meshes -- always reproduce on the actual capture.

## Key conventions

- **Imports**: `from core.geometry import ...` (absolute from project root).
  Tests add the project root to `sys.path` via `conftest.py`.
- **QML properties**: Exposed via `setExposedProperties()` in `ObjectSplitter.py`.
  Each needs a `getFoo` getter; setters should guard with `if value != old`
  and call `self.propertyChanged.emit()`.
- **Two QML files**: `qml/` (UM 1.5) and `qt6/` (UM 1.6) must stay in sync.
  The only difference is the `UM` import version.
- **Style**: PEP 8, Google-style docstrings, type hints throughout.
  Copyright header (LGPLv3) on every file.
- **CI**: GitHub Actions runs `pytest` on push/PR (py3.10-3.12 Linux, py3.12
  Windows). Still run `pytest` locally before pushing.
- **Fallback chains**: mesh_splitter and connectors use priority-ordered
  fallback strategies. Maintain this pattern when adding new strategies.
- **Daemon threads with timeouts**: Expensive algorithms
  (shortest seam, radial) run in daemon threads with a 10s hard timeout.
  Use the cancel_event pattern for new long-running computations.

## Common tasks

### Add a new QML-exposed property

1. Add instance var in `__init__`: `self._foo = default_value`
2. Add `"Foo"` to `setExposedProperties()` list
3. Add `getFoo` getter + `setFoo` setter (guard with `if value != old`,
   emit `propertyChanged`)
4. Add UI control in **both** `qml/` and `qt6/` QML files

### Add a new cut mode

1. Add constant in `ObjectSplitter.py`: `CUT_MODE_NEW = "newmode"`
2. Implement plane calculation in `core/plane_calculator.py`
3. Wire it in `_performCut` in `ObjectSplitter.py`
4. Add to QML combobox model + mode map in **both** QML files
5. Add mode description text in QML `getModeDescription` (both files)
6. Add tests in `tests/test_plane_calculator.py` and `tests/test_integration.py`

### Debug a failing cut

1. Enable "Debug capture" in the UI (or set `_capture_dir` in code)
2. Reproduce the cut -- inputs saved to `captures/`
3. Replay offline:
   ```python
   from core.debug_capture import replay_operation, save_result_meshes
   result = replay_operation("captures/<name>")
   save_result_meshes(result, "output/")
   ```

### Modify the connector system

- Peg/hole assignment: `core/connectors.py` -- `determine_peg_side()`
- Placement logic: `find_connector_position()` (centroid of cut surface)
- Boolean engines tried in order: `manifold` > `blender` > default
- If all booleans fail, connectors are silently skipped (never crash)

### Work with the split strategies

`core/mesh_splitter.py` uses three strategies in order:
1. **Capped slice** (trimesh built-in, needs rtree) -- preferred
2. **Manual cap** (uncapped slice + scipy Delaunay triangulation) -- fallback
3. **Face-based split** (centroid side classification, no cap) -- last resort

When adding a new strategy, insert it in the chain and update `SplitResult`
metadata accordingly.

## File-level reference

| File | Lines | What it does |
|------|-------|-------------|
| `core/geometry.py` | ~400 | Rotation, transforms, preview mesh generation |
| `core/plane_calculator.py` | ~1250 | All 7 cut plane algorithms |
| `core/mesh_splitter.py` | ~615 | Mesh slicing with 3 fallback strategies |
| `core/connectors.py` | ~300 | Peg/hole creation and boolean operations |
| `core/path_cutter.py` | ~340 | Geodesic path + face partition |
| `core/debug_capture.py` | ~280 | Capture/replay serialization |
| `ObjectSplitter.py` | ~1200 | Cura Tool adapter |
| `qml/ObjectSplitter.qml` | ~580 | Qt5 UI panel |
| `qt6/ObjectSplitter.qml` | ~580 | Qt6 UI panel |

## Dependencies

- `trimesh` (required) -- mesh operations. Bundled in `lib/`.
  To refresh: `python scripts/bundle_deps.py`
- `numpy`, `scipy` -- math, Delaunay triangulation fallback
- `manifold3d` (optional) -- faster boolean ops
- `shapely` -- cross-section polygon area (used in smallest/valley search)
- `networkx` -- graph algorithms
- `matplotlib` (optional) -- PNG rendering in viz

## Documentation

- [README.md](README.md) -- User-facing overview, installation, quick start
- [docs/cut-techniques.md](docs/cut-techniques.md) -- Detailed algorithm docs
  for every cut mode with complexity analysis
- [docs/architecture.md](docs/architecture.md) -- Module diagram, data flow,
  QML properties, test structure
- [ROADMAP.md](ROADMAP.md) -- Development phases and planned features

## Skills

Below are Claude Code skill definitions that could be created for this project.
These represent common workflows that benefit from standardized automation.

### `/test` -- Run tests

Run the full test suite and report results.

```bash
pytest -v --tb=short
```

### `/add-cut-mode` -- Add a new cut mode

Interactive workflow:
1. Ask for mode name and algorithm description.
2. Add constant to `ObjectSplitter.py`.
3. Create algorithm function in `core/plane_calculator.py`.
4. Wire in `_performCut`.
5. Update both QML files (combobox model, mode map, description).
6. Generate test stubs in `tests/test_plane_calculator.py`.
7. Run tests.

### `/add-property` -- Add a QML-exposed property

Interactive workflow:
1. Ask for property name, type, and default value.
2. Add instance variable in `ObjectSplitter.__init__`.
3. Add to `setExposedProperties()` list.
4. Create getter and setter methods.
5. Add UI control to both `qml/` and `qt6/` files.

### `sync-qml` -- Sync QML files  *(implemented: `.claude/skills/sync-qml/`)*

Diff `qml/` and `qt6/` panels, flag drift beyond the `import UM` line, and check
delimiter balance. Use after editing either QML file.

### `diagnose-cut` -- Debug a wrong/failing cut  *(implemented: `.claude/skills/diagnose-cut/`)*

Replay a debug capture offline and compare to the Cura log to find why a cut
grazed, hit the wrong feature, or fell back. Reproduces the real runtime env
(no triangulation engine). Use this whenever a cut "doesn't work".

### `/bundle-deps` -- Refresh bundled dependencies

```bash
python scripts/bundle_deps.py    # bundles only trimesh, networkx, rtree
```

Verify the bundle by importing trimesh from `lib/`.

### `/build-package` -- Build the installable .curapackage

```bash
python scripts/build_curapackage.py   # -> dist/ObjectSplitter-<version>.curapackage
```

Reads version/SDK from `plugin.json`; ships only runtime files (no tests/docs/
scripts). The OPC layout and metadata are handled by the script.
