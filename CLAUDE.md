# ObjectSplitter - Claude Code Project Guide

## What is this?

A Cura plugin that splits 3D meshes along planes. All computational logic
lives in `core/` (pure Python, no Cura deps). `ObjectSplitter.py` is a thin
adapter that wires Cura UI events to the core pipeline.

## Architecture

```
core/                  Pure logic - testable outside Cura
  geometry.py          Rotation matrices, coordinate transforms
  plane_calculator.py  Cut plane algorithms (horizontal, vertical, smallest, shortest seam)
  mesh_splitter.py     Mesh slicing with fallback strategies
  connectors.py        Peg & hole creation via boolean ops
  debug_capture.py     Serialize operations for offline replay

ObjectSplitter.py      Cura tool adapter (events, scene nodes, undo, QML properties)
__init__.py            Plugin registration (conditional Cura imports)
qml/                   Qt5 / Cura 4.x UI
qt6/                   Qt6 / Cura 5.x UI (keep both in sync)
viz/                   HTML/PNG visualization for debugging
```

## Running tests

```bash
pytest                          # 82 tests, ~2.5s
pytest tests/test_geometry.py   # single module
```

Config is in `pytest.ini`. Test deps are in `requirements-test.txt`.

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
- **No CI** yet. Run `pytest` locally before pushing.

## Common tasks

**Add a new QML-exposed property:**
1. Add instance var in `__init__`
2. Add to `setExposedProperties()` list
3. Add `getFoo` / `setFoo` methods (emit `propertyChanged`)
4. Add UI control in both `qml/` and `qt6/` QML files

**Add a new cut mode:**
1. Add constant in `ObjectSplitter.py`
2. Implement plane calculation in `core/plane_calculator.py`
3. Wire it in `_performCut` in `ObjectSplitter.py`
4. Add to QML combobox model + mode map in both QML files
5. Add tests in `tests/test_plane_calculator.py` and `tests/test_integration.py`

**Debug a failing cut:**
1. Enable "Debug capture" in the UI (or set `_capture_dir` in code)
2. Reproduce the cut - inputs saved to `captures/`
3. Replay offline:
   ```python
   from core.debug_capture import replay_operation, save_result_meshes
   result = replay_operation("captures/<name>")
   save_result_meshes(result, "output/")
   ```

## Dependencies

- `trimesh` (required) - mesh operations
- `numpy`, `scipy` - math, Delaunay triangulation fallback
- `manifold3d` (optional) - faster boolean ops
- `shapely`, `networkx` - cross-section & seam algorithms
- `matplotlib` (optional) - PNG rendering in viz
