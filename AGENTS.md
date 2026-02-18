# ObjectSplitter Agent Guide

This file is for coding agents working in this repository.

## Project Summary

ObjectSplitter is a Cura plugin that splits 3D meshes for printing.

- `core/` contains pure Python geometry/splitting logic (no Cura imports).
- `ObjectSplitter.py` is the Cura adapter (UI events, scene updates, undo).
- `qml/` (Cura 4.x) and `qt6/` (Cura 5.x) are parallel UIs and must stay in sync.

## Repository Map

```
core/
  geometry.py          transforms and preview geometry
  plane_calculator.py  cut-plane and partition algorithms
  mesh_splitter.py     split strategies and face-set splitting
  connectors.py        peg/hole generation and booleans
  path_cutter.py       geodesic path partitioning
  debug_capture.py     capture/replay for offline debugging

ObjectSplitter.py      Cura tool integration
qml/ObjectSplitter.qml Qt5 UI (UM 1.5)
qt6/ObjectSplitter.qml Qt6 UI (UM 1.6)
tests/                 unit and integration tests
```

## Working Rules

1. Keep `core/` Cura-independent.
2. When changing UI, update both `qml/ObjectSplitter.qml` and `qt6/ObjectSplitter.qml`.
3. Preserve fallback-chain patterns in splitting/boolean code.
4. Run tests before finishing work.

## Test Commands

```bash
pytest -q
pytest -v --tb=short
pytest tests/test_mesh_splitter.py -q
```

Test config is in `pytest.ini`, and test deps are in `requirements-test.txt`.

## Common Change Workflows

### Add a new cut mode

1. Add constant in `ObjectSplitter.py`.
2. Implement algorithm in `core/plane_calculator.py`.
3. Wire mode handling in `_performCut` (`ObjectSplitter.py`).
4. Add mode to both QML files (model, mapping, description).
5. Add tests in `tests/test_plane_calculator.py` and `tests/test_integration.py`.

### Add a new QML-exposed property

1. Add backing field in `ObjectSplitter.__init__`.
2. Add property name in `setExposedProperties(...)`.
3. Implement getter/setter with change guard and `propertyChanged.emit()`.
4. Add matching UI in both QML files.

### Debug a failing cut

1. Enable debug capture in UI.
2. Reproduce cut to create `captures/<name>/`.
3. Replay offline:

```python
from core.debug_capture import replay_operation, save_result_meshes
result = replay_operation("captures/<name>")
save_result_meshes(result, "output/")
```

## Notes on Face-Set Splits

- `split_by_face_sets(...)` supports optional hole filling via `attempt_hole_fill`.
- Shortest/radial/path mode calls intentionally disable hole filling to preserve
  original face-partition counts.

## Dependency Notes

- Core runtime dependencies are bundled in `lib/`.
- To refresh bundled dependencies:

```bash
python scripts/bundle_deps.py
```

## References

- `CLAUDE.md`
- `docs/architecture.md`
- `docs/cut-techniques.md`
- `README.md`
