# ObjectSplitter

A [Cura](https://ultimaker.com/software/ultimaker-cura/) plugin that splits 3D meshes for 3D printing. Decompose large models into smaller, printable pieces with optional interlocking peg-and-hole connectors.

## Features

- **Reliable, user-defined cuts** -- the default modes give a predictable result from where you click, every time:
  - **Multi-point** -- place points; the cut follows a geodesic path through them
  - **Plane** -- one click cuts cleanly *across* the clicked feature (**Along surface**), or pick **Horizontal**, or a **3-point plane** at any angle; optional whole-model cut for stacking splits
  - **Isolate region** -- draw one or more closed loops, pick a side, extract it
- **Clean, watertight cuts** -- a capped-slice local split produces a flat cut face localized to the clicked feature (no triangle-edge sawtooth), with **no triangulation-engine dependency**, so it works in a stock Cura install
- **Interlocking connectors** -- automatic peg-and-hole placement so parts reassemble aligned (planar and along-path cuts)
- **Real-time preview** -- a cut plane or surface-normal arrow follows your cursor, cached for smooth hovering on dense meshes
- **Debug capture & replay** -- serialize any cut to disk and reproduce it offline for diagnosis
- **Experimental search modes** -- Smallest, Shortest seam, Radial, Valley, Valley-seam (behind a "show experimental" toggle; an automatic search that is less reliable on arbitrary meshes)
- **Cura 4.x and 5.x** -- Qt5 and Qt6 UIs kept in sync

## Installation

**From a package (recommended):** build a `.curapackage` and drag it onto Cura
(or Marketplace -> Install from file), then restart.

```bash
python scripts/build_curapackage.py   # -> dist/ObjectSplitter-<version>.curapackage
```

**From source:** copy the `ObjectSplitter` folder into your Cura plugins directory
(`%APPDATA%\cura\<version>\plugins\` on Windows, `~/Library/Application Support/cura/<version>/plugins/` on macOS, `~/.local/share/cura/<version>/plugins/` on Linux) and restart. The **Object Splitter** tool appears in the left toolbar.

The plugin bundles only the dependencies Cura does **not** ship -- `trimesh`
(4.x), `networkx`, and `rtree` -- in `lib/`; it uses Cura's own `numpy`, `scipy`,
and `shapely` for ABI safety. To refresh the bundle:

```bash
python scripts/bundle_deps.py
```

## Quick start

1. Load a model in Cura and select the **Object Splitter** tool.
2. Choose a cut mode (default: **Multi-point**, **Plane**, **Isolate region**).
3. Cut:
   - **Plane / Along surface:** hover (the arrow shows the clicked surface normal), then **left-click** the feature to lop off.
   - **Plane / Horizontal:** set the height, click (or enable *Cut whole model*).
   - **Plane / 3-point plane:** click three points to define the plane; it cuts on the third click.
   - **Multi-point:** place 2+ points, then press **Cut**.
   - **Isolate region:** draw a loop, **Finish Loop**, optionally add loops, **Pick Target Region**, click the side to keep, **Isolate Region**.
4. Optionally enable **connectors** to add interlocking pegs and holes.

For arbitrary cuts on interconnected geometry (e.g. a ladder's rungs), use
**Multi-point** -- a single plane crosses all of several parallel features.

## Cut modes

See [docs/cut-techniques.md](docs/cut-techniques.md) for the algorithm behind
each mode. Summary:

| Mode | What it does | Reliability |
|------|-------------|-------------|
| **Multi-point** | Place points; geodesic path through them | predictable |
| **Plane -- Along surface** | Cuts across the clicked feature (plane found by rotating about the surface normal to the smallest local cross-section) | predictable |
| **Plane -- Horizontal** | Flat cut parallel to the build plate (height %, whole-model for stacking) | predictable |
| **Plane -- 3-point** | Plane through three clicked points, any angle | predictable |
| **Isolate region** | Extract a region bounded by closed loops | predictable |
| *Smallest / Valley* | Search all orientations for the minimum cross-section | experimental |
| *Shortest seam / Radial* | Dual-Dijkstra geodesic surface partition | experimental |
| *Valley-seam* | Concavity-biased geodesic seam | experimental |

The experimental modes use an automatic search that can pick grazing or
unintended planes on real models; they are hidden behind a toggle. The point
modes (Valley / Valley-seam with anchors enabled) instead cut a valley-weighted
geodesic *through your placed points*.

## Connector system

When connectors are enabled, the plugin automatically:

1. Compares the volume of the two halves.
2. Adds a **peg** (cylinder) to the smaller part.
3. Subtracts a **hole** (slightly larger cylinder) from the larger part via boolean difference.

Planar cuts place the peg/hole on the cut plane; along-path cuts (Multi-point
and anchored Valley) place connectors along the cut path. Connectors need a
watertight cut surface; on non-watertight input meshes they may be skipped.

Configurable parameters in the UI:

| Parameter | Default | Range |
|-----------|---------|-------|
| Diameter | 4.0 mm | 2 -- 10 mm |
| Height | 3.0 mm | 1 -- 8 mm |
| Clearance | 0.2 mm | 0.1 -- 0.5 mm |

Boolean operations try engines in order: `manifold` > `blender` > default trimesh.

## Project structure

```
ObjectSplitter/
  core/                  Pure logic (no Cura dependencies)
    geometry.py          Rotation matrices, coordinate transforms, mesh previews
    plane_calculator.py  Cut plane algorithms (planar, search, geodesic seam)
    mesh_splitter.py     Mesh slicing: clean local split + fallback strategies
    connectors.py        Peg & hole creation via boolean ops
    path_cutter.py       Geodesic / valley-weighted path cutting
    debug_capture.py     Serialize operations for offline replay
  ObjectSplitter.py      Cura tool adapter (events, scene nodes, undo, QML)
  __init__.py            Plugin registration (conditional Cura imports)
  qml/                   Qt5 / Cura 4.x UI
  qt6/                   Qt6 / Cura 5.x UI
  scripts/               bundle_deps.py, build_curapackage.py, profiling
  tests/                 ~286 tests (~20 s)
  docs/                  Architecture and algorithm documentation
  .github/workflows/     CI: pytest on py3.10-3.12 (Linux) + py3.12 (Windows)
```

For detailed architecture documentation see [docs/architecture.md](docs/architecture.md).

## Development

### Running tests

```bash
pip install -r requirements-test.txt   # includes rtree, triangle, mapbox-earcut
pytest                                  # ~286 tests, ~20 s
pytest tests/test_clean_local_split.py  # single module
pytest -v                               # verbose output
```

CI runs the suite on Python 3.10-3.12 (Linux) and 3.12 (Windows). A few fork
tessellation tests are skipped when the `triangle` engine has no wheel for the
running Python.

### Debugging a failed cut

1. Enable **Debug capture** in the UI (or set `_capture_dir` in code).
2. Reproduce the cut -- inputs (mesh + params) are saved to `captures/`.
3. Replay offline to inspect what the cut actually did:

```python
from core.debug_capture import replay_operation, save_result_meshes

result = replay_operation("captures/<name>")
save_result_meshes(result, "output/")
# Inspect result_upper.stl and result_lower.stl in any 3D viewer
```

### Adding a new cut mode

See the [Common Tasks](CLAUDE.md#common-tasks) section in CLAUDE.md.

## Dependencies

Bundled in `lib/` (Cura does not ship them):

| Package | Purpose |
|---------|---------|
| `trimesh` (4.x) | Mesh operations, slicing, sections |
| `networkx` | Graph algorithms |
| `rtree` | Accelerated spatial queries for slicing (Windows DLL bundled) |

Provided by Cura at runtime (used, not bundled, for ABI safety): `numpy`,
`scipy`, `shapely`. Optional: `manifold3d` (faster booleans). Test-only:
`triangle`, `mapbox-earcut` (mesh-builder triangulation), `matplotlib` (viz).

## License

LGPLv3 or higher. Copyright (c) 2024-2026 Emanuel Lönnberg.
