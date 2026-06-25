# Architecture

ObjectSplitter is structured as a **pure-logic core** wrapped by a thin
**Cura adapter**. This separation means the algorithms can be tested, debugged,
and iterated on without launching Cura.

## Layer diagram

```
+----------------------------------------------------------+
|  Cura UI (QML panels)                                    |
|    qml/ObjectSplitter.qml   (Qt5 / Cura 4.x)            |
|    qt6/ObjectSplitter.qml   (Qt6 / Cura 5.x)            |
+----------------------------------------------------------+
|  ObjectSplitter.py  (Cura Tool adapter)                  |
|    - Mouse / keyboard event handling                     |
|    - Scene node creation and undo                        |
|    - QML property bridge                                 |
|    - Progress dialogs                                    |
+----------------------------------------------------------+
|  core/  (pure Python, no Cura imports)                   |
|    plane_calculator  ->  mesh_splitter  ->  connectors   |
|    geometry           path_cutter        debug_capture   |
+----------------------------------------------------------+
|  trimesh, networkx, rtree (bundled in lib/);             |
|  numpy, scipy, shapely provided by Cura at runtime       |
+----------------------------------------------------------+
```

Cura's bundled Python ships no mesh-builder triangulation engine
(`mapbox_earcut` / `triangle`), so the core must never rely on one. The clean
local split (`mesh_splitter.clean_local_plane_split`) slices uncapped and caps
via scipy for exactly this reason.

## Data flow: click to split

```
User clicks on mesh
        |
        v
ObjectSplitter.event()
        |
        +-- pick pass: get 3D position + face ID + scene node
        |
        v
_performCut(node, click_position)
        |
        +-- extract trimesh from Cura mesh data
        +-- merge vertices, compute world-to-local transform
        +-- snap click position to mesh surface
        |
        v
  +-- plane_calculator --+
  |  horizontal_cut_plane |  (or vertical, smallest, valley)
  |  find_smallest_cut_plane |
  |  find_valley_cut_plane   |
  |  find_shortest_seam_partition |  (shortest seam / radial)
  +---------------------------+
        |
        v  CutPlane(origin, normal)  or  (face_set_a, face_set_b)
        |
  +-- mesh_splitter ------+
  |  slice_mesh_with_fallback |  (for plane-based modes)
  |  split_by_face_sets       |  (for seam/radial/path modes)
  |  split_by_local_plane     |  (smallest mode with candidates)
  +---------------------------+
        |
        v  SplitResult(upper, lower, capped, strategy_used)
        |
  +-- connectors ----------+
  |  add_connectors          |
  |    determine_peg_side    |
  |    find_connector_position |
  |    create_peg_mesh       |
  |    create_hole_mesh      |
  |    try_boolean_difference |
  +---------------------------+
        |
        v  ConnectorResult(upper, lower, peg_on, hole_on)
        |
        v
Create Cura scene nodes (GroupedOperation for undo)
```

## Module responsibilities

### `core/geometry.py`

Low-level 3D math. No dependencies beyond numpy.

| Function | Purpose |
|----------|---------|
| `rotation_matrix_from_vectors` | Rodrigues' formula: 3x3 matrix rotating vec1 to vec2 |
| `world_to_local_transform` | SVD-based Procrustes alignment from vertex correspondence |
| `transform_point_to_local` | Apply inverse transform: `R^T @ (world - T) / scale` |
| `plane_normal_from_spherical` | Spherical (theta, phi) to unit normal |
| `create_plane_mesh_data` | Generate preview quad for translucent plane overlay |
| `create_marker_mesh_data` | Generate arrow geometry for non-planar mode preview |
| `create_pin_mesh_data` | Generate diamond pin for path-mode waypoints |

### `core/plane_calculator.py`

Determines where to cut. Returns `CutPlane(origin, normal)` or face partitions.

| Function | Modes | Output |
|----------|-------|--------|
| `horizontal_cut_plane` | Horizontal | CutPlane |
| `vertical_cut_plane` | Vertical | CutPlane |
| `find_smallest_cut_plane` | Smallest | SmallestPlaneSearchResult |
| `find_valley_cut_plane` | Valley | SmallestPlaneSearchResult |
| `find_shortest_seam_partition` | Shortest seam, Radial | (set_a, set_b, source, sink) |
| `refine_partition_with_mincut` | Shortest seam | (refined_a, refined_b) |
| `smooth_partition_boundary` | Shortest seam | (smoothed_a, smoothed_b) |
| `snap_point_to_mesh_surface` | All modes | (point, face_id) |

### `core/mesh_splitter.py`

Performs the actual mesh cutting.

| Function | Purpose |
|----------|---------|
| `slice_mesh_with_fallback` | 3-strategy cascade for plane-based splitting |
| `split_by_face_sets` | Split using pre-computed face partitions |
| `split_by_local_plane` | Plane-guided local separation with candidate ranking |
| `split_by_shortest_seam` | Wrapper for face-set splitting (seam mode) |
| `local_plane_partition` | Classify faces by plane side + BFS connectivity |

### `core/connectors.py`

Adds interlocking geometry to split parts.

| Function | Purpose |
|----------|---------|
| `add_connectors` | Main entry: adds peg to smaller part, hole to larger |
| `determine_peg_side` | Volume comparison to assign peg vs hole |
| `find_connector_position` | Cross-section centroid for connector placement |
| `create_peg_mesh` | Generate oriented cylinder peg |
| `create_hole_mesh` | Generate oriented cylinder hole (larger, for boolean) |
| `try_boolean_difference` | Try manifold > blender > default engines |

### `core/path_cutter.py`

Multi-point geodesic cutting.

| Function | Purpose |
|----------|---------|
| `find_geodesic_path` | Vertex-based Dijkstra between two vertices |
| `snap_to_nearest_vertex` | Snap a 3D point to the closest mesh vertex |
| `chain_paths` | Connect waypoints into a single path, handling components |
| `partition_faces_by_path` | Flood-fill face assignment from cutting path |

### `core/debug_capture.py`

Serialize and replay operations for offline debugging.

| Function | Purpose |
|----------|---------|
| `capture_operation` | Save mesh + params to `captures/<name>/` |
| `load_captured_operation` | Load mesh + params from disk |
| `replay_operation` | Re-run full pipeline with captured inputs |
| `save_result_meshes` | Export result STLs for inspection |

### `ObjectSplitter.py`

The Cura `Tool` subclass. Bridges UI events to the core pipeline.

**Key responsibilities:**
- Event handling (mouse move, click, Ctrl+click)
- Extracting trimesh data from Cura scene nodes
- Managing preview overlays (plane, arrow, pins)
- Progress dialog during long operations
- Creating scene nodes from split results
- Undo/redo via `GroupedOperation`
- QML property exposure (getters, setters, `propertyChanged` signals)

### QML files

Both `qml/ObjectSplitter.qml` and `qt6/ObjectSplitter.qml` contain the same
UI layout. The only difference is the import statement:

- `qml/`: `import UM 1.5 as UM` (Cura 4.x)
- `qt6/`: `import UM 1.6 as UM` (Cura 5.x)

**UI layout:**
1. Title + trimesh availability warning
2. Cut mode dropdown (ComboBox)
3. Mode description (dynamic text)
4. Mode-specific controls (height slider, resolution slider, path buttons)
5. Preview toggle
6. Connector controls (enable, diameter, height, clearance sliders)
7. Debug capture toggle
8. Usage instructions

## QML property binding

Properties are exposed via `setExposedProperties()` in `ObjectSplitter.py`.

| Property | Type | Read/Write | Purpose |
|----------|------|-----------|---------|
| CutMode | str | R/W | Selected algorithm |
| CutModes | list | R | Available modes for dropdown |
| CutHeightPercent | float | R/W | 0-100 for horizontal mode |
| ShowPreview | bool | R/W | Toggle preview overlay |
| TrimeshAvailable | bool | R | Library check |
| SearchResolution | int | R/W | 6-36 for smallest/valley mode |
| ConnectorEnabled | bool | R/W | Toggle peg/hole |
| ConnectorDiameter | float | R/W | mm |
| ConnectorHeight | float | R/W | mm |
| ConnectorClearance | float | R/W | mm |
| DebugCaptureEnabled | bool | R/W | Save inputs to disk |
| PathPointCount | int | R | Number of placed waypoints |
| PathCloseLoop | bool | R/W | Loop path back to start |
| TriggerPathCut | bool | W | Execute path cut |
| OpenScadPath | str | R/W | For Blender boolean engine |

**Setter pattern:**
```python
def setFoo(self, value):
    if value != self._foo:
        self._foo = value
        self.propertyChanged.emit()
```

## Test structure

Tests live in `tests/` and use fixtures from `conftest.py`.

| Test file | Coverage |
|-----------|----------|
| `test_geometry.py` | Rotation matrices, transforms, mesh generation |
| `test_plane_calculator.py` | All plane calculation algorithms |
| `test_mesh_splitter.py` | Slicing strategies, fallbacks, partitions |
| `test_connectors.py` | Peg/hole creation, booleans, volume comparison |
| `test_path_cutter.py` | Geodesic paths, waypoint chaining, components |
| `test_integration.py` | End-to-end cutting workflows |
| `test_radial_partition.py` | Dual-Dijkstra, min-cut refinement |
| `test_roundtrip.py` | Capture and replay validation |

**Key fixtures:**
- `cube_mesh`, `tall_box_mesh`, `sphere_mesh`, `cylinder_mesh`, `torus_mesh`
- `l_shaped_mesh`, `translated_cube_mesh`, `scaled_cube_mesh`
- `center_click`, `offset_click`, `surface_click_cube`
- `default_connector_config`, `small_connector_config`, `large_connector_config`

Run with:
```bash
pytest              # all tests
pytest -v           # verbose
pytest -k smallest  # filter by keyword
```
