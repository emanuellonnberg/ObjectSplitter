# ObjectSplitter Plugin - Development Roadmap

## Current Status (v0.1 - Proof of Concept)

- [x] Basic plugin structure
- [x] Horizontal cut mode
- [x] Vertical cut mode
- [x] Smallest cross-section finder algorithm
- [x] QML UI panel with mode selection
- [x] Scene node creation for split parts
- [x] Undo support via GroupedOperation

## Phase 1: Core Improvements

### 1.1 Cut Plane Preview
- [x] Show translucent plane mesh at cut location before committing
- [x] Update preview on mouse move (hover over object)
- [x] Toggle preview on/off in UI
- [ ] Different colors for valid/invalid cuts
- [ ] Preview both resulting parts with slight separation

### 1.2 Interactive Plane Adjustment
- [ ] Allow user to rotate the cut plane after initial placement
- [ ] Drag handles for plane orientation
- [ ] Keyboard shortcuts for 90° rotations
- [ ] "Confirm" / "Cancel" buttons before executing cut

### 1.3 Click Position Improvements
- [ ] For horizontal mode: use click Y position instead of just percentage
- [ ] For vertical mode: orient plane based on view direction
- [ ] Snap to object center option

---

## Phase 2: Connector System

### 2.1 Basic Peg & Hole Connectors

**Concept:** Add interlocking geometry at the cut surface so parts align when assembled.

```
Part A (smaller)     Part B (larger)
    ┌───────┐           ┌───────┐
    │       │           │   ○   │  ← hole
    │   ●   │           │       │
    │       │           │       │
    └───────┘           └───────┘
        ↑
      peg (protrusion)
```

**Implementation:**
- [x] Determine which part is larger (by volume or bounding box)
- [x] Add peg (cylinder) to smaller part's cut face
- [x] Add matching hole (boolean subtraction) to larger part's cut face
- [x] Configurable peg dimensions (diameter, height)
- [x] Slight clearance for fit (e.g., hole 0.2mm larger than peg)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `connector_enabled` | True | Enable/disable connectors |
| `connector_diameter` | 4.0 mm | Peg/hole diameter |
| `connector_height` | 3.0 mm | How deep the peg/hole extends |
| `connector_clearance` | 0.2 mm | Extra space in hole for fit |
| `connector_count` | 1 | Number of connectors (auto or manual) |
| `connector_shape` | "cylinder" | cylinder, square, dovetail |

### 2.2 Smart Connector Placement

- [x] Auto-detect good positions on cut surface (centroid-based)
- [ ] Ensure minimum distance from cut surface boundary
- [ ] For large cuts: multiple connectors in grid pattern
- [ ] Avoid thin areas that can't support a hole

**Algorithm for placement:**
1. Get the cut surface polygon (from trimesh cap)
2. Compute centroid of the cut surface
3. Check if centroid has enough material around it (min radius > connector_diameter)
4. If multiple connectors: use Poisson disk sampling on cut surface
5. Validate each position has sufficient depth on both sides

### 2.3 Volume Comparison Logic

```python
def determine_peg_side(mesh_a, mesh_b):
    """
    Determine which part gets the peg vs hole.
    Peg goes on smaller part, hole on larger part.

    Rationale: Hole removes material, so put it on the part
    with more material to spare.
    """
    volume_a = mesh_a.volume if mesh_a.is_watertight else mesh_a.convex_hull.volume
    volume_b = mesh_b.volume if mesh_b.is_watertight else mesh_b.convex_hull.volume

    if volume_a < volume_b:
        return "a_gets_peg", "b_gets_hole"
    else:
        return "b_gets_peg", "a_gets_hole"
```

### 2.4 Connector Shapes

**Cylinder (default):**
- Simple, works well
- Easy to print
- Requires rotation alignment

**Square/Rectangular:**
- Prevents rotation
- Harder to insert

**Dovetail / Tapered:**
- Self-aligning
- Snug fit
- More complex geometry

---

## Phase 3: Advanced Features

### 3.1 Multi-Cut Support
- [ ] Queue multiple cuts before executing
- [ ] Split into 3+ parts in one operation
- [ ] Tree-based splitting (split A, then split A1, etc.)

### 3.2 Cut Surface Analysis
- [ ] Warn if cut creates non-manifold geometry
- [ ] Warn if resulting parts are too thin
- [ ] Estimate print time/material for each part

### 3.3 Automatic Optimal Split
- [ ] Given a max print volume, find optimal cut(s)
- [ ] Minimize number of cuts needed
- [ ] Maximize structural integrity of joints

### 3.4 Label/Numbering
- [ ] Emboss part numbers on cut surfaces
- [ ] Add alignment marks
- [ ] Generate assembly instructions

---

## Phase 4: UX Polish

### 4.1 Improved UI
- [ ] Visual preview of connector placement
- [ ] Undo/redo for individual operations
- [ ] Presets for common scenarios (half, thirds, etc.)

### 4.2 Settings Persistence
- [ ] Remember last-used settings
- [ ] Save/load split configurations
- [ ] Per-model split history

### 4.3 Error Handling
- [ ] Better error messages for failed cuts
- [ ] Recovery suggestions
- [ ] Mesh repair integration

---

## Technical Notes

### Dependencies
- **trimesh** (required): Mesh cutting, boolean operations
- **manifold3d** (optional): Better boolean operations for connectors
- **numpy** (bundled with Cura): Array operations

### Mesh Cutting Approach
```python
# Current implementation using trimesh
upper = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=True)
lower = trimesh.intersections.slice_mesh_plane(mesh, -normal, origin, cap=True)
```

### Connector Boolean Operations
```python
# For adding peg (union)
peg = trimesh.creation.cylinder(radius=r, height=h)
peg.apply_translation([cx, cy, cz])
part_with_peg = trimesh.boolean.union([part, peg])

# For adding hole (difference)
hole = trimesh.creation.cylinder(radius=r + clearance, height=h + 0.1)
hole.apply_translation([cx, cy, cz])
part_with_hole = trimesh.boolean.difference([part, hole])
```

### Performance Considerations
- "Smallest section" search is O(n²) in search resolution
- Boolean operations can be slow on complex meshes
- Consider caching mesh analysis results
- Progress dialog for long operations

---

## Repository Migration Plan

When ready to move to separate repository:

1. Create new repo: `ObjectSplitter` or `CuraObjectSplitter`
2. Copy `ObjectSplitter/` folder contents to new repo root
3. Add proper README.md with installation instructions
4. Add LICENSE file (LGPLv3)
5. Set up GitHub releases for Cura plugin marketplace
6. Remove `ObjectSplitter/` from MySupportImprover repo (or keep as submodule)

---

## References

- [trimesh documentation](https://trimesh.org/)
- [Cura Plugin Development](https://github.com/Ultimaker/Cura/wiki/Plugin-Directory)
- [Mesh Boolean Operations](https://github.com/elalish/manifold)
- Similar tools: Meshmixer, Blender Boolean modifier
