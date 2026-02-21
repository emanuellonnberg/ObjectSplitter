# Cut Techniques

ObjectSplitter provides seven cut modes, ranging from simple planar cuts to
geodesic surface-following algorithms. This document explains the algorithm
behind each mode, when to use it, and its performance characteristics.

## Overview

The modes fall into three categories:

1. **Planar (simple)** -- Horizontal, Vertical. Instant. Good for basic splits.
2. **Planar (search)** -- Smallest Section, Valley. Search many orientations to
   find the optimal plane. Good for necks, joints, and narrow features.
3. **Non-planar (geodesic)** -- Shortest Seam, Radial, Valley Seam, Path. Cut along the mesh
   surface instead of through a flat plane. Good for organic shapes where no
   single plane produces a clean seam.

---

## 1. Horizontal cut

**Mode string:** `horizontal`
**Source:** `core/plane_calculator.py` -- `horizontal_cut_plane()`

### How it works

Cuts the mesh parallel to the build plate (XZ plane) at a user-specified
percentage of the object's height.

```
origin = [0, y_min + height * percent / 100, 0]
normal = [0, 1, 0]
```

### When to use it

- Splitting a model into top and bottom halves for printing on a flat bed.
- Reducing print height to fit within a printer's Z limit.
- Any time you want a clean, predictable horizontal split.

### Parameters

| Parameter | Description |
|-----------|-------------|
| Height % (0-100) | Where along the Y axis to cut. 50% = middle. |

### Complexity

O(1) -- just arithmetic on the bounding box.

---

## 2. Vertical cut

**Mode string:** `vertical`
**Source:** `core/plane_calculator.py` -- `vertical_cut_plane()`

### How it works

Cuts perpendicular to the build plate through the point where the user clicks.
The default plane normal is `[1, 0, 0]` (X axis), but the plugin can orient it
based on the camera view direction.

```
origin = click_position
normal = [1, 0, 0]  (or view-aligned)
```

### When to use it

- Splitting a model left/right or front/back.
- Dividing wide models that exceed the printer's X or Y build volume.

### Parameters

None beyond the click position.

### Complexity

O(1).

---

## 3. Smallest cross-section

**Mode string:** `smallest`
**Source:** `core/plane_calculator.py` -- `find_smallest_cut_plane()`

This is the core algorithm that makes ObjectSplitter useful for complex models.
It finds the plane orientation that produces the minimum cross-sectional area at
the click point -- automatically finding necks, joints, and narrow features.

### How it works

#### Step 1: Spherical sampling

The algorithm tests `n_theta * 2 * n_theta` plane orientations uniformly
distributed over the unit sphere. For each orientation (theta, phi):

```
normal = [sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)]
```

With the default resolution of 18, this produces 18 * 36 = **648 samples**.

#### Step 2: Cross-section computation

For each candidate normal, the algorithm:

1. Computes the cross-section: `mesh.section(plane_origin, normal)` -> Path3D.
2. Projects the 3D path to 2D on the cutting plane.
3. Extracts the polygon area **nearest to the click point** (not total area).
   This prevents surface-grazing planes from winning because they happen to
   produce a tiny sliver of area far from where you clicked.

#### Step 3: Minimum area threshold

Cross-sections smaller than `5 * average_face_area` are rejected. This filters
out surface-grazing orientations that skim the mesh at a shallow angle and
produce misleadingly small areas.

#### Step 4: Alignment bias (optional)

When a surface normal is available (from the face the user clicked on), the
scoring gently biases toward planes aligned with that direction:

```
score = area * (1 + 0.5 * (1 - |dot(candidate_normal, surface_normal)|))
```

- A perfectly aligned plane: `score = area` (no penalty).
- A perpendicular plane: `score = area * 1.5` (50% penalty).

This means if you click on the side of a neck, the algorithm prefers cutting
through the neck in the direction you clicked from, rather than some arbitrary
angle that happens to be marginally smaller.

#### Step 5: Fallback

If no valid cross-sections are found (e.g. the click is outside the mesh),
the algorithm falls back to axis-aligned normals: Y, X, Z.

### When to use it

- Figurine necks, wrists, ankles.
- Mechanical joints and connectors.
- Any feature where you want the smallest possible cut surface.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Search resolution | Number of elevation angles (azimuth = 2x this) | 18 |

Higher resolution tests more orientations (finer search) but takes longer:

| Resolution | Samples | Approx. time |
|------------|---------|---------------|
| 6 | 72 | ~0.3 s |
| 18 | 648 | ~1-3 s |
| 36 | 2592 | ~5-10 s |

### Complexity

O(n_theta * n_phi * T) where T is the time per cross-section computation,
which depends on mesh complexity.

### Design decisions

**Why use local area instead of total area?**
A plane that barely clips a thin feature far from the click point can have tiny
total area but is useless as a cut. Using only the polygon nearest to the click
ensures the algorithm finds genuinely useful cuts.

**Why the alignment bias?**
Without it, clicking on the side of a neck might select a plane 90 degrees away
that has a marginally smaller area. The bias breaks ties in favor of the
direction you clicked from, which matches user intent.

**Why the minimum area threshold?**
Surface-grazing planes can skim the mesh from hundreds of orientations, each
producing areas under 1 mm^2. The threshold (5x average face area) filters
these out so the search focuses on real through-cuts.

---

## 4. Valley / groove detection

**Mode string:** `valley`
**Source:** `core/plane_calculator.py` -- `find_valley_cut_plane()`

An enhancement of the smallest-section search that can find grooves even when
the user's click is slightly off-center.

### How it works

#### Phase 1: Coarse search

Same as smallest-section: sample all orientations and collect the top 20
candidates by score.

#### Phase 2: Refinement sweep

For each of the top 20 candidates, **sweep the plane origin** along the
candidate normal direction within a window of +/-15% of the mesh diagonal.
Test 11 positions along each sweep and keep the one with the minimum
cross-section area.

```
for offset in linspace(-sweep_distance, +sweep_distance, 11):
    swept_origin = click_position + offset * candidate_normal
    area = compute_section_area(mesh, swept_origin, candidate_normal)
    if area < sweep_best:
        sweep_best = area
```

This "slides" the cutting plane along each axis to find where the groove is
truly narrowest.

#### Phase 3: Proximity bias

A mild penalty (30%) is applied based on how far the optimal position is from
the original click point. This prevents the algorithm from wandering too far
from where the user intended to cut.

```
proximity_penalty = 1.0 + 0.3 * (|offset| / sweep_distance)
final_score = score * proximity_penalty
```

### When to use it

- Figurine necks where clicking exactly at the narrowest point is difficult.
- Natural grooves, valleys, and indentations in organic models.
- Any time smallest-section finds a good orientation but misses the optimal
  position along that axis.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Search resolution | Same as smallest mode | 18 |
| Sweep fraction | How far to sweep (% of mesh diagonal) | 15% |
| Sweep steps | Positions tested per candidate | 11 |
| Top candidates | How many coarse results to refine | 20 |

### Complexity

O(n_theta * n_phi * T) for phase 1, plus O(top_candidates * n_sweep_steps * T)
for phase 2. Roughly 1.5-2x the cost of smallest-section.

### Design decisions

**Why not just use a finer grid for smallest-section?**
Finer sampling improves angular resolution but doesn't help with positional
accuracy. If the narrowest point is 5 mm away from the click, no amount of
angular sampling will find it -- you need to sweep the origin.

**Why limit the sweep to 15%?**
Larger sweeps risk finding unrelated narrow features elsewhere on the model.
15% of the diagonal is enough to correct for typical mis-clicks while staying
local to the intended region.

---

## 5. Shortest seam

**Mode string:** `shortest`
**Source:** `core/plane_calculator.py` -- `find_shortest_seam_partition()`,
`refine_partition_with_mincut()`

This mode finds the shortest path around the mesh surface that separates the
clicked region from the rest. Unlike planar modes, the cut follows the mesh
geometry and can go around corners and through saddle points.

### How it works

The algorithm runs in a daemon thread with a **10-second timeout** to prevent
UI freezes.

#### Step 1: Source face identification

Find the mesh face closest to the click position using trimesh's proximity
query. Falls back to nearest-vertex lookup if the proximity query fails.

#### Step 2: Build face adjacency graph

Compute face centroids and edge lengths between adjacent faces using numpy
vectorized operations. Build a CSR-like adjacency structure for efficient
traversal of meshes with 20k+ faces.

#### Step 3: Dijkstra from source

Run Dijkstra's algorithm on the face adjacency graph from the source face.
Edge weights are centroid-to-centroid distances. This produces a distance
field `dist_src[f]` for every face f.

#### Step 4: Smart sink selection

Select a sink face that is both:
- **Far** from the source (at least 40% of max geodesic distance).
- **Aligned** with the opposite direction from the surface normal at the
  click point (if available).

This is done with a combined score:

```
score = normalized_distance * alignment_score
```

where `alignment_score = (dot(direction_to_face, -surface_normal) + 1) / 2`.

#### Step 5: Dual-Dijkstra partition

Run a second Dijkstra from the sink face to get `dist_sink[f]`. Then compute
a relative score for each face:

```
relative_score = dist_src / (dist_src + dist_sink)
```

- Faces near the source: score close to 0.
- Faces near the sink: score close to 1.
- Faces equidistant: score = 0.5.

#### Step 6: Optimal threshold sweep

Sweep 100 threshold values from 0.1 to 0.9. For each threshold t, partition
faces into set_a (score < t) and set_b (score >= t). Compute the total length
of edges crossing the boundary. Pick the threshold that minimizes this boundary
length -- this is the shortest seam.

### Refinement: min-cut (optional)

After the initial partition, `refine_partition_with_mincut()` can improve the
boundary using Dinic's max-flow algorithm:

1. **Find deep anchors**: BFS from the boundary inward to find the face
   farthest from the boundary in each partition. These become source and sink
   for the flow network.

2. **Build flow graph**: Each face adjacency edge gets capacity equal to its
   edge length plus a small base cost (10% of average edge length). The base
   cost acts as edge-count regularization -- it penalizes jagged paths with
   many short edges, preferring straighter cuts.

3. **Run Dinic's max-flow**: Find the minimum cut between source and sink.
   The reachable set from the source side becomes the refined partition.

4. **Boundary smoothing**: Greedy local optimization that moves boundary faces
   to the other side if doing so shortens the total boundary length. Iterates
   up to 5 times until convergence.

### When to use it

- Organic models where no single plane produces a clean cut (e.g. tentacles,
  tails, irregular protrusions).
- Models with saddle points or complex topology.
- When you want the cut to follow the model's natural geometry.

### Complexity

| Phase | Complexity |
|-------|-----------|
| Dijkstra (x2) | O(F log F) |
| Threshold sweep | O(F + E) |
| Min-cut (Dinic's) | O(V * E) worst case |
| Boundary smoothing | O(iterations * boundary_faces) |

Where F = faces, E = edges, V = vertices in the flow graph.

The 10-second timeout ensures the UI stays responsive even on very large meshes.

### Design decisions

**Why dual-Dijkstra instead of direct max-flow?**
Max-flow on the full mesh is too slow for 20k+ face models. Dual-Dijkstra
produces a good initial partition in O(F log F), and the optional min-cut
refinement only needs to operate near the boundary.

**Why the base cost in min-cut edge weights?**
Pure edge-length weights cause stair-stepping on diagonal cuts -- the algorithm
takes many short edges instead of fewer longer ones. Adding 10% of the average
edge length as a fixed cost per edge penalizes high edge count, producing
smoother boundaries.

**Why boundary smoothing after min-cut?**
Min-cut finds the globally optimal cut through the flow graph, but the result
can still have local irregularities where moving a single face would shorten
the boundary. Greedy smoothing cleans these up.

---

## 6. Radial (geodesic)

**Mode string:** `radial`
**Source:** Uses the same `find_shortest_seam_partition()` as shortest seam,
but without the min-cut refinement step.

### How it works

Identical to shortest seam steps 1-6 (dual-Dijkstra partition with threshold
sweep), but skips the `refine_partition_with_mincut()` step. This makes it
faster but with a potentially rougher boundary.

### When to use it

- When shortest seam is too slow and you're willing to accept a less refined
  boundary.
- As a quick preview of where the geodesic cut would land before committing to
  the more expensive shortest seam computation.

### Complexity

O(F log F) for two Dijkstra passes plus O(F + E) for the threshold sweep.

---

## 7. Path (multi-point geodesic)

**Mode string:** `path`
**Source:** `core/path_cutter.py`

The user manually places waypoints on the mesh surface. The algorithm connects
consecutive waypoints via geodesic shortest paths along mesh edges, then
partitions faces on each side of the resulting path.

### How it works

#### Step 1: Snap waypoints to vertices

Each 3D waypoint is snapped to the nearest mesh vertex. If a waypoint lands on
a different connected component than the first waypoint, it is re-snapped to the
nearest vertex on the correct component.

#### Step 2: Geodesic path computation

For each consecutive pair of waypoints, run vertex-based Dijkstra on the mesh
edge graph (edge weights = Euclidean distances). This produces the shortest path
along mesh edges between the two vertices.

```python
full_path = []
for i in range(len(waypoints) - 1):
    segment = find_geodesic_path(mesh, waypoints[i], waypoints[i+1])
    if full_path and segment:
        # Remove duplicate junction vertex
        full_path.extend(segment[1:])
    else:
        full_path.extend(segment)
```

#### Step 3: Face partition

1. Identify all edges along the path.
2. Build face adjacency, excluding edges that cross the path (those edges
   separate the two sides).
3. For each face adjacent to the path, determine its side using the cross
   product of the path tangent and the face-to-centroid vector.
4. Flood-fill from assigned boundary faces to label all remaining faces.

### When to use it

- When you want precise control over exactly where the cut goes.
- Complex cuts that can't be described by a single plane.
- Cuts that need to follow specific surface features or avoid certain areas.

### Parameters

| Parameter | Description |
|-----------|-------------|
| Close Loop | Connect the last waypoint back to the first |
| Waypoints | 2+ points placed by clicking on the mesh |

### Complexity

O(k * V log V) for k waypoint pairs with V vertices, plus O(F) for face
partition flood-fill.

---

## 8. Valley seam (concavity path)

**Mode string:** `valley_seam`
**Source:** `core/plane_calculator.py` -- `find_valley_seam_partition()`

Valley Seam is a non-planar mode that prefers seams running through concave
surface regions near the click. It is intended for "throat/groove" separations
where planar valley can produce a straight notch.

### How it works

1. Build the face adjacency graph.
2. Compute a weighted edge cost:
   - lower cost on concave adjacencies,
   - higher cost on convex adjacencies,
   - mild locality penalty for edges far from the click.
3. Run dual Dijkstra (source/sink) on this weighted graph.
4. Sweep score thresholds and select the boundary minimizing weighted seam cost.
5. Return face partitions directly (non-planar split path).

### When to use it

- Cutting around a clicked feature along a natural groove/seam.
- Organic or rounded "throat" geometry where a flat valley plane is not ideal.

### Complexity

Similar to radial/shortest coarse partitioning: O(F log F) for two Dijkstra
passes plus O(F + E) threshold evaluation.

---

## Mesh splitting strategies

Once a cut plane or face partition is determined, the actual mesh splitting is
handled by `core/mesh_splitter.py` using three fallback strategies:

### Strategy 1: Capped slicing (preferred)

```python
upper = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=True)
lower = trimesh.intersections.slice_mesh_plane(mesh, -normal, origin, cap=True)
```

Uses trimesh's built-in capped slicing. Produces watertight results.
Requires `rtree` to be available.

### Strategy 2: Manual capping (fallback)

1. Perform uncapped slicing.
2. Compute the cross-section polygon at the cut plane.
3. Triangulate it using scipy Delaunay (or trimesh's own triangulator).
4. Transform the 2D triangulation back to 3D.
5. Ensure the cap normal faces the correct direction.
6. Concatenate the cap with the sliced mesh.

### Strategy 3: Face-based split (last resort)

```python
face_centroids = vertices[faces].mean(axis=1)
face_distances = dot(face_centroids - plane_origin, plane_normal)
upper_faces = faces[face_distances >= 0]
lower_faces = faces[face_distances < 0]
```

Simple centroid-side classification. No capping -- produces open meshes.
Always succeeds but results are not watertight.

### For non-planar modes (seam, radial, path)

These modes produce face partitions directly. The mesh is split into two
submeshes using `mesh.submesh()`, then `fill_holes()` is attempted for
watertightness.

---

## Algorithm comparison

| Mode | Cut shape | Automatic | User control | Best for |
|------|-----------|-----------|-------------|----------|
| Horizontal | Flat plane | Height only | Height slider | Flat splits |
| Vertical | Flat plane | Position only | Click position | Left/right splits |
| Smallest | Flat plane | Full | Click + resolution | Necks, joints |
| Valley | Flat plane | Full | Click + resolution | Grooves, indentations |
| Shortest seam | Surface-following | Full | Click point | Organic protrusions |
| Radial | Surface-following | Full | Click point | Quick geodesic preview |
| Path | Surface-following | Manual | Waypoints | Precise custom cuts |
| Valley seam | Surface-following | Full | Click point | Concave groove/throat seams |
