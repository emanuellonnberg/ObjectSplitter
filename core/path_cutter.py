# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Path-based mesh cutting using geodesic shortest paths.

The user places 2+ waypoints on the mesh surface. For each consecutive pair,
we compute the shortest path along mesh edges (vertex-based Dijkstra).
The full path defines a cutting seam, and faces are flood-filled on each side
to create the partition.
"""

import heapq
import logging
from collections import deque, defaultdict
from typing import List, Tuple, Optional

import numpy

try:
    import trimesh
except ImportError:
    trimesh = None

logger = logging.getLogger("objectsplitter.path_cutter")


# ---------------------------------------------------------------------------
# Vertex-based Dijkstra shortest path
# ---------------------------------------------------------------------------

def find_geodesic_path(
    mesh: "trimesh.Trimesh",
    start_vertex: int,
    end_vertex: int,
) -> List[int]:
    """
    Find the shortest geodesic path between two vertices using Dijkstra's
    algorithm on the mesh edge graph.

    Edge weights are Euclidean distances between connected vertices.

    Args:
        mesh: The trimesh object.
        start_vertex: Index of the start vertex.
        end_vertex: Index of the end vertex.

    Returns:
        Ordered list of vertex indices from start_vertex to end_vertex (inclusive).

    Raises:
        ValueError: If no path exists between the vertices.
    """
    n_verts = len(mesh.vertices)
    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)

    # Build adjacency list from mesh edges
    edges = mesh.edges_unique
    edge_lengths = numpy.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )

    adj = [[] for _ in range(n_verts)]
    for i, (v0, v1) in enumerate(edges):
        w = float(edge_lengths[i])
        adj[v0].append((v1, w))
        adj[v1].append((v0, w))

    # Dijkstra
    dist = numpy.full(n_verts, numpy.inf)
    dist[start_vertex] = 0.0
    prev = numpy.full(n_verts, -1, dtype=numpy.int64)
    heap = [(0.0, start_vertex)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end_vertex:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if numpy.isinf(dist[end_vertex]):
        raise ValueError(
            f"No path between vertex {start_vertex} and {end_vertex}"
        )

    # Reconstruct path
    path = []
    v = end_vertex
    while v != -1:
        path.append(v)
        v = int(prev[v])
    path.reverse()
    return path


def snap_to_nearest_vertex(
    mesh: "trimesh.Trimesh",
    point: numpy.ndarray,
) -> int:
    """Find the mesh vertex closest to a 3D point."""
    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    dists = numpy.linalg.norm(vertices - point, axis=1)
    return int(numpy.argmin(dists))


# ---------------------------------------------------------------------------
# Chain multiple waypoints into a single path
# ---------------------------------------------------------------------------

def chain_paths(
    mesh: "trimesh.Trimesh",
    waypoints: List[numpy.ndarray],
) -> List[int]:
    """
    Given 2+ waypoints (3D positions on mesh surface), snap each to the
    nearest vertex and compute geodesic paths between consecutive pairs.

    Handles disconnected components: if a waypoint snaps to a vertex on a
    different component from the first waypoint, it is re-snapped to the
    nearest vertex on the same component.

    Returns:
        Ordered list of vertex indices forming the full path.
        Duplicate vertices at junctions are removed.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    n_verts = len(vertices)

    # Snap all waypoints to nearest vertices
    vert_indices = [snap_to_nearest_vertex(mesh, wp) for wp in waypoints]

    # --- Connected-component check ---
    # Build adjacency for component labelling
    edges = mesh.edges_unique
    comp_label = numpy.full(n_verts, -1, dtype=numpy.int64)
    adj_quick = [[] for _ in range(n_verts)]
    for v0, v1 in edges:
        adj_quick[v0].append(v1)
        adj_quick[v1].append(v0)

    comp_id = 0
    for seed in range(n_verts):
        if comp_label[seed] >= 0:
            continue
        queue = deque([seed])
        comp_label[seed] = comp_id
        while queue:
            u = queue.popleft()
            for nb in adj_quick[u]:
                if comp_label[nb] < 0:
                    comp_label[nb] = comp_id
                    queue.append(nb)
        comp_id += 1

    # Determine the target component (component of the first waypoint)
    target_comp = int(comp_label[vert_indices[0]])
    target_mask = comp_label == target_comp
    target_verts_idx = numpy.where(target_mask)[0]

    logger.info("Mesh has %d connected components; target component %d has %d vertices",
                comp_id, target_comp, len(target_verts_idx))

    # Re-snap any waypoints that landed on a different component
    for i in range(len(vert_indices)):
        if int(comp_label[vert_indices[i]]) != target_comp:
            # Find nearest vertex on the target component
            dists = numpy.linalg.norm(
                vertices[target_verts_idx] - waypoints[i], axis=1
            )
            best_local = int(numpy.argmin(dists))
            old_v = vert_indices[i]
            vert_indices[i] = int(target_verts_idx[best_local])
            logger.info("Waypoint %d re-snapped: vertex %d (comp %d) → %d (comp %d)",
                        i, old_v, int(comp_label[old_v]),
                        vert_indices[i], target_comp)

    logger.info("Path waypoints: %d points → vertex indices %s",
                len(waypoints), vert_indices)

    # Build chained path
    full_path = []
    for i in range(len(vert_indices) - 1):
        segment = find_geodesic_path(mesh, vert_indices[i], vert_indices[i + 1])
        if full_path and segment:
            # Remove duplicate junction vertex
            full_path.extend(segment[1:])
        else:
            full_path.extend(segment)

    logger.info("Full path: %d vertices", len(full_path))
    return full_path


# ---------------------------------------------------------------------------
# Partition faces by cutting path
# ---------------------------------------------------------------------------

def partition_faces_by_path(
    mesh: "trimesh.Trimesh",
    vertex_path: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Partition mesh faces into two groups based on a vertex path that forms
    a cutting seam across the mesh surface.

    Strategy:
    1. Identify all edges along the path.
    2. Identify faces that are adjacent to the path (boundary faces).
    3. For each boundary face, determine which side of the path it's on
       using the face's position relative to the path direction.
    4. Flood-fill from each side to assign all remaining faces.

    Args:
        mesh: The trimesh object.
        vertex_path: Ordered list of vertex indices forming the cutting path.

    Returns:
        (set_a, set_b): Two lists of face indices forming the partition.
    """
    if len(vertex_path) < 2:
        raise ValueError("Path must have at least 2 vertices")

    n_faces = len(mesh.faces)
    faces = numpy.asarray(mesh.faces, dtype=numpy.int64)

    # 1. Build set of path edges (as frozensets for undirected lookup)
    oriented_path_edges = []
    path_edges = set()
    for i in range(len(vertex_path) - 1):
        u = int(vertex_path[i])
        v = int(vertex_path[i + 1])
        if u == v:
            continue
        oriented_path_edges.append((u, v))
        path_edges.add(frozenset((u, v)))
    oriented_edge_lookup = set(oriented_path_edges)
    path_vertex_set = set(vertex_path)
    is_closed_loop = (
        len(vertex_path) >= 4 and int(vertex_path[0]) == int(vertex_path[-1])
    )

    logger.info("Path has %d edges, %d unique vertices",
                len(path_edges), len(path_vertex_set))

    # 2. Build face adjacency (shared-edge) and identify faces touching path
    #    For each edge of each face, check if it's a path edge
    face_adj = [[] for _ in range(n_faces)]
    face_path_edges = {}  # face_idx → list of path edges it contains

    # Build edge-to-face map
    edge_to_faces = {}
    for fi in range(n_faces):
        f = faces[fi]
        for j in range(3):
            edge = frozenset((int(f[j]), int(f[(j + 1) % 3])))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    # Build adjacency and identify boundary faces
    for edge, face_list in edge_to_faces.items():
        is_path_edge = edge in path_edges
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                fi, fj = face_list[i], face_list[j]
                if is_path_edge:
                    # These faces are on opposite sides of the path — don't connect
                    if fi not in face_path_edges:
                        face_path_edges[fi] = []
                    face_path_edges[fi].append(edge)
                    if fj not in face_path_edges:
                        face_path_edges[fj] = []
                    face_path_edges[fj].append(edge)
                else:
                    # Normal adjacency
                    face_adj[fi].append(fj)
                    face_adj[fj].append(fi)

    # 3. Assign initial sides for boundary faces.
    #    Closed loops get edge-local opposite-side seeds (more stable for
    #    curved/non-linear loops). Open paths keep the global tangent fallback.
    face_side = numpy.full(n_faces, -1, dtype=numpy.int32)  # -1 = unassigned
    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    centroids = mesh.triangles_center
    face_normals = numpy.asarray(mesh.face_normals, dtype=numpy.float64)

    if is_closed_loop:
        # For each path edge, the two incident faces should belong to opposite
        # sides. Use a local edge frame to seed those labels.
        seed_votes = defaultdict(list)
        for edge in path_edges:
            incident = edge_to_faces.get(edge, [])
            if len(incident) < 2:
                continue

            fi = int(incident[0])
            fj = int(incident[1])
            u, v = tuple(edge)
            u = int(u)
            v = int(v)
            if (v, u) in oriented_edge_lookup and (u, v) not in oriented_edge_lookup:
                u, v = v, u

            p0 = vertices[u]
            p1 = vertices[v]
            tangent = p1 - p0
            tangent_len = numpy.linalg.norm(tangent)
            if tangent_len <= 1e-10:
                continue
            tangent /= tangent_len

            n_ref = face_normals[fi] + face_normals[fj]
            n_ref_len = numpy.linalg.norm(n_ref)
            if n_ref_len <= 1e-10:
                n_ref = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
                n_ref_len = 1.0
            n_ref /= n_ref_len

            side_dir = numpy.cross(tangent, n_ref)
            side_len = numpy.linalg.norm(side_dir)
            if side_len <= 1e-10:
                side_dir = numpy.cross(tangent, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64))
                side_len = numpy.linalg.norm(side_dir)
                if side_len <= 1e-10:
                    continue
            side_dir /= side_len

            edge_mid = 0.5 * (p0 + p1)
            si = float(numpy.dot(centroids[fi] - edge_mid, side_dir))
            sj = float(numpy.dot(centroids[fj] - edge_mid, side_dir))
            if si >= sj:
                seed_votes[fi].append(0)
                seed_votes[fj].append(1)
            else:
                seed_votes[fi].append(1)
                seed_votes[fj].append(0)

        for fi, votes in seed_votes.items():
            count_0 = votes.count(0)
            count_1 = votes.count(1)
            face_side[int(fi)] = 0 if count_0 >= count_1 else 1

    needs_global_seed = (
        (not is_closed_loop) or
        (not numpy.any(face_side == 0)) or
        (not numpy.any(face_side == 1))
    )

    if needs_global_seed:
        # Compute path direction as a polyline fallback.
        path_verts = vertices[numpy.asarray(vertex_path, dtype=numpy.int64)]
        if len(path_verts) >= 2:
            path_tangent = path_verts[-1] - path_verts[0]
            tangent_len = numpy.linalg.norm(path_tangent)
            if tangent_len > 1e-10:
                path_tangent /= tangent_len
            else:
                path_tangent = numpy.array([1.0, 0.0, 0.0])
        else:
            path_tangent = numpy.array([1.0, 0.0, 0.0])

        path_center = path_verts.mean(axis=0)
        for fi in face_path_edges:
            face_center = centroids[fi]
            to_face = face_center - path_center
            cross = numpy.cross(path_tangent, to_face)
            side_val = numpy.dot(cross, face_normals[fi])
            face_side[fi] = 0 if side_val >= 0 else 1
    else:
        # Closed-loop path had enough local seeds; assign remaining boundary
        # faces with global fallback to complete the boundary front.
        path_verts = vertices[numpy.asarray(vertex_path, dtype=numpy.int64)]
        path_center = path_verts.mean(axis=0)
        path_tangent = path_verts[-1] - path_verts[0]
        tangent_len = numpy.linalg.norm(path_tangent)
        if tangent_len > 1e-10:
            path_tangent /= tangent_len
        else:
            path_tangent = numpy.array([1.0, 0.0, 0.0])

        for fi in face_path_edges:
            if face_side[fi] >= 0:
                continue
            face_center = centroids[fi]
            to_face = face_center - path_center
            cross = numpy.cross(path_tangent, to_face)
            side_val = numpy.dot(cross, face_normals[fi])
            face_side[fi] = 0 if side_val >= 0 else 1

    # 4. Flood-fill from assigned boundary faces
    for fi in range(n_faces):
        if face_side[fi] >= 0:
            queue = deque([fi])
            side = face_side[fi]
            while queue:
                cur = queue.popleft()
                for nb in face_adj[cur]:
                    if face_side[nb] == -1:
                        face_side[nb] = side
                        queue.append(nb)

    # 5. Assign any remaining unassigned faces to the larger partition
    #    (these are faces in disconnected components)
    count_0 = int(numpy.sum(face_side == 0))
    count_1 = int(numpy.sum(face_side == 1))
    default_side = 0 if count_0 >= count_1 else 1
    face_side[face_side == -1] = default_side

    set_a = numpy.where(face_side == 0)[0].tolist()
    set_b = numpy.where(face_side == 1)[0].tolist()

    # Ensure set_a is the smaller partition
    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    logger.info("Path partition: %d / %d faces", len(set_a), len(set_b))
    return set_a, set_b
