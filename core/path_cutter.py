# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Path-based mesh cutting using geodesic shortest paths.

The user places 2+ waypoints on the mesh surface. For each consecutive pair,
we compute the shortest path along mesh edges (vertex-based Dijkstra).
The full path defines a cutting seam, and faces are flood-filled on each side
to create the partition.
"""

import logging
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial import cKDTree

try:
    import trimesh
except ImportError:
    trimesh = None

logger = logging.getLogger("objectsplitter.path_cutter")


# ---------------------------------------------------------------------------
# Mesh graph cache
# ---------------------------------------------------------------------------

def _concavity_weight(
    mesh: "trimesh.Trimesh",
    edges_unique: numpy.ndarray,
    valley_bias: float,
) -> numpy.ndarray:
    """Per-edge cost multiplier in [~0.05, 1.0]; concave edges get a discount.

    Concave (valley) edges -- where the two adjacent faces fold inward -- are
    made cheaper so a shortest path prefers them, hugging grooves. Convex and
    boundary edges keep their full length (multiplier 1.0).
    """
    n = len(edges_unique)
    weight = numpy.ones(n, dtype=numpy.float64)
    try:
        convex = numpy.asarray(mesh.face_adjacency_convex, dtype=bool)
        concave = ~convex
        if not concave.any():
            return weight
        angles = numpy.asarray(mesh.face_adjacency_angles, dtype=numpy.float64)
        fa_edges = numpy.sort(numpy.asarray(mesh.face_adjacency_edges), axis=1)
        eu = numpy.sort(numpy.asarray(edges_unique), axis=1)

        # Encode each sorted (v0, v1) edge as a single integer key, then match
        # concave adjacency edges to unique-edge rows with searchsorted -- fully
        # vectorized, no per-edge Python loop or dict.
        stride = numpy.int64(int(eu.max()) + 1) if n else numpy.int64(1)
        eu_key = eu[:, 0].astype(numpy.int64) * stride + eu[:, 1]
        order = numpy.argsort(eu_key)
        eu_key_sorted = eu_key[order]

        fk = fa_edges[concave, 0].astype(numpy.int64) * stride + fa_edges[concave, 1]
        strength = numpy.clip(angles[concave] / (numpy.pi / 2.0), 0.0, 1.0)
        discount = numpy.maximum(0.05, 1.0 - valley_bias * strength)

        pos = numpy.searchsorted(eu_key_sorted, fk)
        pos = numpy.clip(pos, 0, len(eu_key_sorted) - 1)
        valid = eu_key_sorted[pos] == fk
        idx = order[pos[valid]]
        # Several concave adjacencies can map to one edge -> keep the strongest.
        numpy.minimum.at(weight, idx, discount[valid])
    except Exception as e:  # noqa: BLE001 - weighting is best-effort
        logger.debug("concavity weighting unavailable: %s", e)
    return weight


class _MeshGraph:
    """Precomputed mesh graph state shared across a chain_paths call.

    Building the sparse CSR adjacency, KD-tree, and component labels once is
    the difference between a fast scipy path and a slow per-segment rebuild
    in pure Python.
    """

    __slots__ = ("vertices", "csr", "kdtree", "component_labels", "n_components")

    def __init__(self, mesh: "trimesh.Trimesh", valley_bias: float = 0.0):
        vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
        edges = numpy.asarray(mesh.edges_unique, dtype=numpy.int64)
        n_verts = len(vertices)

        edge_lengths = numpy.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
        if valley_bias > 0.0:
            # Lower the cost of concave (valley) edges so the shortest path
            # prefers to run along grooves while still passing through the
            # waypoints. Convex/ridge edges keep their full length.
            edge_lengths = edge_lengths * _concavity_weight(mesh, edges, valley_bias)
        rows = numpy.concatenate([edges[:, 0], edges[:, 1]])
        cols = numpy.concatenate([edges[:, 1], edges[:, 0]])
        data = numpy.concatenate([edge_lengths, edge_lengths])
        csr = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_verts, n_verts)
        )

        n_components, labels = scipy.sparse.csgraph.connected_components(
            csr, directed=False, return_labels=True
        )

        self.vertices = vertices
        self.csr = csr
        self.kdtree = cKDTree(vertices)
        self.component_labels = labels
        self.n_components = n_components


def _dijkstra_path(graph: _MeshGraph, start: int, end: int) -> List[int]:
    """Single-source Dijkstra (scipy/C) with predecessor reconstruction."""
    dist, prev = scipy.sparse.csgraph.dijkstra(
        graph.csr,
        directed=False,
        indices=int(start),
        return_predecessors=True,
    )
    if numpy.isinf(dist[end]):
        raise ValueError(f"No path between vertex {start} and {end}")
    path: List[int] = []
    v = int(end)
    while v >= 0:
        path.append(v)
        if v == int(start):
            break
        v = int(prev[v])
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Vertex-based Dijkstra shortest path
# ---------------------------------------------------------------------------

def find_geodesic_path(
    mesh: "trimesh.Trimesh",
    start_vertex: int,
    end_vertex: int,
) -> List[int]:
    """
    Find the shortest geodesic path between two vertices using scipy's
    sparse-graph Dijkstra on the mesh edge graph.

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
    graph = _MeshGraph(mesh)
    return _dijkstra_path(graph, int(start_vertex), int(end_vertex))


def snap_to_nearest_vertex(
    mesh: "trimesh.Trimesh",
    point: numpy.ndarray,
) -> int:
    """Find the mesh vertex closest to a 3D point."""
    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    tree = cKDTree(vertices)
    _, idx = tree.query(numpy.asarray(point, dtype=numpy.float64))
    return int(idx)


# ---------------------------------------------------------------------------
# Chain multiple waypoints into a single path
# ---------------------------------------------------------------------------

def chain_paths(
    mesh: "trimesh.Trimesh",
    waypoints: List[numpy.ndarray],
    valley_bias: float = 0.0,
) -> List[int]:
    """
    Given 2+ waypoints (3D positions on mesh surface), snap each to the
    nearest vertex and compute geodesic paths between consecutive pairs.

    valley_bias > 0 discounts concave (valley) edges so the path hugs grooves
    while still passing through the waypoints (0 = plain geodesic).

    Handles disconnected components: if a waypoint snaps to a vertex on a
    different component from the first waypoint, it is re-snapped to the
    nearest vertex on the same component.

    Returns:
        Ordered list of vertex indices forming the full path.
        Duplicate vertices at junctions are removed.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    graph = _MeshGraph(mesh, valley_bias=valley_bias)
    points = numpy.asarray(waypoints, dtype=numpy.float64).reshape(-1, 3)

    # Bulk-snap waypoints to nearest vertices via KD-tree.
    _, snapped = graph.kdtree.query(points)
    vert_indices: List[int] = snapped.tolist()

    target_comp = int(graph.component_labels[vert_indices[0]])
    target_mask = graph.component_labels == target_comp
    target_verts_idx = numpy.where(target_mask)[0]

    logger.info(
        "Mesh has %d connected components; target component %d has %d vertices",
        graph.n_components, target_comp, len(target_verts_idx),
    )

    # Re-snap any waypoints that landed on a different component.
    if len(target_verts_idx) < len(graph.component_labels):
        target_tree = cKDTree(graph.vertices[target_verts_idx])
        for i in range(len(vert_indices)):
            if int(graph.component_labels[vert_indices[i]]) != target_comp:
                _, local_idx = target_tree.query(points[i])
                old_v = vert_indices[i]
                vert_indices[i] = int(target_verts_idx[int(local_idx)])
                logger.info(
                    "Waypoint %d re-snapped: vertex %d (comp %d) -> %d (comp %d)",
                    i,
                    old_v,
                    int(graph.component_labels[old_v]),
                    vert_indices[i],
                    target_comp,
                )

    logger.info(
        "Path waypoints: %d points -> vertex indices %s",
        len(waypoints),
        vert_indices,
    )

    # Build chained path using the cached graph.
    full_path: List[int] = []
    for i in range(len(vert_indices) - 1):
        segment = _dijkstra_path(graph, vert_indices[i], vert_indices[i + 1])
        if full_path and segment:
            full_path.extend(segment[1:])
        else:
            full_path.extend(segment)

    logger.info("Full path: %d vertices", len(full_path))
    return full_path


# ---------------------------------------------------------------------------
# Helpers shared by partition and isolate
# ---------------------------------------------------------------------------

def _encode_undirected_edges(edges: numpy.ndarray, key_base: int) -> numpy.ndarray:
    """Pack undirected (N,2) vertex-index edges into unique int64 keys."""
    e = numpy.asarray(edges, dtype=numpy.int64)
    a = numpy.minimum(e[:, 0], e[:, 1])
    b = numpy.maximum(e[:, 0], e[:, 1])
    return a * int(key_base) + b


def _build_face_adj_csr(
    mesh: "trimesh.Trimesh",
    seam_edge_keys: Optional[numpy.ndarray],
) -> Tuple[scipy.sparse.csr_matrix, numpy.ndarray, numpy.ndarray, int]:
    """Build a face-adjacency CSR using trimesh's face_adjacency, with the
    given seam edges removed.

    Returns:
        (csr, seam_face_pairs, seam_edge_pairs, key_base)
        - csr: sparse face-adjacency, n_faces x n_faces, unweighted (data=1)
        - seam_face_pairs: (S, 2) pairs of face indices straddling each seam edge
        - seam_edge_pairs: (S, 2) corresponding vertex-pair edges
        - key_base: encoding multiplier used so callers can build matching keys
    """
    fa = numpy.asarray(mesh.face_adjacency, dtype=numpy.int64)
    fae = numpy.asarray(mesh.face_adjacency_edges, dtype=numpy.int64)
    n_faces = len(mesh.faces)
    n_verts = len(mesh.vertices)
    key_base = max(n_verts + 1, 2)

    if seam_edge_keys is not None and len(seam_edge_keys) > 0:
        encoded_fae = _encode_undirected_edges(fae, key_base)
        seam_mask = numpy.isin(encoded_fae, seam_edge_keys)
    else:
        seam_mask = numpy.zeros(len(fa), dtype=bool)

    non_seam = fa[~seam_mask]
    if len(non_seam):
        rows = numpy.concatenate([non_seam[:, 0], non_seam[:, 1]])
        cols = numpy.concatenate([non_seam[:, 1], non_seam[:, 0]])
        data = numpy.ones(len(rows), dtype=numpy.int8)
    else:
        rows = numpy.empty(0, dtype=numpy.int64)
        cols = numpy.empty(0, dtype=numpy.int64)
        data = numpy.empty(0, dtype=numpy.int8)
    csr = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_faces, n_faces),
    )
    return csr, fa[seam_mask], fae[seam_mask], key_base


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
    2. Build a face-adjacency graph with those edges removed (scipy sparse).
    3. For each boundary face, determine which side of the path it's on
       using a local edge frame.
    4. Multi-source Dijkstra on the face graph (C-backed via scipy) flood-
       fills the remaining faces by nearest seeded side.
    """
    if len(vertex_path) < 2:
        raise ValueError("Path must have at least 2 vertices")

    n_faces = len(mesh.faces)

    # 1. Collect path edges.
    oriented_path_edges = []
    seam_edge_pairs_list = []
    for i in range(len(vertex_path) - 1):
        u = int(vertex_path[i])
        v = int(vertex_path[i + 1])
        if u == v:
            continue
        oriented_path_edges.append((u, v))
        seam_edge_pairs_list.append((min(u, v), max(u, v)))
    seam_edge_pairs_list = list(set(seam_edge_pairs_list))  # de-dup
    oriented_edge_lookup = set(oriented_path_edges)
    path_vertex_set = set(int(v) for v in vertex_path)
    is_closed_loop = (
        len(vertex_path) >= 4 and int(vertex_path[0]) == int(vertex_path[-1])
    )

    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    centroids = numpy.asarray(mesh.triangles_center, dtype=numpy.float64)
    face_normals = numpy.asarray(mesh.face_normals, dtype=numpy.float64)
    n_verts = len(vertices)
    key_base = max(n_verts + 1, 2)

    if seam_edge_pairs_list:
        seam_edge_pairs = numpy.asarray(seam_edge_pairs_list, dtype=numpy.int64)
        seam_edge_keys = _encode_undirected_edges(seam_edge_pairs, key_base)
    else:
        seam_edge_pairs = numpy.empty((0, 2), dtype=numpy.int64)
        seam_edge_keys = numpy.empty(0, dtype=numpy.int64)

    logger.info(
        "Path has %d unique edges, %d unique vertices",
        len(seam_edge_pairs),
        len(path_vertex_set),
    )

    # 2. Face adjacency CSR (scipy) minus seam edges.
    face_adj_csr, seam_face_pairs, seam_edge_pairs_kept, _ = _build_face_adj_csr(
        mesh, seam_edge_keys
    )
    seam_faces_array = numpy.unique(seam_face_pairs.ravel()) if len(seam_face_pairs) else numpy.empty(0, dtype=numpy.int64)
    seam_faces_set = set(int(f) for f in seam_faces_array)

    # Closed loops handled topologically via connected components.
    if is_closed_loop and len(seam_faces_array):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            face_adj_csr, directed=False, return_labels=True
        )
        seam_component_ids = sorted(set(int(labels[f]) for f in seam_faces_array))

        if len(seam_component_ids) >= 2:
            component_sizes = {
                cid: int(numpy.sum(labels == cid))
                for cid in seam_component_ids
            }
            chosen_component_id = min(
                seam_component_ids, key=lambda cid: component_sizes[cid]
            )
            set_a = numpy.where(labels == chosen_component_id)[0].tolist()
            set_b = numpy.where(labels != chosen_component_id)[0].tolist()
            if len(set_a) > len(set_b):
                set_a, set_b = set_b, set_a
            logger.info(
                "Path partition: using closed-loop topological components (seam_faces=%d, seam_components=%s) -> %d / %d faces",
                len(seam_faces_array),
                [component_sizes[cid] for cid in seam_component_ids],
                len(set_a),
                len(set_b),
            )
            return set_a, set_b

    # 3. Seed boundary faces using the local edge frame for each seam edge.
    face_side = numpy.full(n_faces, -1, dtype=numpy.int32)
    seed_votes = defaultdict(list)

    for row_index in range(len(seam_face_pairs)):
        fi = int(seam_face_pairs[row_index, 0])
        fj = int(seam_face_pairs[row_index, 1])
        u = int(seam_edge_pairs_kept[row_index, 0])
        v = int(seam_edge_pairs_kept[row_index, 1])
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

    path_verts = vertices[numpy.asarray(vertex_path, dtype=numpy.int64)]
    path_segments = []
    for i in range(len(path_verts) - 1):
        seg_start = numpy.asarray(path_verts[i], dtype=numpy.float64)
        seg_end = numpy.asarray(path_verts[i + 1], dtype=numpy.float64)
        seg_vec = seg_end - seg_start
        seg_len = numpy.linalg.norm(seg_vec)
        if seg_len <= 1e-10:
            continue
        path_segments.append((seg_start, seg_end, seg_vec / seg_len))

    if len(path_verts) >= 2:
        path_tangent = path_verts[-1] - path_verts[0]
        tangent_len = numpy.linalg.norm(path_tangent)
        if tangent_len > 1e-10:
            path_tangent /= tangent_len
        else:
            path_tangent = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
    else:
        path_tangent = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)
    path_center = path_verts.mean(axis=0)

    def _global_side_hint(face_index: int) -> int:
        face_center = centroids[face_index]
        to_face = face_center - path_center
        cross = numpy.cross(path_tangent, to_face)
        side_val = float(numpy.dot(cross, face_normals[face_index]))
        return 0 if side_val >= 0 else 1

    def _local_side_hint(face_index: int) -> int:
        if not path_segments:
            return _global_side_hint(face_index)

        face_center = centroids[face_index]
        normal = face_normals[face_index]
        best_dist = None
        best_side_val = None

        for seg_start, seg_end, tangent in path_segments:
            seg_vec = seg_end - seg_start
            seg_len_sq = float(numpy.dot(seg_vec, seg_vec))
            if seg_len_sq <= 1e-12:
                continue
            t = float(numpy.dot(face_center - seg_start, seg_vec) / seg_len_sq)
            t = min(1.0, max(0.0, t))
            closest = seg_start + seg_vec * t
            offset = face_center - closest
            distance = float(numpy.linalg.norm(offset))

            side_dir = numpy.cross(tangent, normal)
            side_len = numpy.linalg.norm(side_dir)
            if side_len <= 1e-10:
                continue
            side_val = float(numpy.dot(offset, side_dir / side_len))
            if best_dist is None or distance < best_dist:
                best_dist = distance
                best_side_val = side_val

        if best_side_val is None:
            return _global_side_hint(face_index)
        return 0 if best_side_val >= 0 else 1

    needs_global_seed = (
        (not numpy.any(face_side == 0)) or
        (not numpy.any(face_side == 1))
    )

    if needs_global_seed or is_closed_loop:
        for fi in seam_faces_set:
            if face_side[fi] >= 0 and not needs_global_seed:
                continue
            face_side[fi] = _local_side_hint(fi)

    seed_0 = numpy.where(face_side == 0)[0]
    seed_1 = numpy.where(face_side == 1)[0]
    if seed_0.size == 0 or seed_1.size == 0:
        logger.warning(
            "Path partition: local seam seeding was one-sided (side0=%d, side1=%d); "
            "falling back to local path-side hints",
            seed_0.size,
            seed_1.size,
        )
        for fi in range(n_faces):
            face_side[fi] = _local_side_hint(fi)
        seed_0 = numpy.where(face_side == 0)[0]
        seed_1 = numpy.where(face_side == 1)[0]
        if seed_0.size == 0 or seed_1.size == 0:
            raise ValueError("Path partition could not determine both sides of the seam")

    # 4. Multi-source BFS via scipy on the face adjacency CSR. With
    #    min_only=True and unweighted=True we get the BFS distance from each
    #    face to its nearest seed in one C call per side.
    dist_0 = scipy.sparse.csgraph.dijkstra(
        face_adj_csr,
        directed=False,
        indices=seed_0.astype(numpy.int64),
        unweighted=True,
        min_only=True,
    )
    dist_1 = scipy.sparse.csgraph.dijkstra(
        face_adj_csr,
        directed=False,
        indices=seed_1.astype(numpy.int64),
        unweighted=True,
        min_only=True,
    )

    unassigned = face_side == -1
    d0u = dist_0[unassigned]
    d1u = dist_1[unassigned]
    both_inf = numpy.isinf(d0u) & numpy.isinf(d1u)
    d0u_finite = numpy.where(numpy.isinf(d0u), numpy.inf, d0u)
    d1u_finite = numpy.where(numpy.isinf(d1u), numpy.inf, d1u)
    choose_0 = d0u_finite < d1u_finite
    ties = (d0u_finite == d1u_finite) & ~both_inf
    new_side = numpy.where(choose_0, 0, 1).astype(numpy.int32)
    new_side[both_inf] = -1

    unassigned_indices = numpy.where(unassigned)[0]
    face_side[unassigned_indices] = new_side
    tie_count = int(numpy.count_nonzero(ties))
    if tie_count:
        # Rare. Fall back to local hint only for tied faces.
        tied_face_indices = unassigned_indices[ties]
        for fi in tied_face_indices:
            face_side[int(fi)] = _local_side_hint(int(fi))

    # 5. Assign any remaining unassigned faces (disconnected components with
    #    no seeded boundary contact) to the currently larger partition.
    count_0 = int(numpy.sum(face_side == 0))
    count_1 = int(numpy.sum(face_side == 1))
    default_side = 0 if count_0 >= count_1 else 1
    face_side[face_side == -1] = default_side

    logger.info(
        "Path partition seeds: side0=%d side1=%d local_seed_faces=%d ties=%d closed=%s",
        seed_0.size,
        seed_1.size,
        len(seed_votes),
        tie_count,
        str(is_closed_loop),
    )

    set_a = numpy.where(face_side == 0)[0].tolist()
    set_b = numpy.where(face_side == 1)[0].tolist()

    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    logger.info("Path partition: %d / %d faces", len(set_a), len(set_b))
    return set_a, set_b


def isolate_region_by_loops(
    mesh: "trimesh.Trimesh",
    loop_vertex_paths: List[List[int]],
    target_face_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Isolate the connected face region containing target_face_id after removing
    all seam edges from one or more closed geodesic loops.

    Returns:
        (extracted_faces, remainder_faces)
    """
    if not loop_vertex_paths:
        raise ValueError("Need at least one closed loop to isolate a region")

    n_faces = len(mesh.faces)
    if n_faces == 0:
        raise ValueError("Mesh has no faces")
    if target_face_id is None or int(target_face_id) < 0 or int(target_face_id) >= n_faces:
        raise ValueError("Target face is invalid")

    seam_edge_pairs_set: Set[Tuple[int, int]] = set()
    for loop_index, vertex_path in enumerate(loop_vertex_paths):
        if len(vertex_path) < 4:
            raise ValueError(f"Loop {loop_index + 1} needs at least 3 points")
        if int(vertex_path[0]) != int(vertex_path[-1]):
            raise ValueError(f"Loop {loop_index + 1} is not closed")
        if len({int(v) for v in vertex_path[:-1]}) < 3:
            raise ValueError(f"Loop {loop_index + 1} does not contain 3 unique vertices")

        loop_edges = 0
        for start, end in zip(vertex_path[:-1], vertex_path[1:]):
            u = int(start)
            v = int(end)
            if u == v:
                continue
            seam_edge_pairs_set.add((min(u, v), max(u, v)))
            loop_edges += 1
        if loop_edges < 3:
            raise ValueError(f"Loop {loop_index + 1} did not resolve to a usable seam")

    if not seam_edge_pairs_set:
        raise ValueError("Loop seams were empty")

    n_verts = len(mesh.vertices)
    key_base = max(n_verts + 1, 2)
    seam_edge_pairs = numpy.asarray(sorted(seam_edge_pairs_set), dtype=numpy.int64)
    seam_edge_keys = _encode_undirected_edges(seam_edge_pairs, key_base)

    face_adj_csr, seam_face_pairs, _, _ = _build_face_adj_csr(mesh, seam_edge_keys)
    seam_faces_array = (
        numpy.unique(seam_face_pairs.ravel())
        if len(seam_face_pairs)
        else numpy.empty(0, dtype=numpy.int64)
    )
    if int(target_face_id) in {int(f) for f in seam_faces_array}:
        raise ValueError("Target face lies on a seam; pick a face inside the desired region")

    n_components, labels = scipy.sparse.csgraph.connected_components(
        face_adj_csr, directed=False, return_labels=True
    )
    if n_components < 2:
        raise ValueError("Closed loop set does not separate the mesh into multiple regions")

    target_component = int(labels[int(target_face_id)])
    if target_component < 0 or target_component >= n_components:
        raise ValueError("Could not determine the target component")

    extracted_mask = labels == target_component
    extracted_faces = numpy.where(extracted_mask)[0].tolist()
    remainder_faces = numpy.where(~extracted_mask)[0].tolist()

    if not extracted_faces or not remainder_faces:
        raise ValueError("Loop isolation produced a degenerate partition")

    logger.info(
        "Path isolate: loops=%d seam_edges=%d seam_faces=%d components=%d extracted=%d remainder=%d",
        len(loop_vertex_paths),
        len(seam_edge_pairs),
        len(seam_faces_array),
        n_components,
        len(extracted_faces),
        len(remainder_faces),
    )

    return extracted_faces, remainder_faces
