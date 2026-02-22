# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Geometry utilities: rotation matrices, coordinate transforms, plane math.

No Cura dependencies - uses only numpy.
"""

import numpy
import logging
from typing import Optional, Tuple

logger = logging.getLogger("objectsplitter.geometry")


def rotation_matrix_from_vectors(vec1: numpy.ndarray, vec2: numpy.ndarray) -> numpy.ndarray:
    """
    Create a rotation matrix that rotates vec1 to vec2
    using Rodrigues' rotation formula.

    Args:
        vec1: Source direction (3D unit vector or will be normalized).
        vec2: Target direction (3D unit vector or will be normalized).

    Returns:
        3x3 rotation matrix.
    """
    v1 = vec1 / numpy.linalg.norm(vec1)
    v2 = vec2 / numpy.linalg.norm(vec2)

    cross = numpy.cross(v1, v2)
    dot = numpy.dot(v1, v2)

    if numpy.linalg.norm(cross) < 1e-6:
        if dot > 0:
            return numpy.eye(3)
        else:
            # Opposite direction: 180-degree rotation around a perpendicular axis
            if abs(v1[0]) < 0.9:
                perp = numpy.array([1, 0, 0])
            else:
                perp = numpy.array([0, 1, 0])
            perp = perp - numpy.dot(perp, v1) * v1
            perp = perp / numpy.linalg.norm(perp)
            return 2 * numpy.outer(perp, perp) - numpy.eye(3)

    cross_normalized = cross / numpy.linalg.norm(cross)
    angle = numpy.arccos(numpy.clip(dot, -1, 1))

    K = numpy.array([
        [0, -cross_normalized[2], cross_normalized[1]],
        [cross_normalized[2], 0, -cross_normalized[0]],
        [-cross_normalized[1], cross_normalized[0], 0]
    ])

    R = numpy.eye(3) + numpy.sin(angle) * K + (1 - numpy.cos(angle)) * (K @ K)
    return R


def world_to_local_transform(
    local_vertices: numpy.ndarray,
    world_vertices: numpy.ndarray
) -> Optional[Tuple[numpy.ndarray, numpy.ndarray, float]]:
    """
    Estimate the world-to-local coordinate transformation from matching
    vertex pairs (local space vs world space).

    Picks three non-collinear points, estimates uniform scale + rotation + translation,
    and returns (R, T, scale) such that:
        world_point = scale * R @ local_point + T
        local_point = R.T @ (world_point - T) / scale

    Args:
        local_vertices: Vertex positions in local/object space (N x 3).
        world_vertices: Corresponding vertex positions in world space (N x 3).

    Returns:
        (R, T, scale) tuple, or None if the transform cannot be estimated.
    """
    if local_vertices is None or len(local_vertices) < 3:
        return None
    if world_vertices is None or len(world_vertices) < 3:
        return None

    try:
        v0L = numpy.array(local_vertices[0])
        v0W = numpy.array(world_vertices[0])

        # Find v1, v2 not collinear with v0
        d1L = numpy.array(local_vertices[1]) - v0L
        v1_index, v2_index = 1, 2
        for i in range(2, len(local_vertices)):
            d2L = numpy.array(local_vertices[i]) - v0L
            if numpy.linalg.norm(numpy.cross(d1L, d2L)) > 1e-6:
                v2_index = i
                break

        v1L = numpy.array(local_vertices[v1_index])
        v2L = numpy.array(local_vertices[v2_index])
        v1W = numpy.array(world_vertices[v1_index])
        v2W = numpy.array(world_vertices[v2_index])

        d1L = v1L - v0L
        d2L = v2L - v0L
        d1W = v1W - v0W
        d2W = v2W - v0W

        scale_d1 = numpy.linalg.norm(d1W) / max(1e-9, numpy.linalg.norm(d1L))
        scale_d2 = numpy.linalg.norm(d2W) / max(1e-9, numpy.linalg.norm(d2L))
        if numpy.isfinite(scale_d1) and numpy.isfinite(scale_d2):
            scale = (scale_d1 + scale_d2) / 2.0
        else:
            scale = scale_d1

        basis_L = numpy.column_stack((d1L, d2L, numpy.cross(d1L, d2L)))
        basis_W = numpy.column_stack((d1W, d2W, numpy.cross(d1W, d2W)))

        if numpy.linalg.matrix_rank(basis_L) < 3:
            return None

        R_approx = basis_W @ numpy.linalg.inv(basis_L) / (scale if scale != 0 else 1.0)
        U, _, Vt = numpy.linalg.svd(R_approx)
        R = U @ Vt
        if numpy.linalg.det(R) < 0:
            R[:, 2] *= -1

        T = v0W - scale * (R @ v0L)
        return R, T, scale

    except Exception as e:
        logger.debug("world_to_local_transform failed: %s", e)
        return None


def transform_point_to_local(
    point: numpy.ndarray,
    R: numpy.ndarray,
    T: numpy.ndarray,
    scale: float
) -> numpy.ndarray:
    """
    Transform a world-space point to local/object space.

    Args:
        point: 3D point in world space.
        R: 3x3 rotation matrix (local-to-world).
        T: 3D translation vector (local-to-world).
        scale: Uniform scale factor.

    Returns:
        3D point in local space.
    """
    return R.T @ (point - T) / (scale if scale != 0 else 1.0)


def plane_normal_from_spherical(theta: float, phi: float) -> numpy.ndarray:
    """
    Convert spherical coordinates to a unit normal vector.

    Args:
        theta: Elevation angle in radians (0 = +Y pole, pi = -Y pole).
        phi: Azimuth angle in radians (0 to 2*pi).

    Returns:
        3D unit vector [x, y, z].
    """
    return numpy.array([
        numpy.sin(theta) * numpy.cos(phi),
        numpy.cos(theta),
        numpy.sin(theta) * numpy.sin(phi)
    ])


def create_plane_mesh_data(
    origin: numpy.ndarray,
    normal: numpy.ndarray,
    size: float
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Generate vertices and face indices for a flat preview plane.

    Args:
        origin: Center point of the plane (3D).
        normal: Normal vector of the plane (3D).
        size: Side length of the square plane.

    Returns:
        (vertices, indices) as numpy arrays.
        vertices: (4, 3) float32, indices: (4, 3) int32 for double-sided quad.
    """
    normal_arr = normal / numpy.linalg.norm(normal)

    if abs(normal_arr[1]) < 0.9:
        up = numpy.array([0, 1, 0])
    else:
        up = numpy.array([1, 0, 0])

    tangent1 = numpy.cross(normal_arr, up)
    tangent1 = tangent1 / numpy.linalg.norm(tangent1)
    tangent2 = numpy.cross(normal_arr, tangent1)

    half = size / 2.0
    t1 = tangent1 * half
    t2 = tangent2 * half

    center = numpy.asarray(origin, dtype=numpy.float64)
    vertices = numpy.array([
        center - t1 - t2,
        center + t1 - t2,
        center + t1 + t2,
        center - t1 + t2,
    ], dtype=numpy.float32)

    indices = numpy.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 2, 1],
        [0, 3, 2],
    ], dtype=numpy.int32)

    return vertices, indices


def create_marker_mesh_data(
    center: numpy.ndarray,
    size: float = 2.0,
    direction: Optional[numpy.ndarray] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Generate vertices and face indices for an arrow marker pointing at the
    click location along the surface normal direction.

    The arrow tip touches `center`. The shaft extends away from the surface
    in the `direction` the normal points (outward from the mesh).

    Args:
        center: 3D point where the arrow tip touches (the click location).
        size: Scale factor. Arrow head radius = size * 0.5.
        direction: Direction the arrow points FROM (i.e. surface normal).
                   Arrow tip is at center, body extends opposite to this.
                   Defaults to [0, 1, 0] (arrow points down from above).

    Returns:
        (vertices, indices) as numpy arrays.
    """
    s = float(size)
    c = numpy.asarray(center, dtype=numpy.float64)

    # Default direction: surface normal pointing outward (arrow comes from outside)
    if direction is not None:
        d = numpy.asarray(direction, dtype=numpy.float64)
        norm = numpy.linalg.norm(d)
        if norm > 1e-10:
            d = d / norm
        else:
            d = numpy.array([0.0, 1.0, 0.0])
    else:
        d = numpy.array([0.0, 1.0, 0.0])

    # Build arrow along +Y axis, then rotate to align with `d`
    # Arrow proportions (built along Y axis, tip at origin):
    head_radius = s * 0.5
    head_length = s * 1.5
    shaft_radius = s * 0.15
    shaft_length = s * 3.0

    n_sides = 8
    angles = numpy.linspace(0, 2 * numpy.pi, n_sides, endpoint=False)

    verts_local = []
    faces = []

    # --- Cone head (tip at origin, base at Y = head_length) ---
    verts_local.append(numpy.array([0.0, 0.0, 0.0]))  # tip
    for a in angles:
        verts_local.append(numpy.array([
            head_radius * numpy.cos(a), head_length, head_radius * numpy.sin(a)
        ]))
    cone_base_center_idx = len(verts_local)
    verts_local.append(numpy.array([0.0, head_length, 0.0]))

    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([0, 1 + i, 1 + j])
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([cone_base_center_idx, 1 + j, 1 + i])

    # --- Shaft (thin prism from head_length to head_length + shaft_length) ---
    shaft_base_idx = len(verts_local)
    for a in angles:
        verts_local.append(numpy.array([
            shaft_radius * numpy.cos(a), head_length, shaft_radius * numpy.sin(a)
        ]))
    for a in angles:
        verts_local.append(numpy.array([
            shaft_radius * numpy.cos(a), head_length + shaft_length,
            shaft_radius * numpy.sin(a)
        ]))
    shaft_top_center_idx = len(verts_local)
    verts_local.append(numpy.array([0.0, head_length + shaft_length, 0.0]))

    sb = shaft_base_idx
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([sb + i, sb + j, sb + n_sides + j])
        faces.append([sb + i, sb + n_sides + j, sb + n_sides + i])
    # Shaft top cap
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([shaft_top_center_idx, sb + n_sides + i, sb + n_sides + j])

    # --- Rotate from +Y to the surface normal direction ---
    # The arrow is built along +Y; we need to rotate so +Y aligns with `d`
    R = rotation_matrix_from_vectors(numpy.array([0.0, 1.0, 0.0]), d)

    vertices = numpy.array([
        c + R @ v for v in verts_local
    ], dtype=numpy.float32)
    indices = numpy.array(faces, dtype=numpy.int32)
    return vertices, indices


def create_pin_mesh_data(
    center: numpy.ndarray,
    size: float = 1.0,
    direction: Optional[numpy.ndarray] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Generate vertices and face indices for a pin/dot marker (diamond shape)
    placed at a surface point. The pin has a small diamond head sitting on the
    surface and a thin spike poking into the surface for anchoring.

    Args:
        center: 3D point on the surface.
        size: Scale factor. Diamond radius = size * 0.4.
        direction: Surface normal direction. Pin head extends outward.
                   Defaults to [0, 1, 0].

    Returns:
        (vertices, indices) as numpy arrays.
    """
    s = float(size)
    c = numpy.asarray(center, dtype=numpy.float64)

    if direction is not None:
        d = numpy.asarray(direction, dtype=numpy.float64)
        norm = numpy.linalg.norm(d)
        if norm > 1e-10:
            d = d / norm
        else:
            d = numpy.array([0.0, 1.0, 0.0])
    else:
        d = numpy.array([0.0, 1.0, 0.0])

    # Diamond proportions (built along +Y axis, base at origin)
    diamond_radius = s * 0.4
    diamond_height_up = s * 0.6    # height of upper half above base
    diamond_height_down = s * 0.25  # height of lower half (into surface)
    spike_length = s * 0.3          # thin spike going into surface

    n_sides = 6  # hexagonal cross-section
    angles = numpy.linspace(0, 2 * numpy.pi, n_sides, endpoint=False)

    verts_local = []
    faces = []

    # Base ring (equator of diamond, at Y=0 = surface)
    for a in angles:
        verts_local.append(numpy.array([
            diamond_radius * numpy.cos(a), 0.0, diamond_radius * numpy.sin(a)
        ]))

    # Top point (above surface)
    top_idx = len(verts_local)
    verts_local.append(numpy.array([0.0, diamond_height_up, 0.0]))

    # Bottom point (into surface)
    bot_idx = len(verts_local)
    verts_local.append(numpy.array([0.0, -(diamond_height_down + spike_length), 0.0]))

    # Upper cone faces (base ring to top)
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([i, j, top_idx])

    # Lower cone faces (base ring to bottom, reversed winding)
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([i, bot_idx, j])

    # Rotate from +Y to surface normal direction
    R = rotation_matrix_from_vectors(numpy.array([0.0, 1.0, 0.0]), d)

    vertices = numpy.array([
        c + R @ v for v in verts_local
    ], dtype=numpy.float32)
    indices = numpy.array(faces, dtype=numpy.int32)
    return vertices, indices


def create_dot_mesh_data(
    center: numpy.ndarray,
    size: float = 0.8,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Generate a simple octahedral dot marker centered on a point.

    Args:
        center: 3D point for dot center.
        size: Dot radius-like scale.

    Returns:
        (vertices, indices) as numpy arrays.
    """
    c = numpy.asarray(center, dtype=numpy.float64)
    s = float(size)

    # Octahedron vertices (axis-aligned)
    verts = numpy.array([
        [0.0, s, 0.0],    # top
        [0.0, -s, 0.0],   # bottom
        [s, 0.0, 0.0],    # +x
        [-s, 0.0, 0.0],   # -x
        [0.0, 0.0, s],    # +z
        [0.0, 0.0, -s],   # -z
    ], dtype=numpy.float64)
    verts = (verts + c).astype(numpy.float32)

    faces = numpy.array([
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ], dtype=numpy.int32)

    return verts, faces
