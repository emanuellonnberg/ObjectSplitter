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
