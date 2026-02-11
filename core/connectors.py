# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Connector (peg & hole) creation and placement.

Adds interlocking geometry at cut surfaces so split parts can be
reassembled with proper alignment.
No Cura dependencies - uses only trimesh and numpy.
"""

import numpy
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from .geometry import rotation_matrix_from_vectors

logger = logging.getLogger("objectsplitter.connectors")

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class ConnectorConfig:
    """Configuration for peg-and-hole connectors."""
    enabled: bool = True
    diameter: float = 4.0       # mm
    height: float = 3.0         # mm
    clearance: float = 0.2      # mm - extra space in hole
    sides: int = 16             # cylinder approximation segments


@dataclass
class ConnectorResult:
    """Result of adding connectors to mesh halves."""
    upper: "trimesh.Trimesh"
    lower: "trimesh.Trimesh"
    connector_position: Optional[numpy.ndarray] = None
    peg_on: Optional[str] = None  # "upper" or "lower"
    hole_on: Optional[str] = None  # "upper" or "lower"
    hole_engine: Optional[str] = None  # which boolean engine succeeded
    skipped_reason: Optional[str] = None

    @property
    def connectors_added(self) -> bool:
        return self.peg_on is not None


def get_mesh_volume(mesh: "trimesh.Trimesh") -> float:
    """Get mesh volume, falling back to convex hull or bounding box."""
    try:
        if mesh.is_watertight:
            return abs(mesh.volume)
        else:
            return abs(mesh.convex_hull.volume)
    except Exception:
        bounds = mesh.bounds
        return numpy.prod(bounds[1] - bounds[0])


def determine_peg_side(
    mesh_a: "trimesh.Trimesh",
    mesh_b: "trimesh.Trimesh"
) -> Tuple[str, str]:
    """
    Determine which part gets peg vs hole based on volume.
    Peg goes on the smaller part, hole on the larger part.

    Returns:
        ("peg", "hole") or ("hole", "peg") for (mesh_a_role, mesh_b_role).
    """
    volume_a = get_mesh_volume(mesh_a)
    volume_b = get_mesh_volume(mesh_b)
    logger.debug("Volume comparison: a=%.2f mm^3, b=%.2f mm^3", volume_a, volume_b)

    if volume_a <= volume_b:
        return ("peg", "hole")
    else:
        return ("hole", "peg")


def find_connector_position(
    mesh: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray
) -> Optional[numpy.ndarray]:
    """
    Find the centroid of the cut surface for connector placement.

    Args:
        mesh: One half of the split mesh.
        plane_origin: Cut plane origin point.
        plane_normal: Cut plane normal vector.

    Returns:
        3D position for the connector, or None if no valid position found.
    """
    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            logger.debug("No cross-section for connector placement")
            return None

        if not hasattr(section, 'vertices') or len(section.vertices) == 0:
            logger.debug("Cross-section has no vertices")
            return None

        centroid = section.vertices.mean(axis=0)
        # Project onto the plane for precision
        dist = numpy.dot(centroid - plane_origin, plane_normal)
        centroid = centroid - dist * plane_normal

        logger.debug("Connector position: %s", centroid)
        return centroid

    except Exception as e:
        logger.debug("Error finding connector position: %s", e)
        return None


def create_peg_mesh(
    position: numpy.ndarray,
    normal: numpy.ndarray,
    diameter: float,
    height: float,
    sides: int = 16
) -> "trimesh.Trimesh":
    """
    Create a cylindrical peg mesh at the given position, oriented along the normal.

    The peg base sits on the cut surface and extends outward along the normal.
    """
    radius = diameter / 2.0
    peg = trimesh.creation.cylinder(radius=radius, height=height, sections=sides)

    # Shift base to origin (cylinder is centered, move up by half height)
    peg.apply_translation([0, 0, height / 2])

    # Rotate from Z-axis to the plane normal
    z_axis = numpy.array([0, 0, 1])
    normal_normalized = normal / numpy.linalg.norm(normal)
    R = rotation_matrix_from_vectors(z_axis, normal_normalized)
    transform = numpy.eye(4)
    transform[:3, :3] = R
    peg.apply_transform(transform)

    peg.apply_translation(position)

    logger.debug("Created peg: d=%.2f, h=%.2f at %s", diameter, height, position)
    return peg


def create_hole_mesh(
    position: numpy.ndarray,
    normal: numpy.ndarray,
    diameter: float,
    height: float,
    clearance: float,
    sides: int = 16
) -> "trimesh.Trimesh":
    """
    Create a cylindrical hole mesh (to be boolean-subtracted) at the given position.

    The hole is slightly larger than the peg (by clearance) and slightly deeper
    for a clean boolean subtraction.
    """
    radius = diameter / 2.0 + clearance
    hole_height = height + 0.2

    hole = trimesh.creation.cylinder(radius=radius, height=hole_height, sections=sides)

    # Shift so the top is at Z=0 (hole goes into the part)
    hole.apply_translation([0, 0, -hole_height / 2])

    # Rotate Z-axis to negative normal (hole goes inward)
    z_axis = numpy.array([0, 0, 1])
    normal_normalized = normal / numpy.linalg.norm(normal)
    R = rotation_matrix_from_vectors(z_axis, -normal_normalized)
    transform = numpy.eye(4)
    transform[:3, :3] = R
    hole.apply_transform(transform)

    hole.apply_translation(position)

    logger.debug("Created hole: d=%.2f (clearance=%.2f), h=%.2f at %s",
                 diameter + clearance * 2, clearance, hole_height, position)
    return hole


def try_boolean_difference(
    mesh: "trimesh.Trimesh",
    tool: "trimesh.Trimesh"
) -> Tuple[Optional["trimesh.Trimesh"], Optional[str]]:
    """
    Try boolean difference using available engines in priority order.

    Returns:
        (result_mesh, engine_name) or (None, None) if all fail.
    """
    engines = ['manifold', 'blender', None]  # None = default engine
    for engine in engines:
        engine_name = engine or 'default'
        try:
            kwargs = {"engine": engine} if engine else {}
            result = trimesh.boolean.difference([mesh, tool], **kwargs)
            if result is not None and len(result.vertices) > 0:
                logger.debug("Boolean difference succeeded with '%s' engine", engine_name)
                return result, engine_name
        except Exception as e:
            logger.debug("Boolean difference with '%s' failed: %s", engine_name, e)

    logger.warning("All boolean difference engines failed")
    return None, None


def add_connectors(
    mesh_upper: "trimesh.Trimesh",
    mesh_lower: "trimesh.Trimesh",
    plane_origin: numpy.ndarray,
    plane_normal: numpy.ndarray,
    config: ConnectorConfig = None
) -> ConnectorResult:
    """
    Add a peg to the smaller part and a hole to the larger part.

    The peg is added via mesh concatenation (fast, no boolean needed since
    it sits on the cut surface). The hole is subtracted via boolean difference.
    If the hole fails, connectors are skipped entirely.

    Args:
        mesh_upper: Upper half mesh.
        mesh_lower: Lower half mesh.
        plane_origin: Cut plane origin.
        plane_normal: Cut plane normal.
        config: Connector configuration. Uses defaults if None.

    Returns:
        ConnectorResult with the (possibly modified) meshes and metadata.
    """
    if config is None:
        config = ConnectorConfig()

    if not config.enabled:
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            skipped_reason="connectors disabled")

    upper_role, lower_role = determine_peg_side(mesh_upper, mesh_lower)

    connector_pos = find_connector_position(mesh_upper, plane_origin, plane_normal)
    if connector_pos is None:
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            skipped_reason="no valid connector position found")

    peg = create_peg_mesh(connector_pos, plane_normal, config.diameter, config.height, config.sides)
    hole = create_hole_mesh(connector_pos, plane_normal, config.diameter, config.height,
                            config.clearance, config.sides)

    try:
        if upper_role == "peg":
            hole_mesh, hole_engine = try_boolean_difference(mesh_lower, hole)
            if hole_mesh is not None and len(hole_mesh.vertices) > 0:
                result_upper = trimesh.util.concatenate([mesh_upper, peg])
                return ConnectorResult(
                    upper=result_upper, lower=hole_mesh,
                    connector_position=connector_pos,
                    peg_on="upper", hole_on="lower",
                    hole_engine=hole_engine)
            else:
                return ConnectorResult(
                    upper=mesh_upper, lower=mesh_lower,
                    connector_position=connector_pos,
                    skipped_reason="boolean difference for hole failed")
        else:
            hole_mesh, hole_engine = try_boolean_difference(mesh_upper, hole)
            if hole_mesh is not None and len(hole_mesh.vertices) > 0:
                result_lower = trimesh.util.concatenate([mesh_lower, peg])
                return ConnectorResult(
                    upper=hole_mesh, lower=result_lower,
                    connector_position=connector_pos,
                    peg_on="lower", hole_on="upper",
                    hole_engine=hole_engine)
            else:
                return ConnectorResult(
                    upper=mesh_upper, lower=mesh_lower,
                    connector_position=connector_pos,
                    skipped_reason="boolean difference for hole failed")

    except Exception as e:
        logger.error("Error adding connectors: %s", e)
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            connector_position=connector_pos,
            skipped_reason=str(e))
