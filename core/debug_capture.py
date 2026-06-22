# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Debug capture and replay system for ObjectSplitter.

Serializes mesh inputs and parameters to disk so that cutting operations
can be replayed and debugged outside of Cura. This enables:
  - Capturing a failing cut with its exact inputs
  - Replaying the operation in a unit test or standalone script
  - Comparing results across algorithm changes
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import numpy

logger = logging.getLogger("objectsplitter.debug_capture")

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class CapturedOperation:
    """All inputs needed to replay a cut operation."""
    # Identification
    timestamp: str
    name: str  # e.g. "horizontal_cut_cube"

    # Input mesh (stored as file path relative to capture dir)
    mesh_file: str  # .stl or .ply file

    # Cut parameters
    cut_mode: str  # "horizontal", "vertical", "smallest", "valley", "shortest", "valley_seam"
    click_position: list  # [x, y, z]
    height_percent: float  # for horizontal mode
    search_resolution: int  # for smallest/valley mode

    # Connector parameters
    connector_enabled: bool
    connector_diameter: float
    connector_height: float
    connector_clearance: float
    valley_sdf_bias_enabled: bool = False
    anchor_points: Optional[list] = None

    # Optional: plane override (for replaying with a known plane)
    plane_origin: Optional[list] = None
    plane_normal: Optional[list] = None

    # Path / multi-point mode parameters
    waypoints: Optional[list] = None  # list of [x, y, z], exact points passed to the cut
    path_close_loop: bool = True
    path_cap_ends: bool = True


def get_capture_dir(base_dir: Optional[str] = None) -> Path:
    """Get or create the capture directory."""
    if base_dir:
        capture_dir = Path(base_dir)
    else:
        capture_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


def capture_operation(
    mesh: "trimesh.Trimesh",
    cut_mode: str,
    click_position: numpy.ndarray,
    height_percent: float = 50.0,
    search_resolution: int = 18,
    valley_sdf_bias_enabled: bool = False,
    anchor_points: Optional[list] = None,
    connector_enabled: bool = True,
    connector_diameter: float = 4.0,
    connector_height: float = 3.0,
    connector_clearance: float = 0.2,
    plane_origin: Optional[numpy.ndarray] = None,
    plane_normal: Optional[numpy.ndarray] = None,
    waypoints: Optional[list] = None,
    path_close_loop: bool = True,
    path_cap_ends: bool = True,
    name: Optional[str] = None,
    capture_dir: Optional[str] = None
) -> str:
    """
    Save a mesh and all operation parameters to disk for later replay.

    Args:
        mesh: The input trimesh object.
        cut_mode: The cutting mode string.
        click_position: 3D click position.
        height_percent: Horizontal cut height percentage.
        search_resolution: Search resolution for smallest/valley mode.
        valley_sdf_bias_enabled: Enables SDF bias for valley/valley_seam replay.
        anchor_points: Optional list of 3D anchor points used by the cut mode.
        connector_enabled: Whether connectors are enabled.
        connector_diameter: Connector peg diameter.
        connector_height: Connector peg height.
        connector_clearance: Connector hole clearance.
        plane_origin: Optional pre-computed plane origin.
        plane_normal: Optional pre-computed plane normal.
        name: Optional descriptive name for the capture.
        capture_dir: Optional directory override.

    Returns:
        Path to the capture directory for this operation.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if name is None:
        name = f"{cut_mode}_{timestamp}"

    base = get_capture_dir(capture_dir)
    op_dir = base / name
    op_dir.mkdir(parents=True, exist_ok=True)

    # Save mesh
    mesh_file = "input_mesh.stl"
    mesh.export(str(op_dir / mesh_file))

    # Save parameters
    op = CapturedOperation(
        timestamp=timestamp,
        name=name,
        mesh_file=mesh_file,
        cut_mode=cut_mode,
        click_position=click_position.tolist() if hasattr(click_position, 'tolist') else list(click_position),
        height_percent=height_percent,
        search_resolution=search_resolution,
        valley_sdf_bias_enabled=valley_sdf_bias_enabled,
        anchor_points=(
            [numpy.asarray(p, dtype=numpy.float64).reshape(3).tolist() for p in anchor_points]
            if anchor_points else None
        ),
        connector_enabled=connector_enabled,
        connector_diameter=connector_diameter,
        connector_height=connector_height,
        connector_clearance=connector_clearance,
        plane_origin=plane_origin.tolist() if plane_origin is not None else None,
        plane_normal=plane_normal.tolist() if plane_normal is not None else None,
        waypoints=(
            [numpy.asarray(p, dtype=numpy.float64).reshape(3).tolist() for p in waypoints]
            if waypoints else None
        ),
        path_close_loop=path_close_loop,
        path_cap_ends=path_cap_ends,
    )

    with open(op_dir / "params.json", "w") as f:
        json.dump(asdict(op), f, indent=2)

    logger.info("Captured operation '%s' to %s", name, op_dir)
    return str(op_dir)


def load_captured_operation(capture_path: str) -> tuple:
    """
    Load a captured operation from disk.

    Args:
        capture_path: Path to the capture directory.

    Returns:
        (mesh, params) tuple where mesh is a trimesh.Trimesh and
        params is a CapturedOperation dataclass.
    """
    op_dir = Path(capture_path)

    with open(op_dir / "params.json", "r") as f:
        data = json.load(f)

    params = CapturedOperation(**data)

    mesh = trimesh.load(str(op_dir / params.mesh_file))

    logger.info("Loaded captured operation '%s' from %s", params.name, op_dir)
    return mesh, params


def replay_operation(capture_path: str, preloaded: Optional[tuple] = None) -> dict:
    """
    Replay a captured operation and return the results.

    This is the main entry point for re-running a previously captured
    cut operation with the exact same inputs.

    Args:
        capture_path: Path to the capture directory.
        preloaded: Optional ``(mesh, params)`` from a prior
            ``load_captured_operation`` call. When given, the mesh is not
            re-read from disk, so callers timing/profiling the cut can keep
            disk I/O and STL parsing out of their measurements.

    Returns:
        Dictionary with keys:
            - 'split_result': SplitResult from mesh_splitter
            - 'connector_result': ConnectorResult (if connectors enabled)
            - 'plane': CutPlane used
            - 'params': The CapturedOperation parameters
    """
    from .plane_calculator import (
        horizontal_cut_plane, vertical_cut_plane,
        find_smallest_cut_plane, find_shortest_seam_partition,
        find_valley_cut_plane, find_valley_seam_partition,
        snap_point_to_mesh_surface, CutPlane
    )
    from .mesh_splitter import (
        slice_mesh_with_fallback,
        split_by_shortest_seam,
        split_by_local_plane,
        split_by_face_sets,
    )
    from .path_cutter import chain_paths, partition_faces_by_path
    from .connectors import add_connectors, ConnectorConfig

    if preloaded is not None:
        mesh, params = preloaded
    else:
        mesh, params = load_captured_operation(capture_path)
    click_pos = numpy.array(params.click_position)

    # Path (multi-point) mode replays purely from waypoints; it never
    # computes a plane, so handle it up front and return early.
    if params.cut_mode == "path":
        waypoints = [
            numpy.asarray(p, dtype=numpy.float64).reshape(3)
            for p in (params.waypoints or [])
        ]
        if len(waypoints) < 2:
            raise ValueError("Path replay needs at least 2 waypoints")
        vertex_path = chain_paths(mesh, waypoints)
        face_set_a, face_set_b = partition_faces_by_path(mesh, vertex_path)
        split_result = split_by_face_sets(
            mesh,
            face_set_a,
            face_set_b,
            strategy_name="path_cut",
            attempt_hole_fill=(params.connector_enabled or params.path_cap_ends),
        )

        connector_result = None
        if params.connector_enabled and split_result.success:
            from .connectors import add_path_connectors
            path_points = numpy.asarray(mesh.vertices[vertex_path], dtype=numpy.float64)
            connector_result = add_path_connectors(
                split_result.upper,
                split_result.lower,
                path_points,
                split_result.cap_faces_upper,
                split_result.cap_faces_lower,
                config=ConnectorConfig(
                    enabled=True,
                    diameter=params.connector_diameter,
                    height=params.connector_height,
                    clearance=params.connector_clearance,
                ),
                face_count=len(mesh.faces),
            )

        return {
            'split_result': split_result,
            'connector_result': connector_result,
            'plane': None,
            'params': params,
        }
    anchor_points = None
    if getattr(params, "anchor_points", None):
        anchor_points = [numpy.asarray(p, dtype=numpy.float64).reshape(3) for p in params.anchor_points]
    search_result = None
    valley_face_id = None

    # Determine cut plane
    if params.plane_origin is not None and params.plane_normal is not None:
        plane = CutPlane(
            origin=numpy.array(params.plane_origin),
            normal=numpy.array(params.plane_normal))
    elif params.cut_mode == "horizontal":
        plane = horizontal_cut_plane(mesh, params.height_percent)
    elif params.cut_mode == "vertical":
        plane = vertical_cut_plane(click_pos)
    elif params.cut_mode == "smallest":
        search_result = find_smallest_cut_plane(
            mesh, click_pos, params.search_resolution, collect_all_samples=True)
        plane = search_result.plane
    elif params.cut_mode == "valley":
        valley_click = click_pos
        if anchor_points and len(anchor_points) > 0:
            valley_click = numpy.asarray(anchor_points[0], dtype=numpy.float64)
        snap_point, valley_face_id = snap_point_to_mesh_surface(mesh, valley_click)
        valley_surface_normal = None
        if valley_face_id is not None and valley_face_id < len(mesh.face_normals):
            valley_surface_normal = numpy.array(
                mesh.face_normals[valley_face_id], dtype=numpy.float64
            )
        search_result = find_valley_cut_plane(
            mesh,
            snap_point,
            params.search_resolution,
            surface_normal=valley_surface_normal,
            use_sdf_bias=bool(getattr(params, "valley_sdf_bias_enabled", False)),
            anchor_points=anchor_points,
            debug_trace_path=os.path.join(capture_path, "valley_trace_replay.json"),
        )
        plane = search_result.plane
    elif params.cut_mode == "shortest":
        plane = None  # Handled differently
    elif params.cut_mode == "valley_seam":
        plane = None  # Handled differently
    else:
        plane = horizontal_cut_plane(mesh, params.height_percent)

    # Perform the split
    if params.cut_mode == "shortest":
        face_set_a, face_set_b, _, _ = find_shortest_seam_partition(mesh, click_pos)
        split_result = split_by_shortest_seam(mesh, face_set_a, face_set_b)
    elif params.cut_mode == "valley":
        if valley_face_id is None:
            valley_click = click_pos
            if anchor_points and len(anchor_points) > 0:
                valley_click = numpy.asarray(anchor_points[0], dtype=numpy.float64)
            _, valley_face_id = snap_point_to_mesh_surface(mesh, valley_click)
        if valley_face_id is None:
            valley_face_id = 0
        candidate_normals = (
            [n for _, n in search_result.top_candidates]
            if search_result is not None and search_result.top_candidates
            else [plane.normal]
        )
        split_result = split_by_local_plane(
            mesh, plane.origin, candidate_normals, valley_face_id
        )
    elif params.cut_mode == "valley_seam":
        source_face_hint = None
        target_face_hint = None
        seam_click = click_pos
        seam_surface_normal = None
        if anchor_points and len(anchor_points) > 0:
            seam_click = numpy.asarray(anchor_points[0], dtype=numpy.float64)
            seam_click, source_face_hint = snap_point_to_mesh_surface(mesh, seam_click)
            if source_face_hint is not None and source_face_hint < len(mesh.face_normals):
                seam_surface_normal = numpy.array(mesh.face_normals[source_face_hint], dtype=numpy.float64)
            if len(anchor_points) >= 2:
                _, target_face_hint = snap_point_to_mesh_surface(
                    mesh, numpy.asarray(anchor_points[-1], dtype=numpy.float64)
                )
        face_set_a, face_set_b, _, _ = find_valley_seam_partition(
            mesh,
            seam_click,
            source_face_hint=source_face_hint,
            target_face_hint=target_face_hint,
            surface_normal=seam_surface_normal,
            use_sdf_bias=bool(getattr(params, "valley_sdf_bias_enabled", False)),
            anchor_points=anchor_points,
            debug_trace_path=os.path.join(capture_path, "valley_seam_trace_replay.json"),
        )
        split_result = split_by_face_sets(
            mesh,
            face_set_a,
            face_set_b,
            strategy_name="valley_seam",
            attempt_hole_fill=False,
        )
    else:
        split_result = slice_mesh_with_fallback(mesh, plane.origin, plane.normal)

    result = {
        'split_result': split_result,
        'connector_result': None,
        'plane': plane,
        'params': params,
    }

    # Add connectors if enabled
    if params.connector_enabled and split_result.success and split_result.capped:
        if params.cut_mode not in ("shortest", "valley_seam") and plane is not None:
            config = ConnectorConfig(
                enabled=True,
                diameter=params.connector_diameter,
                height=params.connector_height,
                clearance=params.connector_clearance)
            connector_result = add_connectors(
                split_result.upper, split_result.lower,
                plane.origin, plane.normal, config)
            result['connector_result'] = connector_result

    return result


def save_result_meshes(
    result: dict,
    output_dir: str
) -> list:
    """
    Save the result meshes from a replay to disk for inspection.

    Args:
        result: Dictionary from replay_operation().
        output_dir: Directory to save meshes into.

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    split = result.get('split_result')
    connector = result.get('connector_result')

    # Use connector result meshes if available, otherwise split result
    if connector is not None and connector.connectors_added:
        upper = connector.upper
        lower = connector.lower
    elif split is not None and split.success:
        upper = split.upper
        lower = split.lower
    else:
        return saved

    if upper is not None:
        path = str(out / "result_upper.stl")
        upper.export(path)
        saved.append(path)

    if lower is not None:
        path = str(out / "result_lower.stl")
        lower.export(path)
        saved.append(path)

    return saved
