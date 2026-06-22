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
    attempts: int = 0  # number of placement attempts made

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
    # True volume is only meaningful when both halves are watertight. For
    # surface/path cuts the pieces are often open shells, and convex-hull
    # volume is unstable for thin wide pieces -- a tiny edit could flip the
    # comparison and move the peg to the larger piece between near-identical
    # cuts. Fall back to face count, a stable, monotonic proxy that matches
    # the visible "smaller piece".
    if mesh_a.is_watertight and mesh_b.is_watertight:
        size_a = abs(get_mesh_volume(mesh_a))
        size_b = abs(get_mesh_volume(mesh_b))
        metric = "volume"
    else:
        size_a = float(len(mesh_a.faces))
        size_b = float(len(mesh_b.faces))
        metric = "face_count"
    logger.debug("Peg-side comparison (%s): a=%.2f, b=%.2f", metric, size_a, size_b)

    if size_a <= size_b:
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


_BOOLEAN_AVAILABLE: Optional[bool] = None


def boolean_engine_available() -> bool:
    """Whether any trimesh boolean engine actually works in this environment.

    Probed once with a tiny difference and cached. When no engine is present
    (e.g. manifold3d/blender not installed), every boolean connector attempt
    is doomed and slow, so callers should skip the boolean path entirely
    instead of retrying it dozens of times.
    """
    global _BOOLEAN_AVAILABLE
    if _BOOLEAN_AVAILABLE is None:
        try:
            a = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            b = trimesh.creation.box(extents=(1.0, 1.0, 3.0))
            result = trimesh.boolean.difference([a, b])
            _BOOLEAN_AVAILABLE = result is not None and len(result.vertices) > 0
        except Exception as e:
            logger.info("No boolean engine available (%s); boolean connectors disabled", e)
            _BOOLEAN_AVAILABLE = False
    return bool(_BOOLEAN_AVAILABLE)


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


def _prepare_mesh_for_boolean(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Return a cleaned copy of a mesh for more robust boolean operations.
    """
    prepared = mesh.copy()
    try:
        if hasattr(prepared, "unique_faces"):
            prepared.update_faces(prepared.unique_faces())
        if hasattr(prepared, "nondegenerate_faces"):
            prepared.update_faces(prepared.nondegenerate_faces())
        prepared.remove_unreferenced_vertices()
        prepared.merge_vertices(digits_vertex=7)
        # validate=True runs internal cleanup routines in trimesh.
        prepared.process(validate=True)
        prepared.fix_normals()
    except Exception as e:
        logger.debug("Boolean pre-cleanup failed (continuing): %s", e)
    return prepared


def _cap_vertex_indices(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
) -> Optional[numpy.ndarray]:
    """Return unique vertex indices referenced by cap faces."""
    if cap_faces is None or len(cap_faces) == 0:
        return None
    faces = numpy.asarray(mesh.faces, dtype=numpy.int64)
    valid_faces = [int(f) for f in cap_faces if 0 <= int(f) < len(faces)]
    if not valid_faces:
        return None
    vids = numpy.unique(faces[numpy.asarray(valid_faces, dtype=numpy.int64)].reshape(-1))
    if len(vids) == 0:
        return None
    return numpy.asarray(vids, dtype=numpy.int64)


def _snap_point_to_cap(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
    point: numpy.ndarray,
) -> Optional[numpy.ndarray]:
    """Snap a 3D point to the nearest point on the cap *surface*.

    Snapping to the nearest cap vertex pulls the connector to the cap rim
    when the cap is a boundary-only triangulation (no interior vertices),
    placing the peg at the edge of the cut. Snapping to the nearest surface
    point keeps a centered query centered.
    """
    vids = _cap_vertex_indices(mesh, cap_faces)
    if vids is None:
        return None
    p = numpy.asarray(point, dtype=numpy.float64).reshape(3)

    try:
        cap = mesh.submesh([cap_faces], append=True)
        from trimesh.proximity import closest_point
        cp, _, _ = closest_point(cap, p[None, :])
        if cp is not None and len(cp) > 0:
            return numpy.asarray(cp[0], dtype=numpy.float64)
    except Exception as e:
        logger.debug("closest_point cap snap failed, using nearest vertex: %s", e)

    pts = numpy.asarray(mesh.vertices[vids], dtype=numpy.float64)
    d = numpy.linalg.norm(pts - p[None, :], axis=1)
    if len(d) == 0:
        return None
    return numpy.asarray(pts[int(numpy.argmin(d))], dtype=numpy.float64)


def _select_cap_patch_vertices(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
    center: numpy.ndarray,
    radius: float,
) -> Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]:
    """
    Select cap vertices near a center point and return (vertex_indices, distances).
    """
    if radius <= 1e-6:
        return None, None

    cap_vids = _cap_vertex_indices(mesh, cap_faces)
    if cap_vids is None:
        return None, None

    pts = numpy.asarray(mesh.vertices[cap_vids], dtype=numpy.float64)
    dists = numpy.linalg.norm(pts - center[None, :], axis=1)
    chosen_mask = dists <= float(radius)
    chosen_vids = cap_vids[chosen_mask]
    chosen_dists = dists[chosen_mask]

    # If radius misses everything on coarse caps, take nearest cap vertices.
    if len(chosen_vids) < 12:
        order = numpy.argsort(dists)
        k = int(min(max(12, len(chosen_vids)), len(cap_vids)))
        chosen_vids = cap_vids[order[:k]]
        chosen_dists = dists[order[:k]]

    if len(chosen_vids) < 3:
        return None, None
    return numpy.asarray(chosen_vids, dtype=numpy.int64), numpy.asarray(chosen_dists, dtype=numpy.float64)


def _boss_profile(dist: numpy.ndarray, radius: float) -> numpy.ndarray:
    """Radial connector profile: flat core, smooth shoulder, zero past radius.

    Returns a weight in [0, 1] for each distance: 1.0 within the flat core,
    a quintic-smoothstep falloff through the shoulder, and exactly 0.0 at and
    beyond ``radius``. Multiplying by the connector depth gives a blunt,
    cylinder-like boss instead of a spike.
    """
    r = max(float(radius), 1e-6)
    u = numpy.clip(numpy.asarray(dist, dtype=numpy.float64) / r, 0.0, 1.0)
    core_ratio = 0.72
    w = numpy.ones_like(u)
    shoulder = u > core_ratio
    if numpy.any(shoulder):
        t = (u[shoulder] - core_ratio) / (1.0 - core_ratio)
        s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        w[shoulder] = 1.0 - s
    return w


def _fit_cap_plane(points: numpy.ndarray):
    """PCA best-fit plane for a set of 3D points -> (origin, u, v, normal)."""
    pts = numpy.asarray(points, dtype=numpy.float64)
    if len(pts) < 3:
        return None
    origin = pts.mean(axis=0)
    centered = pts - origin
    try:
        _, vecs = numpy.linalg.eigh(centered.T @ centered)
    except numpy.linalg.LinAlgError:
        return None
    normal = vecs[:, 0]
    u = vecs[:, 2]
    un = numpy.linalg.norm(u)
    if un < 1e-12:
        return None
    u = u / un
    v = numpy.cross(normal, u)
    vn = numpy.linalg.norm(v)
    if vn < 1e-12:
        return None
    v = v / vn
    return origin, u, v, normal


def _cap_plane_normal(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
) -> Optional[numpy.ndarray]:
    """Unit normal of the best-fit plane through a cap's vertices."""
    vids = _cap_vertex_indices(mesh, cap_faces)
    if vids is None or len(vids) < 3:
        return None
    basis = _fit_cap_plane(numpy.asarray(mesh.vertices[vids], dtype=numpy.float64))
    if basis is None:
        return None
    normal = numpy.asarray(basis[3], dtype=numpy.float64)
    n = numpy.linalg.norm(normal)
    if n < 1e-9:
        return None
    return normal / n


def _deform_cap_patch_dense(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
    center: numpy.ndarray,
    direction: numpy.ndarray,
    radius: float,
    depth: float,
) -> Optional[Tuple["trimesh.Trimesh", Optional[str]]]:
    """Rebuild the cap region with interior points, then form a smooth boss.

    The default cap is a boundary-only triangulation (no interior vertices),
    so displacing it produces a sharp tent rather than a peg. This re-meshes
    the connector region with a dense interior point grid (constrained to the
    cap outline), displaces only the interior by the boss profile, and welds
    the rebuilt cap back onto the wall. Returns None to signal the caller to
    fall back to the simple displacement path.
    """
    try:
        from scipy.spatial import Delaunay
        from shapely.geometry import Polygon, Point
    except Exception:
        return None

    try:
        from .mesh_splitter import _orient_caps_to_boundary
    except Exception:
        _orient_caps_to_boundary = None

    cap_vids = _cap_vertex_indices(mesh, cap_faces)
    if cap_vids is None or len(cap_vids) < 3:
        return None

    direction = numpy.asarray(direction, dtype=numpy.float64).reshape(3)
    dn = numpy.linalg.norm(direction)
    if dn < 1e-9:
        return None
    direction = direction / dn
    center = numpy.asarray(center, dtype=numpy.float64).reshape(3)

    # Cap outline loop(s) -> pick the largest as the patch boundary.
    cap = mesh.submesh([cap_faces], append=True)
    try:
        outline = cap.outline()
        loops = outline.discrete if (outline is not None and hasattr(outline, "discrete")) else None
    except Exception:
        loops = None
    if not loops:
        return None
    loop3d = numpy.asarray(max(loops, key=len), dtype=numpy.float64)
    if len(loop3d) >= 2 and numpy.allclose(loop3d[0], loop3d[-1]):
        loop3d = loop3d[:-1]
    if len(loop3d) < 3:
        return None

    basis = _fit_cap_plane(loop3d)
    if basis is None:
        return None
    origin, u, v, _ = basis
    loop2d = numpy.column_stack([(loop3d - origin) @ u, (loop3d - origin) @ v])
    center2d = numpy.array([(center - origin) @ u, (center - origin) @ v])

    try:
        poly = Polygon(loop2d)
    except Exception:
        return None
    if not poly.is_valid or poly.area <= 1e-9:
        return None

    # Interior sample grid inside the cap, clustered around the connector.
    rr = float(radius) * 1.3
    step = max(float(radius) / 5.0, 1e-3)
    coords = numpy.arange(-rr, rr + step, step)
    interior = []
    for dx in coords:
        for dy in coords:
            if dx * dx + dy * dy > rr * rr:
                continue
            x = center2d[0] + dx
            y = center2d[1] + dy
            pt = Point(x, y)
            if poly.contains(pt) and poly.boundary.distance(pt) > step * 0.5:
                interior.append((x, y))
    if len(interior) < 6:
        return None
    interior2d = numpy.asarray(interior, dtype=numpy.float64)

    all2d = numpy.vstack([loop2d, interior2d])
    n_boundary = len(loop2d)
    try:
        tri = Delaunay(all2d)
    except Exception:
        return None
    keep_faces = []
    for simplex in tri.simplices:
        cen = all2d[simplex].mean(axis=0)
        if poly.contains(Point(cen[0], cen[1])):
            keep_faces.append(simplex)
    if not keep_faces:
        return None
    new_faces = numpy.asarray(keep_faces, dtype=numpy.int64)

    # 3D positions: boundary keeps exact original coords (weld seam),
    # interior lifted onto the fit plane.
    interior3d = (
        origin[None, :]
        + interior2d[:, 0:1] * u[None, :]
        + interior2d[:, 1:2] * v[None, :]
    )
    new_cap_verts = numpy.vstack([loop3d, interior3d])

    # Displace only interior vertices to form the boss; seam stays put.
    dist_c = numpy.linalg.norm(new_cap_verts - center[None, :], axis=1)
    w = _boss_profile(dist_c, radius)
    w[:n_boundary] = 0.0
    max_disp = float(w.max() * float(depth)) if len(w) else 0.0
    if max_disp < max(0.03, float(depth) * 0.08):
        return None
    new_cap_verts = new_cap_verts + (w * float(depth))[:, None] * direction[None, :]

    new_cap = trimesh.Trimesh(
        vertices=new_cap_verts, faces=new_faces, process=False, validate=False
    )

    # Wall = original mesh minus old cap faces; weld the rebuilt cap on.
    all_mesh_faces = numpy.asarray(mesh.faces, dtype=numpy.int64)
    drop = numpy.zeros(len(all_mesh_faces), dtype=bool)
    valid_cap = numpy.asarray(cap_faces, dtype=numpy.int64)
    valid_cap = valid_cap[(valid_cap >= 0) & (valid_cap < len(all_mesh_faces))]
    drop[valid_cap] = True
    wall = trimesh.Trimesh(
        vertices=numpy.asarray(mesh.vertices, dtype=numpy.float64),
        faces=all_mesh_faces[~drop],
        process=False,
        validate=False,
    )
    base_faces = len(wall.faces)
    combined = trimesh.util.concatenate([wall, new_cap])
    try:
        combined.merge_vertices(digits_vertex=7)
        combined.remove_unreferenced_vertices()
    except Exception:
        pass

    if _orient_caps_to_boundary is not None:
        try:
            _orient_caps_to_boundary(
                combined, base_faces, [(base_faces, base_faces + len(new_faces))]
            )
        except Exception:
            try:
                combined.fix_normals()
            except Exception:
                pass
    else:
        try:
            combined.fix_normals()
        except Exception:
            pass

    return combined, None


def _deform_cap_patch(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
    center: numpy.ndarray,
    direction: numpy.ndarray,
    radius: float,
    depth: float,
) -> Tuple[Optional["trimesh.Trimesh"], Optional[str]]:
    """Form a connector boss on the cap.

    Prefers a dense re-mesh (smooth, peg-like). Falls back to the simple
    in-place vertex displacement if the dense path is unavailable or fails.
    """
    dense = _deform_cap_patch_dense(mesh, cap_faces, center, direction, radius, depth)
    if dense is not None and dense[0] is not None:
        return dense
    return _deform_cap_patch_simple(mesh, cap_faces, center, direction, radius, depth)


def _deform_cap_patch_simple(
    mesh: "trimesh.Trimesh",
    cap_faces: list,
    center: numpy.ndarray,
    direction: numpy.ndarray,
    radius: float,
    depth: float,
) -> Tuple[Optional["trimesh.Trimesh"], Optional[str]]:
    """
    Deform a circular patch on cap faces by displacing vertices along direction.
    """
    vids, dists = _select_cap_patch_vertices(mesh, cap_faces, center, radius)
    if vids is None or dists is None:
        return None, "insufficient cap patch vertices"

    direction = numpy.asarray(direction, dtype=numpy.float64).reshape(3)
    dn = numpy.linalg.norm(direction)
    if dn < 1e-9:
        return None, "invalid deformation direction"
    direction = direction / dn

    r = max(float(radius), 1e-6)
    # If too few vertices are within the nominal radius, expand it locally
    # so we still create a visible, smooth connector patch.
    inside = int(numpy.sum(dists <= r))
    if inside < 6 and len(dists) >= 6:
        k = min(8, len(dists))
        kth = float(numpy.partition(dists, k - 1)[k - 1])
        r = max(r, kth * 1.05 + 1e-6)
    verts = numpy.asarray(mesh.vertices, dtype=numpy.float64).copy()

    # Blunt profile (wider flat core + smooth shoulder), so cap-native
    # connectors are less spike-like and easier to fit/print.
    u = numpy.clip(dists / r, 0.0, 1.0)
    core_ratio = 0.72
    w = numpy.ones_like(u)
    shoulder_mask = u > core_ratio
    if numpy.any(shoulder_mask):
        v = (u[shoulder_mask] - core_ratio) / (1.0 - core_ratio)
        # Quintic smoothstep shoulder from 1 -> 0.
        s = v * v * v * (v * (v * 6.0 - 15.0) + 10.0)
        w[shoulder_mask] = 1.0 - s
    max_disp = float(w.max() * float(depth)) if len(w) > 0 else 0.0
    if max_disp < max(0.03, float(depth) * 0.08):
        return None, "cap-native displacement too small"
    displacements = (w * float(depth))[:, None] * direction[None, :]
    verts[vids] += displacements

    out = trimesh.Trimesh(
        vertices=verts,
        faces=numpy.asarray(mesh.faces, dtype=numpy.int64).copy(),
        process=False,
        validate=False,
    )
    try:
        out.remove_unreferenced_vertices()
        out.fix_normals()
    except Exception:
        pass
    return out, None


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

    connector_pos = find_connector_position(mesh_upper, plane_origin, plane_normal)
    if connector_pos is None:
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            skipped_reason="no valid connector position found")

    return add_connectors_at_position(
        mesh_upper=mesh_upper,
        mesh_lower=mesh_lower,
        connector_position=connector_pos,
        connector_normal=plane_normal,
        config=config,
    )


def add_connectors_at_position(
    mesh_upper: "trimesh.Trimesh",
    mesh_lower: "trimesh.Trimesh",
    connector_position: numpy.ndarray,
    connector_normal: numpy.ndarray,
    config: ConnectorConfig = None,
) -> ConnectorResult:
    """
    Add a peg/hole connector at a caller-provided point and direction.

    Useful for non-planar splits (e.g. path cuts) where no single cut plane
    exists, but we still have a seam point and an estimated split direction.
    """
    if config is None:
        config = ConnectorConfig()

    if not config.enabled:
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            skipped_reason="connectors disabled")

    pos = numpy.asarray(connector_position, dtype=numpy.float64).reshape(3)
    normal = numpy.asarray(connector_normal, dtype=numpy.float64).reshape(3)
    normal_norm = numpy.linalg.norm(normal)
    if normal_norm < 1e-9:
        return ConnectorResult(
            upper=mesh_upper, lower=mesh_lower,
            connector_position=pos,
            skipped_reason="invalid connector normal")
    normal = normal / normal_norm

    # Clean meshes before booleans; this helps on dense scanned/organic models.
    upper_prepared = _prepare_mesh_for_boolean(mesh_upper)
    lower_prepared = _prepare_mesh_for_boolean(mesh_lower)

    preferred_upper_role, _ = determine_peg_side(mesh_upper, mesh_lower)
    role_orders = [preferred_upper_role]
    role_orders.append("hole" if preferred_upper_role == "peg" else "peg")

    if mesh_upper.is_watertight and not mesh_lower.is_watertight:
        role_orders = ["hole", "peg"]
    elif mesh_lower.is_watertight and not mesh_upper.is_watertight:
        role_orders = ["peg", "hole"]

    role_errors = []

    for upper_role in role_orders:
        try:
            if upper_role == "peg":
                # Upper peg / lower hole.
                # `normal` is treated as lower->upper, so upper peg should extend
                # toward lower (-normal), while lower hole should subtract inward
                # toward lower (-hole_normal = -normal).
                peg_normal = -normal
                hole_normal = normal
                hole_target = "lower"

                peg = create_peg_mesh(
                    pos, peg_normal, config.diameter, config.height, config.sides
                )
                hole_inset = max(0.05, min(0.4, float(config.height) * 0.12))
                hole_pos = pos + (-hole_normal) * hole_inset
                hole = create_hole_mesh(
                    hole_pos,
                    hole_normal,
                    config.diameter,
                    config.height,
                    config.clearance,
                    config.sides,
                )

                # Try standard pocket first, then a deeper through-hole fallback.
                try:
                    lower_diag = float(
                        numpy.linalg.norm(lower_prepared.bounds[1] - lower_prepared.bounds[0])
                    )
                except Exception:
                    lower_diag = float(config.height) * 3.0
                through_height = max(config.height + 0.2, lower_diag * 0.35)
                hole_through = create_hole_mesh(
                    hole_pos,
                    hole_normal,
                    config.diameter,
                    through_height,
                    config.clearance,
                    config.sides,
                )

                hole_mesh, hole_engine = try_boolean_difference(lower_prepared, hole)
                if (hole_mesh is None or len(hole_mesh.vertices) == 0):
                    hole_mesh, hole_engine = try_boolean_difference(lower_prepared, hole_through)

                if hole_mesh is not None and len(hole_mesh.vertices) > 0:
                    result_upper = trimesh.util.concatenate([mesh_upper, peg])
                    return ConnectorResult(
                        upper=result_upper, lower=hole_mesh,
                        connector_position=pos,
                        peg_on="upper", hole_on="lower",
                        hole_engine=hole_engine)
                role_errors.append(f"boolean difference for hole failed on {hole_target}")
            else:
                # Lower peg / upper hole.
                # With `normal` as lower->upper, lower peg should extend toward
                # upper (+normal), while upper hole should subtract inward toward
                # upper (-hole_normal = +normal) so hole_normal = -normal.
                peg_normal = normal
                hole_normal = -normal
                hole_target = "upper"

                peg = create_peg_mesh(
                    pos, peg_normal, config.diameter, config.height, config.sides
                )
                hole_inset = max(0.05, min(0.4, float(config.height) * 0.12))
                hole_pos = pos + (-hole_normal) * hole_inset
                hole = create_hole_mesh(
                    hole_pos,
                    hole_normal,
                    config.diameter,
                    config.height,
                    config.clearance,
                    config.sides,
                )

                try:
                    upper_diag = float(
                        numpy.linalg.norm(upper_prepared.bounds[1] - upper_prepared.bounds[0])
                    )
                except Exception:
                    upper_diag = float(config.height) * 3.0
                through_height = max(config.height + 0.2, upper_diag * 0.35)
                hole_through = create_hole_mesh(
                    hole_pos,
                    hole_normal,
                    config.diameter,
                    through_height,
                    config.clearance,
                    config.sides,
                )

                hole_mesh, hole_engine = try_boolean_difference(upper_prepared, hole)
                if (hole_mesh is None or len(hole_mesh.vertices) == 0):
                    hole_mesh, hole_engine = try_boolean_difference(upper_prepared, hole_through)

                if hole_mesh is not None and len(hole_mesh.vertices) > 0:
                    result_lower = trimesh.util.concatenate([mesh_lower, peg])
                    return ConnectorResult(
                        upper=hole_mesh, lower=result_lower,
                        connector_position=pos,
                        peg_on="lower", hole_on="upper",
                        hole_engine=hole_engine)
                role_errors.append(f"boolean difference for hole failed on {hole_target}")
        except Exception as e:
            role_errors.append(str(e))
            logger.error("Error adding connectors (%s peg side): %s", upper_role, e)

    last_error = " | ".join(role_errors) if role_errors else None
    return ConnectorResult(
        upper=mesh_upper, lower=mesh_lower,
        connector_position=pos,
        skipped_reason=last_error or "boolean difference for hole failed on both sides")


def add_cap_native_connectors(
    mesh_upper: "trimesh.Trimesh",
    mesh_lower: "trimesh.Trimesh",
    connector_position: numpy.ndarray,
    connector_normal: numpy.ndarray,
    cap_faces_upper: list,
    cap_faces_lower: list,
    config: ConnectorConfig = None,
) -> ConnectorResult:
    """
    Add connectors without booleans by deforming matching cap patches.

    This is more robust on dense organic meshes because it avoids global
    boolean subtraction and only edits vertices on the split cap faces.
    """
    if config is None:
        config = ConnectorConfig()
    if not config.enabled:
        return ConnectorResult(
            upper=mesh_upper,
            lower=mesh_lower,
            skipped_reason="connectors disabled",
        )

    pos = numpy.asarray(connector_position, dtype=numpy.float64).reshape(3)
    normal = numpy.asarray(connector_normal, dtype=numpy.float64).reshape(3)
    nn = numpy.linalg.norm(normal)
    if nn < 1e-9:
        return ConnectorResult(
            upper=mesh_upper,
            lower=mesh_lower,
            connector_position=pos,
            skipped_reason="invalid connector normal",
        )
    normal = normal / nn

    if not cap_faces_upper or not cap_faces_lower:
        return ConnectorResult(
            upper=mesh_upper,
            lower=mesh_lower,
            connector_position=pos,
            skipped_reason="missing cap-face metadata for cap-native connector",
        )

    # Snap placement to cap geometry so deformation always targets the actual
    # split surface (not nearby non-cap triangles).
    upper_snap = _snap_point_to_cap(mesh_upper, cap_faces_upper, pos)
    lower_snap = _snap_point_to_cap(mesh_lower, cap_faces_lower, pos)
    if upper_snap is not None and lower_snap is not None:
        pos = 0.5 * (upper_snap + lower_snap)
    elif upper_snap is not None:
        pos = upper_snap
    elif lower_snap is not None:
        pos = lower_snap

    # The connector must push perpendicular to the cut surface ("straight in"),
    # not along the caller's rough normal. On organic closed-loop carves the
    # passed normal (e.g. a centroid difference) can be 60-90 deg off the cut
    # plane, which drives the peg sideways under the surface. Use the cap's
    # best-fit plane normal as the true axis, keeping the caller's normal only
    # to choose the sign (which side is "out"). Both halves share this single
    # world axis so the peg and hole stay aligned.
    cap_axis = _cap_plane_normal(mesh_upper, cap_faces_upper)
    if cap_axis is None:
        cap_axis = _cap_plane_normal(mesh_lower, cap_faces_lower)
    if cap_axis is not None:
        if float(numpy.dot(cap_axis, normal)) < 0.0:
            cap_axis = -cap_axis
        normal = cap_axis

    upper_role, _ = determine_peg_side(mesh_upper, mesh_lower)
    # connector_normal is interpreted as lower->upper.
    # lower peg / upper hole => +normal direction
    # upper peg / lower hole => -normal direction
    direction_sign = 1.0 if upper_role == "hole" else -1.0
    direction = normal * direction_sign

    peg_radius = max(1.0, float(config.diameter) * 0.85)
    hole_radius = peg_radius + max(0.20, float(config.clearance) * 2.5)
    # Shorter default cap-native deformation for a flatter, sturdier profile.
    # User height control still scales this.
    peg_depth = max(0.20, float(config.height) * 0.42)
    hole_depth = max(peg_depth + float(config.clearance), peg_depth * 1.05)

    if upper_role == "peg":
        upper_new, err_upper = _deform_cap_patch(
            mesh_upper, cap_faces_upper, pos, direction, peg_radius, peg_depth
        )
        lower_new, err_lower = _deform_cap_patch(
            mesh_lower, cap_faces_lower, pos, direction, hole_radius, hole_depth
        )
        if upper_new is None or lower_new is None:
            return ConnectorResult(
                upper=mesh_upper,
                lower=mesh_lower,
                connector_position=pos,
                skipped_reason=(
                    f"cap-native failed (upper={err_upper or 'ok'}, lower={err_lower or 'ok'})"
                ),
            )
        return ConnectorResult(
            upper=upper_new,
            lower=lower_new,
            connector_position=pos,
            peg_on="upper",
            hole_on="lower",
            hole_engine="cap_native",
        )

    upper_new, err_upper = _deform_cap_patch(
        mesh_upper, cap_faces_upper, pos, direction, hole_radius, hole_depth
    )
    lower_new, err_lower = _deform_cap_patch(
        mesh_lower, cap_faces_lower, pos, direction, peg_radius, peg_depth
    )
    if upper_new is None or lower_new is None:
        return ConnectorResult(
            upper=mesh_upper,
            lower=mesh_lower,
            connector_position=pos,
            skipped_reason=(
                f"cap-native failed (upper={err_upper or 'ok'}, lower={err_lower or 'ok'})"
            ),
        )
    return ConnectorResult(
        upper=upper_new,
        lower=lower_new,
        connector_position=pos,
        peg_on="lower",
        hole_on="upper",
        hole_engine="cap_native",
    )


def add_path_connectors(
    mesh_upper: "trimesh.Trimesh",
    mesh_lower: "trimesh.Trimesh",
    path_points: numpy.ndarray,
    cap_faces_upper: list,
    cap_faces_lower: list,
    config: ConnectorConfig = None,
    face_count: int = 0,
) -> ConnectorResult:
    """Place a connector along a path-cut seam.

    Path/surface cuts are non-planar: there is no single cut plane, so this
    tries several candidate points along the seam, two normal directions, a
    few lateral offsets and decreasing sizes, preferring cap-native
    deformation and only falling back to boolean subtraction when a boolean
    engine is actually available. Returns a ConnectorResult with the
    (possibly modified) halves and placement metadata, or a skip reason.

    Pure geometry: no Cura dependencies, so it can be unit-tested and run in
    offline replay/profiling.
    """
    if config is None:
        config = ConnectorConfig()

    result_upper = mesh_upper
    result_lower = mesh_lower
    path_points = numpy.asarray(path_points, dtype=numpy.float64)

    if not config.enabled:
        return ConnectorResult(
            upper=result_upper, lower=result_lower,
            skipped_reason="connectors disabled",
        )
    if len(path_points) < 2:
        return ConnectorResult(
            upper=result_upper, lower=result_lower,
            skipped_reason="path too short",
        )

    base_normal = numpy.asarray(
        result_upper.centroid - result_lower.centroid, dtype=numpy.float64
    )
    if numpy.linalg.norm(base_normal) < 1e-9:
        base_normal = numpy.asarray(path_points[-1] - path_points[0], dtype=numpy.float64)
    if numpy.linalg.norm(base_normal) < 1e-9:
        base_normal = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)

    base_diameter = float(config.diameter)
    base_height = float(config.height)
    base_clearance = float(config.clearance)
    size_scales = [1.0, 0.85, 0.7]
    if base_diameter <= 1.2 or base_height <= 1.0:
        size_scales = [1.0]

    # Candidate points along the seam (avoid the exact endpoints).
    n_path = len(path_points)
    if n_path <= 2:
        candidate_indices = [n_path // 2]
    else:
        if face_count > 200000:
            fraction_seeds = [0.50, 0.33, 0.67]
        else:
            fraction_seeds = [0.50, 0.25, 0.75]
        candidate_indices = []
        for frac in fraction_seeds:
            idx = int(round((n_path - 1) * frac))
            idx = max(1, min(n_path - 2, idx))
            if idx not in candidate_indices:
                candidate_indices.append(idx)
        if not candidate_indices:
            candidate_indices = [n_path // 2]

    attempt_failures = []
    total_attempts = 0

    for idx in candidate_indices:
        connector_pos = numpy.asarray(path_points[idx], dtype=numpy.float64)

        prev_idx = max(0, idx - 1)
        next_idx = min(n_path - 1, idx + 1)
        tangent = numpy.asarray(path_points[next_idx] - path_points[prev_idx], dtype=numpy.float64)

        local_normal = numpy.asarray(base_normal, dtype=numpy.float64)
        tan_len = numpy.linalg.norm(tangent)
        if tan_len > 1e-9:
            tangent = tangent / tan_len
            projected = local_normal - numpy.dot(local_normal, tangent) * tangent
            if numpy.linalg.norm(projected) > 1e-9:
                local_normal = projected

        for normal_variant in (local_normal, -local_normal):
            refined_pos = connector_pos
            upper_pos = lower_pos = None
            try:
                upper_pos = find_connector_position(result_upper, connector_pos, normal_variant)
            except Exception:
                pass
            try:
                lower_pos = find_connector_position(result_lower, connector_pos, normal_variant)
            except Exception:
                pass
            if upper_pos is not None and lower_pos is not None:
                refined_pos = 0.5 * (
                    numpy.asarray(upper_pos, dtype=numpy.float64)
                    + numpy.asarray(lower_pos, dtype=numpy.float64)
                )
            elif upper_pos is not None:
                refined_pos = numpy.asarray(upper_pos, dtype=numpy.float64)
            elif lower_pos is not None:
                refined_pos = numpy.asarray(lower_pos, dtype=numpy.float64)

            position_variants = [numpy.asarray(refined_pos, dtype=numpy.float64)]
            side_dir = numpy.cross(normal_variant, tangent)
            side_len = numpy.linalg.norm(side_dir)
            if side_len > 1e-9:
                side_dir = side_dir / side_len
                lateral = max(0.15, base_diameter * 0.15)
                position_variants.append(numpy.asarray(refined_pos, dtype=numpy.float64) + side_dir * lateral)
                position_variants.append(numpy.asarray(refined_pos, dtype=numpy.float64) - side_dir * lateral)

            for scale in size_scales:
                for pos_variant in position_variants:
                    total_attempts += 1
                    attempt_config = ConnectorConfig(
                        enabled=True,
                        diameter=max(0.6, base_diameter * scale),
                        height=max(0.6, base_height * scale),
                        clearance=base_clearance,
                        sides=config.sides,
                    )

                    # Preferred: cap-native deformation (no boolean).
                    if cap_faces_upper and cap_faces_lower:
                        cap_native_result = add_cap_native_connectors(
                            mesh_upper=result_upper,
                            mesh_lower=result_lower,
                            connector_position=pos_variant,
                            connector_normal=normal_variant,
                            cap_faces_upper=cap_faces_upper,
                            cap_faces_lower=cap_faces_lower,
                            config=attempt_config,
                        )
                        if cap_native_result.connectors_added:
                            logger.info(
                                "Path connectors added: peg on %s, hole on %s "
                                "(engine: %s, attempts=%d, scale=%.2f)",
                                cap_native_result.peg_on, cap_native_result.hole_on,
                                cap_native_result.hole_engine, total_attempts, scale,
                            )
                            return ConnectorResult(
                                upper=cap_native_result.upper,
                                lower=cap_native_result.lower,
                                connector_position=pos_variant,
                                peg_on=cap_native_result.peg_on,
                                hole_on=cap_native_result.hole_on,
                                hole_engine=cap_native_result.hole_engine,
                                attempts=total_attempts,
                            )

                    # Boolean fallback only when an engine actually exists.
                    if not boolean_engine_available():
                        attempt_failures.append(
                            f"idx={idx},s={scale:.2f}: cap-native failed, no boolean engine for fallback"
                        )
                        continue

                    connector_result = add_connectors_at_position(
                        mesh_upper=result_upper,
                        mesh_lower=result_lower,
                        connector_position=pos_variant,
                        connector_normal=normal_variant,
                        config=attempt_config,
                    )
                    if connector_result.connectors_added:
                        logger.info(
                            "Path connectors added: peg on %s, hole on %s "
                            "(engine: %s, attempts=%d, scale=%.2f)",
                            connector_result.peg_on, connector_result.hole_on,
                            connector_result.hole_engine, total_attempts, scale,
                        )
                        return ConnectorResult(
                            upper=connector_result.upper,
                            lower=connector_result.lower,
                            connector_position=pos_variant,
                            peg_on=connector_result.peg_on,
                            hole_on=connector_result.hole_on,
                            hole_engine=connector_result.hole_engine,
                            attempts=total_attempts,
                        )

                    attempt_failures.append(
                        f"idx={idx},s={scale:.2f}: {connector_result.skipped_reason}"
                    )

    preview = "; ".join(attempt_failures[:4])
    if len(attempt_failures) > 4:
        preview += f"; ... ({len(attempt_failures)} attempts)"
    logger.warning(
        "Path connectors skipped after %d attempts: %s",
        total_attempts, preview if preview else "no connector candidate succeeded",
    )
    return ConnectorResult(
        upper=result_upper, lower=result_lower,
        skipped_reason=preview or "no connector candidate succeeded",
        attempts=total_attempts,
    )
