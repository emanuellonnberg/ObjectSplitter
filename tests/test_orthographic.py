import importlib.util

import numpy as np
import pytest
import trimesh
from scripts.visual_verification import create_fork

# The legacy sharp-tooth fork (create_fork) pins the `triangle` tessellation
# engine so its Dijkstra-seam cuts are deterministic and match the orthographic
# baselines. When `triangle` has no wheel for the running Python, create_fork
# falls back to a different tessellation that the baselines don't describe, so
# skip the tessellation-sensitive fork tests rather than report false failures.
requires_triangle = pytest.mark.skipif(
    importlib.util.find_spec("triangle") is None,
    reason="needs the 'triangle' engine for a deterministic fork tessellation",
)
from core.plane_calculator import (
    find_shortest_seam_partition,
    find_valley_seam_partition,
    snap_point_to_mesh_surface,
    horizontal_cut_plane,
    vertical_cut_plane,
    find_smallest_cut_plane,
    find_valley_cut_plane,
)
from core.mesh_splitter import slice_mesh_with_fallback, split_by_local_plane, split_by_face_sets
from core.path_cutter import chain_paths, partition_faces_by_path, isolate_region_by_loops


def generate_projection(mesh: trimesh.Trimesh, view_axis: int = 2, resolution: int = 20, custom_bounds=None) -> np.ndarray:
    """
    Generate a boolean 2D mask (low-resolution orthographic projection) of the mesh.
    view_axis: 0 for X, 1 for Y, 2 for Z
    """
    bounds = custom_bounds if custom_bounds is not None else mesh.bounds
    
    # Determine the two axes forming the image plane
    axes = [0, 1, 2]
    axes.remove(view_axis)
    ax1, ax2 = axes
    
    # Create a grid across the bounding box area (adding a small padding)
    pad1 = (bounds[1][ax1] - bounds[0][ax1]) * 0.05
    pad2 = (bounds[1][ax2] - bounds[0][ax2]) * 0.05
    
    linspace1 = np.linspace(bounds[0][ax1] - pad1, bounds[1][ax1] + pad1, resolution)
    linspace2 = np.linspace(bounds[0][ax2] - pad2, bounds[1][ax2] + pad2, resolution)
    
    grid1, grid2 = np.meshgrid(linspace1, linspace2, indexing='ij')
    
    # Setup ray origins and directions
    ray_origins = np.zeros((resolution * resolution, 3))
    ray_origins[:, ax1] = grid1.flatten()
    ray_origins[:, ax2] = grid2.flatten()
    
    # Start rays slightly outside the bounding box corresponding to the viewing axis
    ray_origins[:, view_axis] = bounds[0][view_axis] - max(5.0, (bounds[1][view_axis]-bounds[0][view_axis])*0.1)
    
    ray_directions = np.zeros((resolution * resolution, 3))
    ray_directions[:, view_axis] = 1.0
    
    # Cast rays through the mesh to generate the boolean image mask
    hits = mesh.ray.intersects_any(ray_origins, ray_directions)
    
    # Reshape to 2D image
    image = hits.reshape((resolution, resolution))
    return image

def print_image(image):
    # Print max Y at the top, max X on the right
    res_x, res_y = image.shape
    output = ""
    for y in range(res_y - 1, -1, -1):
        row = ""
        for x in range(res_x):
            row += "XX" if image[x, y] else ".."
        output += row + "\n"
    return output


def _count_true_runs(row: np.ndarray) -> int:
    """Count contiguous True-runs in a 1D boolean row."""
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    transitions = np.diff(padded)
    run_starts = np.where(transitions == 1)[0]
    return int(len(run_starts))


def _filled_indices(mask: np.ndarray, dim: str) -> np.ndarray:
    """Return occupied pixel indices along x or y for a 2D boolean mask."""
    if dim == "x":
        return np.where(mask.any(axis=1))[0]
    if dim == "y":
        return np.where(mask.any(axis=0))[0]
    raise ValueError(f"Unsupported dim: {dim}")


def _assert_axis_separation(mask_a: np.ndarray, mask_b: np.ndarray, axis: str, min_gap: float = 2.0):
    """Assert two projected masks are separated along one image axis."""
    idx_a = _filled_indices(mask_a, axis)
    idx_b = _filled_indices(mask_b, axis)
    assert idx_a.size > 0 and idx_b.size > 0, "Expected both masks to have occupied pixels."
    center_gap = abs(float(idx_a.mean()) - float(idx_b.mean()))
    assert center_gap >= min_gap, f"Expected separation along {axis}, got center gap={center_gap:.2f}"


def _assert_axis_coverage_similar(mask_a: np.ndarray, mask_b: np.ndarray, axis: str, max_span_delta: int = 3):
    """Assert projected masks have similar footprint span on one axis."""
    idx_a = _filled_indices(mask_a, axis)
    idx_b = _filled_indices(mask_b, axis)
    assert idx_a.size > 0 and idx_b.size > 0, "Expected both masks to have occupied pixels."
    span_a = int(idx_a.max() - idx_a.min())
    span_b = int(idx_b.max() - idx_b.min())
    assert abs(span_a - span_b) <= max_span_delta, (
        f"Expected similar {axis}-span, got {span_a} vs {span_b}"
    )


def _true_runs_1d(row: np.ndarray):
    """Return contiguous True runs as (start, end) index pairs."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(row.astype(bool)):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(row) - 1))
    return runs


def _central_gap_centers(mask: np.ndarray, rows_from_top: int = 6):
    """
    Return x-centers of the interior gap nearest image center for top rows with
    multiple occupied runs. Rows with <2 runs are skipped.
    """
    centers = []
    width = mask.shape[0]
    cx = (width - 1) / 2.0
    y_start = mask.shape[1] - 2
    y_end = max(-1, y_start - rows_from_top)
    for y in range(y_start, y_end, -1):
        row = mask[:, y]
        runs = _true_runs_1d(row)
        if len(runs) < 2:
            continue
        gaps = []
        for i in range(len(runs) - 1):
            gap_start = runs[i][1] + 1
            gap_end = runs[i + 1][0] - 1
            if gap_start <= gap_end:
                gaps.append((gap_start, gap_end))
        if not gaps:
            continue
        # Pick interior gap closest to image center.
        gap = min(gaps, key=lambda g: abs(((g[0] + g[1]) / 2.0) - cx))
        centers.append((gap[0] + gap[1]) / 2.0)
    return centers


def _assert_center_gap_axis_aligned(mask: np.ndarray, max_drift_ratio: float = 0.05, rows_from_top: int = 6):
    """
    Assert the central gap remains axis-aligned (stable x-position) near the top.
    max_drift_ratio is drift as fraction of projection width.
    """
    centers = _central_gap_centers(mask, rows_from_top=rows_from_top)
    assert len(centers) >= 3, "Need at least 3 top rows with a multi-column gap."
    drift_px = float(max(centers) - min(centers))
    drift_ratio = drift_px / float(mask.shape[0])
    assert drift_ratio <= max_drift_ratio, (
        f"Central gap drift too high: {drift_ratio:.2%} "
        f"(drift_px={drift_px:.2f}, width={mask.shape[0]})."
    )


def _rows_to_mask(rows_top_to_bottom):
    """Convert top-to-bottom '0'/'1' row strings into an [x, y] boolean mask."""
    height = len(rows_top_to_bottom)
    width = len(rows_top_to_bottom[0]) if height else 0
    mask = np.zeros((width, height), dtype=bool)
    for row_idx, row in enumerate(rows_top_to_bottom):
        y = height - 1 - row_idx
        for x, value in enumerate(row):
            mask[x, y] = (value == "1")
    return mask


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union for two boolean masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _create_rounded_fork():
    """
    Create a rounded/cylindrical-style fork footprint by unioning buffered
    rectangles, then extruding. This shape approximates smooth tooth throats.
    """
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        # Reuse the legacy fork as a soft fallback when shapely isn't available.
        return create_fork()

    def _capsule(x0: float, y0: float, x1: float, y1: float, radius: float = 1.8):
        return box(x0, y0, x1, y1).buffer(radius, join_style=1)

    poly = unary_union([
        _capsule(-4.0, -19.0, 4.0, 5.0),        # handle
        _capsule(-16.0, 5.0, 16.0, 17.0),       # base
        _capsule(-16.0, 17.0, -10.0, 41.0),     # left tooth
        _capsule(-3.0, 17.0, 3.0, 41.0),        # middle tooth
        _capsule(10.0, 17.0, 16.0, 41.0),       # right tooth
    ]).buffer(0)

    fork = trimesh.creation.extrude_polygon(poly, height=5.0)
    vertices, faces = trimesh.remesh.subdivide(fork.vertices, fork.faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _create_double_bridge_plate():
    """Create a closed solid with an upper plate attached by two narrow bridges."""
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        pytest.skip("shapely not available")

    poly = unary_union([
        box(-30.0, -20.0, 30.0, 12.0),
        box(-18.0, 24.0, 18.0, 40.0),
        box(-14.0, 12.0, -6.0, 24.0),
        box(6.0, 12.0, 14.0, 24.0),
    ]).buffer(0)

    mesh = trimesh.creation.extrude_polygon(poly, height=6.0)
    vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


_FORK_SMALLEST_BASELINE_SMALL_ROWS = [
    "00000000000000000000",
    "00000000111100000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
    "00000000000000000000",
]

_FORK_SMALLEST_BASELINE_BIG_ROWS = [
    "00000000000000000000",
    "01110000111100001110",
    "01110000111100001110",
    "01110000111100001110",
    "01110000111100001110",
    "01110000111100001110",
    "01110000111100001110",
    "01110000111100001110",
    "01111111111111111110",
    "01111111111111111110",
    "01111111111111111110",
    "01111111111111111110",
    "00000001111110000000",
    "00000001111110000000",
    "00000001111110000000",
    "00000001111110000000",
    "00000001111110000000",
    "00000001111110000000",
    "00000001111110000000",
    "00000000000000000000",
]


@pytest.fixture(scope="module")
def fork_split_meshes():
    """Return (original_mesh, middle_tooth_mesh, fork_body_mesh)."""
    mesh = create_fork()
    click_pt = np.array([0, 39, 0], dtype=float)
    surface_normal = np.array([0, 0, -1], dtype=float)
    set_a, set_b, _, _ = find_shortest_seam_partition(mesh, click_pt, surface_normal=surface_normal)

    assert len(set_a) > 10, "Set A is suspiciously small, cut likely failed."
    assert len(set_b) > 10, "Set B is suspiciously small, cut likely failed."

    mask_a = np.zeros(len(mesh.faces), dtype=bool)
    mask_a[set_a] = True
    mesh_a = mesh.submesh([mask_a], append=True)
    mesh_b = mesh.submesh([~mask_a], append=True)
    return mesh, mesh_a, mesh_b


@requires_triangle
def test_fork_middle_tooth_cut_orthographic_projection(fork_split_meshes):
    """
    Validates that a shortest seam cut across the middle tooth of a fork mesh 
    results in the correct piece shape, using low-res orthographic projection 
    comparisons instead of unstable vertex matching.
    """
    mesh, mesh_a, mesh_b = fork_split_meshes
    
    # Piece A should be the cut middle tooth. 
    # Use spatial resolution 20 to ensure it gets enough pixels inside unified bounds
    img_a_z = generate_projection(mesh_a, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    # Verification checks
    assert img_a_z.sum() > 4, f"Piece A projection is missing density!\n{print_image(img_a_z)}"
    assert np.all(~img_a_z[0, :]), f"Piece A projection spills to left edge!\n{print_image(img_a_z)}"
    assert np.all(~img_a_z[-1, :]), f"Piece A projection spills to right edge!\n{print_image(img_a_z)}"
    
    # Verify Piece B (The Fork body missing the middle tooth)
    img_b_z = generate_projection(mesh_b, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    assert img_b_z.sum() > 40, f"Piece B projection missing density!\n{print_image(img_b_z)}"
    
    # The top-middle portion of piece B should have a "gap" where Piece A was removed.
    # In the projection array [X, Y], Y goes from 0 (base) to end (tips of teeth).
    assert np.any(~img_b_z[:, -1]), f"Piece B has a fully solid top edge!\n{print_image(img_b_z)}"
    
    # Gap check: at the very top row, center area should be empty.
    middle_x_start, middle_x_end = 8, 12  # approx center of 20px grid
    top_center_row = img_b_z[middle_x_start:middle_x_end, -1]
    assert np.all(~top_center_row), f"Piece B top-center gap missing.\n{print_image(img_b_z)}"
    
    # Base check: base should be solid in the middle
    base_region = img_b_z[:, 4]
    assert base_region.sum() > 4, "Piece B's base projection is too thin or has holes in it!"


@pytest.mark.parametrize("resolution", [16, 20, 24])
@requires_triangle
def test_fork_projection_signature_stable_across_resolutions(fork_split_meshes, resolution):
    """Top row signature should remain stable: tooth-only piece=1 run, body piece=2 side runs."""
    mesh, mesh_a, mesh_b = fork_split_meshes

    img_a_z = generate_projection(mesh_a, view_axis=2, resolution=resolution, custom_bounds=mesh.bounds)
    img_b_z = generate_projection(mesh_b, view_axis=2, resolution=resolution, custom_bounds=mesh.bounds)

    rows_a = np.where(img_a_z.any(axis=0))[0]
    rows_b = np.where(img_b_z.any(axis=0))[0]
    assert rows_a.size > 0 and rows_b.size > 0, "Expected both projections to have filled pixels."

    top_row = int(min(rows_a.max(), rows_b.max()))
    row_a = img_a_z[:, top_row]
    row_b = img_b_z[:, top_row]

    assert _count_true_runs(row_a) == 1, f"Piece A expected one top run.\n{print_image(img_a_z)}"
    assert _count_true_runs(row_b) >= 2, f"Piece B expected split top runs.\n{print_image(img_b_z)}"

    center = resolution // 2
    half_width = max(1, resolution // 10)
    center_slice = slice(max(0, center - half_width), min(resolution, center + half_width + 1))

    assert row_a[center_slice].any(), "Piece A should occupy the center at the top."
    assert not row_b[center_slice].any(), "Piece B should have a center gap at the top."


def test_horizontal_cut_orthographic_signature(cube_mesh):
    """Horizontal center cut should separate upper/lower halves along Y in top view."""
    plane = horizontal_cut_plane(cube_mesh, 50.0)
    result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
    assert result.success

    img_upper = generate_projection(result.upper, view_axis=2, resolution=24, custom_bounds=cube_mesh.bounds)
    img_lower = generate_projection(result.lower, view_axis=2, resolution=24, custom_bounds=cube_mesh.bounds)

    assert img_upper.sum() > 20
    assert img_lower.sum() > 20
    _assert_axis_separation(img_upper, img_lower, axis="y", min_gap=2.0)
    _assert_axis_coverage_similar(img_upper, img_lower, axis="x", max_span_delta=2)


def test_vertical_cut_orthographic_signature(cube_mesh):
    """Vertical center cut should separate left/right halves along X in top view."""
    plane = vertical_cut_plane(cube_mesh.centroid)
    result = slice_mesh_with_fallback(cube_mesh, plane.origin, plane.normal)
    assert result.success

    img_upper = generate_projection(result.upper, view_axis=2, resolution=24, custom_bounds=cube_mesh.bounds)
    img_lower = generate_projection(result.lower, view_axis=2, resolution=24, custom_bounds=cube_mesh.bounds)

    assert img_upper.sum() > 20
    assert img_lower.sum() > 20
    _assert_axis_separation(img_upper, img_lower, axis="x", min_gap=2.0)
    _assert_axis_coverage_similar(img_upper, img_lower, axis="y", max_span_delta=2)


def test_smallest_cut_orthographic_signature(cylinder_mesh):
    """
    Smallest-cut mode on a Z cylinder should split into two parts separated along Z
    when viewed from Y (XZ projection).
    """
    search = find_smallest_cut_plane(cylinder_mesh, cylinder_mesh.centroid, search_resolution=8)
    result = slice_mesh_with_fallback(cylinder_mesh, search.plane.origin, search.plane.normal)
    assert result.success

    img_upper = generate_projection(result.upper, view_axis=1, resolution=28, custom_bounds=cylinder_mesh.bounds)
    img_lower = generate_projection(result.lower, view_axis=1, resolution=28, custom_bounds=cylinder_mesh.bounds)

    assert img_upper.sum() > 30
    assert img_lower.sum() > 30
    _assert_axis_separation(img_upper, img_lower, axis="y", min_gap=2.0)  # y-axis here is Z in XZ view
    _assert_axis_coverage_similar(img_upper, img_lower, axis="x", max_span_delta=3)


def test_valley_cut_orthographic_signature(cylinder_mesh):
    """
    Valley mode on a cylinder should also produce two halves with clear Z-separation
    in an XZ orthographic projection.
    """
    click = cylinder_mesh.centroid + np.array([0.0, 3.0, 0.0])
    search = find_valley_cut_plane(cylinder_mesh, click, search_resolution=8)
    result = slice_mesh_with_fallback(cylinder_mesh, search.plane.origin, search.plane.normal)
    assert result.success

    img_upper = generate_projection(result.upper, view_axis=1, resolution=28, custom_bounds=cylinder_mesh.bounds)
    img_lower = generate_projection(result.lower, view_axis=1, resolution=28, custom_bounds=cylinder_mesh.bounds)

    assert img_upper.sum() > 30
    assert img_lower.sum() > 30
    _assert_axis_separation(img_upper, img_lower, axis="y", min_gap=2.0)  # y-axis here is Z in XZ view
    _assert_axis_coverage_similar(img_upper, img_lower, axis="x", max_span_delta=3)


@requires_triangle
def test_fork_smallest_cut_orthographic_signature():
    """
    Smallest mode on a fork should avoid tiny surface-graze slivers and produce
    a meaningful separation near the clicked tooth region.
    """
    mesh = create_fork()
    click_pt = np.array([0, 39, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    search = find_smallest_cut_plane(
        mesh,
        snap_pt,
        search_resolution=18,
        surface_normal=click_surface_normal,
    )

    candidate_normals = [n for _, n in search.top_candidates] if search.top_candidates else [search.plane.normal]
    result = split_by_local_plane(mesh, search.plane.origin, candidate_normals, click_face_id)
    assert result.success

    small_faces = min(len(result.upper.faces), len(result.lower.faces))
    ratio = small_faces / len(mesh.faces)
    assert ratio >= 0.02, (
        f"Smallest fork cut produced a tiny sliver ({small_faces}/{len(mesh.faces)} faces, {ratio:.2%})."
    )

    img_a = generate_projection(result.upper, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    img_b = generate_projection(result.lower, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    # The smaller piece should have visible body and the larger piece should show a top-center gap.
    if img_a.sum() > img_b.sum():
        img_a, img_b = img_b, img_a

    baseline_small = _rows_to_mask(_FORK_SMALLEST_BASELINE_SMALL_ROWS)
    baseline_big = _rows_to_mask(_FORK_SMALLEST_BASELINE_BIG_ROWS)
    iou_small = _mask_iou(img_a, baseline_small)
    iou_big = _mask_iou(img_b, baseline_big)
    assert iou_small >= 0.60, f"Small-piece IoU too low: {iou_small:.3f}"
    assert iou_big >= 0.90, f"Remainder IoU too low: {iou_big:.3f}"

    assert img_a.sum() >= 4, f"Smallest-mode separated piece is too sparse.\n{print_image(img_a)}"
    center_slice = slice(8, 12)
    assert not img_b[center_slice, -1].any(), f"Expected top-center gap in remainder.\n{print_image(img_b)}"


@requires_triangle
def test_fork_valley_cut_orthographic_signature():
    """
    Legacy planar valley mode on the fork should remain stable against
    baseline masks and avoid degenerate tiny slivers.
    """
    mesh = create_fork()
    click_pt = np.array([0, 39, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    search = find_valley_cut_plane(
        mesh,
        snap_pt,
        search_resolution=18,
        surface_normal=click_surface_normal,
    )

    candidate_normals = [n for _, n in search.top_candidates] if search.top_candidates else [search.plane.normal]
    result = split_by_local_plane(mesh, search.plane.origin, candidate_normals, click_face_id)
    assert result.success

    small_faces = min(len(result.upper.faces), len(result.lower.faces))
    ratio = small_faces / len(mesh.faces)
    assert ratio >= 0.01, (
        f"Valley fork cut produced a tiny sliver ({small_faces}/{len(mesh.faces)} faces, {ratio:.2%})."
    )

    img_a = generate_projection(result.upper, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    img_b = generate_projection(result.lower, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    if img_a.sum() > img_b.sum():
        img_a, img_b = img_b, img_a

    # Valley cuts ACROSS the clicked middle tooth. With grazing planes now
    # penalized it may cut deeper than the smallest mode, so check structurally
    # (the separated piece covers the middle tooth) rather than matching the
    # smallest baseline mask.
    baseline_big = _rows_to_mask(_FORK_SMALLEST_BASELINE_BIG_ROWS)
    assert _mask_iou(img_b, baseline_big) >= 0.90, "Valley remainder changed shape unexpectedly."

    assert img_a.sum() >= 4, f"Valley separated piece is too sparse.\n{print_image(img_a)}"
    center_slice = slice(8, 12)
    assert img_a[center_slice, :].any(), (
        f"Valley piece should cover the clicked middle tooth.\n{print_image(img_a)}")
    assert not img_b[center_slice, -1].any(), f"Expected top-center gap in valley remainder.\n{print_image(img_b)}"


@requires_triangle
def test_fork_valley_seam_cut_orthographic_signature():
    """
    Valley seam mode should improve over planar valley on fork geometry by
    removing a larger clicked-feature partition and producing a deeper cutout.
    """
    mesh = create_fork()
    click_pt = np.array([0, 39, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    set_a, set_b, _, _ = find_valley_seam_partition(
        mesh,
        snap_pt,
        source_face_hint=click_face_id,
        surface_normal=click_surface_normal,
    )
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="valley_seam",
        attempt_hole_fill=False,
    )
    assert result.success

    small_faces = min(len(result.upper.faces), len(result.lower.faces))
    ratio = small_faces / len(mesh.faces)
    assert ratio >= 0.015, (
        f"Valley seam fork cut did not remove enough of the clicked feature "
        f"({small_faces}/{len(mesh.faces)} faces, {ratio:.2%})."
    )

    valley_search = find_valley_cut_plane(
        mesh,
        snap_pt,
        search_resolution=18,
        surface_normal=click_surface_normal,
    )
    valley_candidate_normals = (
        [n for _, n in valley_search.top_candidates]
        if valley_search.top_candidates
        else [valley_search.plane.normal]
    )
    valley_result = split_by_local_plane(
        mesh,
        valley_search.plane.origin,
        valley_candidate_normals,
        click_face_id,
    )
    assert valley_result.success
    valley_ratio = min(len(valley_result.upper.faces), len(valley_result.lower.faces)) / len(mesh.faces)
    assert ratio >= valley_ratio + 0.003, (
        f"Valley seam should improve over planar valley. "
        f"valley_seam={ratio:.2%}, valley={valley_ratio:.2%}"
    )

    img_a = generate_projection(result.upper, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    img_b = generate_projection(result.lower, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    if img_a.sum() > img_b.sum():
        img_a, img_b = img_b, img_a

    small_x = np.where(img_a.any(axis=1))[0]
    small_y = np.where(img_a.any(axis=0))[0]
    assert img_a.sum() >= 4, f"Valley seam separated piece is too sparse.\n{print_image(img_a)}"
    assert small_x.size >= 2, f"Valley seam removed piece too narrow.\n{print_image(img_a)}"
    assert small_y.size >= 2, f"Valley seam removed piece not deep enough.\n{print_image(img_a)}"

    center_slice = slice(8, 12)
    assert not img_b[center_slice, -1].any(), (
        f"Expected top-center gap in valley seam remainder.\n{print_image(img_b)}"
    )


def test_fork_valley_seam_rounded_throat_orthographic_signature():
    """
    Rounded/cylindrical fork throat clicks should avoid tiny seam slivers and
    produce a focused clicked-feature partition.
    """
    mesh = _create_rounded_fork()
    click_pt = np.array([0, 22, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    set_a, set_b, src, _ = find_valley_seam_partition(
        mesh,
        snap_pt,
        source_face_hint=click_face_id,
        surface_normal=click_surface_normal,
    )
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="valley_seam",
        attempt_hole_fill=False,
    )
    assert result.success

    small_set = set_a if len(set_a) <= len(set_b) else set_b
    assert src in small_set, "Clicked source face should remain in the smaller partition."

    if len(result.upper.faces) <= len(result.lower.faces):
        small_mesh, big_mesh = result.upper, result.lower
    else:
        small_mesh, big_mesh = result.lower, result.upper

    ratio = len(small_mesh.faces) / len(mesh.faces)
    assert 0.05 <= ratio <= 0.35, (
        f"Rounded fork valley seam should be a focused partition, got {ratio:.2%} "
        f"({len(small_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_small = generate_projection(small_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    img_big = generate_projection(big_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)

    assert img_small.sum() >= 20, f"Rounded seam piece too sparse.\n{print_image(img_small)}"
    x_idx = np.where(img_small.any(axis=1))[0]
    y_idx = np.where(img_small.any(axis=0))[0]
    assert x_idx.size >= 4, f"Rounded seam piece too narrow.\n{print_image(img_small)}"
    assert y_idx.size >= 8, f"Rounded seam piece too shallow.\n{print_image(img_small)}"

    # The remaining fork should still project all three upper tooth columns
    # for a throat-local seam, rather than collapsing into a half-object split.
    top_band = [img_big[:, -2], img_big[:, -3], img_big[:, -4]]
    max_runs = max(_count_true_runs(row) for row in top_band)
    assert max_runs >= 3, f"Rounded seam remainder lost top tooth structure.\n{print_image(img_big)}"
    _assert_center_gap_axis_aligned(img_big, max_drift_ratio=0.05, rows_from_top=6)


def test_fork_valley_seam_rounded_throat_with_sdf_bias():
    """
    Experimental SDF bias should preserve rounded-throat valley-seam behavior:
    focused partition near click and stable top-center gap alignment.
    """
    mesh = _create_rounded_fork()
    click_pt = np.array([0, 22, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    set_a, set_b, src, _ = find_valley_seam_partition(
        mesh,
        snap_pt,
        source_face_hint=click_face_id,
        surface_normal=click_surface_normal,
        use_sdf_bias=True,
    )
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="valley_seam",
        attempt_hole_fill=False,
    )
    assert result.success

    small_set = set_a if len(set_a) <= len(set_b) else set_b
    assert src in small_set, "SDF-biased seam should keep source face in smaller partition."

    if len(result.upper.faces) <= len(result.lower.faces):
        small_mesh, big_mesh = result.upper, result.lower
    else:
        small_mesh, big_mesh = result.lower, result.upper

    ratio = len(small_mesh.faces) / len(mesh.faces)
    assert 0.04 <= ratio <= 0.40, (
        f"SDF-biased rounded fork seam should stay focused, got {ratio:.2%} "
        f"({len(small_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_small = generate_projection(small_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    img_big = generate_projection(big_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)

    assert img_small.sum() >= 16, f"SDF-biased seam piece too sparse.\n{print_image(img_small)}"
    top_band = [img_big[:, -2], img_big[:, -3], img_big[:, -4]]
    max_runs = max(_count_true_runs(row) for row in top_band)
    assert max_runs >= 3, f"SDF-biased seam remainder lost top tooth structure.\n{print_image(img_big)}"
    _assert_center_gap_axis_aligned(img_big, max_drift_ratio=0.05, rows_from_top=6)


def test_fork_shortest_rounded_throat_orthographic_signature():
    """
    Shortest mode on the rounded fork should produce a focused non-planar split
    around the clicked throat region (not a half-object split).
    """
    mesh = _create_rounded_fork()
    click_pt = np.array([0, 30, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    set_a, set_b, src, _ = find_shortest_seam_partition(
        mesh,
        snap_pt,
        source_face_hint=click_face_id,
        surface_normal=click_surface_normal,
    )
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="shortest_seam",
        attempt_hole_fill=False,
    )
    assert result.success

    small_set = set_a if len(set_a) <= len(set_b) else set_b
    assert src in small_set, "Shortest-mode source face should stay in the smaller partition."

    if len(result.upper.faces) <= len(result.lower.faces):
        small_mesh, big_mesh = result.upper, result.lower
    else:
        small_mesh, big_mesh = result.lower, result.upper

    ratio = len(small_mesh.faces) / len(mesh.faces)
    assert 0.02 <= ratio <= 0.25, (
        f"Rounded fork shortest cut should stay focused, got {ratio:.2%} "
        f"({len(small_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_small = generate_projection(small_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    img_big = generate_projection(big_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    assert img_small.sum() >= 12, f"Rounded shortest piece too sparse.\n{print_image(img_small)}"
    x_idx = np.where(img_small.any(axis=1))[0]
    y_idx = np.where(img_small.any(axis=0))[0]
    assert x_idx.size >= 4, f"Rounded shortest piece too narrow.\n{print_image(img_small)}"
    assert y_idx.size >= 4, f"Rounded shortest piece too shallow.\n{print_image(img_small)}"

    top_band = [img_big[:, -2], img_big[:, -3], img_big[:, -4]]
    max_runs = max(_count_true_runs(row) for row in top_band)
    assert max_runs >= 3, f"Rounded shortest remainder lost top tooth structure.\n{print_image(img_big)}"
    _assert_center_gap_axis_aligned(img_big, max_drift_ratio=0.05, rows_from_top=6)


def test_fork_radial_rounded_throat_orthographic_signature():
    """
    Radial mode should preserve the same rounded-throat non-planar behavior as
    the radial backend path used by the tool.
    """
    mesh = _create_rounded_fork()
    click_pt = np.array([0, 30, 0], dtype=float)
    snap_pt, click_face_id = snap_point_to_mesh_surface(mesh, click_pt)
    assert click_face_id is not None

    click_surface_normal = np.array(mesh.face_normals[click_face_id], dtype=np.float64)
    set_a, set_b, src, _ = find_shortest_seam_partition(
        mesh,
        snap_pt,
        source_face_hint=click_face_id,
        surface_normal=click_surface_normal,
    )
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="radial",
        attempt_hole_fill=False,
    )
    assert result.success

    small_set = set_a if len(set_a) <= len(set_b) else set_b
    assert src in small_set, "Radial-mode source face should stay in the smaller partition."

    if len(result.upper.faces) <= len(result.lower.faces):
        small_mesh, big_mesh = result.upper, result.lower
    else:
        small_mesh, big_mesh = result.lower, result.upper

    ratio = len(small_mesh.faces) / len(mesh.faces)
    assert 0.02 <= ratio <= 0.25, (
        f"Rounded fork radial cut should stay focused, got {ratio:.2%} "
        f"({len(small_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_small = generate_projection(small_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    img_big = generate_projection(big_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    assert img_small.sum() >= 12, f"Rounded radial piece too sparse.\n{print_image(img_small)}"

    top_band = [img_big[:, -2], img_big[:, -3], img_big[:, -4]]
    max_runs = max(_count_true_runs(row) for row in top_band)
    assert max_runs >= 3, f"Rounded radial remainder lost top tooth structure.\n{print_image(img_big)}"
    _assert_center_gap_axis_aligned(img_big, max_drift_ratio=0.05, rows_from_top=6)


def test_fork_path_rounded_throat_orthographic_signature():
    """
    Path mode with a closed 3D throat loop should remove a meaningful
    middle-throat region on the rounded fork.
    """
    mesh = _create_rounded_fork()
    waypoints = [
        # Closed loop around the middle-throat side ring (front/back + left/right)
        np.array([-4.8, 18.6, 0.0], dtype=float),
        np.array([-4.8, 18.6, 5.0], dtype=float),
        np.array([4.8, 18.6, 5.0], dtype=float),
        np.array([4.8, 18.6, 0.0], dtype=float),
        np.array([-4.8, 18.6, 0.0], dtype=float),
    ]

    vertex_path = chain_paths(mesh, waypoints)
    assert len(vertex_path) >= 12, f"Expected a non-trivial geodesic path, got {len(vertex_path)} vertices."

    set_a, set_b = partition_faces_by_path(mesh, vertex_path)
    result = split_by_face_sets(
        mesh,
        set_a,
        set_b,
        strategy_name="path_cut",
        attempt_hole_fill=False,
    )
    assert result.success

    if len(result.upper.faces) <= len(result.lower.faces):
        small_mesh, big_mesh = result.upper, result.lower
    else:
        small_mesh, big_mesh = result.lower, result.upper

    ratio = len(small_mesh.faces) / len(mesh.faces)
    assert 0.12 <= ratio <= 0.30, (
        f"Rounded fork path cut should remove a meaningful throat region, got {ratio:.2%} "
        f"({len(small_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_small = generate_projection(small_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    img_big = generate_projection(big_mesh, view_axis=2, resolution=24, custom_bounds=mesh.bounds)
    assert img_small.sum() >= 20, f"Rounded path piece too sparse.\n{print_image(img_small)}"
    x_idx = np.where(img_small.any(axis=1))[0]
    y_idx = np.where(img_small.any(axis=0))[0]
    assert x_idx.size >= 4, f"Rounded path piece too narrow.\n{print_image(img_small)}"
    assert y_idx.size >= 6, f"Rounded path piece too shallow.\n{print_image(img_small)}"

    top_band = [img_big[:, -3], img_big[:, -4], img_big[:, -5]]
    max_runs = max(_count_true_runs(row) for row in top_band)
    assert max_runs == 2, f"Rounded path remainder should show two top tooth columns.\n{print_image(img_big)}"
    _assert_center_gap_axis_aligned(img_big, max_drift_ratio=0.05, rows_from_top=6)


def test_path_isolate_double_bridge_orthographic_signature():
    """
    Multi-loop path isolate should extract the upper plate from a part with two
    separate bridge connections, leaving the base in the remainder.
    """
    mesh = _create_double_bridge_plate()
    left_loop = [
        np.array([-14.5, 18.0, 0.0], dtype=float),
        np.array([-14.5, 18.0, 6.0], dtype=float),
        np.array([-5.5, 18.0, 6.0], dtype=float),
        np.array([-5.5, 18.0, 0.0], dtype=float),
        np.array([-14.5, 18.0, 0.0], dtype=float),
    ]
    right_loop = [
        np.array([5.5, 18.0, 0.0], dtype=float),
        np.array([5.5, 18.0, 6.0], dtype=float),
        np.array([14.5, 18.0, 6.0], dtype=float),
        np.array([14.5, 18.0, 0.0], dtype=float),
        np.array([5.5, 18.0, 0.0], dtype=float),
    ]

    loop_vertex_paths = [chain_paths(mesh, left_loop), chain_paths(mesh, right_loop)]
    for vertex_path in loop_vertex_paths:
        assert len(vertex_path) >= 12
        assert vertex_path[0] == vertex_path[-1]

    _, target_face_id = snap_point_to_mesh_surface(mesh, np.array([0.0, 32.0, 3.0], dtype=float))
    assert target_face_id is not None

    extracted_faces, remainder_faces = isolate_region_by_loops(mesh, loop_vertex_paths, target_face_id)
    result = split_by_face_sets(
        mesh,
        extracted_faces,
        remainder_faces,
        strategy_name="path_isolate",
        attempt_hole_fill=False,
    )
    assert result.success

    extracted_mesh = result.upper
    remainder_mesh = result.lower
    ratio = len(extracted_mesh.faces) / len(mesh.faces)
    assert 0.45 <= ratio <= 0.60, (
        f"Expected a meaningful isolated upper plate, got {ratio:.2%} "
        f"({len(extracted_mesh.faces)}/{len(mesh.faces)} faces)."
    )

    img_extracted = generate_projection(extracted_mesh, view_axis=2, resolution=28, custom_bounds=mesh.bounds)
    img_remainder = generate_projection(remainder_mesh, view_axis=2, resolution=28, custom_bounds=mesh.bounds)
    assert img_extracted.sum() > 20
    assert img_remainder.sum() > 40

    extracted_y = np.where(img_extracted.any(axis=0))[0]
    remainder_y = np.where(img_remainder.any(axis=0))[0]
    assert extracted_y.size > 0 and remainder_y.size > 0
    assert extracted_y.mean() > remainder_y.mean() + 4.0, (
        f"Expected isolated region to sit above the remainder.\n"
        f"Extracted:\n{print_image(img_extracted)}\nRemainder:\n{print_image(img_remainder)}"
    )

    top_band_sum = img_extracted[:, -6:].sum()
    bottom_band_sum = img_extracted[:, :6].sum()
    assert top_band_sum > bottom_band_sum * 2, (
        f"Expected isolated region to occupy the upper plate band.\n{print_image(img_extracted)}"
    )
