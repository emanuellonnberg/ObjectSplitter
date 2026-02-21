import numpy as np
import pytest
import trimesh
from scripts.visual_verification import create_fork
from core.plane_calculator import find_shortest_seam_partition

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

def test_fork_middle_tooth_cut_orthographic_projection():
    """
    Validates that a shortest seam cut across the middle tooth of a fork mesh 
    results in the correct piece shape, using low-res orthographic projection 
    comparisons instead of unstable vertex matching.
    """
    # 1. Generate the test mesh
    mesh = create_fork()
    
    # The middle tooth is roughly near Y=39, with surface normal facing down (-Z)
    click_pt = np.array([0, 39, 0], dtype=float)
    surface_normal = np.array([0, 0, -1], dtype=float)
    
    # 2. Perform the shortest seam distance cut
    set_a, set_b, _, _ = find_shortest_seam_partition(mesh, click_pt, surface_normal=surface_normal)
    
    # Validate the cut actually resulted in two pieces
    assert len(set_a) > 10, "Set A is suspiciously small, cut likely failed."
    assert len(set_b) > 10, "Set B is suspiciously small, cut likely failed."
    
    mask_a = np.zeros(len(mesh.faces), dtype=bool)
    mask_a[set_a] = True
    
    # 3. Create submeshes for the partitioned pieces
    mesh_a = mesh.submesh([mask_a], append=True)
    mesh_b = mesh.submesh([~mask_a], append=True)
    
    # Piece A should be the cut middle tooth. 
    # Use spatial resolution 20 to ensure it gets enough pixels inside unified bounds
    img_a_z = generate_projection(mesh_a, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    print("Piece A projection:")
    print(print_image(img_a_z))
    
    # Verification checks
    assert img_a_z.sum() > 4, "Piece A projection is missing density! Middle tooth not isolated."
    assert np.all(~img_a_z[0, :]), "Piece A projection spills to left edge! Cut may be angled."
    assert np.all(~img_a_z[-1, :]), "Piece A projection spills to right edge! Cut may be angled."
    
    # Verify Piece B (The Fork body missing the middle tooth)
    img_b_z = generate_projection(mesh_b, view_axis=2, resolution=20, custom_bounds=mesh.bounds)
    print("Piece B projection:")
    print(print_image(img_b_z))
    
    assert img_b_z.sum() > 40, "Piece B projection missing density!"
    
    # The top-middle portion of piece B should have a "gap" where Piece A was removed.
    # In the projection array [X, Y], Y goes from 0 (base) to end (tips of teeth).
    assert np.any(~img_b_z[:, -1]), "Piece B has a fully solid top edge! The middle tooth was NOT cut out."
    
    # Gap check: at top, middle area should be empty
    middle_x_start, middle_x_end = 8, 12 # approx center of 20px grid
    top_y = 15 # top quater
    gap_region = img_b_z[middle_x_start:middle_x_end, top_y:]
    assert np.all(~gap_region), "Piece B's projection does not have the expected empty space where the middle tooth was!"
    
    # Base check: base should be solid in the middle
    base_region = img_b_z[:, 4]
    assert base_region.sum() > 4, "Piece B's base projection is too thin or has holes in it!"
