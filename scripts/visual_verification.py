import sys
import os
import argparse
import numpy as np

# Ensure the project root (ObjectSplitter/) is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import trimesh
from core.plane_calculator import (
    horizontal_cut_plane,
    vertical_cut_plane,
    find_smallest_cut_plane,
    find_valley_cut_plane,
    find_shortest_seam_partition
)
from core.mesh_splitter import slice_mesh_with_fallback, split_by_shortest_seam
from core.connectors import add_connectors, ConnectorConfig
from viz.visualizer import generate_html_viewer, generate_report

def create_l_shape():
    """Create an L-shaped mesh."""
    box1 = trimesh.creation.box(extents=[20, 40, 10])
    box2 = trimesh.creation.box(extents=[20, 10, 10])
    box2.apply_translation([20, -15, 0])
    return trimesh.util.concatenate([box1, box2])

def create_dumbbell():
    """Create a dumbbell shape (two spheres connected by a cylinder)."""
    sphere1 = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    sphere1.apply_translation([-20, 0, 0])
    
    sphere2 = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    sphere2.apply_translation([20, 0, 0])
    
    cylinder = trimesh.creation.cylinder(radius=4.0, height=40.0)
    # Rotate cylinder to connect spheres along X axis
    transform = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    cylinder.apply_transform(transform)
    
    return trimesh.util.concatenate([sphere1, cylinder, sphere2])

def create_box_with_hole():
    """Create a box with a hole through it."""
    box = trimesh.creation.box(extents=[30, 30, 10])
    cylinder = trimesh.creation.cylinder(radius=5.0, height=20.0)
    # Use boolean difference to create a hole
    try:
        result = box.difference(cylinder)
        if hasattr(result, 'is_empty') and not result.is_empty:
            return result
    except Exception as e:
        print(f"Warning: Boolean difference failed ({e}), returning simple box")
    return box

def create_fork():
    """Create a unified, perfectly manifold fork shape using 2D polygon extrusion."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("Warning: Shapely not installed. Fork test may fail.")
        return trimesh.creation.box()
        
    # Define a 2D footprint of the fork
    pts = [
        (-5, -19), (-5, 5), (-17, 5), (-17, 41),
        (-11, 41), (-11, 17), (-3, 17), (-3, 41),
        (3, 41), (3, 17), (11, 17), (11, 41),
        (17, 41), (17, 5), (5, 5), (5, -19)
    ]
    poly = Polygon(pts)
    # Pin the triangulation engine: the shortest-seam Dijkstra mincut is
    # sensitive to the cap tessellation, and the orthographic baselines were
    # authored against the `triangle` engine's topology. mapbox_earcut produces
    # a different tessellation that degenerates the seam to a tiny sliver, so we
    # prefer `triangle` to keep the fork (and its golden cuts) deterministic.
    # Fall back to the default engine if `triangle` is unavailable (e.g. no
    # wheel for the running Python) so the viz CLI still works; the
    # tessellation-sensitive tests guard on `triangle` being importable.
    try:
        fork = trimesh.creation.extrude_polygon(poly, height=5.0, engine="triangle")
    except Exception:
        fork = trimesh.creation.extrude_polygon(poly, height=5.0)

    # Subdivide multiple times so the Dijkstra seam can trace smoothly
    # without crossing massive quadrilateral faces
    vertices, faces = trimesh.remesh.subdivide(fork.vertices, fork.faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def verify_split(original_mesh, split_result):
    """
    Verify basic properties of a split result.
    Returns (success, messages)
    """
    if not split_result.success:
        return False, ["Split operation failed"]
        
    messages = []
    success = True
    
    # Check if halves are empty
    if len(split_result.upper.faces) == 0 or len(split_result.lower.faces) == 0:
        success = False
        messages.append("One or both halves are empty")
        
    # Verify connected component count to ensure we aren't shredding the object
    upper_pieces = len(split_result.upper.split()) if len(split_result.upper.faces) > 0 else 0
    lower_pieces = len(split_result.lower.split()) if len(split_result.lower.faces) > 0 else 0
    
    if upper_pieces > 1:
        messages.append(f"Warning: Upper half separated into {upper_pieces} disconnected pieces")
    if lower_pieces > 1:
        messages.append(f"Warning: Lower half separated into {lower_pieces} disconnected pieces")
        
    # Check if original was watertight, the pieces should be watertight (if capped)
    if original_mesh.is_watertight and getattr(split_result, 'capped', False):
        if not split_result.upper.is_watertight:
            messages.append("Warning: Upper half is not watertight")
        if not split_result.lower.is_watertight:
            messages.append("Warning: Lower half is not watertight")
            
    # Collision check: Upper and Lower halves should NOT intersect
    try:
        collide = trimesh.collision.CollisionManager()
        collide.add_object('upper', split_result.upper)
        collide.add_object('lower', split_result.lower)
        is_collision = collide.in_collision_internal()
        if is_collision:
            messages.append("Warning: Upper and Lower halves are intersecting/overlapping")
    except ValueError as e:
        messages.append(f"Warning: CollisionManager check failed ({e})")
            
    return success, messages

def run_tests(output_dir):
    """Run visual tests and generate reports."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    test_cases = [
        {
            "name": "l_shape_vertical_center",
            "mesh": create_l_shape(),
            "cut_fn": lambda m: vertical_cut_plane(m.centroid),
            "description": "Vertical cut through the centroid of an L-shape"
        },
        {
            "name": "dumbbell_valley",
            "mesh": create_dumbbell(),
            "cut_fn": lambda m: find_valley_cut_plane(m, m.centroid).plane,
            "description": "Valley cut to separate dumbbell bells"
        },
        {
            "name": "box_with_hole_horizontal_50",
            "mesh": create_box_with_hole(),
            "cut_fn": lambda m: horizontal_cut_plane(m, 50.0),
            "description": "Horizontal cut at 50% height on a box with a hole"
        },
        {
            "name": "torus_smallest_section",
            "mesh": trimesh.creation.torus(major_radius=15.0, minor_radius=5.0),
            "cut_fn": lambda m: find_smallest_cut_plane(m, m.centroid + np.array([15.0, 0.0, 0.0])).plane,
            "description": "Smallest cut plane on a Torus ring"
        },
        {
            "name": "fork_middle_tooth_seam",
            "mesh": create_fork(),
            "cut_fn": lambda m: find_shortest_seam_partition(m, np.array([0, 25.0, 0]), surface_normal=np.array([0., 0., -1.])),
            "description": "Shortest seam cut to separate the middle tooth of a fork",
            "is_seam": True
        }
    ]
    
    print(f"Running {len(test_cases)} visual verification tests...")
    
    for tc in test_cases:
        print(f"Running: {tc['name']}...")
        mesh = tc["mesh"]
        
        if tc.get("is_seam"):
            # Seam calculation returns (set_a, set_b, best_origin, best_normal)
            set_a, set_b, _, _ = tc["cut_fn"](mesh)
            split_result = split_by_shortest_seam(mesh, set_a, set_b)
            plane_origin = None
            plane_normal = None
        else:
            plane = tc["cut_fn"](mesh)
            plane_origin = plane.origin
            plane_normal = plane.normal
            split_result = slice_mesh_with_fallback(mesh, plane_origin, plane_normal)
        
        # Connectors (Optional, try on Horizontal cut)
        if "horizontal" in tc["name"] and split_result.success and getattr(split_result, 'capped', False):
            config = ConnectorConfig(diameter=3.0, height=2.0)
            cr = add_connectors(split_result.upper, split_result.lower, plane_origin, plane_normal, config)
            # the result of add_connectors is a ConnectorResult, we update the SplitResult to match
            split_result.upper = cr.upper
            split_result.lower = cr.lower
        
        # Verify basic programmatic properties
        is_valid, messages = verify_split(mesh, split_result)
        
        # Generate HTML Viewer
        viewer_filename = f"{tc['name']}.html"
        viewer_path = os.path.join(output_dir, viewer_filename)
        
        if split_result.success:
            generate_html_viewer(
                split_result.upper, split_result.lower, viewer_path,
                title=f"Verification: {tc['name']}",
                metadata={"Description": tc["description"]},
                plane_origin=plane_origin,
                plane_normal=plane_normal
            )
        
        results.append({
            "name": tc["name"],
            "status": "pass" if is_valid else "fail",
            "split_result": split_result,
            "viewer_path": viewer_filename, # Relative path for the index report
            "details": messages
        })

    # Generate Index Report
    report_path = os.path.join(output_dir, "verification_report.html")
    generate_report(results, report_path, title="Visual Verification Report")
    
    # Print Summary
    passes = sum(1 for r in results if r["status"] == "pass")
    print(f"\nVerification Complete: {passes}/{len(results)} passed.")
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run visual verification tests for ObjectSplitter")
    parser.add_argument("--output", default="test_output", help="Directory to output HTML reports")
    args = parser.parse_args()
    
    output_dir = os.path.join(_project_root, args.output)
    run_tests(output_dir)
