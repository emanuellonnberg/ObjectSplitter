# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
Visualization module for ObjectSplitter test results.

Generates:
  - Static PNG images of split results (via matplotlib, if available)
  - Interactive HTML pages with 3D views (via three.js / embedded STL viewer)
  - Summary HTML report across multiple test cases
"""

import json
import os
import base64
import logging
from pathlib import Path
from typing import Optional

import numpy

logger = logging.getLogger("objectsplitter.visualizer")

try:
    import trimesh
except ImportError:
    trimesh = None


def render_split_image(
    upper: "trimesh.Trimesh",
    lower: "trimesh.Trimesh",
    output_path: str,
    title: str = "",
    plane_origin: Optional[numpy.ndarray] = None,
    plane_normal: Optional[numpy.ndarray] = None,
    separation: float = 5.0
) -> bool:
    """
    Render a static PNG image showing the two split halves separated.

    Uses matplotlib for 3D plotting if available. Falls back to
    trimesh's built-in scene rendering.

    Args:
        upper: Upper mesh half.
        lower: Lower mesh half.
        output_path: File path for the output PNG.
        title: Optional title text for the image.
        plane_origin: Optional cut plane origin for annotation.
        plane_normal: Optional cut plane normal for annotation.
        separation: Distance to separate the two halves for visualization.

    Returns:
        True if image was generated successfully.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        logger.warning("matplotlib not available, skipping image render")
        return False

    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Compute separation direction (along the cut normal, or default Y)
        if plane_normal is not None:
            sep_dir = plane_normal / numpy.linalg.norm(plane_normal)
        else:
            sep_dir = numpy.array([0, 1, 0])

        offset = sep_dir * separation / 2.0

        # Plot upper mesh (shifted up)
        _plot_mesh(ax, upper, offset, color='#4A90D9', alpha=0.7, label='Upper')
        # Plot lower mesh (shifted down)
        _plot_mesh(ax, lower, -offset, color='#D94A4A', alpha=0.7, label='Lower')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            ax.set_title(title, fontsize=14)
        ax.legend()

        # Equal aspect ratio
        _set_equal_axes(ax, upper, lower)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info("Rendered split image: %s", output_path)
        return True

    except Exception as e:
        logger.error("Failed to render image: %s", e)
        return False


def _plot_mesh(ax, mesh, offset, color, alpha, label):
    """Plot a trimesh on a matplotlib 3D axis with a position offset."""
    vertices = mesh.vertices + offset
    faces = mesh.faces

    # Subsample faces if mesh is very large (matplotlib can't handle huge meshes)
    max_faces = 5000
    if len(faces) > max_faces:
        indices = numpy.random.choice(len(faces), max_faces, replace=False)
        faces = faces[indices]

    polygons = vertices[faces]
    collection = Poly3DCollection(polygons, alpha=alpha, linewidths=0.1, edgecolors='gray')
    collection.set_facecolor(color)
    ax.add_collection3d(collection)


def _set_equal_axes(ax, *meshes):
    """Set equal axis scaling on a 3D matplotlib axis."""
    all_verts = numpy.vstack([m.vertices for m in meshes if m is not None])
    max_range = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2.0
    mid = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def _mesh_to_stl_data_uri(mesh: "trimesh.Trimesh") -> str:
    """Convert a trimesh to a base64-encoded STL data URI for embedding in HTML."""
    stl_bytes = mesh.export(file_type='stl')
    b64 = base64.b64encode(stl_bytes).decode('ascii')
    return f"data:application/octet-stream;base64,{b64}"


def generate_html_viewer(
    upper: "trimesh.Trimesh",
    lower: "trimesh.Trimesh",
    output_path: str,
    title: str = "ObjectSplitter Result",
    metadata: Optional[dict] = None,
    plane_origin: Optional[numpy.ndarray] = None,
    plane_normal: Optional[numpy.ndarray] = None
) -> bool:
    """
    Generate an interactive HTML page with a 3D viewer for the split result.

    Uses Three.js loaded from CDN with an embedded STL loader.
    The meshes are base64-encoded directly into the HTML so the file is
    fully self-contained.

    Args:
        upper: Upper mesh half.
        lower: Lower mesh half.
        output_path: File path for the output HTML.
        title: Page title.
        metadata: Optional dictionary of key-value metadata to display.
        plane_origin: Optional cut plane origin for display.
        plane_normal: Optional cut plane normal for display.

    Returns:
        True if HTML was generated successfully.
    """
    try:
        upper_uri = _mesh_to_stl_data_uri(upper)
        lower_uri = _mesh_to_stl_data_uri(lower)

        meta_rows = ""
        if metadata:
            for k, v in metadata.items():
                meta_rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>\n"

        if plane_origin is not None:
            meta_rows += f"<tr><td><strong>Plane Origin</strong></td><td>[{plane_origin[0]:.3f}, {plane_origin[1]:.3f}, {plane_origin[2]:.3f}]</td></tr>\n"
        if plane_normal is not None:
            meta_rows += f"<tr><td><strong>Plane Normal</strong></td><td>[{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]</td></tr>\n"

        # Compute bounding info for camera setup
        all_verts = numpy.vstack([upper.vertices, lower.vertices])
        center = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2.0
        extent = (all_verts.max(axis=0) - all_verts.min(axis=0)).max()

        html = _THREE_JS_TEMPLATE.format(
            title=title,
            upper_data_uri=upper_uri,
            lower_data_uri=lower_uri,
            meta_rows=meta_rows,
            center_x=center[0], center_y=center[1], center_z=center[2],
            camera_distance=extent * 2.0,
            upper_verts=len(upper.vertices), upper_faces=len(upper.faces),
            lower_verts=len(lower.vertices), lower_faces=len(lower.faces),
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info("Generated HTML viewer: %s", output_path)
        return True

    except Exception as e:
        logger.error("Failed to generate HTML viewer: %s", e)
        return False


def generate_report(
    test_results: list,
    output_path: str,
    title: str = "ObjectSplitter Test Report"
) -> bool:
    """
    Generate a summary HTML report from multiple test results.

    Args:
        test_results: List of dicts, each with keys:
            - 'name': Test name
            - 'status': 'pass' or 'fail'
            - 'split_result': SplitResult
            - 'image_path': Optional path to PNG image (relative to report)
            - 'viewer_path': Optional path to HTML viewer (relative to report)
            - 'details': Optional dict of extra info
        output_path: File path for the output HTML.
        title: Report title.

    Returns:
        True if report was generated successfully.
    """
    try:
        rows = ""
        pass_count = sum(1 for r in test_results if r.get('status') == 'pass')
        fail_count = len(test_results) - pass_count

        for r in test_results:
            status_class = 'pass' if r.get('status') == 'pass' else 'fail'
            status_icon = 'PASS' if r.get('status') == 'pass' else 'FAIL'

            links = ""
            if r.get('image_path'):
                links += f'<a href="{r["image_path"]}">Image</a> '
            if r.get('viewer_path'):
                links += f'<a href="{r["viewer_path"]}">3D Viewer</a> '

            split = r.get('split_result')
            summary = split.summary() if split else 'N/A'

            rows += f"""
            <tr class="{status_class}">
                <td class="status">{status_icon}</td>
                <td>{r.get('name', 'unnamed')}</td>
                <td>{summary}</td>
                <td>{links}</td>
            </tr>"""

        html = _REPORT_TEMPLATE.format(
            title=title,
            pass_count=pass_count,
            fail_count=fail_count,
            total=len(test_results),
            rows=rows
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info("Generated test report: %s", output_path)
        return True

    except Exception as e:
        logger.error("Failed to generate report: %s", e)
        return False


# ==========================================================================
# HTML Templates
# ==========================================================================

_THREE_JS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
  #info {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 15px; border-radius: 8px; max-width: 400px; z-index: 10; }}
  #info h2 {{ margin: 0 0 10px 0; font-size: 16px; }}
  #info table {{ font-size: 13px; border-collapse: collapse; }}
  #info td {{ padding: 2px 8px; }}
  #controls {{ position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 8px; z-index: 10; }}
  #controls label {{ margin-right: 15px; cursor: pointer; }}
  canvas {{ display: block; }}
  .mesh-info {{ font-size: 12px; color: #aaa; margin-top: 8px; }}
</style>
</head>
<body>
<div id="info">
  <h2>{title}</h2>
  <table>{meta_rows}</table>
  <div class="mesh-info">
    Upper: {upper_verts} verts / {upper_faces} faces<br>
    Lower: {lower_verts} verts / {lower_faces} faces
  </div>
</div>
<div id="controls">
  <label><input type="checkbox" id="showUpper" checked> Upper (blue)</label>
  <label><input type="checkbox" id="showLower" checked> Lower (red)</label>
  <label><input type="checkbox" id="wireframe"> Wireframe</label>
  <label>Separation: <input type="range" id="separation" min="0" max="50" value="5" style="width:120px"></label>
</div>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ STLLoader }} from 'three/addons/loaders/STLLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 10000);
camera.position.set({center_x} + {camera_distance}, {center_y} + {camera_distance} * 0.7, {center_z} + {camera_distance});
camera.lookAt({center_x}, {center_y}, {center_z});

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set({center_x}, {center_y}, {center_z});
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0x404040, 2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(1, 2, 1).normalize();
scene.add(dirLight);
scene.add(new THREE.DirectionalLight(0xffffff, 0.5).position.set(-1, -1, -1).normalize());

// Grid
const gridHelper = new THREE.GridHelper(200, 20, 0x444466, 0x333355);
scene.add(gridHelper);

// Load STL meshes
const loader = new STLLoader();
let upperMesh, lowerMesh;
const upperMat = new THREE.MeshPhongMaterial({{ color: 0x4A90D9, specular: 0x111111, shininess: 50 }});
const lowerMat = new THREE.MeshPhongMaterial({{ color: 0xD94A4A, specular: 0x111111, shininess: 50 }});

function loadDataURI(uri) {{
  const b64 = uri.split(',')[1];
  const binary = atob(b64);
  const buffer = new ArrayBuffer(binary.length);
  const view = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
  return buffer;
}}

const upperGeom = loader.parse(loadDataURI("{upper_data_uri}"));
upperGeom.computeVertexNormals();
upperMesh = new THREE.Mesh(upperGeom, upperMat);
scene.add(upperMesh);

const lowerGeom = loader.parse(loadDataURI("{lower_data_uri}"));
lowerGeom.computeVertexNormals();
lowerMesh = new THREE.Mesh(lowerGeom, lowerMat);
scene.add(lowerMesh);

// Controls
document.getElementById('showUpper').addEventListener('change', e => {{ upperMesh.visible = e.target.checked; }});
document.getElementById('showLower').addEventListener('change', e => {{ lowerMesh.visible = e.target.checked; }});
document.getElementById('wireframe').addEventListener('change', e => {{
  upperMat.wireframe = e.target.checked;
  lowerMat.wireframe = e.target.checked;
}});
document.getElementById('separation').addEventListener('input', e => {{
  const s = parseFloat(e.target.value) / 2;
  upperMesh.position.y = s;
  lowerMesh.position.y = -s;
}});

// Initial separation
upperMesh.position.y = 2.5;
lowerMesh.position.y = -2.5;

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #2c3e50; }}
  .summary {{ font-size: 18px; margin-bottom: 20px; }}
  .summary .pass-count {{ color: #27ae60; font-weight: bold; }}
  .summary .fail-count {{ color: #e74c3c; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr.pass .status {{ color: #27ae60; font-weight: bold; }}
  tr.fail .status {{ color: #e74c3c; font-weight: bold; }}
  tr.fail {{ background: #fdf2f2; }}
  a {{ color: #3498db; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="summary">
  <span class="pass-count">{pass_count} passed</span> /
  <span class="fail-count">{fail_count} failed</span> /
  {total} total
</div>
<table>
  <thead>
    <tr><th>Status</th><th>Test</th><th>Result Summary</th><th>Links</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</body>
</html>"""
