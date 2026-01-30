# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

import sys
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QProgressDialog

from UM.Logger import Logger
from UM.Application import Application
from UM.Math.Vector import Vector
from UM.Math.Matrix import Matrix
from UM.Tool import Tool
from UM.Event import Event, MouseEvent
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Scene.Selection import Selection
from UM.Scene.SceneNode import SceneNode
from UM.View.GL.OpenGL import OpenGL

from cura.CuraApplication import CuraApplication
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.PickingPass import PickingPass

from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.AddSceneNodeOperation import AddSceneNodeOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from cura.Operations.SetParentOperation import SetParentOperation

from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator

import numpy
import math
from typing import Optional, Tuple, List

# Log Python environment info for debugging
Logger.log("i", "ObjectSplitter: Python executable: %s", sys.executable)
Logger.log("i", "ObjectSplitter: Python version: %s", sys.version)
Logger.log("i", "ObjectSplitter: Python path: %s", os.pathsep.join(sys.path[:3]))  # First 3 paths

# Try to import trimesh - it's optional but required for cutting
try:
    import trimesh
    TRIMESH_AVAILABLE = True
    Logger.log("i", "ObjectSplitter: trimesh version: %s", trimesh.__version__)
except ImportError:
    TRIMESH_AVAILABLE = False
    Logger.log("w", "trimesh not available - Object Splitter cutting functionality disabled")

# Check for rtree availability
try:
    import rtree
    RTREE_AVAILABLE = True
    Logger.log("i", "ObjectSplitter: rtree is available")
except ImportError:
    RTREE_AVAILABLE = False
    Logger.log("w", "ObjectSplitter: rtree not available - will use fallback triangulation")

# Check for scipy (used for fallback triangulation)
try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
    Logger.log("i", "ObjectSplitter: scipy is available for triangulation")
except ImportError:
    SCIPY_AVAILABLE = False
    Logger.log("w", "ObjectSplitter: scipy not available - triangulation may fail")


class ObjectSplitter(Tool):
    """Tool for splitting 3D objects into multiple parts by cutting along planes."""

    # Cut mode constants
    CUT_MODE_HORIZONTAL = "horizontal"      # Cut parallel to build plate
    CUT_MODE_VERTICAL = "vertical"          # Cut perpendicular to build plate
    CUT_MODE_SMALLEST = "smallest"          # Find smallest cross-section
    CUT_MODE_CUSTOM = "custom"              # User-defined plane orientation
    CUT_MODE_SHORTEST = "shortest"          # Shortest seam (geodesic loop)

    def __init__(self):
        super().__init__()
        self._plugin_id = "ObjectSplitter"
        if hasattr(self, "setPluginId"):
            try:
                self.setPluginId(self._plugin_id)
            except Exception:
                pass
        self._shortcut_key = Qt.Key.Key_K  # K for "Kut" (avoiding conflicts)
        self._controller = self.getController()

        # Cut settings
        self._cut_mode = self.CUT_MODE_HORIZONTAL
        self._cut_height = 0.0  # For horizontal cuts: Z position (relative to object)
        self._cut_height_percent = 50.0  # Percentage of object height
        self._plane_normal = numpy.array([0.0, 1.0, 0.0])  # Y-up in Cura
        self._plane_origin = numpy.array([0.0, 0.0, 0.0])

        # Preview settings
        self._show_preview = True
        self._preview_node = None
        self._preview_size = 100.0  # Size of preview plane (will be adjusted to mesh)

        # Connector settings
        self._connector_enabled = True
        self._connector_diameter = 4.0  # mm - diameter of peg/hole
        self._connector_height = 3.0  # mm - how deep the peg/hole extends
        self._connector_clearance = 0.2  # mm - extra space in hole for fit
        self._connector_sides = 16  # Number of sides for cylinder approximation

        prefs = Application.getInstance().getPreferences()
        prefs.addPreference("objectsplitter/openscad_path", "")
        self._openscad_path = prefs.getValue("objectsplitter/openscad_path")
        # Search settings for smallest cut
        self._search_resolution = 18  # Number of angles to search

        # State
        self._selection_pass = None
        self._last_picked_node = None
        self._last_picked_position = None
        self._hover_node = None  # Node currently being hovered over
        self._picking_pass = None  # Cached picking pass
        self._progress_dialog = None  # Progress dialog for long operations

        self.setExposedProperties(
            "CutMode",
            "CutModes",
            "CutHeightPercent",
            "ShowPreview",
            "TrimeshAvailable",
            "SearchResolution",
            # Connector properties
            "ConnectorEnabled",
            "OpenScadPath",
            "ConnectorDiameter",
            "ConnectorHeight",
            "ConnectorClearance"
            # (Note: ConnectorSides not exposed to QML to avoid undefined warnings)
        )

        Logger.log("d", "Object Splitter Tool initialized (trimesh available: %s)", str(TRIMESH_AVAILABLE))

        CuraApplication.getInstance().globalContainerStackChanged.connect(self._updateEnabled)
        Selection.selectionChanged.connect(self._onSelectionChanged)

    def _updateEnabled(self):
        """Update whether the tool is enabled based on current state."""
        plugin_enabled = True

        global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if global_container_stack:
            plugin_enabled = True  # Could add conditions here

        Application.getInstance().getController().toolEnabledChanged.emit(self._plugin_id, plugin_enabled)

    def _onSelectionChanged(self):
        """Handle selection changes."""
        pass  # Could update preview here


    # ==========================================================================
    # Progress Dialog
    # ==========================================================================

    def _showProgress(self, title: str, message: str, minimum: int = 0, maximum: int = 100) -> QProgressDialog:
        """Show a progress dialog for long operations."""
        app = QApplication.instance()
        if app is None:
            return None

        dialog = self._progress_dialog
        if dialog is None:
            dialog = QProgressDialog(message, None, minimum, maximum)
            dialog.setWindowTitle(title)
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            dialog.setCancelButton(None)  # No cancel button
            dialog.setMinimumDuration(0)  # Show immediately
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            dialog.setValue(minimum)
            self._progress_dialog = dialog
        else:
            dialog.setLabelText(message)
            dialog.setRange(minimum, maximum)
            dialog.setWindowTitle(title)
            dialog.setValue(minimum)

        dialog.show()
        QApplication.processEvents()
        return dialog

    def _updateProgress(self, message: str, value: int = None) -> None:
        """Update the progress dialog."""
        dialog = self._progress_dialog
        if dialog is None:
            return
        if message:
            dialog.setLabelText(message)
        if value is not None:
            dialog.setValue(value)
        QApplication.processEvents()

    def _closeProgress(self) -> None:
        """Close the progress dialog."""
        dialog = self._progress_dialog
        if dialog is None:
            return
        dialog.close()
        self._progress_dialog = None

    # ==========================================================================
    # Properties for QML
    # ==========================================================================
    def getId(self) -> str:
        return self._plugin_id

    def getPluginId(self) -> str:
        return self._plugin_id

    def getQmlPath(self):
        """Return the path to the QML file for the tool panel."""
        qml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qml", "ObjectSplitter.qml")
        Logger.log("d", f"QML path: {qml_path}")
        return qml_path
    
    def getCutMode(self) -> str:
        return self._cut_mode

    def setCutMode(self, mode: str) -> None:
        if mode != self._cut_mode:
            self._cut_mode = mode
            Logger.log("d", "Cut mode changed to: %s", mode)
            self.propertyChanged.emit()

    def getCutModes(self) -> list:
        """Return available cut modes for QML dropdown."""
        return [
            {"value": self.CUT_MODE_HORIZONTAL, "text": "Horizontal (parallel to bed)"},
            {"value": self.CUT_MODE_VERTICAL, "text": "Vertical"},
            {"value": self.CUT_MODE_SMALLEST, "text": "Smallest cross-section"},
            {"value": self.CUT_MODE_SHORTEST, "text": "Shortest seam (surface loop)"}
            # {"value": self.CUT_MODE_CUSTOM, "text": "Custom angle"},  # Future
        ]

    def getCutHeightPercent(self) -> float:
        return self._cut_height_percent

    def setCutHeightPercent(self, value: float) -> None:
        if value != self._cut_height_percent:
            self._cut_height_percent = float(value)
            Logger.log("d", "Cut height percent changed to: %s", str(value))
            self.propertyChanged.emit()

    def getShowPreview(self) -> bool:
        return self._show_preview

    def setShowPreview(self, value: bool) -> None:
        if value != self._show_preview:
            self._show_preview = value
            if not value:
                self._removePreview()
            self.propertyChanged.emit()

    def getTrimeshAvailable(self) -> bool:
        return TRIMESH_AVAILABLE

    def getSearchResolution(self) -> int:
        return self._search_resolution

    def setSearchResolution(self, value: int) -> None:
        if value != self._search_resolution:
            self._search_resolution = int(value)
            self.propertyChanged.emit()

    # Connector properties
    def getConnectorEnabled(self) -> bool:
        return self._connector_enabled

    def setConnectorEnabled(self, value: bool) -> None:
        if value != self._connector_enabled:
            self._connector_enabled = value
            Logger.log("d", "Connector enabled changed to: %s", str(value))
            self.propertyChanged.emit()

    def getConnectorDiameter(self) -> float:
        return self._connector_diameter

    def setConnectorDiameter(self, value: float) -> None:
        if value != self._connector_diameter:
            self._connector_diameter = float(value)
            Logger.log("d", "Connector diameter changed to: %s", str(value))
            self.propertyChanged.emit()

    def getConnectorHeight(self) -> float:
        return self._connector_height

    def setConnectorHeight(self, value: float) -> None:
        if value != self._connector_height:
            self._connector_height = float(value)
            Logger.log("d", "Connector height changed to: %s", str(value))
            self.propertyChanged.emit()

    def getConnectorClearance(self) -> float:
        return self._connector_clearance

    def setConnectorClearance(self, value: float) -> None:
        if value != self._connector_clearance:
            self._connector_clearance = float(value)
            Logger.log("d", "Connector clearance changed to: %s", str(value))
            self.propertyChanged.emit()

    def setOpenScadPath(self, value: str) -> None:
        value = value or ""
        if value != (self._openscad_path or ""):
            self._openscad_path = value
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/openscad_path", value)
            Logger.log("i", "ObjectSplitter: OpenSCAD path set to: %s", value)
            self._configureBooleanEngines()  # Ensure trimesh uses the updated OpenSCAD path
            self.propertyChanged.emit()

    # ==========================================================================
    # Event Handling
    # ==========================================================================

    def event(self, event):
        super().event(event)
        modifiers = QApplication.keyboardModifiers()
        ctrl_is_active = modifiers & Qt.KeyboardModifier.ControlModifier

        # Handle mouse move for preview
        if event.type == Event.MouseMoveEvent and self._show_preview:
            self._updatePreview(event.x, event.y)
            return

        if event.type == Event.MousePressEvent and MouseEvent.LeftButton in event.buttons and self._controller.getToolsEnabled():
            if ctrl_is_active:
                self._controller.setActiveTool("TranslateTool")
                return

            if not TRIMESH_AVAILABLE:
                Logger.log("e", "Cannot split: trimesh library not available. Install with: pip install trimesh")
                return

            # Hide preview before cutting
            self._removePreview()

            # Get the object under the mouse
            if self._selection_pass is None:
                self._selection_pass = Application.getInstance().getRenderer().getRenderPass("selection")

            picked_node = self._controller.getScene().findObject(
                self._selection_pass.getIdAtPosition(event.x, event.y)
            )

            if not picked_node:
                Logger.log("d", "No object picked")
                return

            # Check if it's a regular mesh (not a modifier volume)
            node_stack = picked_node.callDecoration("getStack")
            if node_stack:
                if (node_stack.getProperty("support_mesh", "value") or
                    node_stack.getProperty("anti_overhang_mesh", "value") or
                    node_stack.getProperty("infill_mesh", "value") or
                    node_stack.getProperty("cutting_mesh", "value")):
                    Logger.log("d", "Cannot split modifier meshes")
                    return

            # Get 3D click position
            active_camera = self._controller.getScene().getActiveCamera()
            picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
            picking_pass.render()
            picked_position = picking_pass.getPickedPosition(event.x, event.y)

            Logger.log("i", "Splitting object '%s' at position %s", picked_node.getName(), str(picked_position))

            # Store for potential preview updates
            self._last_picked_node = picked_node
            self._last_picked_position = picked_position

            # Perform the cut
            self._performCut(picked_node, picked_position)

    def setEnabled(self, enable: bool) -> None:
        """Called when the tool is enabled/disabled."""
        super().setEnabled(enable)
        if not enable:
            self._removePreview()

    # ==========================================================================
    # Preview Handling
    # ==========================================================================

    def _updatePreview(self, mouse_x: float, mouse_y: float):
        """Update the preview plane based on mouse position."""
        if not self._show_preview:
            self._removePreview()
            return

        # Get the object under the mouse
        if self._selection_pass is None:
            self._selection_pass = Application.getInstance().getRenderer().getRenderPass("selection")

        picked_node = self._controller.getScene().findObject(
            self._selection_pass.getIdAtPosition(mouse_x, mouse_y)
        )

        # Don't show preview on preview node itself
        if picked_node == self._preview_node:
            return

        if not picked_node:
            self._removePreview()
            return

        # Check if it's a regular mesh (not a modifier volume)
        node_stack = picked_node.callDecoration("getStack")
        if node_stack:
            if (node_stack.getProperty("support_mesh", "value") or
                node_stack.getProperty("anti_overhang_mesh", "value") or
                node_stack.getProperty("infill_mesh", "value") or
                node_stack.getProperty("cutting_mesh", "value")):
                self._removePreview()
                return

        # Get 3D position under mouse
        active_camera = self._controller.getScene().getActiveCamera()
        if self._picking_pass is None:
            self._picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
        self._picking_pass.render()
        picked_position = self._picking_pass.getPickedPosition(mouse_x, mouse_y)

        if picked_position is None:
            self._removePreview()
            return

        # Get mesh data for plane calculation
        mesh_data = picked_node.getMeshData()
        if mesh_data is None:
            self._removePreview()
            return

        # Calculate plane parameters based on cut mode
        transformed_mesh = mesh_data.getTransformed(picked_node.getWorldTransformation())
        vertices = transformed_mesh.getVertices()

        # Get bounding box for plane size
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        mesh_size = max_bounds - min_bounds
        plane_size = max(mesh_size[0], mesh_size[2]) * 1.2  # 20% larger than mesh

        # Determine plane normal and origin based on mode
        if self._cut_mode == self.CUT_MODE_HORIZONTAL:
            min_y = min_bounds[1]
            max_y = max_bounds[1]
            height = max_y - min_y
            cut_y = min_y + (height * self._cut_height_percent / 100.0)
            plane_origin = Vector(0, cut_y, 0)
            plane_normal = Vector(0, 1, 0)
        elif self._cut_mode == self.CUT_MODE_VERTICAL:
            plane_origin = picked_position
            plane_normal = Vector(1, 0, 0)  # Cut along X axis
        elif self._cut_mode == self.CUT_MODE_SMALLEST:
            # For smallest mode, just show horizontal plane at click position as hint
            # The actual smallest cut is computed on click
            plane_origin = picked_position
            plane_normal = Vector(0, 1, 0)
        elif self._cut_mode == self.CUT_MODE_SHORTEST:
            # For shortest seam mode, no simple planar preview – use a vertical plane as a placeholder
            plane_origin = picked_position
            plane_normal = Vector(0, 1, 0)
        else:
            plane_origin = picked_position
            plane_normal = Vector(0, 1, 0)


        # Create or update preview
        self._createOrUpdatePreview(plane_origin, plane_normal, plane_size)
        self._hover_node = picked_node

    def _createOrUpdatePreview(self, origin: Vector, normal: Vector, size: float):
        """Create or update the preview plane mesh."""
        if self._preview_node is None:
            self._preview_node = self._createPreviewNode()

        # Update preview mesh geometry
        mesh_builder = self._createPlaneMesh(origin, normal, size)
        mesh_data = mesh_builder.build()
        self._preview_node.setMeshData(mesh_data)

        # Make sure it's in the scene
        scene_root = self._controller.getScene().getRoot()
        if self._preview_node.getParent() != scene_root:
            self._preview_node.setParent(scene_root)

    def _createPreviewNode(self) -> SceneNode:
        """Create a new preview node (non-selectable, non-sliceable)."""
        node = SceneNode()
        node.setName("ObjectSplitter_Preview")
        node.setSelectable(False)
        node.setCalculateBoundingBox(False)

        # Set rendering to be translucent
        # Note: The actual transparency depends on Cura's rendering pipeline
        # We'll use a special mesh type or shader if available

        return node

    def _createPlaneMesh(self, origin: Vector, normal: Vector, size: float) -> MeshBuilder:
        """Create a flat plane mesh at the given position and orientation."""
        mesh = MeshBuilder()

        # Normalize the normal vector
        normal_arr = numpy.array([normal.x, normal.y, normal.z])
        normal_arr = normal_arr / numpy.linalg.norm(normal_arr)

        # Find two perpendicular vectors to the normal
        if abs(normal_arr[1]) < 0.9:
            up = numpy.array([0, 1, 0])
        else:
            up = numpy.array([1, 0, 0])

        tangent1 = numpy.cross(normal_arr, up)
        tangent1 = tangent1 / numpy.linalg.norm(tangent1)
        tangent2 = numpy.cross(normal_arr, tangent1)

        # Scale by half size
        half_size = size / 2.0
        t1 = tangent1 * half_size
        t2 = tangent2 * half_size

        # Create 4 corners of the plane
        center = numpy.array([origin.x, origin.y, origin.z])
        corners = [
            center - t1 - t2,  # Bottom-left
            center + t1 - t2,  # Bottom-right
            center + t1 + t2,  # Top-right
            center - t1 + t2,  # Top-left
        ]

        # Create vertices (we need 6 for 2 triangles, but we'll use indexed)
        vertices = numpy.array(corners, dtype=numpy.float32)

        # Create indices for 2 triangles (both sides for visibility)
        indices = numpy.array([
            [0, 1, 2],  # Front face triangle 1
            [0, 2, 3],  # Front face triangle 2
            [0, 2, 1],  # Back face triangle 1
            [0, 3, 2],  # Back face triangle 2
        ], dtype=numpy.int32)

        mesh.setVertices(vertices)
        mesh.setIndices(indices)

        # Set a distinct color for the preview (orange/red for visibility)
        # Colors are RGBA per vertex
        colors = numpy.array([
            [1.0, 0.3, 0.0, 0.5],  # Orange, semi-transparent
            [1.0, 0.3, 0.0, 0.5],
            [1.0, 0.3, 0.0, 0.5],
            [1.0, 0.3, 0.0, 0.5],
        ], dtype=numpy.float32)
        mesh.setColors(colors)

        mesh.calculateNormals()

        return mesh

    def _removePreview(self):
        """Remove the preview plane from the scene."""
        if self._preview_node is not None:
            if self._preview_node.getParent() is not None:
                self._preview_node.setParent(None)
            self._preview_node = None
        self._hover_node = None

    # ==========================================================================
    # Cutting Logic
    # ==========================================================================

    def _performCut(self, node: CuraSceneNode, click_position: Vector):
        self._showProgress("Object Splitter", "Preparing mesh...", 0, 100)
        try:
            mesh_data = node.getMeshData()
            if mesh_data is None:
                Logger.log("e", "Node has no mesh data")
                self._closeProgress()
                return

            # Load mesh data in world coordinates
            self._updateProgress("Loading mesh data...", 10)
            transformed_mesh = mesh_data.getTransformed(node.getWorldTransformation())
            vertices = transformed_mesh.getVertices()
            indices = transformed_mesh.getIndices()
            if indices is None:
                indices = numpy.arange(len(vertices)).reshape(-1, 3).astype(numpy.int32)
            tm = trimesh.Trimesh(vertices=vertices, faces=indices)

            # Convert click position to mesh-local coordinates for accurate cutting logic
            click_pos_local = None
            try:
                # Approximate inverse transformation to get local coordinates of click_position
                local_vertices = mesh_data.getVertices()
                if local_vertices is not None and len(local_vertices) >= 3:
                    world_vertices = vertices  # already obtained above
                    # Pick three non-collinear points (v0, v1, v2) for transform estimation
                    v0L = numpy.array(local_vertices[0]); v0W = numpy.array(world_vertices[0])
                    # Find v1 and v2 not collinear with v0
                    v1_index, v2_index = 1, 2
                    d1L = numpy.array(local_vertices[1]) - v0L
                    for i in range(2, len(local_vertices)):
                        d2L = numpy.array(local_vertices[i]) - v0L
                        if numpy.linalg.norm(numpy.cross(d1L, d2L)) > 1e-6:
                            v1_index = 1; v2_index = i; break
                    v1L = numpy.array(local_vertices[v1_index]); v2L = numpy.array(local_vertices[v2_index])
                    v1W = numpy.array(world_vertices[v1_index]); v2W = numpy.array(world_vertices[v2_index])
                    # Compute scale (assume uniform)
                    d1L = v1L - v0L; d2L = v2L - v0L
                    d1W = v1W - v0W; d2W = v2W - v0W
                    scale_d1 = numpy.linalg.norm(d1W) / max(1e-9, numpy.linalg.norm(d1L))
                    scale_d2 = numpy.linalg.norm(d2W) / max(1e-9, numpy.linalg.norm(d2L))
                    scale = (scale_d1 + scale_d2) / 2.0 if numpy.isfinite(scale_d1) and numpy.isfinite(scale_d2) else scale_d1
                    # Solve for rotation matrix R (3x3) using 3 basis vectors
                    basis_L = numpy.column_stack((d1L, d2L, numpy.cross(d1L, d2L)))
                    basis_W = numpy.column_stack((d1W, d2W, numpy.cross(d1W, d2W)))
                    if numpy.linalg.matrix_rank(basis_L) >= 3:
                        R_approx = basis_W @ numpy.linalg.inv(basis_L) / (scale if scale != 0 else 1.0)
                        # Orthonormalize R_approx via SVD to get a proper rotation
                        U, _, Vt = numpy.linalg.svd(R_approx)
                        R = U @ Vt
                        if numpy.linalg.det(R) < 0:  # ensure right-hand rotation
                            R[:, 2] *= -1
                        else:
                            R = R  # already proper rotation
                        T = v0W - scale * (R @ v0L)
                        # Compute click_position in local coords: local = R^T * (world - T) / scale
                        click_pos_arr = numpy.array([click_position.x, click_position.y, click_position.z])
                        click_local_arr = R.T @ (click_pos_arr - T) / (scale if scale != 0 else 1.0)
                        click_pos_local = Vector(click_local_arr[0], click_local_arr[1], click_local_arr[2])
            except Exception as e:
                Logger.log("d", "Local coordinate conversion failed: %s", str(e))
                click_pos_local = None

            # Determine cut plane based on mode
            self._updateProgress("Calculating cut plane...", 20)
            if self._cut_mode == self.CUT_MODE_HORIZONTAL:
                plane_normal, plane_origin = self._getHorizontalCutPlane(tm, click_position)
            elif self._cut_mode == self.CUT_MODE_VERTICAL:
                # Use mesh-local orientation for vertical plane normal
                if click_pos_local:
                    plane_normal_local = numpy.array([1.0, 0.0, 0.0])
                    # Transform local X-axis to world (rotation only)
                    plane_normal_arr = (R @ plane_normal_local) if 'R' in locals() else numpy.array([1.0, 0.0, 0.0])
                    plane_normal = Vector(float(plane_normal_arr[0]), float(plane_normal_arr[1]), float(plane_normal_arr[2]))
                else:
                    plane_normal = Vector(1, 0, 0)
                plane_origin = numpy.array([click_position.x, click_position.y, click_position.z])
            elif self._cut_mode == self.CUT_MODE_SMALLEST:
                self._updateProgress("Searching for smallest cross-section...", 20)
                # If available, use local coordinates for click position to improve accuracy
                if click_pos_local:
                    plane_normal, plane_origin = self._findSmallestCutPlane(tm, click_pos_local)
                else:
                    plane_normal, plane_origin = self._findSmallestCutPlane(tm, click_position)
            elif self._cut_mode == self.CUT_MODE_SHORTEST:
                self._updateProgress("Computing shortest seam...", 20)
                plane_normal, plane_origin = Vector(0, 0, 1), Vector(click_position.x, click_position.y, click_position.z)  # Placeholder for logging
                mesh_upper, mesh_lower, capped = self._cutShortestLoop(tm, click_position)
            else:
                plane_normal, plane_origin = self._getHorizontalCutPlane(tm, click_position)

            if self._cut_mode != self.CUT_MODE_SHORTEST:  # Log plane for planar cuts
                Logger.log("d", "Cut plane: origin=%s, normal=%s", str(plane_origin), str(plane_normal))
            # Perform the cut
            self._updateProgress("Splitting mesh...", 40)
            if self._cut_mode == self.CUT_MODE_SHORTEST:
                # Geodesic loop cut already computed above
                pass  # mesh_upper, mesh_lower obtained, continue with result
            else:
                mesh_upper, mesh_lower, capped = self._sliceMeshWithFallback(tm, plane_origin, plane_normal)
            if mesh_upper is None or mesh_lower is None:
                Logger.log("e", "Cut operation failed")
                self._closeProgress()
                return
            if len(mesh_upper.vertices) == 0:
                Logger.log("w", "Upper mesh is empty after cut")
                self._closeProgress()
                return
            if len(mesh_lower.vertices) == 0:
                Logger.log("w", "Lower mesh is empty after cut")
                self._closeProgress()
                return

            Logger.log("i", "Cut successful: upper=%d verts, lower=%d verts, capped=%s",
                       len(mesh_upper.vertices), len(mesh_lower.vertices), str(capped))
            # Add connectors if enabled and applicable
            if self._connector_enabled:
                if self._cut_mode == self.CUT_MODE_SHORTEST:
                    Logger.log("w", "Skipping connectors - not supported for shortest seam mode")
                elif capped:
                    self._updateProgress("Adding connectors...", 60)
                    mesh_upper, mesh_lower = self._addConnectors(mesh_upper, mesh_lower, plane_origin, plane_normal)
                    Logger.log("i", "After connectors: upper=%d verts, lower=%d verts",
                               len(mesh_upper.vertices), len(mesh_lower.vertices))
                else:
                    Logger.log("w", "Skipping connectors - mesh was not capped (open edges)")
            # ... [object creation and finalization unchanged] ...
        except Exception as e:
            Logger.log("e", "Error during cut operation: %s", str(e))
        finally:
            self._closeProgress()

    def _cutShortestLoop(self, mesh: "trimesh.Trimesh", click_pos: Vector) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", bool]:
        """Compute a geodesic shortest seam (closed loop) around the clicked point. Returns upper and lower meshes and whether they were capped."""
        # Find the face on which the click lies (or nearest face)
        point = numpy.array([click_pos.x, click_pos.y, click_pos.z]).reshape(1, -1)
        face_index = None
        try:
            from trimesh.proximity import ProximityQuery
            pq = ProximityQuery(mesh)
            _, _, face_ids = pq.on_surface(point)
            face_index = int(face_ids[0]) if face_ids is not None else None
        except Exception as e:
            Logger.log("d", "Proximity query failed: %s", str(e))
        if face_index is None:
            # Fallback: pick nearest vertex's face
            nearest_idx = mesh.vertices.shape[0] - 1  # default to last vertex
            if mesh.vertices.shape[0] > 0:
                distances = numpy.linalg.norm(mesh.vertices - point, axis=1)
                nearest_idx = int(numpy.argmin(distances))
            faces_with_vertex = numpy.where(mesh.faces == nearest_idx)[0]
            face_index = int(faces_with_vertex[0]) if faces_with_vertex.size > 0 else 0
        # Use face connectivity to perform minimum cut separating face_index from opposite side
        faces_count = len(mesh.faces)
        adj_pairs = mesh.face_adjacency  # (N,2) array of adjacent face indices
        adj_edges = mesh.face_adjacency_edges  # (N,2) array of vertex indices for shared edge
        # Build flow network for min-cut (face graph)
        graph = [[] for _ in range(faces_count)]
        def _add_edge(u, v, cap):
            graph[u].append({"v": v, "cap": cap, "rev": len(graph[v])})
            graph[v].append({"v": u, "cap": 0, "rev": len(graph[u]) - 1})
        # Add edges for each adjacent face pair with weight = shared edge length
        for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
            edge_length = float(numpy.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
            _add_edge(int(f1), int(f2), edge_length)
            _add_edge(int(f2), int(f1), edge_length)
        # Choose sink as the farthest face from source (using Dijkstra distances)
        dist = [float("inf")] * faces_count; dist[face_index] = 0.0
        import heapq
        pq = [(0.0, face_index)]
        while pq:
            d, f = heapq.heappop(pq)
            if d > dist[f]: continue
            for edge in graph[f]:
                if edge["cap"] > 0:  # neighbor edge with weight as cap
                    nd = d + edge["cap"]
                    if nd < dist[edge["v"]]:
                        dist[edge["v"]] = nd
                        heapq.heappush(pq, (nd, edge["v"]))
        sink_face = int(numpy.argmax(dist)) if faces_count > 0 else face_index
        # Max-flow (Dinic's algorithm) from source to sink
        flow = 0
        def _bfs_level():
            level = [-1] * faces_count; queue = [face_index]; level[face_index] = 0
            for u in queue:
                for edge in graph[u]:
                    if level[edge["v"]] < 0 and edge["cap"] > 0:
                        level[edge["v"]] = level[u] + 1
                        queue.append(edge["v"])
            return level
        def _dfs_flow(u, sink, f, level, it):
            if u == sink: return f
            for i in range(it[u], len(graph[u])):
                it[u] = i
                edge = graph[u][i]
                if edge["cap"] <= 0 or level[edge["v"]] != level[u] + 1:
                    continue
                ret = _dfs_flow(edge["v"], sink, min(f, edge["cap"]), level, it)
                if ret > 0:
                    edge["cap"] -= ret
                    graph[edge["v"]][edge["rev"]]["cap"] += ret
                    return ret
            return 0
        # Compute max flow and residual graph
        while True:
            level = _bfs_level()
            if level[sink_face] < 0: break
            it = [0] * faces_count
            while True:
                pushed = _dfs_flow(face_index, sink_face, float("inf"), level, it)
                if pushed <= 1e-9: break
                flow += pushed
        # After max flow, find reachable faces from source in residual network
        reachable = [False] * faces_count
        stack = [face_index]; reachable[face_index] = True
        while stack:
            u = stack.pop()
            for edge in graph[u]:
                if edge["cap"] > 0 and not reachable[edge["v"]]:
                    reachable[edge["v"]] = True
                    stack.append(edge["v"])
        set_A = [i for i, r in enumerate(reachable) if r]
        set_B = [i for i, r in enumerate(reachable) if not r]
        # Ensure source side is smaller piece (label as "upper")
        if len(set_A) > len(set_B):
            set_A, set_B = set_B, set_A
        # Separate mesh into two parts by faces sets
        upper_mesh = mesh.submesh([set_A], append=True)
        lower_mesh = mesh.submesh([set_B], append=True)
        # Attempt to cap open seams by filling holes for watertightness
        capped = False
        try:
            if upper_mesh is not None and lower_mesh is not None:
                if not upper_mesh.is_watertight or not lower_mesh.is_watertight:
                    upper_mesh_filled = upper_mesh.copy(); lower_mesh_filled = lower_mesh.copy()
                    upper_mesh_filled.fill_holes(); lower_mesh_filled.fill_holes()
                    if upper_mesh_filled.is_watertight and lower_mesh_filled.is_watertight:
                        upper_mesh = upper_mesh_filled; lower_mesh = lower_mesh_filled
                        capped = True
        except Exception as e:
            Logger.log("d", "Hole filling exception: %s", str(e))
        return upper_mesh, lower_mesh, capped


        """Perform the cut operation on the given node."""

        # Show progress dialog
        self._showProgress("Object Splitter", "Preparing mesh...", 0, 100)

        try:
            mesh_data = node.getMeshData()
            if mesh_data is None:
                Logger.log("e", "Node has no mesh data")
                self._closeProgress()
                return

            # Get mesh in world coordinates
            self._updateProgress("Loading mesh data...", 10)
            transformed_mesh = mesh_data.getTransformed(node.getWorldTransformation())
            vertices = transformed_mesh.getVertices()
            indices = transformed_mesh.getIndices()

            if indices is None:
                # Non-indexed mesh - create indices
                indices = numpy.arange(len(vertices)).reshape(-1, 3).astype(numpy.int32)

            # Convert to trimesh
            tm = trimesh.Trimesh(vertices=vertices, faces=indices)

            # Determine cut plane based on mode
            self._updateProgress("Calculating cut plane...", 20)
            if self._cut_mode == self.CUT_MODE_HORIZONTAL:
                plane_normal, plane_origin = self._getHorizontalCutPlane(tm, click_position)
            elif self._cut_mode == self.CUT_MODE_VERTICAL:
                plane_normal, plane_origin = self._getVerticalCutPlane(tm, click_position)
            elif self._cut_mode == self.CUT_MODE_SMALLEST:
                self._updateProgress("Searching for smallest cross-section...", 20)
                plane_normal, plane_origin = self._findSmallestCutPlane(tm, click_position)
            else:
                plane_normal, plane_origin = self._getHorizontalCutPlane(tm, click_position)

            Logger.log("d", "Cut plane: origin=%s, normal=%s", str(plane_origin), str(plane_normal))

            # Perform the cut - try with capping first, fallback to no cap
            self._updateProgress("Splitting mesh...", 40)
            mesh_upper, mesh_lower, capped = self._sliceMeshWithFallback(tm, plane_origin, plane_normal)

            if mesh_upper is None or mesh_lower is None:
                Logger.log("e", "Cut operation failed")
                self._closeProgress()
                return

            # Check if we got valid meshes
            if len(mesh_upper.vertices) == 0:
                Logger.log("w", "Upper mesh is empty after cut")
                self._closeProgress()
                return
            if len(mesh_lower.vertices) == 0:
                Logger.log("w", "Lower mesh is empty after cut")
                self._closeProgress()
                return

            Logger.log("i", "Cut successful: upper=%d verts, lower=%d verts, capped=%s",
                       len(mesh_upper.vertices), len(mesh_lower.vertices), str(capped))

            # Add connectors if enabled (only if mesh was capped properly)
            if self._connector_enabled and capped:
                self._updateProgress("Adding connectors...", 60)
                mesh_upper, mesh_lower = self._addConnectors(
                    mesh_upper, mesh_lower, plane_origin, plane_normal
                )
                Logger.log("i", "After connectors: upper=%d verts, lower=%d verts",
                           len(mesh_upper.vertices), len(mesh_lower.vertices))
            elif self._connector_enabled and not capped:
                Logger.log("w", "Skipping connectors - mesh was not capped (open edges)")

            self._updateProgress("Creating new objects...", 80)

            # Create new scene nodes for both parts
            original_name = node.getName()

            op = GroupedOperation()

            # Create upper part
            node_upper = self._createMeshNode(
                mesh_upper.vertices,
                mesh_upper.faces,
                f"{original_name}_part1"
            )

            # Create lower part
            node_lower = self._createMeshNode(
                mesh_lower.vertices,
                mesh_lower.faces,
                f"{original_name}_part2"
            )

            # Add new nodes and remove original
            self._updateProgress("Finalizing...", 90)
            scene_root = self._controller.getScene().getRoot()

            op.addOperation(AddSceneNodeOperation(node_upper, scene_root))
            op.addOperation(AddSceneNodeOperation(node_lower, scene_root))
            op.addOperation(RemoveSceneNodeOperation(node))

            op.push()

            # Emit scene changed
            CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node_upper)

            self._updateProgress("Done!", 100)
            Logger.log("i", "Split complete: created '%s' and '%s'",
                       node_upper.getName(), node_lower.getName())

        except Exception as e:
            Logger.log("e", "Error during cut operation: %s", str(e))
        finally:
            self._closeProgress()

    def _sliceMeshWithFallback(self, mesh: "trimesh.Trimesh", plane_origin: numpy.ndarray,
                                plane_normal: numpy.ndarray) -> Tuple[Optional["trimesh.Trimesh"], Optional["trimesh.Trimesh"], bool]:
        """
        Slice mesh with multiple fallback strategies for robustness.

        Returns:
            Tuple of (upper_mesh, lower_mesh, was_capped)
            was_capped is True if the cut surfaces were closed (watertight result)
        """
        mesh_upper = None
        mesh_lower = None
        capped = False

        # Strategy 1: Try with capping (ideal case - watertight mesh with rtree)
        try:
            mesh_upper = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                cap=True
            )
            mesh_lower = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=-plane_normal,
                plane_origin=plane_origin,
                cap=True
            )
            if mesh_upper is not None and mesh_lower is not None:
                capped = True
                Logger.log("d", "Slicing with cap=True succeeded")
                return mesh_upper, mesh_lower, capped
        except ImportError as e:
            # rtree not available
            Logger.log("w", "Capping requires 'rtree' library: %s. Trying manual capping.", str(e))
        except Exception as e:
            error_msg = str(e).lower()
            if "watertight" in error_msg:
                Logger.log("w", "Mesh is not watertight, cannot use built-in cap. Trying manual capping.")
            elif "rtree" in error_msg:
                Logger.log("w", "rtree library missing: %s. Trying manual capping.", str(e))
            else:
                Logger.log("w", "Capped slicing failed: %s. Trying manual capping.", str(e))

        # Strategy 2: Slice without capping, then manually cap
        try:
            mesh_upper = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                cap=False
            )
            mesh_lower = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=-plane_normal,
                plane_origin=plane_origin,
                cap=False
            )
            if mesh_upper is not None and mesh_lower is not None:
                # Try to manually cap the meshes
                mesh_upper_capped = self._manualCapMesh(mesh_upper, plane_origin, plane_normal)
                mesh_lower_capped = self._manualCapMesh(mesh_lower, plane_origin, -plane_normal)

                if mesh_upper_capped is not None and mesh_lower_capped is not None:
                    Logger.log("i", "Slicing with manual capping succeeded")
                    return mesh_upper_capped, mesh_lower_capped, True
                else:
                    Logger.log("w", "Manual capping failed, using uncapped meshes")
                    return mesh_upper, mesh_lower, False
        except Exception as e:
            Logger.log("e", "Uncapped slicing failed: %s", str(e))

        # Strategy 3: Manual vertex-based splitting as last resort
        try:
            mesh_upper, mesh_lower = self._manualMeshSplit(mesh, plane_origin, plane_normal)
            if mesh_upper is not None and mesh_lower is not None:
                Logger.log("i", "Manual mesh splitting succeeded")
                return mesh_upper, mesh_lower, False
        except Exception as e:
            Logger.log("e", "Manual splitting failed: %s", str(e))

        return None, None, False

    def _manualCapMesh(self, mesh: "trimesh.Trimesh", plane_origin: numpy.ndarray,
                        plane_normal: numpy.ndarray) -> Optional["trimesh.Trimesh"]:
        """
        Manually cap a mesh by finding the boundary edges on the cut plane
        and triangulating them to close the surface.
        Uses scipy Delaunay triangulation to avoid rtree dependency.
        """
        try:
            # Get the cross-section path at the cut plane
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            if section is None:
                Logger.log("w", "Could not get cross-section for capping")
                return None

            # Convert to 2D for triangulation
            path_2d, transform = section.to_planar()
            if path_2d is None:
                Logger.log("w", "Could not convert section to 2D")
                return None

            # Get vertices from the path
            # The path contains discrete curves - we need to extract vertices
            vertices_2d = None
            faces_2d = None

            # Try scipy Delaunay triangulation first (doesn't need rtree)
            if SCIPY_AVAILABLE:
                try:
                    # Get all vertices from path entities
                    all_vertices = []
                    for entity in path_2d.entities:
                        points = path_2d.vertices[entity.points]
                        all_vertices.extend(points)

                    if len(all_vertices) < 3:
                        Logger.log("w", "Not enough vertices for triangulation")
                        return None

                    vertices_2d = numpy.array(all_vertices)

                    # Remove duplicate vertices
                    vertices_2d = numpy.unique(vertices_2d, axis=0)

                    if len(vertices_2d) < 3:
                        Logger.log("w", "Not enough unique vertices for triangulation")
                        return None

                    # Use scipy Delaunay triangulation
                    tri = Delaunay(vertices_2d)
                    faces_2d = tri.simplices

                    Logger.log("d", "Scipy Delaunay triangulation: %d vertices, %d faces",
                               len(vertices_2d), len(faces_2d))

                except Exception as e:
                    Logger.log("w", "Scipy triangulation failed: %s", str(e))
                    vertices_2d = None
                    faces_2d = None

            # Fallback to trimesh triangulation if scipy failed
            if vertices_2d is None or faces_2d is None:
                try:
                    vertices_2d, faces_2d = path_2d.triangulate()
                except Exception as e:
                    Logger.log("w", "Trimesh triangulation also failed: %s", str(e))
                    return None

            if vertices_2d is None or len(vertices_2d) == 0 or faces_2d is None or len(faces_2d) == 0:
                Logger.log("w", "Triangulation produced empty result")
                return None

            # Transform triangulated vertices back to 3D
            vertices_3d_homogeneous = numpy.column_stack([
                vertices_2d,
                numpy.zeros(len(vertices_2d)),
                numpy.ones(len(vertices_2d))
            ])
            transform_inv = numpy.linalg.inv(transform)
            vertices_3d = (transform_inv @ vertices_3d_homogeneous.T).T[:, :3]

            # Create cap mesh
            cap_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_2d)

            # Ensure cap normal faces the right direction (away from the part)
            # The cap should face in the direction of the plane normal
            if len(cap_mesh.face_normals) > 0:
                cap_normal = cap_mesh.face_normals.mean(axis=0)
                norm = numpy.linalg.norm(cap_normal)
                if norm > 1e-6:
                    cap_normal = cap_normal / norm
                    if numpy.dot(cap_normal, plane_normal) < 0:
                        # Flip the faces
                        cap_mesh.faces = cap_mesh.faces[:, ::-1]

            # Combine original mesh with cap
            combined = trimesh.util.concatenate([mesh, cap_mesh])

            Logger.log("d", "Manual capping added %d cap vertices, %d cap faces",
                       len(vertices_3d), len(faces_2d))

            return combined

        except Exception as e:
            Logger.log("w", "Manual capping error: %s", str(e))
            return None

    def _manualMeshSplit(self, mesh: "trimesh.Trimesh", plane_origin: numpy.ndarray,
                          plane_normal: numpy.ndarray) -> Tuple[Optional["trimesh.Trimesh"], Optional["trimesh.Trimesh"]]:
        """
        Manually split mesh by separating faces based on which side of the plane they're on.
        This is a simple approach that doesn't handle faces crossing the plane perfectly,
        but works as a fallback when trimesh's slice_mesh_plane fails.
        """
        vertices = mesh.vertices
        faces = mesh.faces

        # Compute signed distance of each vertex to the plane
        distances = numpy.dot(vertices - plane_origin, plane_normal)

        # For each face, determine which side it's on based on centroid
        face_centroids = vertices[faces].mean(axis=1)
        face_distances = numpy.dot(face_centroids - plane_origin, plane_normal)

        # Split faces
        upper_mask = face_distances >= 0
        lower_mask = face_distances < 0

        upper_faces = faces[upper_mask]
        lower_faces = faces[lower_mask]

        if len(upper_faces) == 0 or len(lower_faces) == 0:
            Logger.log("w", "Manual split resulted in empty mesh on one side")
            return None, None

        # Create new meshes (reusing all vertices, trimesh will clean up unused ones)
        mesh_upper = trimesh.Trimesh(vertices=vertices.copy(), faces=upper_faces)
        mesh_lower = trimesh.Trimesh(vertices=vertices.copy(), faces=lower_faces)

        # Remove unreferenced vertices
        mesh_upper.remove_unreferenced_vertices()
        mesh_lower.remove_unreferenced_vertices()

        return mesh_upper, mesh_lower

    def _getHorizontalCutPlane(self, mesh: "trimesh.Trimesh", click_pos: Vector) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get a horizontal cut plane at the specified height percentage."""
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_y = bounds[0][1]
        max_y = bounds[1][1]
        height = max_y - min_y

        # Calculate cut height based on percentage
        cut_y = min_y + (height * self._cut_height_percent / 100.0)

        # Or use click position Y if closer to it
        # cut_y = click_pos.y  # Could use this instead

        plane_origin = numpy.array([0.0, cut_y, 0.0])
        plane_normal = numpy.array([0.0, 1.0, 0.0])  # Y-up

        return plane_normal, plane_origin

    def _getVerticalCutPlane(self, mesh: "trimesh.Trimesh", click_pos: Vector) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get a vertical cut plane through the click position."""
        # Use click position as origin
        plane_origin = numpy.array([click_pos.x, click_pos.y, click_pos.z])

        # Default to cutting along X axis (YZ plane)
        plane_normal = numpy.array([1.0, 0.0, 0.0])

        return plane_normal, plane_origin

    def _findSmallestCutPlane(self, mesh: "trimesh.Trimesh", click_pos: Vector) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Find the plane orientation that produces the smallest cross-sectional area."""

        plane_origin = numpy.array([click_pos.x, click_pos.y, click_pos.z])

        best_normal = numpy.array([0.0, 1.0, 0.0])  # Default to horizontal
        best_area = float('inf')

        # Sample orientations in spherical coordinates
        n_theta = self._search_resolution
        n_phi = self._search_resolution * 2

        for i in range(n_theta):
            theta = numpy.pi * i / n_theta  # 0 to pi (elevation)
            for j in range(n_phi):
                phi = 2 * numpy.pi * j / n_phi  # 0 to 2pi (azimuth)

                # Convert spherical to Cartesian
                normal = numpy.array([
                    numpy.sin(theta) * numpy.cos(phi),
                    numpy.cos(theta),
                    numpy.sin(theta) * numpy.sin(phi)
                ])

                # Get cross-section at this orientation
                try:
                    section = mesh.section(plane_origin=plane_origin, plane_normal=normal)
                    if section is not None:
                        # Get 2D area of the cross-section
                        path_2d, _ = section.to_planar()
                        area = abs(path_2d.area)

                        if area < best_area and area > 0:
                            best_area = area
                            best_normal = normal.copy()
                except Exception:
                    continue

        Logger.log("d", "Smallest cut found: area=%.2f mm², normal=%s", best_area, str(best_normal))

        return best_normal, plane_origin

    def _createMeshNode(self, vertices: numpy.ndarray, faces: numpy.ndarray, name: str) -> CuraSceneNode:
        """Create a new CuraSceneNode from vertices and faces."""

        # Build mesh using MeshBuilder
        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices.astype(numpy.float32))
        mesh_builder.setIndices(faces.astype(numpy.int32))
        mesh_builder.calculateNormals()

        mesh_data = mesh_builder.build()

        # Create scene node
        node = CuraSceneNode()
        node.setName(name)
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        node.setMeshData(mesh_data)
        node.calculateBoundingBoxMesh()

        # Add decorators for Cura integration
        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        node.addDecorator(BuildPlateDecorator(active_build_plate))
        node.addDecorator(SliceableObjectDecorator())

        return node

    # ==========================================================================
    # Connector Logic
    # ==========================================================================

    def _getMeshVolume(self, mesh: "trimesh.Trimesh") -> float:
        """Get the volume of a mesh. Uses convex hull if mesh is not watertight."""
        try:
            if mesh.is_watertight:
                return abs(mesh.volume)
            else:
                return abs(mesh.convex_hull.volume)
        except Exception:
            # Fallback to bounding box volume
            bounds = mesh.bounds
            return numpy.prod(bounds[1] - bounds[0])

    def _determinePegSide(self, mesh_a: "trimesh.Trimesh", mesh_b: "trimesh.Trimesh") -> Tuple[str, str]:
        """
        Determine which part gets the peg vs hole based on volume.
        Peg goes on smaller part, hole on larger part.

        Returns:
            Tuple of ("peg", "hole") or ("hole", "peg") indicating what mesh_a and mesh_b get.
        """
        volume_a = self._getMeshVolume(mesh_a)
        volume_b = self._getMeshVolume(mesh_b)

        Logger.log("d", "Volume comparison: mesh_a=%.2f mm³, mesh_b=%.2f mm³", volume_a, volume_b)

        if volume_a <= volume_b:
            return ("peg", "hole")  # mesh_a gets peg, mesh_b gets hole
        else:
            return ("hole", "peg")  # mesh_a gets hole, mesh_b gets peg

    def _findConnectorPosition(self, mesh: "trimesh.Trimesh", plane_origin: numpy.ndarray,
                                plane_normal: numpy.ndarray) -> Optional[numpy.ndarray]:
        """
        Find a suitable position for the connector on the cut surface.
        Returns the centroid of the cut surface if valid, None otherwise.
        """
        try:
            # Get the cross-section at the cut plane
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            if section is None:
                Logger.log("w", "Could not get cross-section for connector placement")
                return None

            # Get the 3D vertices directly from the section
            # The section is a Path3D object with vertices in 3D space
            if not hasattr(section, 'vertices') or len(section.vertices) == 0:
                Logger.log("w", "Cross-section has no vertices")
                return None

            # Calculate centroid directly from 3D vertices
            vertices_3d = section.vertices
            centroid_3d = vertices_3d.mean(axis=0)

            # Project centroid onto the cut plane to ensure it's exactly on the plane
            # (it should already be close, but this ensures precision)
            dist_to_plane = numpy.dot(centroid_3d - plane_origin, plane_normal)
            centroid_3d = centroid_3d - dist_to_plane * plane_normal

            Logger.log("d", "Connector position (3D centroid): %s", str(centroid_3d))

            return centroid_3d

        except Exception as e:
            Logger.log("w", "Error finding connector position: %s", str(e))
            return None

    def _createPegMesh(self, position: numpy.ndarray, normal: numpy.ndarray,
                       diameter: float, height: float) -> "trimesh.Trimesh":
        """Create a cylinder mesh for the peg at the given position."""
        radius = diameter / 2.0

        # Create cylinder along Z axis, then transform
        peg = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=self._connector_sides
        )

        # The cylinder is centered at origin along Z
        # We need to:
        # 1. Move it so the base is at Z=0 (shift up by height/2)
        # 2. Rotate to align with the plane normal
        # 3. Translate to the connector position

        # Shift so base is at origin
        peg.apply_translation([0, 0, height / 2])

        # Create rotation to align Z axis with the plane normal
        z_axis = numpy.array([0, 0, 1])
        normal_normalized = normal / numpy.linalg.norm(normal)

        # Rotation matrix from Z to normal
        rotation_matrix = self._rotationMatrixFromVectors(z_axis, normal_normalized)
        transform = numpy.eye(4)
        transform[:3, :3] = rotation_matrix
        peg.apply_transform(transform)

        # Translate to position
        peg.apply_translation(position)

        Logger.log("d", "Created peg: diameter=%.2f, height=%.2f, position=%s",
                   diameter, height, str(position))

        return peg

    def _createHoleMesh(self, position: numpy.ndarray, normal: numpy.ndarray,
                        diameter: float, height: float, clearance: float) -> "trimesh.Trimesh":
        """Create a cylinder mesh for the hole (to be subtracted) at the given position."""
        # Hole is slightly larger than peg for clearance
        radius = diameter / 2.0 + clearance
        # Hole is slightly deeper to ensure clean subtraction
        hole_height = height + 0.2

        hole = trimesh.creation.cylinder(
            radius=radius,
            height=hole_height,
            sections=self._connector_sides
        )

        # Shift so the top of the cylinder is at Z=0 (hole goes into the part)
        hole.apply_translation([0, 0, -hole_height / 2])

        # Create rotation to align Z axis with the negative plane normal (hole goes in)
        z_axis = numpy.array([0, 0, 1])
        normal_normalized = normal / numpy.linalg.norm(normal)

        rotation_matrix = self._rotationMatrixFromVectors(z_axis, -normal_normalized)
        transform = numpy.eye(4)
        transform[:3, :3] = rotation_matrix
        hole.apply_transform(transform)

        # Translate to position
        hole.apply_translation(position)

        Logger.log("d", "Created hole: diameter=%.2f (with clearance=%.2f), height=%.2f, position=%s",
                   diameter + clearance * 2, clearance, hole_height, str(position))

        return hole

    def _rotationMatrixFromVectors(self, vec1: numpy.ndarray, vec2: numpy.ndarray) -> numpy.ndarray:
        """
        Create a rotation matrix that rotates vec1 to vec2.
        Uses Rodrigues' rotation formula.
        """
        vec1 = vec1 / numpy.linalg.norm(vec1)
        vec2 = vec2 / numpy.linalg.norm(vec2)

        # Check if vectors are parallel
        cross = numpy.cross(vec1, vec2)
        dot = numpy.dot(vec1, vec2)

        if numpy.linalg.norm(cross) < 1e-6:
            if dot > 0:
                # Same direction, identity rotation
                return numpy.eye(3)
            else:
                # Opposite direction, 180 degree rotation
                # Find a perpendicular vector
                if abs(vec1[0]) < 0.9:
                    perp = numpy.array([1, 0, 0])
                else:
                    perp = numpy.array([0, 1, 0])
                perp = perp - numpy.dot(perp, vec1) * vec1
                perp = perp / numpy.linalg.norm(perp)
                # Rodrigues for 180 degree rotation around perp
                return 2 * numpy.outer(perp, perp) - numpy.eye(3)

        # Rodrigues' formula
        cross_normalized = cross / numpy.linalg.norm(cross)
        angle = numpy.arccos(numpy.clip(dot, -1, 1))

        K = numpy.array([
            [0, -cross_normalized[2], cross_normalized[1]],
            [cross_normalized[2], 0, -cross_normalized[0]],
            [-cross_normalized[1], cross_normalized[0], 0]
        ])

        R = numpy.eye(3) + numpy.sin(angle) * K + (1 - numpy.cos(angle)) * (K @ K)
        return R

    def _addConnectors(self, mesh_upper: "trimesh.Trimesh", mesh_lower: "trimesh.Trimesh",
                       plane_origin: numpy.ndarray, plane_normal: numpy.ndarray) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh"]:
        """
        Add peg to smaller part and hole to larger part.
        Returns the modified meshes.

        Uses simple mesh concatenation for pegs (faster and more reliable than boolean union
        since the peg sits on the cut surface without overlapping).
        For holes, tries boolean difference with fallback to skipping if it fails.
        """
        if not self._connector_enabled:
            return mesh_upper, mesh_lower

        # Determine which part gets peg vs hole
        upper_role, lower_role = self._determinePegSide(mesh_upper, mesh_lower)

        # Find connector position on the cut surface
        # Use the original mesh (upper) for finding position since both share the cut surface
        connector_pos = self._findConnectorPosition(mesh_upper, plane_origin, plane_normal)

        if connector_pos is None:
            Logger.log("w", "Could not find valid connector position, skipping connectors")
            return mesh_upper, mesh_lower

        # Create peg and hole meshes
        peg = self._createPegMesh(
            connector_pos, plane_normal,
            self._connector_diameter, self._connector_height
        )

        hole = self._createHoleMesh(
            connector_pos, plane_normal,
            self._connector_diameter, self._connector_height, self._connector_clearance
        )

        # Apply to the appropriate meshes
        # Try hole first - if it fails, skip both peg and hole (peg without hole is useless)
        try:
            if upper_role == "peg":
                # Lower gets hole (boolean difference) - try this first
                mesh_lower_with_hole = self._tryBooleanDifference(mesh_lower, hole)
                if mesh_lower_with_hole is not None and len(mesh_lower_with_hole.vertices) > 0:
                    Logger.log("i", "Added hole to lower part via boolean difference")
                    # Hole succeeded, now add the peg to upper
                    mesh_upper_result = trimesh.util.concatenate([mesh_upper, peg])
                    Logger.log("i", "Added peg to upper part via concatenation")
                    return mesh_upper_result, mesh_lower_with_hole
                else:
                    Logger.log("w", "Could not create hole - skipping connectors entirely")
                    return mesh_upper, mesh_lower
            else:
                # Upper gets hole (boolean difference) - try this first
                mesh_upper_with_hole = self._tryBooleanDifference(mesh_upper, hole)
                if mesh_upper_with_hole is not None and len(mesh_upper_with_hole.vertices) > 0:
                    Logger.log("i", "Added hole to upper part via boolean difference")
                    # Hole succeeded, now add the peg to lower
                    mesh_lower_result = trimesh.util.concatenate([mesh_lower, peg])
                    Logger.log("i", "Added peg to lower part via concatenation")
                    return mesh_upper_with_hole, mesh_lower_result
                else:
                    Logger.log("w", "Could not create hole - skipping connectors entirely")
                    return mesh_upper, mesh_lower

        except Exception as e:
            Logger.log("e", "Error adding connectors: %s. Using meshes without connectors.", str(e))
            return mesh_upper, mesh_lower

    def _tryBooleanDifference(self, mesh: "trimesh.Trimesh", tool: "trimesh.Trimesh") -> Optional["trimesh.Trimesh"]:
        """
        Try to perform boolean difference using available engines.
        Returns None if all methods fail.
        """
        # Try manifold3d first (best option if available)
        try:
            result = trimesh.boolean.difference([mesh, tool], engine='manifold')
            if result is not None and len(result.vertices) > 0:
                Logger.log("d", "Boolean difference succeeded with manifold engine")
                return result
        except Exception as e:
            Logger.log("d", "Manifold boolean failed: %s", str(e))

        # Try blender engine
        try:
            result = trimesh.boolean.difference([mesh, tool], engine='blender')
            if result is not None and len(result.vertices) > 0:
                Logger.log("d", "Boolean difference succeeded with blender engine")
                return result
        except Exception as e:
            Logger.log("d", "Blender boolean failed: %s", str(e))

        # Try default engine (may use OpenSCAD)
        try:
            result = trimesh.boolean.difference([mesh, tool])
            if result is not None and len(result.vertices) > 0:
                Logger.log("d", "Boolean difference succeeded with default engine")
                return result
        except Exception as e:
            Logger.log("d", "Default boolean failed: %s", str(e))

        Logger.log("w", "All boolean difference methods failed")
        return None
