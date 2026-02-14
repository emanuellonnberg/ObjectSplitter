# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""
ObjectSplitter Cura Tool - thin integration layer.

All computational logic lives in the `core` package (geometry, plane_calculator,
mesh_splitter, connectors) so it can be tested independently of Cura.
This file handles only Cura-specific concerns: UI events, scene management,
progress dialogs, and property binding.
"""

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
from typing import Optional, Tuple

# Log Python environment info for debugging
Logger.log("i", "ObjectSplitter: Python executable: %s", sys.executable)
Logger.log("i", "ObjectSplitter: Python version: %s", sys.version)
Logger.log("i", "ObjectSplitter: Python path: %s", os.pathsep.join(sys.path[:3]))

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

# Import core modules (Cura-independent algorithms)
from .core.geometry import (
    world_to_local_transform,
    transform_point_to_local,
    create_plane_mesh_data,
    create_marker_mesh_data,
)
from .core.plane_calculator import (
    horizontal_cut_plane,
    vertical_cut_plane,
    find_smallest_cut_plane,
    find_shortest_seam_partition,
)
from .core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
    split_by_local_plane,
)
from .core.connectors import (
    add_connectors,
    ConnectorConfig,
)


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

        # Debug capture (derived from _capture_dir: None = off, path = on)
        self._capture_dir = None

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
            "ConnectorClearance",
            # (Note: ConnectorSides not exposed to QML to avoid undefined warnings)
            # Debug capture
            "DebugCaptureEnabled",
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

        try:
            plugin_id = self.getPluginId()
            Application.getInstance().getController().toolEnabledChanged.emit(plugin_id, plugin_enabled)
        except ValueError:
            # Plugin ID not set yet during initialization, skip the emit
            pass

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

    def getOpenScadPath(self) -> str:
        return self._openscad_path or ""

    def setOpenScadPath(self, value: str) -> None:
        value = value or ""
        if value != (self._openscad_path or ""):
            self._openscad_path = value
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/openscad_path", value)
            Logger.log("i", "ObjectSplitter: OpenSCAD path set to: %s", value)
            self._configureBooleanEngines()  # Ensure trimesh uses the updated OpenSCAD path
            self.propertyChanged.emit()

    def getDebugCaptureEnabled(self) -> bool:
        return self._capture_dir is not None

    def setDebugCaptureEnabled(self, value: bool) -> None:
        if value != self.getDebugCaptureEnabled():
            if value:
                self._capture_dir = os.path.join(os.path.dirname(__file__), "captures")
                Logger.log("i", "Debug capture enabled, saving to: %s", self._capture_dir)
            else:
                self._capture_dir = None
                Logger.log("i", "Debug capture disabled")
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
        Logger.log("d", "ObjectSplitter.setEnabled: %s", enable)
        super().setEnabled(enable)
        if enable:
            Logger.log("d", "ObjectSplitter: Tool enabled, panel should be visible")
        else:
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

        # For smallest/shortest seam: show arrow at click point along surface normal
        if self._cut_mode in (self.CUT_MODE_SMALLEST, self.CUT_MODE_SHORTEST):
            center = numpy.array([picked_position.x, picked_position.y, picked_position.z])
            mesh_size_max = max(mesh_size[0], mesh_size[1], mesh_size[2])
            marker_size = max(0.5, mesh_size_max * 0.012)  # 1.2% of mesh or 0.5mm

            # Compute face normal at hover point for arrow direction
            face_normal = None
            try:
                indices = transformed_mesh.getIndices()
                verts_arr = numpy.asarray(vertices, dtype=numpy.float64)
                if indices is not None:
                    indices_arr = numpy.asarray(indices, dtype=numpy.int32)
                else:
                    indices_arr = numpy.arange(len(verts_arr), dtype=numpy.int32).reshape(-1, 3)
                tm_preview = trimesh.Trimesh(vertices=verts_arr, faces=indices_arr)
                tm_preview.merge_vertices()
                from .core.plane_calculator import snap_point_to_mesh_surface
                _, face_id = snap_point_to_mesh_surface(tm_preview, center)
                if face_id is not None and face_id < len(tm_preview.face_normals):
                    face_normal = numpy.array(tm_preview.face_normals[face_id], dtype=numpy.float64)
            except Exception:
                pass  # Fall back to default Y-up arrow

            self._createOrUpdateMarker(center, marker_size, face_normal)
            self._hover_node = picked_node
            return

        # Planar modes: show the actual cut plane
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
        else:
            plane_origin = picked_position
            plane_normal = Vector(0, 1, 0)

        self._createOrUpdatePreview(plane_origin, plane_normal, plane_size)
        self._hover_node = picked_node

    def _createOrUpdateMarker(self, center: numpy.ndarray, size: float,
                              direction: numpy.ndarray = None):
        """Create or update an arrow marker at the click location, oriented along surface normal."""
        if self._preview_node is None:
            self._preview_node = self._createPreviewNode()

        vertices, indices = create_marker_mesh_data(center, size, direction)
        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices)
        mesh_builder.setIndices(indices)
        n_verts = len(vertices)
        colors = numpy.array([[1.0, 0.3, 0.0, 0.7]] * n_verts, dtype=numpy.float32)
        mesh_builder.setColors(colors)
        mesh_builder.calculateNormals()
        self._preview_node.setMeshData(mesh_builder.build())

        scene_root = self._controller.getScene().getRoot()
        if self._preview_node.getParent() != scene_root:
            self._preview_node.setParent(scene_root)

    def _createOrUpdatePreview(self, origin: Vector, normal: Vector, size: float):
        """Create or update the preview plane mesh."""
        if self._preview_node is None:
            self._preview_node = self._createPreviewNode()

        origin_arr = numpy.array([origin.x, origin.y, origin.z])
        normal_arr = numpy.array([normal.x, normal.y, normal.z])
        vertices, indices = create_plane_mesh_data(origin_arr, normal_arr, size)

        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices)
        mesh_builder.setIndices(indices)

        colors = numpy.array([
            [1.0, 0.3, 0.0, 0.5],
            [1.0, 0.3, 0.0, 0.5],
            [1.0, 0.3, 0.0, 0.5],
            [1.0, 0.3, 0.0, 0.5],
        ], dtype=numpy.float32)
        mesh_builder.setColors(colors)
        mesh_builder.calculateNormals()

        self._preview_node.setMeshData(mesh_builder.build())

        scene_root = self._controller.getScene().getRoot()
        if self._preview_node.getParent() != scene_root:
            self._preview_node.setParent(scene_root)

    def _createPreviewNode(self) -> SceneNode:
        """Create a new preview node (non-selectable, non-sliceable)."""
        node = SceneNode()
        node.setName("ObjectSplitter_Preview")
        node.setSelectable(False)
        node.setCalculateBoundingBox(False)
        return node

    def _removePreview(self):
        """Remove the preview plane from the scene."""
        if self._preview_node is not None:
            if self._preview_node.getParent() is not None:
                self._preview_node.setParent(None)
            self._preview_node = None
        self._hover_node = None

    # ==========================================================================
    # Cutting Logic (delegates to core modules)
    # ==========================================================================

    def _extractTrimesh(self, node: CuraSceneNode) -> Optional[Tuple["trimesh.Trimesh", numpy.ndarray, numpy.ndarray]]:
        """
        Extract a trimesh object from a Cura scene node.

        Returns:
            (trimesh, local_vertices, world_vertices) or None if extraction fails.
        """
        mesh_data = node.getMeshData()
        if mesh_data is None:
            Logger.log("e", "Node has no mesh data")
            return None

        transformed_mesh = mesh_data.getTransformed(node.getWorldTransformation())
        vertices = transformed_mesh.getVertices()
        indices = transformed_mesh.getIndices()

        # Ensure numpy arrays with correct shape (Cura may return different formats)
        vertices = numpy.asarray(vertices, dtype=numpy.float64)
        if vertices.ndim == 1:
            vertices = vertices.reshape(-1, 3)

        if indices is None:
            # Triangle soup: every 3 consecutive vertices form a face
            n_verts = len(vertices)
            if n_verts % 3 != 0:
                Logger.log("e", "Mesh has %d vertices, not divisible by 3", n_verts)
                return None
            indices = numpy.arange(n_verts, dtype=numpy.int32).reshape(-1, 3)
        else:
            indices = numpy.asarray(indices, dtype=numpy.int32)
            if indices.ndim == 1:
                indices = indices.reshape(-1, 3)

        local_vertices = mesh_data.getVertices()
        if local_vertices is not None:
            local_vertices = numpy.asarray(local_vertices, dtype=numpy.float64)
            if local_vertices.ndim == 1:
                local_vertices = local_vertices.reshape(-1, 3)
        else:
            local_vertices = vertices.copy()

        tm = trimesh.Trimesh(vertices=vertices, faces=indices)

        # Cura often stores meshes as "triangle soup" with no shared vertices.
        # merge_vertices() deduplicates vertices so trimesh can build proper
        # edge-face adjacency, which is required for section(), split(), etc.
        tm.merge_vertices()

        Logger.log("d", "Extracted trimesh: %d vertices, %d faces (watertight=%s)",
                   len(tm.vertices), len(tm.faces), str(tm.is_watertight))

        return tm, local_vertices, vertices

    def _performCut(self, node: CuraSceneNode, click_position: Vector):
        """Perform the cut operation on the given node using core algorithms."""
        self._showProgress("Object Splitter", "Preparing mesh...", 0, 100)

        try:
            # Extract mesh data
            self._updateProgress("Loading mesh data...", 10)
            extraction = self._extractTrimesh(node)
            if extraction is None:
                self._closeProgress()
                return
            tm, local_vertices, world_vertices = extraction

            click_pos_arr = numpy.array([click_position.x, click_position.y, click_position.z])

            # Compute local coordinate transform for accurate cutting
            R = None
            click_pos_local_arr = None
            transform_result = world_to_local_transform(local_vertices, world_vertices)
            if transform_result is not None:
                R, T, scale = transform_result
                click_pos_local_arr = transform_point_to_local(click_pos_arr, R, T, scale)

            # Optionally capture inputs for later replay
            if self._capture_dir:
                try:
                    from .core.debug_capture import capture_operation
                    capture_operation(
                        mesh=tm,
                        cut_mode=self._cut_mode,
                        click_position=click_pos_arr,
                        height_percent=self._cut_height_percent,
                        search_resolution=self._search_resolution,
                        connector_enabled=self._connector_enabled,
                        connector_diameter=self._connector_diameter,
                        connector_height=self._connector_height,
                        connector_clearance=self._connector_clearance,
                        capture_dir=self._capture_dir,
                    )
                except Exception as e:
                    Logger.log("w", "Debug capture failed: %s", str(e))

            # Determine cut plane based on mode (delegates to core.plane_calculator)
            self._updateProgress("Calculating cut plane...", 20)

            split_result = None

            from .core.plane_calculator import snap_point_to_mesh_surface
            _, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)

            if self._cut_mode == self.CUT_MODE_HORIZONTAL:
                plane = horizontal_cut_plane(tm, self._cut_height_percent)
            elif self._cut_mode == self.CUT_MODE_VERTICAL:
                if R is not None:
                    # Transform local X-axis to world for the vertical plane normal
                    plane_normal_world = R @ numpy.array([1.0, 0.0, 0.0])
                    plane = vertical_cut_plane(click_pos_arr, plane_normal_world)
                else:
                    plane = vertical_cut_plane(click_pos_arr)
            elif self._cut_mode == self.CUT_MODE_SMALLEST:
                self._updateProgress("Searching for smallest cross-section...", 20)
                snap_point, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)

                # Get the surface normal at the click point — this tells us
                # the direction the user is clicking from, and strongly biases
                # the search toward that orientation.
                click_surface_normal = None
                if click_face_id is not None and click_face_id < len(tm.face_normals):
                    click_surface_normal = numpy.array(
                        tm.face_normals[click_face_id], dtype=numpy.float64
                    )
                    Logger.log("d", "Click surface normal: %s", click_surface_normal)

                search_result = find_smallest_cut_plane(
                    tm, snap_point, self._search_resolution,
                    surface_normal=click_surface_normal,
                )
                plane = search_result.plane
                Logger.log("i", "Smallest cut: area=%.2f mm^2, tested %d orientations, normal=%s",
                           search_result.area, search_result.samples_tested,
                           str(search_result.plane.normal))
            elif self._cut_mode == self.CUT_MODE_SHORTEST:
                self._updateProgress("Computing shortest seam...", 20)
                snap_point, source_face = snap_point_to_mesh_surface(tm, click_pos_arr)
                click_face_id = source_face
                face_set_a, face_set_b, src, sink = find_shortest_seam_partition(
                    tm, snap_point, source_face_hint=source_face
                )
                Logger.log("d", "Shortest seam: source=%d, sink=%d, partitions=%d/%d faces",
                           src, sink, len(face_set_a), len(face_set_b))
                plane = None  # No planar cut for shortest seam
            else:
                plane = horizontal_cut_plane(tm, self._cut_height_percent)

            if plane is not None:
                Logger.log("d", "Cut plane: %s", plane)

            # Perform the split (delegates to core.mesh_splitter)
            self._updateProgress("Splitting mesh...", 40)
            if self._cut_mode == self.CUT_MODE_SHORTEST:
                split_result = split_by_shortest_seam(tm, face_set_a, face_set_b)
            elif self._cut_mode == self.CUT_MODE_SMALLEST:
                # Use graph-based local separation with candidate fallback.
                # If the best plane only grazes the surface (single-triangle cut),
                # try the next-best candidates until we get a meaningful partition.
                candidate_normals = []
                if search_result and search_result.top_candidates:
                    candidate_normals = [n for _, n in search_result.top_candidates]
                if not candidate_normals:
                    candidate_normals = [plane.normal]

                split_result = split_by_local_plane(
                    tm, plane.origin, candidate_normals, click_face_id
                )
            else:
                split_result = slice_mesh_with_fallback(
                    tm, plane.origin, plane.normal, face_id=click_face_id
                )

            Logger.log("i", "Split result: %s", split_result.summary())

            if not split_result.success:
                Logger.log("e", "Cut operation failed: %s", split_result.error)
                self._closeProgress()
                return

            mesh_upper = split_result.upper
            mesh_lower = split_result.lower

            # Add connectors if enabled (delegates to core.connectors)
            if self._connector_enabled:
                if self._cut_mode == self.CUT_MODE_SHORTEST:
                    Logger.log("w", "Skipping connectors - not supported for shortest seam mode")
                elif split_result.capped and plane is not None:
                    self._updateProgress("Adding connectors...", 60)
                    config = ConnectorConfig(
                        enabled=True,
                        diameter=self._connector_diameter,
                        height=self._connector_height,
                        clearance=self._connector_clearance,
                        sides=self._connector_sides,
                    )
                    connector_result = add_connectors(
                        mesh_upper, mesh_lower,
                        plane.origin, plane.normal, config
                    )
                    if connector_result.connectors_added:
                        mesh_upper = connector_result.upper
                        mesh_lower = connector_result.lower
                        Logger.log("i", "Connectors added: peg on %s, hole on %s (engine: %s)",
                                   connector_result.peg_on, connector_result.hole_on,
                                   connector_result.hole_engine)
                    else:
                        Logger.log("w", "Connectors skipped: %s", connector_result.skipped_reason)
                else:
                    Logger.log("w", "Skipping connectors - mesh was not capped (open edges)")

            # Create Cura scene nodes from result meshes
            self._updateProgress("Creating new objects...", 80)
            original_name = node.getName()

            op = GroupedOperation()

            node_upper = self._createMeshNode(
                mesh_upper.vertices, mesh_upper.faces,
                f"{original_name}_part1"
            )
            node_lower = self._createMeshNode(
                mesh_lower.vertices, mesh_lower.faces,
                f"{original_name}_part2"
            )

            self._updateProgress("Finalizing...", 90)
            scene_root = self._controller.getScene().getRoot()

            op.addOperation(AddSceneNodeOperation(node_upper, scene_root))
            op.addOperation(AddSceneNodeOperation(node_lower, scene_root))
            op.addOperation(RemoveSceneNodeOperation(node))

            op.push()

            CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node_upper)

            self._updateProgress("Done!", 100)
            Logger.log("i", "Split complete: created '%s' and '%s'",
                       node_upper.getName(), node_lower.getName())

        except Exception as e:
            Logger.log("e", "Error during cut operation: %s", str(e))
        finally:
            self._closeProgress()

    def _createMeshNode(self, vertices: numpy.ndarray, faces: numpy.ndarray, name: str) -> CuraSceneNode:
        """Create a new CuraSceneNode from vertices and faces."""
        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices.astype(numpy.float32))
        mesh_builder.setIndices(faces.astype(numpy.int32))
        mesh_builder.calculateNormals()

        mesh_data = mesh_builder.build()

        node = CuraSceneNode()
        node.setName(name)
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        node.setMeshData(mesh_data)
        node.calculateBoundingBoxMesh()

        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        node.addDecorator(BuildPlateDecorator(active_build_plate))
        node.addDecorator(SliceableObjectDecorator())

        return node
