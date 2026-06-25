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
from UM.Mesh.MeshData import MeshData
from UM.Scene.Selection import Selection
from UM.Scene.SceneNode import SceneNode
from UM.View.GL.OpenGL import OpenGL
from UM.Resources import Resources

from cura.CuraApplication import CuraApplication
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.PickingPass import PickingPass

from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.Operation import Operation
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
    create_dot_mesh_data,
)
from .core.plane_calculator import (
    CutPlane,
    horizontal_cut_plane,
    vertical_cut_plane,
    find_plane_along_normal,
    find_smallest_cut_plane,
    find_valley_cut_plane,
    find_valley_seam_partition,
    find_shortest_seam_partition,
    refine_partition_with_mincut,
    snap_point_to_mesh_surface,
)
from .core.mesh_splitter import (
    slice_mesh_with_fallback,
    split_by_shortest_seam,
    split_by_local_plane,
    clean_local_plane_split,
    local_plane_partition,
    split_by_face_sets,
    prune_small_components,
)
from .core.connectors import (
    add_connectors,
    add_connectors_at_position,
    add_cap_native_connectors,
    add_path_connectors,
    boolean_engine_available,
    find_connector_position,
    ConnectorConfig,
)
from .core.path_editing import (
    append_waypoint,
    insert_waypoint_nearest_segment,
    remove_selected_waypoint,
)


class _PathStateOperation(Operation):
    """Undoable operation for ObjectSplitter path-editing state only."""

    def __init__(self, tool, undo_state=None, redo_state=None) -> None:
        super().__init__()
        self._tool = tool
        self._undo_state = undo_state
        self._redo_state = redo_state

    def undo(self) -> None:
        if self._tool is not None and self._undo_state is not None:
            self._tool._restorePathToolState(self._undo_state)

    def redo(self) -> None:
        if self._tool is not None and self._redo_state is not None:
            self._tool._restorePathToolState(self._redo_state)


class _PathStateGroupedOperation(GroupedOperation):
    """Grouped scene operation that also restores ObjectSplitter path state on undo/redo."""

    def __init__(self, tool, undo_state=None, redo_state=None) -> None:
        super().__init__()
        self._tool = tool
        self._undo_state = undo_state
        self._redo_state = redo_state

    def undo(self) -> None:
        super().undo()
        if self._tool is not None and self._undo_state is not None:
            self._tool._restorePathToolState(self._undo_state)

    def redo(self) -> None:
        super().redo()
        if self._tool is not None and self._redo_state is not None:
            self._tool._restorePathToolState(self._redo_state)


class _ColoredMarkerNode(SceneNode):
    """Scene node rendered with a uniform RGBA color via Uranium's color.shader.

    The default Cura mesh renderer ignores per-vertex colors and paints all
    model nodes with the theme model color (yellow on the build plate). This
    node overrides render() to queue itself with color.shader and a u_color
    uniform so we get the requested marker color.
    """

    _shader_cache = None

    def __init__(self, rgba):
        super().__init__()
        self.setName("ObjectSplitter_Marker")
        self.setSelectable(False)
        self.setCalculateBoundingBox(False)
        self._rgba = numpy.asarray(rgba, dtype=numpy.float32).reshape(4)

    def setColor(self, rgba) -> None:
        self._rgba = numpy.asarray(rgba, dtype=numpy.float32).reshape(4)

    def render(self, renderer) -> bool:
        if _ColoredMarkerNode._shader_cache is None:
            try:
                shader = OpenGL.getInstance().createShaderProgram(
                    Resources.getPath(Resources.Shaders, "color.shader")
                )
                # color.shader does not declare a binding for u_color; register
                # one so RenderBatch.addItem(uniforms={"u_color": ...}) actually
                # forwards the value via ShaderProgram.updateBindings().
                shader.addBinding("u_color", "u_color")
                _ColoredMarkerNode._shader_cache = shader
            except Exception as exc:
                Logger.log("w", "ObjectSplitter: failed to load color.shader: %s", exc)
                _ColoredMarkerNode._shader_cache = False
        shader = _ColoredMarkerNode._shader_cache
        if not shader or self.getMeshData() is None:
            return False
        # u_color MUST be a plain Python list of 4 floats. Cura's
        # ShaderProgram._setUniformValueDirect dispatches on `type(value) is
        # list` to build a QVector4D; a numpy.ndarray fails that strict check
        # and is passed straight to Qt's setUniformValue, which raises
        # "TypeError: arguments did not match any overloaded call" and crashes
        # the render loop. Do not "optimise" this to pass self._rgba directly.
        rgba = [float(c) for c in self._rgba]
        renderer.queueNode(
            self,
            shader=shader,
            transparent=(rgba[3] < 1.0),
            uniforms={"u_color": rgba},
        )
        return True


class ObjectSplitter(Tool):
    """Tool for splitting 3D objects into multiple parts by cutting along planes."""

    # Cut mode constants
    CUT_MODE_PLANE = "plane"                # Predictable plane cut across the clicked surface
    CUT_MODE_HORIZONTAL = "horizontal"      # Cut parallel to build plate
    CUT_MODE_VERTICAL = "vertical"          # Cut perpendicular to build plate
    CUT_MODE_SMALLEST = "smallest"          # Find smallest cross-section
    CUT_MODE_CUSTOM = "custom"              # User-defined plane orientation
    CUT_MODE_SHORTEST = "shortest"          # Shortest seam (plane search + refine)
    CUT_MODE_RADIAL = "radial"              # Radial geodesic cut (dual-Dijkstra)
    CUT_MODE_PATH = "path"                  # Multi-point geodesic path cut
    CUT_MODE_PATH_ISOLATE = "path_isolate"  # Multi-loop isolation cut
    CUT_MODE_VALLEY = "valley"              # Geographic valley/groove detection
    CUT_MODE_VALLEY_SEAM = "valley_seam"    # Concavity-biased seam cut

    def __init__(self):
        super().__init__()
        self._shortcut_key = Qt.Key.Key_K  # K for "Kut" (avoiding conflicts)
        self._controller = self.getController()

        # Cut settings
        self._cut_mode = self.CUT_MODE_PATH
        self._cut_height = 0.0  # For horizontal cuts: Z position (relative to object)
        self._cut_height_percent = 50.0  # Percentage of object height
        self._plane_orientation = "surface"  # surface | horizontal | vertical
        self._cut_whole_model = False  # Plane mode: cut whole model vs clicked feature
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
        prefs.addPreference("objectsplitter/valley_sdf_bias_enabled", False)
        prefs.addPreference("objectsplitter/multi_point_anchors_enabled", False)
        prefs.addPreference("objectsplitter/path_close_loop", True)
        prefs.addPreference("objectsplitter/path_small_markers", False)
        prefs.addPreference("objectsplitter/path_marker_color", "cyan")
        prefs.addPreference("objectsplitter/path_cap_ends", True)
        prefs.addPreference("objectsplitter/path_isolate_prune_tiny_fragments", True)
        prefs.addPreference("objectsplitter/path_isolate_tiny_fragment_face_threshold", 80)
        self._openscad_path = prefs.getValue("objectsplitter/openscad_path")
        valley_sdf_pref = prefs.getValue("objectsplitter/valley_sdf_bias_enabled")
        self._valley_sdf_bias_enabled = (
            valley_sdf_pref if isinstance(valley_sdf_pref, bool)
            else str(valley_sdf_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        anchor_pref = prefs.getValue("objectsplitter/multi_point_anchors_enabled")
        self._multi_point_anchors_enabled = (
            anchor_pref if isinstance(anchor_pref, bool)
            else str(anchor_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        close_loop_pref = prefs.getValue("objectsplitter/path_close_loop")
        self._path_close_loop = (
            close_loop_pref if isinstance(close_loop_pref, bool)
            else str(close_loop_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        small_markers_pref = prefs.getValue("objectsplitter/path_small_markers")
        self._path_small_markers = (
            small_markers_pref if isinstance(small_markers_pref, bool)
            else str(small_markers_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        marker_color_pref = str(prefs.getValue("objectsplitter/path_marker_color") or "cyan").strip().lower()
        self._path_marker_color = (
            marker_color_pref
            if marker_color_pref in ("cyan", "yellow", "white", "black", "magenta", "green")
            else "cyan"
        )
        path_cap_pref = prefs.getValue("objectsplitter/path_cap_ends")
        self._path_cap_ends = (
            path_cap_pref if isinstance(path_cap_pref, bool)
            else str(path_cap_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        prune_fragments_pref = prefs.getValue("objectsplitter/path_isolate_prune_tiny_fragments")
        self._path_isolate_prune_tiny_fragments = (
            prune_fragments_pref if isinstance(prune_fragments_pref, bool)
            else str(prune_fragments_pref).strip().lower() in ("1", "true", "yes", "on")
        )
        prune_threshold_pref = prefs.getValue("objectsplitter/path_isolate_tiny_fragment_face_threshold")
        try:
            self._path_isolate_tiny_fragment_face_threshold = max(0, int(prune_threshold_pref))
        except (TypeError, ValueError):
            self._path_isolate_tiny_fragment_face_threshold = 80
        # Search settings for smallest cut
        self._search_resolution = 18  # Number of angles to search

        # Debug capture (derived from _capture_dir: None = off, path = on)
        self._capture_dir = None

        # Path mode state
        self._path_waypoints = []  # List of 3D positions (numpy arrays)
        self._path_node = None  # The node being path-cut
        self._path_preview_nodes = []  # Marker nodes for each waypoint
        self._path_suggestion_nodes = []  # Marker nodes for suggested path
        self._path_insert_mode = False
        self._selected_path_waypoint_index = None
        # Whether to close the path loop. Default is on and persisted in prefs.
        # (Value already loaded above.)
        self._drag_waypoint_index = None
        self._drag_waypoint_node = None
        self._drag_waypoint_undo_state = None
        self._path_isolate_loops = []  # Finalized closed waypoint loops
        self._path_isolate_preview_nodes = []  # Preview markers for finalized loops
        self._path_isolate_target_pick_active = False
        self._path_isolate_target_point = None
        self._path_isolate_target_face_id = None
        self._path_isolate_target_preview_node = None

        # State
        self._selection_pass = None
        self._last_picked_node = None
        self._last_picked_position = None
        self._hover_node = None  # Node currently being hovered over
        # Cached world-space trimesh for the hovered node, so the surface-normal
        # arrow can be computed cheaply per mouse-move (rebuilt only when the
        # node or its transform changes).
        self._hover_tm = None
        self._hover_tm_node = None
        self._hover_tm_xform = None
        self._hover_kdtree = None  # KD-tree of hover trimesh vertices (fast normal lookup)
        self._picking_pass = None  # Cached picking pass
        self._progress_dialog = None  # Progress dialog for long operations

        self.setExposedProperties(
            "CutMode",
            "CutModes",
            "CutHeightPercent",
            "PlaneOrientation",
            "CutWholeModel",
            "ShowPreview",
            "TrimeshAvailable",
            "SearchResolution",
            "ValleySdfBiasEnabled",
            # Connector properties
            "ConnectorEnabled",
            "OpenScadPath",
            "ConnectorDiameter",
            "ConnectorHeight",
            "ConnectorClearance",
            # (Note: ConnectorSides not exposed to QML to avoid undefined warnings)
            # Debug capture
            "DebugCaptureEnabled",
            # Path mode
            "PathPointCount",
            "PathLoopCount",
            "CurrentLoopPointCount",
            "PathCloseLoop",
            "PathCapEnds",
            "PathSmallMarkers",
            "PathMarkerColor",
            "PathInsertMode",
            "HasSelectedPathPoint",
            "SelectedPathPointIndex",
            "PathIsolatePruneTinyFragments",
            "PathIsolateTinyFragmentFaceThreshold",
            "PathIsolateTargetPicked",
            "PathIsolateTargetPickActive",
            "TriggerPathCut",
            "TriggerPathIsolate",
            "TriggerSuggestPath",
            "TriggerStartNewPathLoop",
            "TriggerClearCurrentPathLoop",
            "TriggerPickPathIsolateTarget",
            "RemoveSelectedPathPoint",
            # Multi-point anchoring for valley/seam modes
            "MultiPointAnchorsEnabled",
            "TriggerAnchoredCut",
            "ClearPathPoints",
        )

        Logger.log("d", "Object Splitter Tool initialized (trimesh available: %s)", str(TRIMESH_AVAILABLE))

        CuraApplication.getInstance().globalContainerStackChanged.connect(self._updateEnabled)
        Selection.selectionChanged.connect(self._onSelectionChanged)
        # Clear path markers if the object they belong to is deleted from the
        # scene (otherwise the waypoint dots are left floating).
        try:
            CuraApplication.getInstance().getController().getScene().sceneChanged.connect(self._onSceneChanged)
        except Exception as e:
            Logger.log("w", "Object Splitter: could not connect sceneChanged: %s", str(e))

    def _onSceneChanged(self, *args):
        """Drop path state when the tracked object is removed from the scene."""
        node = self._path_node
        if node is None:
            return
        try:
            removed = node.getParent() is None
        except Exception:
            removed = True
        if removed:
            Logger.log("i", "Object Splitter: tracked object removed; clearing path markers")
            self._clearPathWaypoints()

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
            self._clearPathWaypoints()  # Clear path state on mode change
            Logger.log("d", "Cut mode changed to: %s", mode)
            self.propertyChanged.emit()

    def getCutModes(self) -> list:
        """Return available cut modes for QML dropdown."""
        return [
            {"value": self.CUT_MODE_HORIZONTAL, "text": "Horizontal (parallel to bed)"},
            {"value": self.CUT_MODE_VERTICAL, "text": "Vertical"},
            {"value": self.CUT_MODE_SMALLEST, "text": "Smallest cross-section"},
            {"value": self.CUT_MODE_SHORTEST, "text": "Shortest seam (surface loop)"},
            {"value": self.CUT_MODE_RADIAL, "text": "Radial (geodesic)"},
            {"value": self.CUT_MODE_PATH, "text": "Path (multi-point)"},
            {"value": self.CUT_MODE_PATH_ISOLATE, "text": "Path Isolate (multi-loop)"},
            {"value": self.CUT_MODE_VALLEY, "text": "Valley (geographic groove)"},
            {"value": self.CUT_MODE_VALLEY_SEAM, "text": "Valley Seam (concavity path)"},
        ]

    def getCutHeightPercent(self) -> float:
        return self._cut_height_percent

    def setCutHeightPercent(self, value: float) -> None:
        if value != self._cut_height_percent:
            self._cut_height_percent = float(value)
            Logger.log("d", "Cut height percent changed to: %s", str(value))
            self.propertyChanged.emit()

    def getPlaneOrientation(self) -> str:
        return self._plane_orientation

    def setPlaneOrientation(self, value: str) -> None:
        value = str(value)
        if value != self._plane_orientation:
            self._plane_orientation = value
            self._clearPathWaypoints()  # clear any placed 3-point markers
            self._removePreview()
            Logger.log("d", "Plane orientation changed to: %s", value)
            self.propertyChanged.emit()

    def getCutWholeModel(self) -> bool:
        return self._cut_whole_model

    def setCutWholeModel(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._cut_whole_model:
            self._cut_whole_model = enabled
            Logger.log("d", "Cut whole model: %s", enabled)
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

    def getValleySdfBiasEnabled(self) -> bool:
        return self._valley_sdf_bias_enabled

    def setValleySdfBiasEnabled(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._valley_sdf_bias_enabled:
            self._valley_sdf_bias_enabled = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/valley_sdf_bias_enabled", enabled)
            Logger.log("i", "Valley SDF bias enabled: %s", str(enabled))
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
    # Path Mode Properties
    # ==========================================================================

    def getPathPointCount(self) -> int:
        return len(self._path_waypoints)

    def getPathLoopCount(self) -> int:
        return len(self._path_isolate_loops)

    def getCurrentLoopPointCount(self) -> int:
        return len(self._path_waypoints)

    def getPathCloseLoop(self) -> bool:
        return self._path_close_loop

    def setPathCloseLoop(self, value: bool):
        enabled = bool(value)
        if enabled != self._path_close_loop:
            self._path_close_loop = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_close_loop", enabled)
            # Suggested path preview can become stale when loop setting flips.
            self._clearSuggestedPath()
            self.propertyChanged.emit()

    def getPathCapEnds(self) -> bool:
        return self._path_cap_ends

    def setPathCapEnds(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._path_cap_ends:
            self._path_cap_ends = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_cap_ends", enabled)
            Logger.log("i", "Path cut capping: %s", "on" if enabled else "off")
            self.propertyChanged.emit()

    def getPathSmallMarkers(self) -> bool:
        return self._path_small_markers

    def setPathSmallMarkers(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._path_small_markers:
            self._path_small_markers = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_small_markers", enabled)
            self._rebuildPathWaypointMarkers()
            self._rebuildPathIsolateLoopMarkers()
            self._rebuildPathIsolateTargetMarker()
            Logger.log("i", "Path markers: %s", "small" if enabled else "default")
            self.propertyChanged.emit()

    def getPathMarkerColor(self) -> str:
        return self._path_marker_color

    def setPathMarkerColor(self, value: str) -> None:
        color_name = str(value or "cyan").strip().lower()
        if color_name not in ("cyan", "yellow", "white", "black", "magenta", "green"):
            color_name = "cyan"
        if color_name != self._path_marker_color:
            self._path_marker_color = color_name
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_marker_color", color_name)
            self._rebuildPathWaypointMarkers()
            Logger.log("i", "Path marker color: %s", color_name)
            self.propertyChanged.emit()

    def getPathIsolatePruneTinyFragments(self) -> bool:
        return self._path_isolate_prune_tiny_fragments

    def setPathIsolatePruneTinyFragments(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._path_isolate_prune_tiny_fragments:
            self._path_isolate_prune_tiny_fragments = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_isolate_prune_tiny_fragments", enabled)
            Logger.log("i", "Path isolate tiny-fragment cleanup: %s", "on" if enabled else "off")
            self.propertyChanged.emit()

    def getPathIsolateTinyFragmentFaceThreshold(self) -> int:
        return int(self._path_isolate_tiny_fragment_face_threshold)

    def setPathIsolateTinyFragmentFaceThreshold(self, value: int) -> None:
        try:
            threshold = max(0, int(value))
        except (TypeError, ValueError):
            threshold = 0
        if threshold != self._path_isolate_tiny_fragment_face_threshold:
            self._path_isolate_tiny_fragment_face_threshold = threshold
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/path_isolate_tiny_fragment_face_threshold", threshold)
            Logger.log("i", "Path isolate tiny-fragment threshold: %d faces", threshold)
            self.propertyChanged.emit()

    def getPathInsertMode(self) -> bool:
        return self._path_insert_mode

    def setPathInsertMode(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._path_insert_mode:
            self._path_insert_mode = enabled
            Logger.log("i", "Path insert mode: %s", "on" if enabled else "off")
            self.propertyChanged.emit()

    def getHasSelectedPathPoint(self) -> bool:
        return (
            self._isPathSelectionEnabled() and
            self._selected_path_waypoint_index is not None and
            0 <= self._selected_path_waypoint_index < len(self._path_waypoints)
        )

    def getSelectedPathPointIndex(self) -> int:
        if not self.getHasSelectedPathPoint():
            return -1
        return int(self._selected_path_waypoint_index)

    def getTriggerPathCut(self) -> bool:
        return False  # Write-only property, always reads as False

    def setTriggerPathCut(self, value: bool):
        """Called from QML to trigger the path cut."""
        if value:
            self._executePathCut()

    def getTriggerPathIsolate(self) -> bool:
        return False

    def setTriggerPathIsolate(self, value: bool):
        if value:
            self._executePathIsolate()

    def getTriggerSuggestPath(self) -> bool:
        return False  # Write-only property

    def setTriggerSuggestPath(self, value: bool):
        """Called from QML to compute and show a suggested geodesic path."""
        if value:
            self._suggestPathFromPoints()

    def getRemoveSelectedPathPoint(self) -> bool:
        return False  # Write-only property

    def setRemoveSelectedPathPoint(self, value: bool):
        if value:
            self._removeSelectedPathWaypoint()

    def getPathIsolateTargetPicked(self) -> bool:
        return (
            self._cut_mode == self.CUT_MODE_PATH_ISOLATE and
            self._path_isolate_target_face_id is not None and
            self._path_isolate_target_point is not None
        )

    def getPathIsolateTargetPickActive(self) -> bool:
        return (
            self._cut_mode == self.CUT_MODE_PATH_ISOLATE and
            self._path_isolate_target_pick_active
        )

    def getTriggerStartNewPathLoop(self) -> bool:
        return False

    def setTriggerStartNewPathLoop(self, value: bool):
        if value:
            self._startNewPathLoop()

    def getTriggerClearCurrentPathLoop(self) -> bool:
        return False

    def setTriggerClearCurrentPathLoop(self, value: bool):
        if value:
            self._clearCurrentPathLoopUndoable()

    def getTriggerPickPathIsolateTarget(self) -> bool:
        return False

    def setTriggerPickPathIsolateTarget(self, value: bool):
        if value:
            self._armPathIsolateTargetPick()

    def getMultiPointAnchorsEnabled(self) -> bool:
        return self._multi_point_anchors_enabled

    def setMultiPointAnchorsEnabled(self, value: bool) -> None:
        enabled = bool(value)
        if enabled != self._multi_point_anchors_enabled:
            self._multi_point_anchors_enabled = enabled
            prefs = Application.getInstance().getPreferences()
            prefs.setValue("objectsplitter/multi_point_anchors_enabled", enabled)
            Logger.log("i", "Multi-point anchors enabled: %s", str(enabled))
            self.propertyChanged.emit()

    def getTriggerAnchoredCut(self) -> bool:
        return False  # Write-only property

    def setTriggerAnchoredCut(self, value: bool):
        """Called from QML to trigger an anchored valley/valley-seam cut."""
        if value:
            self._executeAnchoredCut()

    def getClearPathPoints(self) -> bool:
        return False  # Write-only property

    def setClearPathPoints(self, value: bool):
        """Called from QML to clear any placed points."""
        if value:
            self._clearPathWaypointsUndoable()

    def _clearPathWaypoints(self):
        """Clear all path waypoints and remove preview markers."""
        self._path_waypoints.clear()
        self._path_node = None
        self._selected_path_waypoint_index = None
        self._drag_waypoint_index = None
        self._drag_waypoint_node = None
        self._drag_waypoint_undo_state = None
        self._path_isolate_loops.clear()
        self._path_isolate_target_pick_active = False
        self._path_isolate_target_point = None
        self._path_isolate_target_face_id = None
        self._clearSuggestedPath()
        self._clearWaypointPreviewNodes()
        self._clearPathIsolatePreviewNodes()
        self._clearPathIsolateTargetPreview()
        self._removePreview()
        self.propertyChanged.emit()

    def _clearPathWaypointsUndoable(self):
        undo_state = self._capturePathToolState()
        self._clearPathWaypoints()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _pushPathStateUndoOperation(self, undo_state: Optional[dict], redo_state: Optional[dict]) -> None:
        """Push a path-state-only operation onto Cura's undo stack."""
        if undo_state is None or redo_state is None:
            return
        try:
            op = _PathStateOperation(self, undo_state=undo_state, redo_state=redo_state)
            op.push()
        except Exception as e:
            Logger.log("w", "Could not push path edit undo operation: %s", str(e))

    def _clearSuggestedPath(self):
        """Remove suggested path preview markers."""
        for node in self._path_suggestion_nodes:
            try:
                if node.getParent():
                    node.setParent(None)
            except Exception:
                pass
        self._path_suggestion_nodes.clear()

    def _clearWaypointPreviewNodes(self):
        for node in self._path_preview_nodes:
            try:
                if node.getParent():
                    node.setParent(None)
            except Exception:
                pass
        self._path_preview_nodes.clear()

    def _clearPathIsolatePreviewNodes(self):
        for node in self._path_isolate_preview_nodes:
            try:
                if node.getParent():
                    node.setParent(None)
            except Exception:
                pass
        self._path_isolate_preview_nodes.clear()

    def _clearPathIsolateTargetPreview(self):
        if self._path_isolate_target_preview_node is None:
            return
        try:
            if self._path_isolate_target_preview_node.getParent():
                self._path_isolate_target_preview_node.setParent(None)
        except Exception:
                pass
        self._path_isolate_target_preview_node = None

    def _capturePathToolState(self) -> dict:
        """Capture the current path/path-isolate editing state for undo/redo restore."""
        return {
            "cut_mode": self._cut_mode,
            "path_node": self._path_node,
            "path_waypoints": [
                numpy.asarray(point, dtype=numpy.float64).copy()
                for point in self._path_waypoints
            ],
            "selected_path_waypoint_index": self._selected_path_waypoint_index,
            "path_isolate_loops": [
                [numpy.asarray(point, dtype=numpy.float64).copy() for point in loop]
                for loop in self._path_isolate_loops
            ],
            "path_isolate_target_pick_active": bool(self._path_isolate_target_pick_active),
            "path_isolate_target_point": (
                None if self._path_isolate_target_point is None
                else numpy.asarray(self._path_isolate_target_point, dtype=numpy.float64).copy()
            ),
            "path_isolate_target_face_id": self._path_isolate_target_face_id,
        }

    def _restorePathToolState(self, state: Optional[dict]) -> None:
        """Restore previously captured path/path-isolate editing state."""
        if state is None:
            return

        self._clearSuggestedPath()
        self._clearWaypointPreviewNodes()
        self._clearPathIsolatePreviewNodes()
        self._clearPathIsolateTargetPreview()
        self._removePreview()

        self._cut_mode = state.get("cut_mode", self._cut_mode)
        self._path_node = state.get("path_node")
        self._path_waypoints = [
            numpy.asarray(point, dtype=numpy.float64).copy()
            for point in state.get("path_waypoints", [])
        ]
        self._selected_path_waypoint_index = state.get("selected_path_waypoint_index")
        self._drag_waypoint_index = None
        self._drag_waypoint_node = None
        self._path_isolate_loops = [
            [numpy.asarray(point, dtype=numpy.float64).copy() for point in loop]
            for loop in state.get("path_isolate_loops", [])
        ]
        self._path_isolate_target_pick_active = bool(
            state.get("path_isolate_target_pick_active", False)
        )
        target_point = state.get("path_isolate_target_point")
        self._path_isolate_target_point = (
            None if target_point is None
            else numpy.asarray(target_point, dtype=numpy.float64).copy()
        )
        self._path_isolate_target_face_id = state.get("path_isolate_target_face_id")

        self._rebuildPathWaypointMarkers()
        self._rebuildPathIsolateLoopMarkers()
        self._rebuildPathIsolateTargetMarker()
        Logger.log(
            "i",
            "Restored path editing state: mode=%s, waypoints=%d, loops=%d, target=%s",
            self._cut_mode,
            len(self._path_waypoints),
            len(self._path_isolate_loops),
            "yes" if self._path_isolate_target_face_id is not None else "no",
        )
        self.propertyChanged.emit()

    def _isPathSelectionEnabled(self) -> bool:
        return self._cut_mode in (self.CUT_MODE_PATH, self.CUT_MODE_PATH_ISOLATE)

    def _getWaypointModeLabel(self) -> str:
        if self._cut_mode == self.CUT_MODE_PATH_ISOLATE:
            return "Path isolate"
        if self._cut_mode == self.CUT_MODE_PATH:
            return "Path mode"
        return "Point mode"

    def _setSelectedPathWaypoint(self, index: Optional[int]) -> None:
        if not self._isPathSelectionEnabled():
            index = None
        elif index is not None and (index < 0 or index >= len(self._path_waypoints)):
            index = None

        if index == self._selected_path_waypoint_index:
            return

        self._selected_path_waypoint_index = index
        self._rebuildPathWaypointMarkers()
        if index is None:
            Logger.log("i", "%s: cleared point selection", self._getWaypointModeLabel())
        else:
            Logger.log("i", "%s: selected waypoint %d", self._getWaypointModeLabel(), index + 1)
        self.propertyChanged.emit()

    _SELECTED_MARKER_COLOR = numpy.array([0.82, 0.18, 0.18, 1.0], dtype=numpy.float32)
    _MARKER_COLORS = {
        "cyan": numpy.array([0.0, 0.9, 1.0, 1.0], dtype=numpy.float32),
        "yellow": numpy.array([1.0, 0.88, 0.0, 1.0], dtype=numpy.float32),
        "white": numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float32),
        "black": numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float32),
        "magenta": numpy.array([1.0, 0.0, 0.85, 1.0], dtype=numpy.float32),
        "green": numpy.array([0.0, 0.95, 0.25, 1.0], dtype=numpy.float32),
    }

    def _getWaypointMarkerColor(self, index: int) -> numpy.ndarray:
        if self._isPathSelectionEnabled() and index == self._selected_path_waypoint_index:
            return self._SELECTED_MARKER_COLOR
        return self._MARKER_COLORS.get(self._path_marker_color, self._MARKER_COLORS["cyan"])

    def _rebuildPathWaypointMarkers(self):
        self._clearWaypointPreviewNodes()
        if self._path_node is None or not self._path_waypoints:
            return
        if not self._path_node.getMeshData():
            return

        dot_size = self._getWaypointDotSize(self._path_node)
        scene_root = self._controller.getScene().getRoot()

        for index, point in enumerate(self._path_waypoints):
            color = self._getWaypointMarkerColor(index)
            marker_node = self._createColoredMarkerNode(color)
            vertices, indices = create_dot_mesh_data(
                numpy.asarray(point, dtype=numpy.float64),
                dot_size,
            )
            mesh_builder = MeshBuilder()
            mesh_builder.setVertices(vertices)
            mesh_builder.setIndices(indices)
            mesh_builder.calculateNormals()
            marker_node.setMeshData(mesh_builder.build())
            marker_node.setParent(scene_root)
            self._path_preview_nodes.append(marker_node)

    def _shouldInsertPathWaypoint(self) -> bool:
        return (
            self._cut_mode == self.CUT_MODE_PATH and
            self._path_insert_mode and
            len(self._path_waypoints) >= 2
        )

    def _addPathWaypoint(self, position, picked_node):
        """Add a waypoint for path/anchor modes and show a visible dot marker."""
        undo_state = self._capturePathToolState()
        pos_arr = numpy.array([position.x, position.y, position.z])

        # Ensure all waypoints are on the same node
        if self._path_node is not None and self._path_node != picked_node:
            Logger.log("w", "%s: clicked a different object, clearing point state", self._getWaypointModeLabel())
            self._clearPathWaypoints()

        self._path_node = picked_node
        if self._cut_mode == self.CUT_MODE_PATH_ISOLATE and self._path_isolate_target_face_id is not None:
            self._path_isolate_target_face_id = None
            self._path_isolate_target_point = None
            self._clearPathIsolateTargetPreview()
            Logger.log("i", "Path isolate: cleared target selection because loop geometry changed")
        if self._shouldInsertPathWaypoint():
            self._path_waypoints, inserted_index = insert_waypoint_nearest_segment(
                self._path_waypoints,
                pos_arr,
            )
            Logger.log("i", "%s: inserted waypoint %d at %s", self._getWaypointModeLabel(), inserted_index + 1, pos_arr)
        else:
            self._path_waypoints, inserted_index = append_waypoint(self._path_waypoints, pos_arr)
            Logger.log("i", "%s: appended waypoint %d at %s", self._getWaypointModeLabel(), inserted_index + 1, pos_arr)

        if self._isPathSelectionEnabled():
            self._selected_path_waypoint_index = inserted_index
        else:
            self._selected_path_waypoint_index = None
        self._clearSuggestedPath()  # Invalidate previous suggestion
        self._rebuildPathWaypointMarkers()
        self.propertyChanged.emit()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _isPointPlacementMode(self) -> bool:
        return (
            self._cut_mode == self.CUT_MODE_PATH or
            self._cut_mode == self.CUT_MODE_PATH_ISOLATE or
            (
                self._cut_mode == self.CUT_MODE_PLANE and
                self._plane_orientation == "points"
            ) or
            (
                self._multi_point_anchors_enabled and
                self._cut_mode in (self.CUT_MODE_VALLEY, self.CUT_MODE_VALLEY_SEAM)
            )
        )

    def _getWaypointDotSize(self, node) -> float:
        mesh_data = node.getMeshData()
        if mesh_data is None:
            return 0.18 if self._path_small_markers else 0.8
        transformed = mesh_data.getTransformed(node.getWorldTransformation())
        verts = transformed.getVertices()
        if verts is None or len(verts) == 0:
            return 0.18 if self._path_small_markers else 0.8
        mesh_size = verts.max(axis=0) - verts.min(axis=0)
        mesh_size_max = max(mesh_size[0], mesh_size[1], mesh_size[2])
        scale = 0.0032 if self._path_small_markers else 0.01
        minimum = 0.12 if self._path_small_markers else 0.4
        return max(minimum, float(mesh_size_max) * scale)

    def _findNearbyWaypointIndex(self, position, picked_node) -> Optional[int]:
        if self._path_node is None or self._path_node != picked_node:
            return None
        if not self._path_waypoints:
            return None

        pos_arr = numpy.array([position.x, position.y, position.z], dtype=numpy.float64)
        waypoints = numpy.asarray(self._path_waypoints, dtype=numpy.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            return None

        dists = numpy.linalg.norm(waypoints - pos_arr[None, :], axis=1)
        idx = int(numpy.argmin(dists))
        min_pick_radius = 0.22 if self._path_small_markers else 0.9
        radius_scale = 1.7 if self._path_small_markers else 2.5
        pick_radius = max(min_pick_radius, self._getWaypointDotSize(picked_node) * radius_scale)
        if float(dists[idx]) <= pick_radius:
            return idx
        return None

    def _updateWaypointMarker(self, index: int):
        if self._path_node is None:
            return
        if index < 0 or index >= len(self._path_waypoints):
            return
        if index >= len(self._path_preview_nodes):
            return

        pos_arr = numpy.asarray(self._path_waypoints[index], dtype=numpy.float64)
        dot_size = self._getWaypointDotSize(self._path_node)
        marker_node = self._path_preview_nodes[index]

        vertices, indices = create_dot_mesh_data(pos_arr, dot_size)
        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices)
        mesh_builder.setIndices(indices)
        mesh_builder.calculateNormals()
        marker_node.setMeshData(mesh_builder.build())
        color = self._getWaypointMarkerColor(index)
        if isinstance(marker_node, _ColoredMarkerNode):
            marker_node.setColor(color)

        scene_root = self._controller.getScene().getRoot()
        if marker_node.getParent() != scene_root:
            marker_node.setParent(scene_root)

    def _movePathWaypoint(self, index: int, position):
        if index < 0 or index >= len(self._path_waypoints):
            return
        pos_arr = numpy.array([position.x, position.y, position.z], dtype=numpy.float64)
        self._path_waypoints[index] = pos_arr
        self._clearSuggestedPath()
        self._updateWaypointMarker(index)

    def _rebuildPathIsolateLoopMarkers(self):
        self._clearPathIsolatePreviewNodes()
        if self._path_node is None or not self._path_isolate_loops:
            return
        if not self._path_node.getMeshData():
            return

        dot_size = self._getWaypointDotSize(self._path_node) * 0.9
        scene_root = self._controller.getScene().getRoot()
        loop_color = numpy.array([0.08, 0.55, 0.18, 0.95], dtype=numpy.float32)

        for loop in self._path_isolate_loops:
            for point in loop:
                marker_node = self._createColoredMarkerNode(loop_color)
                vertices, indices = create_dot_mesh_data(
                    numpy.asarray(point, dtype=numpy.float64),
                    dot_size,
                )
                mesh_builder = MeshBuilder()
                mesh_builder.setVertices(vertices)
                mesh_builder.setIndices(indices)
                mesh_builder.calculateNormals()
                marker_node.setMeshData(mesh_builder.build())
                marker_node.setParent(scene_root)
                self._path_isolate_preview_nodes.append(marker_node)

    def _rebuildPathIsolateTargetMarker(self):
        self._clearPathIsolateTargetPreview()
        if self._path_node is None or self._path_isolate_target_point is None:
            return
        dot_size = self._getWaypointDotSize(self._path_node) * 1.15
        target_color = numpy.array([0.95, 0.45, 0.05, 1.0], dtype=numpy.float32)
        marker_node = self._createColoredMarkerNode(target_color)
        vertices, indices = create_dot_mesh_data(
            numpy.asarray(self._path_isolate_target_point, dtype=numpy.float64),
            dot_size,
        )
        mesh_builder = MeshBuilder()
        mesh_builder.setVertices(vertices)
        mesh_builder.setIndices(indices)
        mesh_builder.calculateNormals()
        marker_node.setMeshData(mesh_builder.build())
        scene_root = self._controller.getScene().getRoot()
        marker_node.setParent(scene_root)
        self._path_isolate_target_preview_node = marker_node

    def _clearCurrentPathLoop(self):
        self._path_waypoints.clear()
        self._selected_path_waypoint_index = None
        self._drag_waypoint_index = None
        self._drag_waypoint_node = None
        self._drag_waypoint_undo_state = None
        self._clearSuggestedPath()
        self._clearWaypointPreviewNodes()
        self.propertyChanged.emit()

    def _clearCurrentPathLoopUndoable(self):
        undo_state = self._capturePathToolState()
        self._clearCurrentPathLoop()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _startNewPathLoop(self):
        if self._cut_mode != self.CUT_MODE_PATH_ISOLATE:
            return
        if len(self._path_waypoints) < 3:
            Logger.log("w", "Path isolate: need at least 3 points before starting a new loop")
            return

        undo_state = self._capturePathToolState()
        loop = [numpy.asarray(point, dtype=numpy.float64).copy() for point in self._path_waypoints]
        if not numpy.allclose(loop[0], loop[-1]):
            loop.append(loop[0].copy())
        if self._path_isolate_target_face_id is not None:
            self._path_isolate_target_face_id = None
            self._path_isolate_target_point = None
            self._clearPathIsolateTargetPreview()
            Logger.log("i", "Path isolate: cleared target selection because loop set changed")
        self._path_isolate_loops.append(loop)
        Logger.log("i", "Path isolate: finalized loop %d with %d points", len(self._path_isolate_loops), len(loop) - 1)
        self._clearCurrentPathLoop()
        self._rebuildPathIsolateLoopMarkers()
        self.propertyChanged.emit()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _armPathIsolateTargetPick(self):
        if self._cut_mode != self.CUT_MODE_PATH_ISOLATE:
            return
        # If at least one loop is already finished and a complete extra loop is
        # pending, finalize it automatically so the last loop needs no explicit
        # Finish Loop. The first loop must still be finished by hand, so this
        # never fires before/while the first loop is being drawn.
        if self._path_isolate_loops and len(self._path_waypoints) >= 3:
            self._startNewPathLoop()
        if not self._path_isolate_loops:
            Logger.log("w", "Path isolate: add at least one closed loop before picking a target")
            return
        if self._path_waypoints:
            Logger.log("w", "Path isolate: finish or clear the current loop before picking a target")
            return
        undo_state = self._capturePathToolState()
        self._path_isolate_target_pick_active = True
        Logger.log("i", "Path isolate: click the region you want to extract")
        self.propertyChanged.emit()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _setPathIsolateTargetFromClick(self, picked_node, picked_position) -> None:
        if self._cut_mode != self.CUT_MODE_PATH_ISOLATE:
            return
        undo_state = self._capturePathToolState()
        if self._path_node is not None and self._path_node != picked_node:
            Logger.log("w", "Path isolate: clicked a different object while picking target, clearing state")
            self._clearPathWaypoints()
            redo_state = self._capturePathToolState()
            self._pushPathStateUndoOperation(undo_state, redo_state)
            return
        if self._path_node is None:
            self._path_node = picked_node

        extraction = self._extractTrimesh(picked_node)
        if extraction is None:
            self._path_isolate_target_pick_active = False
            self.propertyChanged.emit()
            redo_state = self._capturePathToolState()
            self._pushPathStateUndoOperation(undo_state, redo_state)
            return

        tm, _, _ = extraction
        point_arr = numpy.array([picked_position.x, picked_position.y, picked_position.z], dtype=numpy.float64)
        snapped_point, face_id = snap_point_to_mesh_surface(tm, point_arr)
        if face_id is None:
            Logger.log("w", "Path isolate: could not determine a target face from that click")
            self._path_isolate_target_pick_active = False
            self.propertyChanged.emit()
            redo_state = self._capturePathToolState()
            self._pushPathStateUndoOperation(undo_state, redo_state)
            return

        self._path_isolate_target_point = numpy.asarray(snapped_point, dtype=numpy.float64)
        self._path_isolate_target_face_id = int(face_id)
        self._path_isolate_target_pick_active = False
        self._rebuildPathIsolateTargetMarker()
        Logger.log("i", "Path isolate: target selected on face %d at %s", int(face_id), self._path_isolate_target_point)
        self.propertyChanged.emit()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _removeSelectedPathWaypoint(self):
        if not self.getHasSelectedPathPoint():
            return
        undo_state = self._capturePathToolState()
        removed_index = int(self._selected_path_waypoint_index)
        self._path_waypoints, self._selected_path_waypoint_index = remove_selected_waypoint(
            self._path_waypoints,
            removed_index,
        )
        self._clearSuggestedPath()
        self._rebuildPathWaypointMarkers()
        if self._selected_path_waypoint_index is None:
            Logger.log("i", "%s: removed waypoint %d, selection cleared", self._getWaypointModeLabel(), removed_index + 1)
        else:
            Logger.log(
                "i",
                "%s: removed waypoint %d, selected waypoint %d",
                self._getWaypointModeLabel(),
                removed_index + 1,
                self._selected_path_waypoint_index + 1,
            )
        if not self._path_waypoints:
            if not self._path_isolate_loops:
                self._path_node = None
        self.propertyChanged.emit()
        redo_state = self._capturePathToolState()
        self._pushPathStateUndoOperation(undo_state, redo_state)

    def _dragWaypointToMouse(self, mouse_x: float, mouse_y: float):
        if self._drag_waypoint_index is None or self._drag_waypoint_node is None:
            return

        if self._selection_pass is None:
            self._selection_pass = Application.getInstance().getRenderer().getRenderPass("selection")
        picked_node = self._controller.getScene().findObject(
            self._selection_pass.getIdAtPosition(mouse_x, mouse_y)
        )
        if picked_node != self._drag_waypoint_node:
            return

        active_camera = self._controller.getScene().getActiveCamera()
        if self._picking_pass is None:
            self._picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
        self._picking_pass.render()
        picked_position = self._picking_pass.getPickedPosition(mouse_x, mouse_y)
        if picked_position is None:
            return
        self._movePathWaypoint(self._drag_waypoint_index, picked_position)

    def _finishWaypointDrag(self):
        if self._drag_waypoint_index is None:
            return
        undo_state = self._drag_waypoint_undo_state
        self._drag_waypoint_index = None
        self._drag_waypoint_node = None
        self._drag_waypoint_undo_state = None
        if undo_state is not None:
            redo_state = self._capturePathToolState()
            self._pushPathStateUndoOperation(undo_state, redo_state)

    def _suggestPathFromPoints(self):
        """Compute geodesic path previews from placed points and show as dots.

        In path mode this previews the single open/closed path through the
        current waypoints. In path_isolate mode it previews every finalized
        loop plus the current in-progress loop (closed) so the user can see
        what the cut will actually follow before committing.
        """
        if self._path_node is None:
            Logger.log("w", "Suggest path: no node selected")
            return

        loop_sequences = self._collectSuggestPathSequences()
        if not loop_sequences:
            Logger.log("w", "Suggest path: need at least 2 points")
            return

        try:
            self._showProgress("Object Splitter", "Computing suggested path...", 0, 100)
            extraction = self._extractTrimesh(self._path_node)
            if extraction is None:
                return
            tm, _, _ = extraction

            from .core.path_cutter import chain_paths
            path_points_list = []
            total = len(loop_sequences)
            for i, sequence in enumerate(loop_sequences):
                progress = int(20 + 60 * (i / max(total - 1, 1)))
                self._updateProgress(
                    "Tracing geodesic path %d/%d..." % (i + 1, total),
                    progress,
                )
                vertex_path = chain_paths(tm, sequence)
                if not vertex_path:
                    continue
                points = numpy.asarray(tm.vertices[vertex_path], dtype=numpy.float64)
                path_points_list.append(points)

            if not path_points_list:
                Logger.log("w", "Suggest path: no path returned")
                return

            self._updateProgress("Rendering path preview...", 85)
            self._showSuggestedPathSet(path_points_list)
            total_verts = sum(len(p) for p in path_points_list)
            Logger.log(
                "i",
                "Suggested path displayed with %d vertices across %d loop(s)",
                total_verts,
                len(path_points_list),
            )
        except Exception as e:
            Logger.log("w", "Suggest path failed: %s", str(e))
        finally:
            self._closeProgress()

    def _collectSuggestPathSequences(self):
        """Return list of waypoint sequences (each a list of numpy points) to preview."""
        sequences = []
        if self._cut_mode == self.CUT_MODE_PATH_ISOLATE:
            for loop in self._path_isolate_loops:
                pts = [numpy.asarray(p, dtype=numpy.float64).copy() for p in loop]
                if len(pts) >= 2:
                    sequences.append(pts)
            if len(self._path_waypoints) >= 3:
                current = [
                    numpy.asarray(p, dtype=numpy.float64).copy()
                    for p in self._path_waypoints
                ]
                if not numpy.allclose(current[0], current[-1]):
                    current.append(current[0].copy())
                sequences.append(current)
        else:
            if len(self._path_waypoints) >= 2:
                pts = [
                    numpy.asarray(p, dtype=numpy.float64).copy()
                    for p in self._path_waypoints
                ]
                if self._path_close_loop and len(pts) >= 3:
                    if not numpy.allclose(pts[0], pts[-1]):
                        pts.append(pts[0].copy())
                sequences.append(pts)
        return sequences

    _SUGGEST_PATH_COLOR = numpy.array([0.05, 0.45, 1.0, 0.95], dtype=numpy.float32)

    def _showSuggestedPath(self, path_points: numpy.ndarray):
        """Show a dotted preview of a single suggested path (clears previous)."""
        self._clearSuggestedPath()
        if path_points is None or len(path_points) == 0:
            return
        dot_size = self._suggestedPathDotSize(path_points)
        self._appendSuggestedPathDots(path_points, dot_size)

    def _showSuggestedPathSet(self, path_points_list):
        """Show dotted previews for multiple suggested paths (clears previous)."""
        self._clearSuggestedPath()
        if not path_points_list:
            return
        valid_paths = [p for p in path_points_list if p is not None and len(p) > 0]
        if not valid_paths:
            return
        all_points = numpy.concatenate(valid_paths, axis=0)
        dot_size = self._suggestedPathDotSize(all_points)
        for path_points in valid_paths:
            self._appendSuggestedPathDots(path_points, dot_size)

    def _suggestedPathDotSize(self, path_points: numpy.ndarray) -> float:
        mesh_size_max = max(
            1.0,
            float(numpy.linalg.norm(path_points.max(axis=0) - path_points.min(axis=0)))
        )
        return max(0.2, mesh_size_max * 0.005)

    def _appendSuggestedPathDots(self, path_points: numpy.ndarray, dot_size: float):
        max_dots = 80
        step = max(1, int(numpy.ceil(len(path_points) / max_dots)))
        sampled = path_points[::step]
        if len(sampled) == 0 or not numpy.allclose(sampled[-1], path_points[-1]):
            sampled = numpy.vstack([sampled, path_points[-1]])

        scene_root = self._controller.getScene().getRoot()
        for pt in sampled:
            marker_node = self._createColoredMarkerNode(self._SUGGEST_PATH_COLOR)
            vertices, indices = create_dot_mesh_data(pt, dot_size)
            mesh_builder = MeshBuilder()
            mesh_builder.setVertices(vertices)
            mesh_builder.setIndices(indices)
            mesh_builder.calculateNormals()
            marker_node.setMeshData(mesh_builder.build())
            marker_node.setParent(scene_root)
            self._path_suggestion_nodes.append(marker_node)

    def _executePathCut(self):
        """Execute the cut along the accumulated path waypoints."""
        if self._cut_mode != self.CUT_MODE_PATH:
            Logger.log("w", "Path cut trigger is only valid in path mode")
            return
        if len(self._path_waypoints) < 2:
            Logger.log("w", "Path mode: need at least 2 points to cut")
            return
        if self._path_node is None:
            Logger.log("w", "Path mode: no node selected")
            return

        node = self._path_node
        undo_state = self._capturePathToolState()
        waypoints = list(self._path_waypoints)  # Copy before clearing
        # Optionally close the loop
        if self._path_close_loop and len(waypoints) >= 3:
            waypoints.append(waypoints[0].copy())
        self._clearPathWaypoints()
        redo_state = self._capturePathToolState()
        self._performPathCut(node, waypoints, undo_state=undo_state, redo_state=redo_state)

    def _executePathIsolate(self):
        """Execute the multi-loop region isolation split."""
        if self._cut_mode != self.CUT_MODE_PATH_ISOLATE:
            Logger.log("w", "Path isolate trigger is only valid in path isolate mode")
            return
        if not self._path_isolate_loops:
            Logger.log("w", "Path isolate: add at least one closed loop before isolating")
            return
        if self._path_waypoints:
            Logger.log("w", "Path isolate: finish or clear the current loop before isolating")
            return
        if self._path_node is None:
            Logger.log("w", "Path isolate: no node selected")
            return
        if self._path_isolate_target_face_id is None or self._path_isolate_target_point is None:
            Logger.log("w", "Path isolate: pick a target region before isolating")
            return

        node = self._path_node
        undo_state = self._capturePathToolState()
        loops = [
            [numpy.asarray(point, dtype=numpy.float64).copy() for point in loop]
            for loop in self._path_isolate_loops
        ]
        target_face_id = int(self._path_isolate_target_face_id)
        target_point = numpy.asarray(self._path_isolate_target_point, dtype=numpy.float64).copy()
        self._clearPathWaypoints()
        redo_state = self._capturePathToolState()
        self._performPathIsolate(
            node,
            loops,
            target_face_id,
            target_point,
            undo_state=undo_state,
            redo_state=redo_state,
        )

    def _executeAnchoredCut(self):
        """Execute valley/valley-seam cut using 1+ placed anchor points."""
        if self._cut_mode not in (self.CUT_MODE_VALLEY, self.CUT_MODE_VALLEY_SEAM):
            Logger.log("w", "Anchored cut is only supported for valley modes")
            return
        if len(self._path_waypoints) < 1:
            Logger.log("w", "Anchored cut: place at least 1 point")
            return
        if self._path_node is None:
            Logger.log("w", "Anchored cut: no node selected")
            return

        node = self._path_node
        anchor_points = [pt.copy() for pt in self._path_waypoints]
        click_point = anchor_points[0]
        self._clearPathWaypoints()
        self._performCut(node, Vector(*click_point.tolist()), anchor_points=anchor_points)

    def _planeFromThreePoints(self, p1, p2, p3):
        """Cut plane through three clicked points (any orientation).

        Normal = (p2-p1) x (p3-p1); origin = the centroid of the three points.
        Accepts Vectors or numpy arrays. Returns (origin, normal) numpy arrays.
        """
        def _arr(p):
            if hasattr(p, "x"):
                return numpy.array([p.x, p.y, p.z], dtype=numpy.float64)
            return numpy.asarray(p, dtype=numpy.float64).reshape(3)
        a, b, c = _arr(p1), _arr(p2), _arr(p3)
        normal = numpy.cross(b - a, c - a)
        n = numpy.linalg.norm(normal)
        normal = (normal / n) if n > 1e-9 else numpy.array([0.0, 1.0, 0.0])
        origin = (a + b + c) / 3.0
        return origin, normal

    def event(self, event):
        super().event(event)
        modifiers = QApplication.keyboardModifiers()
        ctrl_is_active = modifiers & Qt.KeyboardModifier.ControlModifier

        if event.type == Event.MouseMoveEvent:
            if self._drag_waypoint_index is not None:
                buttons = getattr(event, "buttons", [])
                if MouseEvent.LeftButton in buttons:
                    self._dragWaypointToMouse(event.x, event.y)
                    return
                self._finishWaypointDrag()

            if self._show_preview:
                self._updatePreview(event.x, event.y)
            return

        if event.type == Event.MouseReleaseEvent and self._drag_waypoint_index is not None:
            self._finishWaypointDrag()
            return

        # (Right-click no longer triggers path cut — use the Cut button instead)

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

            if picked_position is None:
                Logger.log("d", "No picked position")
                return

            if self._isPointPlacementMode():
                if (
                    self._cut_mode == self.CUT_MODE_PATH_ISOLATE and
                    self._path_isolate_target_pick_active
                ):
                    self._setPathIsolateTargetFromClick(picked_node, picked_position)
                    return
                hit_idx = self._findNearbyWaypointIndex(picked_position, picked_node)
                if hit_idx is not None:
                    if self._isPathSelectionEnabled():
                        self._setSelectedPathWaypoint(hit_idx)
                    self._drag_waypoint_index = hit_idx
                    self._drag_waypoint_node = picked_node
                    self._drag_waypoint_undo_state = self._capturePathToolState()
                    Logger.log("d", "Dragging point %d", hit_idx + 1)
                    return
                self._addPathWaypoint(picked_position, picked_node)
                # Plane + 3-point: once three points are placed, cut along the
                # plane through them (any orientation).
                if (self._cut_mode == self.CUT_MODE_PLANE
                        and self._plane_orientation == "points"
                        and len(self._path_waypoints) >= 3):
                    pts = list(self._path_waypoints[:3])
                    origin, normal = self._planeFromThreePoints(*pts)
                    self._clearPathWaypoints()
                    self._removePreview()
                    self._performCut(picked_node, picked_position,
                                     plane_override=(origin, normal))
                return

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
            self._clearSuggestedPath()
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

        # Path and anchor-placement modes do not need a hover arrow/normal preview.
        # Hiding it avoids visual lag and click-position confusion.
        if self._cut_mode in (self.CUT_MODE_PATH, self.CUT_MODE_PATH_ISOLATE):
            self._removePreview()
            self._hover_node = picked_node
            return
        if (self._cut_mode == self.CUT_MODE_PLANE
                and self._plane_orientation == "points"):
            self._removePreview()
            self._hover_node = picked_node
            return
        if (
            self._multi_point_anchors_enabled and
            self._cut_mode in (self.CUT_MODE_VALLEY, self.CUT_MODE_VALLEY_SEAM)
        ):
            self._removePreview()
            self._hover_node = picked_node
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

        # Use the node's cached world-space bounding box instead of transforming
        # every vertex on each hover. Transforming a dense mesh (tens of thousands
        # of vertices) per mouse-move was the cause of laggy hover previews.
        bbox = picked_node.getBoundingBox()
        if bbox is None:
            self._removePreview()
            return
        min_bounds = numpy.array([bbox.minimum.x, bbox.minimum.y, bbox.minimum.z])
        max_bounds = numpy.array([bbox.maximum.x, bbox.maximum.y, bbox.maximum.z])
        mesh_size = max_bounds - min_bounds
        plane_size = max(mesh_size[0], mesh_size[2]) * 1.2  # 20% larger than mesh

        # Surface-relative modes (and Plane "along surface"): show the normal arrow.
        arrow_modes = (
            self.CUT_MODE_SMALLEST,
            self.CUT_MODE_SHORTEST,
            self.CUT_MODE_RADIAL,
            self.CUT_MODE_VALLEY,
            self.CUT_MODE_VALLEY_SEAM,
        )
        plane_surface = (self._cut_mode == self.CUT_MODE_PLANE
                         and self._plane_orientation == "surface")
        if self._cut_mode in arrow_modes or plane_surface:
            center = numpy.array([picked_position.x, picked_position.y, picked_position.z])
            mesh_size_max = max(mesh_size[0], mesh_size[1], mesh_size[2])
            marker_size = max(0.5, mesh_size_max * 0.012)  # 1.2% of mesh or 0.5mm
            # Orient the marker along the surface normal at the hover point. The
            # trimesh is cached per node, so the nearest-face lookup stays cheap.
            normal_dir = None
            try:
                tm = self._getCachedHoverTrimesh(picked_node)
                if tm is not None and self._hover_kdtree is not None:
                    # Nearest-vertex normal: microsecond KD-tree query instead of
                    # the slow nearest-face proximity. Good enough for the arrow.
                    _, vidx = self._hover_kdtree.query(center)
                    vnormals = tm.vertex_normals
                    if vidx is not None and vidx < len(vnormals):
                        normal_dir = numpy.asarray(vnormals[vidx], dtype=numpy.float64)
            except Exception as e:
                Logger.log("d", "Hover normal lookup failed: %s", e)
            self._createOrUpdateMarker(center, marker_size, normal_dir)
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
        colors = numpy.array([[0.05, 0.05, 0.05, 0.85]] * n_verts, dtype=numpy.float32)
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

    def _createColoredMarkerNode(self, rgba) -> SceneNode:
        """Create a marker scene node rendered with the given RGBA color.

        Uses color.shader so the marker shows its true color instead of the
        Cura model theme color.
        """
        return _ColoredMarkerNode(rgba)

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

    def _getCachedHoverTrimesh(self, node: CuraSceneNode):
        """Return a world-space trimesh for the hovered node, cached.

        Extracting the trimesh transforms every vertex, so it is rebuilt only
        when the hovered node or its world transform changes -- keeping the
        per-hover surface-normal lookup cheap on dense meshes.
        """
        try:
            xform = node.getWorldTransformation().getData().tobytes()
        except Exception:
            xform = None
        if (self._hover_tm is not None and self._hover_tm_node is node
                and self._hover_tm_xform == xform):
            return self._hover_tm
        extraction = self._extractTrimesh(node)
        if extraction is None:
            self._hover_tm = None
            self._hover_tm_node = None
            self._hover_tm_xform = None
            self._hover_kdtree = None
            return None
        self._hover_tm = extraction[0]
        self._hover_tm_node = node
        self._hover_tm_xform = xform
        # Cache a KD-tree of vertices so the hover arrow can find the nearest
        # vertex normal in microseconds, instead of the ~4 ms nearest-FACE
        # proximity query (which also builds a ~100 ms triangle tree).
        try:
            from scipy.spatial import cKDTree
            self._hover_kdtree = cKDTree(self._hover_tm.vertices)
        except Exception:
            self._hover_kdtree = None
        return self._hover_tm

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

    def _performCut(
        self,
        node: CuraSceneNode,
        click_position: Vector,
        anchor_points: Optional[list] = None,
        plane_override: Optional[tuple] = None,
    ):
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
            anchor_points_arr = None
            if anchor_points:
                anchor_points_arr = [
                    numpy.asarray(p, dtype=numpy.float64).reshape(3)
                    for p in anchor_points
                ]
            capture_path = None

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
                    capture_path = capture_operation(
                        mesh=tm,
                        cut_mode=self._cut_mode,
                        click_position=click_pos_arr,
                        height_percent=self._cut_height_percent,
                        search_resolution=self._search_resolution,
                        valley_sdf_bias_enabled=self._valley_sdf_bias_enabled,
                        anchor_points=anchor_points_arr,
                        connector_enabled=self._connector_enabled,
                        connector_diameter=self._connector_diameter,
                        connector_height=self._connector_height,
                        connector_clearance=self._connector_clearance,
                        capture_dir=self._capture_dir,
                    )
                    Logger.log("i", "Debug capture written to: %s", capture_path)
                except Exception as e:
                    Logger.log("w", "Debug capture failed: %s", str(e))

            # Determine cut plane based on mode (delegates to core.plane_calculator)
            self._updateProgress("Calculating cut plane...", 20)

            split_result = None

            from .core.plane_calculator import snap_point_to_mesh_surface
            _, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)

            # Point-based modes: when the user has placed anchor points, the cut
            # follows those points deterministically (a geodesic path through
            # them, like Multi-point) instead of seeding the unreliable search.
            used_anchor_path = False
            if (self._multi_point_anchors_enabled
                    and self._cut_mode in (self.CUT_MODE_VALLEY, self.CUT_MODE_VALLEY_SEAM)
                    and anchor_points_arr and len(anchor_points_arr) >= 2):
                try:
                    from .core.path_cutter import chain_paths, partition_faces_by_path
                    self._updateProgress("Computing geodesic path through points...", 20)
                    anchor_pts = numpy.asarray(anchor_points_arr, dtype=numpy.float64)
                    # Valley modes hug concave grooves between the points; plain
                    # Multi-point would use valley_bias=0.
                    vertex_path = chain_paths(tm, anchor_pts, valley_bias=0.75)
                    face_set_a, face_set_b = partition_faces_by_path(tm, vertex_path)
                    used_anchor_path = True
                    plane = None
                    Logger.log("i", "Point mode: valley-weighted path cut through %d anchor points",
                               len(anchor_pts))
                except Exception as e:
                    Logger.log("w", "Anchor path cut failed (%s); using search instead", e)

            if used_anchor_path:
                pass  # face sets already computed from the geodesic path
            elif plane_override is not None:
                # Caller supplied the plane directly (e.g. 2-click vertical cut).
                snap_point, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)
                plane = CutPlane(
                    origin=numpy.asarray(plane_override[0], dtype=numpy.float64),
                    normal=numpy.asarray(plane_override[1], dtype=numpy.float64),
                )
            elif self._cut_mode == self.CUT_MODE_PLANE:
                snap_point, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)
                if self._plane_orientation == "horizontal":
                    # Bed-parallel (Y up). Whole-model uses the height slider;
                    # local passes through the clicked feature's height.
                    if self._cut_whole_model:
                        plane = horizontal_cut_plane(tm, self._cut_height_percent)
                    else:
                        plane = CutPlane(origin=numpy.asarray(snap_point, dtype=numpy.float64),
                                         normal=numpy.array([0.0, 1.0, 0.0]))
                elif self._plane_orientation == "vertical":
                    # Perpendicular to the bed, view-aligned through the click.
                    if R is not None:
                        plane = vertical_cut_plane(snap_point, R @ numpy.array([1.0, 0.0, 0.0]))
                    else:
                        plane = vertical_cut_plane(snap_point)
                else:
                    # "surface": cut ALONG the hover arrow (the clicked surface
                    # normal). The plane CONTAINS the arrow (excludes the grazing
                    # tangent plane); rotating about it to the smallest local
                    # cross-section gives the feature's neck. Deterministic.
                    arrow = numpy.array([0.0, 1.0, 0.0])
                    if click_face_id is not None and click_face_id < len(tm.face_normals):
                        arrow = numpy.array(tm.face_normals[click_face_id], dtype=numpy.float64)
                    plane = find_plane_along_normal(tm, snap_point, arrow)
            elif self._cut_mode == self.CUT_MODE_HORIZONTAL:
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
            elif self._cut_mode == self.CUT_MODE_VALLEY:
                self._updateProgress("Detecting valley/groove...", 20)
                snap_point, click_face_id = snap_point_to_mesh_surface(tm, click_pos_arr)

                valley_surface_normal = None
                if click_face_id is not None and click_face_id < len(tm.face_normals):
                    valley_surface_normal = numpy.array(
                        tm.face_normals[click_face_id], dtype=numpy.float64
                    )
                    Logger.log("d", "Valley click surface normal: %s", valley_surface_normal)

                search_result = find_valley_cut_plane(
                    tm, snap_point, self._search_resolution,
                    surface_normal=valley_surface_normal,
                    use_sdf_bias=self._valley_sdf_bias_enabled,
                    anchor_points=anchor_points_arr,
                    debug_trace_path=(
                        os.path.join(capture_path, "valley_trace.json")
                        if capture_path else None
                    ),
                )
                plane = search_result.plane
                Logger.log("i", "Valley cut: area=%.2f mm^2, tested %d orientations, "
                           "normal=%s, origin=%s, sdf_bias=%s",
                           search_result.area, search_result.samples_tested,
                           str(search_result.plane.normal),
                           str(search_result.plane.origin),
                           str(self._valley_sdf_bias_enabled))
            elif self._cut_mode == self.CUT_MODE_VALLEY_SEAM:
                # Concavity-biased seam path around the clicked region.
                self._updateProgress("Tracing concave valley seam...", 15)
                snap_point, source_face = snap_point_to_mesh_surface(tm, click_pos_arr)
                click_face_id = source_face

                valley_seam_surface_normal = None
                if source_face is not None and source_face < len(tm.face_normals):
                    valley_seam_surface_normal = numpy.array(
                        tm.face_normals[source_face], dtype=numpy.float64
                    )
                target_face_hint = None
                if anchor_points_arr and len(anchor_points_arr) >= 2:
                    _, target_face = snap_point_to_mesh_surface(
                        tm, numpy.asarray(anchor_points_arr[-1], dtype=numpy.float64)
                    )
                    if target_face is not None:
                        target_face_hint = int(target_face)

                try:
                    face_set_a, face_set_b, src, sink = find_valley_seam_partition(
                        tm,
                        snap_point,
                        source_face_hint=source_face,
                        target_face_hint=target_face_hint,
                        surface_normal=valley_seam_surface_normal,
                        use_sdf_bias=self._valley_sdf_bias_enabled,
                        anchor_points=anchor_points_arr,
                        debug_trace_path=(
                            os.path.join(capture_path, "valley_seam_trace.json")
                            if capture_path else None
                        ),
                    )
                    Logger.log(
                        "i",
                        "Valley seam returned. A=%d, B=%d, src=%d, sink=%d, sdf_bias=%s",
                        len(face_set_a),
                        len(face_set_b),
                        src,
                        sink,
                        str(self._valley_sdf_bias_enabled),
                    )
                except Exception as e:
                    Logger.log("w", "Valley seam error: %s", e)
                    import traceback
                    Logger.log("w", traceback.format_exc())
                    face_set_a, face_set_b = [], []

                plane = None  # Non-planar seam mode

            elif self._cut_mode == self.CUT_MODE_SHORTEST:
                # Shortest Seam: plane search + refine_partition_with_mincut
                self._updateProgress("Finding shortest seam...", 15)
                snap_point, source_face = snap_point_to_mesh_surface(tm, click_pos_arr)
                click_face_id = source_face

                seam_surface_normal = None
                if source_face is not None and source_face < len(tm.face_normals):
                    seam_surface_normal = numpy.array(
                        tm.face_normals[source_face], dtype=numpy.float64
                    )

                self._updateProgress("Searching for split plane...", 30)
                search_result = find_smallest_cut_plane(
                    tm, snap_point, self._search_resolution,
                    surface_normal=seam_surface_normal,
                )
                Logger.log("d", "Shortest seam plane search: area=%.2f, normal=%s",
                           search_result.area, search_result.plane.normal)

                # Get plane-based partition
                candidate_normals = []
                if search_result.top_candidates:
                    candidate_normals = [n for _, n in search_result.top_candidates]
                if not candidate_normals:
                    candidate_normals = [search_result.plane.normal]

                flat_len = len(tm.faces)
                face_set_a, face_set_b = [], []
                min_faces_plane = max(10, int(flat_len * 0.005))

                for ci, cn in enumerate(candidate_normals):
                    cn = numpy.asarray(cn, dtype=numpy.float64)
                    pa, pb = local_plane_partition(
                        tm, snap_point, cn, click_face_id
                    )
                    if len(pa) >= min_faces_plane and len(pb) >= min_faces_plane:
                        face_set_a, face_set_b = pa, pb
                        Logger.log("d", "Shortest seam: plane candidate %d accepted (%d/%d faces)",
                                   ci, len(pa), len(pb))
                        break

                if face_set_a:
                    # Refine the plane partition
                    Logger.log("d", "Shortest seam: refining plane partition "
                               "(%d/%d) with min-cut...",
                               len(face_set_a), len(face_set_b))
                    face_set_a, face_set_b = refine_partition_with_mincut(
                        tm, face_set_a, face_set_b
                    )
                else:
                    Logger.log("e", "Shortest seam: plane strategy failed.")
                    face_set_a, face_set_b = [], []

                plane = None  # No planar cut for shortest seam

            elif self._cut_mode == self.CUT_MODE_RADIAL:
                # Radial: dual-Dijkstra geodesic partition
                self._updateProgress("Computing radial cut...", 15)
                snap_point, source_face = snap_point_to_mesh_surface(tm, click_pos_arr)
                click_face_id = source_face

                radial_surface_normal = None
                if source_face is not None and source_face < len(tm.face_normals):
                    radial_surface_normal = numpy.array(
                        tm.face_normals[source_face], dtype=numpy.float64
                    )

                try:
                    Logger.log("i", "Calling find_shortest_seam_partition (radial)...")
                    face_set_a, face_set_b, src, sink = find_shortest_seam_partition(
                        tm, snap_point, source_face_hint=source_face,
                        surface_normal=radial_surface_normal,
                    )
                    Logger.log("i", "Radial search returned. A=%d, B=%d", len(face_set_a), len(face_set_b))
                except Exception as e:
                    Logger.log("w", "Radial search error: %s", e)
                    import traceback
                    Logger.log("w", traceback.format_exc())
                    face_set_a, face_set_b = [], []

                plane = None  # No planar cut for radial mode
            else:
                plane = horizontal_cut_plane(tm, self._cut_height_percent)

            if plane is not None:
                Logger.log("d", "Cut plane: %s", plane)

            # Perform the split (delegates to core.mesh_splitter)
            self._updateProgress("Splitting mesh...", 40)
            if used_anchor_path:
                # Geodesic path cut through the placed points (Multi-point style).
                # Cap the cut like path mode so the faces seal and connectors work.
                split_result = split_by_face_sets(
                    tm, face_set_a, face_set_b,
                    strategy_name="anchor_path",
                    attempt_hole_fill=(self._connector_enabled or self._path_cap_ends),
                )
            elif self._cut_mode == self.CUT_MODE_PLANE:
                # Clean local split: capped slice + clicked-component selection
                # gives a flat watertight cut localized to the clicked feature
                # (or the whole model when CutWholeModel is on).
                split_result = clean_local_plane_split(
                    tm, plane.origin, plane.normal, click_face_id,
                    whole_model=self._cut_whole_model
                )
            elif self._cut_mode in (
                self.CUT_MODE_SHORTEST,
                self.CUT_MODE_RADIAL,
                self.CUT_MODE_VALLEY_SEAM,
            ):
                if self._cut_mode == self.CUT_MODE_SHORTEST:
                    strategy_name = "shortest_seam"
                elif self._cut_mode == self.CUT_MODE_RADIAL:
                    strategy_name = "radial"
                else:
                    strategy_name = "valley_seam"
                split_result = split_by_face_sets(
                    tm, face_set_a, face_set_b,
                    strategy_name=strategy_name,
                    attempt_hole_fill=False,
                )
            elif self._cut_mode in (self.CUT_MODE_SMALLEST, self.CUT_MODE_VALLEY):
                # The search now penalizes grazing planes, so plane.normal cuts
                # across the clicked feature. Use the clean split for a flat,
                # watertight cut (no triangle-edge sawtooth, engine-free).
                split_result = clean_local_plane_split(
                    tm, plane.origin, plane.normal, click_face_id
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
                if used_anchor_path:
                    # Non-planar geodesic path cut through the points: place
                    # connectors along the path (same as Multi-point), not the
                    # planar peg/hole which needs a single cut plane.
                    try:
                        self._updateProgress("Adding connectors...", 60)
                        path_points = numpy.asarray(
                            tm.vertices[vertex_path], dtype=numpy.float64)
                        cr = add_path_connectors(
                            mesh_upper, mesh_lower, path_points,
                            split_result.cap_faces_upper,
                            split_result.cap_faces_lower,
                            config=ConnectorConfig(
                                enabled=True,
                                diameter=float(self._connector_diameter),
                                height=float(self._connector_height),
                                clearance=float(self._connector_clearance),
                                sides=self._connector_sides,
                            ),
                            face_count=len(tm.faces),
                        )
                        if cr.connectors_added:
                            mesh_upper = cr.upper
                            mesh_lower = cr.lower
                            Logger.log("i", "Path connectors added (anchor cut)")
                        else:
                            Logger.log("w", "Connectors skipped: %s", cr.skipped_reason)
                    except Exception as e:
                        Logger.log("w", "Anchor-path connectors failed: %s", e)
                elif self._cut_mode in (
                    self.CUT_MODE_SHORTEST,
                    self.CUT_MODE_RADIAL,
                    self.CUT_MODE_VALLEY_SEAM,
                ):
                    Logger.log("w", "Skipping connectors - not supported for non-planar seam modes")
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
                f"{original_name}_part1",
                normals=self._safeVertexNormals(mesh_upper),
            )
            node_lower = self._createMeshNode(
                mesh_lower.vertices, mesh_lower.faces,
                f"{original_name}_part2",
                normals=self._safeVertexNormals(mesh_lower),
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
    def _performPathCut(self, node, waypoints, undo_state=None, redo_state=None):
        """Execute a path-based cut using multiple waypoints."""
        completed = False
        try:
            self._showProgress("Object Splitter", "Computing path cut...", 0, 100)

            mesh_data = node.getMeshData()
            if not mesh_data:
                Logger.log("e", "Path cut: no mesh data")
                return

            # Transform to world space
            world_transform = node.getWorldTransformation()
            transformed = mesh_data.getTransformed(world_transform)
            world_vertices = transformed.getVertices()
            world_faces = (transformed.getIndices() if transformed.getIndices() is not None
                           else numpy.arange(len(world_vertices)).reshape(-1, 3))

            # Create trimesh — merge vertices for proper edge connectivity
            tm = trimesh.Trimesh(vertices=world_vertices, faces=world_faces,
                                 process=False, validate=False)
            tm.merge_vertices()

            Logger.log("i", "Path cut: %d waypoints on mesh with %d verts, %d faces",
                       len(waypoints), len(tm.vertices), len(tm.faces))

            # Optionally capture inputs for later offline replay / profiling.
            if self._capture_dir:
                try:
                    from .core.debug_capture import capture_operation
                    capture_path = capture_operation(
                        mesh=tm,
                        cut_mode=self._cut_mode,
                        click_position=numpy.asarray(waypoints[0], dtype=numpy.float64),
                        waypoints=waypoints,
                        path_close_loop=self._path_close_loop,
                        path_cap_ends=self._path_cap_ends,
                        connector_enabled=self._connector_enabled,
                        connector_diameter=self._connector_diameter,
                        connector_height=self._connector_height,
                        connector_clearance=self._connector_clearance,
                        capture_dir=self._capture_dir,
                    )
                    Logger.log("i", "Debug capture (path) written to: %s", capture_path)
                except Exception as e:
                    Logger.log("w", "Debug capture (path) failed: %s", str(e))

            self._updateProgress("Computing geodesic path...", 20)

            from .core.path_cutter import chain_paths, partition_faces_by_path

            # Find geodesic path through all waypoints
            vertex_path = chain_paths(tm, waypoints)
            Logger.log("i", "Path cut: geodesic path has %d vertices", len(vertex_path))
            is_closed_path = (
                len(vertex_path) >= 4 and int(vertex_path[0]) == int(vertex_path[-1])
            )

            self._updateProgress("Partitioning faces...", 50)

            # Partition faces along the path
            face_set_a, face_set_b = partition_faces_by_path(tm, vertex_path)
            smaller_faces = min(len(face_set_a), len(face_set_b))
            total_faces = max(1, len(tm.faces))
            partition_ratio = float(smaller_faces) / float(total_faces)
            Logger.log(
                "i",
                "Path cut: partition %d / %d faces (smaller=%.4f%%, closed=%s)",
                len(face_set_a),
                len(face_set_b),
                partition_ratio * 100.0,
                str(is_closed_path),
            )

            if not face_set_a or not face_set_b:
                Logger.log("e", "Path cut: failed to create valid partition")
                self._closeProgress()
                return

            # Guard against pathological ribbon/necklace partitions where the
            # path only carves out a tiny strip of triangles instead of
            # producing a meaningful separation.
            min_partition_ratio = 0.0005  # 0.05% of faces
            min_partition_faces = 200
            if smaller_faces < max(min_partition_faces, int(total_faces * min_partition_ratio)):
                Logger.log(
                    "w",
                    "Path cut aborted: pathological tiny partition (%d/%d faces = %.4f%%). "
                    "This usually means the path formed a narrow ribbon on the surface rather "
                    "than a true separating seam.",
                    smaller_faces,
                    total_faces,
                    partition_ratio * 100.0,
                )
                self._logPathHalfDiagnostics("input", tm)
                Logger.log(
                    "w",
                    "Path cut hint: try a larger/cleaner closed loop, or repair the mesh if it has "
                    "non-manifold/self-touching regions.",
                )
                return

            # Split using face sets (same as shortest/radial modes)
            self._updateProgress("Splitting mesh...", 70)
            split_result = split_by_face_sets(
                tm, face_set_a, face_set_b,
                strategy_name="path_cut",
                # Path capping is controlled independently from connector
                # generation so users can request sealed ends without pegs/holes.
                attempt_hole_fill=(self._connector_enabled or self._path_cap_ends),
            )

            Logger.log("i", "Path cut split result: %s", split_result.summary())

            if not split_result.success:
                Logger.log("e", "Path cut failed: %s", split_result.error)
                self._closeProgress()
                return

            mesh_upper = split_result.upper
            mesh_lower = split_result.lower

            if self._connector_enabled and (
                not bool(mesh_upper.is_watertight) or
                not bool(mesh_lower.is_watertight)
            ):
                self._logPathHalfDiagnostics("upper", mesh_upper)
                self._logPathHalfDiagnostics("lower", mesh_lower)

            # Path-mode connectors (non-planar): delegate the candidate search
            # to core.connectors.add_path_connectors so the logic is testable
            # and exercised by offline replay/profiling.
            if self._connector_enabled:
                if len(vertex_path) >= 2:
                    try:
                        Logger.log(
                            "i",
                            "Path connectors: attempting (capped=%s, upper_watertight=%s, lower_watertight=%s)",
                            str(split_result.capped),
                            str(bool(mesh_upper.is_watertight)),
                            str(bool(mesh_lower.is_watertight)),
                        )
                        self._updateProgress("Adding connectors...", 80)
                        path_points = numpy.asarray(tm.vertices[vertex_path], dtype=numpy.float64)
                        cr = add_path_connectors(
                            mesh_upper,
                            mesh_lower,
                            path_points,
                            split_result.cap_faces_upper,
                            split_result.cap_faces_lower,
                            config=ConnectorConfig(
                                enabled=True,
                                diameter=float(self._connector_diameter),
                                height=float(self._connector_height),
                                clearance=float(self._connector_clearance),
                                sides=self._connector_sides,
                            ),
                            face_count=len(tm.faces),
                        )
                        if cr.connectors_added:
                            mesh_upper = cr.upper
                            mesh_lower = cr.lower
                            Logger.log(
                                "i",
                                "Path connectors added: peg on %s, hole on %s (engine: %s, attempts=%d)",
                                cr.peg_on, cr.hole_on, cr.hole_engine, cr.attempts,
                            )
                        else:
                            Logger.log(
                                "w",
                                "Path connectors skipped after %d attempts: %s",
                                cr.attempts, cr.skipped_reason,
                            )
                    except Exception as e:
                        Logger.log("w", "Path connectors error: %s", str(e))
                else:
                    Logger.log("w", "Path connectors skipped: path too short")

            # Create scene nodes and replace original
            node_upper = self._createMeshNode(
                mesh_upper.vertices, mesh_upper.faces,
                node.getName() + " (A)",
                normals=self._safeVertexNormals(mesh_upper))
            node_lower = self._createMeshNode(
                mesh_lower.vertices, mesh_lower.faces,
                node.getName() + " (B)",
                normals=self._safeVertexNormals(mesh_lower))

            op = _PathStateGroupedOperation(self, undo_state=undo_state, redo_state=redo_state)
            scene_root = self._controller.getScene().getRoot()
            op.addOperation(AddSceneNodeOperation(node_upper, scene_root))
            op.addOperation(AddSceneNodeOperation(node_lower, scene_root))
            op.addOperation(RemoveSceneNodeOperation(node))
            op.push()
            completed = True

            CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node_upper)

            self._updateProgress("Done!", 100)
            Logger.log("i", "Path cut complete: created '%s' and '%s'",
                       node_upper.getName(), node_lower.getName())

        except Exception as e:
            Logger.log("e", "Error during path cut: %s", str(e))
            import traceback
            Logger.log("e", traceback.format_exc())
        finally:
            if not completed and undo_state is not None:
                Logger.log("i", "Path cut did not complete; restoring path editing state")
                self._restorePathToolState(undo_state)
            self._closeProgress()

    def _performPathIsolate(
        self,
        node,
        loops,
        target_face_id: int,
        target_point: numpy.ndarray,
        undo_state=None,
        redo_state=None,
    ):
        """Execute a multi-loop region isolation cut."""
        completed = False
        try:
            self._showProgress("Object Splitter", "Computing isolated region...", 0, 100)

            extraction = self._extractTrimesh(node)
            if extraction is None:
                Logger.log("e", "Path isolate: failed to extract mesh")
                return
            tm, _, _ = extraction

            Logger.log(
                "i",
                "Path isolate: %d loops on mesh with %d verts, %d faces (target_face=%d)",
                len(loops),
                len(tm.vertices),
                len(tm.faces),
                int(target_face_id),
            )

            from .core.path_cutter import chain_paths, isolate_region_by_loops

            loop_vertex_paths = []
            self._updateProgress("Tracing geodesic loops...", 25)
            for loop_index, loop_waypoints in enumerate(loops):
                if len(loop_waypoints) < 4:
                    raise ValueError(f"Loop {loop_index + 1} needs at least 3 points")
                loop_points = [numpy.asarray(point, dtype=numpy.float64).copy() for point in loop_waypoints]
                if not numpy.allclose(loop_points[0], loop_points[-1]):
                    loop_points.append(loop_points[0].copy())
                vertex_path = chain_paths(tm, loop_points)
                if len(vertex_path) < 4 or int(vertex_path[0]) != int(vertex_path[-1]):
                    raise ValueError(f"Loop {loop_index + 1} did not resolve to a closed geodesic seam")
                loop_vertex_paths.append(vertex_path)
                Logger.log("i", "Path isolate: loop %d traced with %d vertices", loop_index + 1, len(vertex_path))

            self._updateProgress("Isolating target component...", 55)
            extracted_faces, remainder_faces = isolate_region_by_loops(
                tm,
                loop_vertex_paths,
                int(target_face_id),
            )
            Logger.log(
                "i",
                "Path isolate: extracted %d faces, remainder %d faces (target=%s)",
                len(extracted_faces),
                len(remainder_faces),
                str(numpy.asarray(target_point, dtype=numpy.float64)),
            )

            self._updateProgress("Splitting mesh...", 75)
            split_result = split_by_face_sets(
                tm,
                extracted_faces,
                remainder_faces,
                strategy_name="path_isolate",
                attempt_hole_fill=True,
            )
            Logger.log("i", "Path isolate split result: %s", split_result.summary())
            if not split_result.success:
                Logger.log("e", "Path isolate failed: %s", split_result.error)
                return

            mesh_extracted = split_result.upper
            mesh_remainder = split_result.lower

            if self._path_isolate_prune_tiny_fragments:
                threshold = int(self._path_isolate_tiny_fragment_face_threshold)
                mesh_extracted, removed_extracted, kept_extracted = prune_small_components(
                    mesh_extracted,
                    min_faces=threshold,
                )
                mesh_remainder, removed_remainder, kept_remainder = prune_small_components(
                    mesh_remainder,
                    min_faces=threshold,
                )
                Logger.log(
                    "i",
                    "Path isolate cleanup: threshold=%d faces | extracted removed=%d kept=%d | remainder removed=%d kept=%d",
                    threshold,
                    removed_extracted,
                    kept_extracted,
                    removed_remainder,
                    kept_remainder,
                )

            if self._connector_enabled:
                Logger.log("i", "Path isolate: connectors are disabled in this mode")

            self._updateProgress("Creating isolated objects...", 90)
            node_extracted = self._createMeshNode(
                mesh_extracted.vertices,
                mesh_extracted.faces,
                node.getName() + " (Isolated)",
                normals=self._safeVertexNormals(mesh_extracted),
            )
            node_remainder = self._createMeshNode(
                mesh_remainder.vertices,
                mesh_remainder.faces,
                node.getName() + " (Remainder)",
                normals=self._safeVertexNormals(mesh_remainder),
            )

            op = _PathStateGroupedOperation(self, undo_state=undo_state, redo_state=redo_state)
            scene_root = self._controller.getScene().getRoot()
            op.addOperation(AddSceneNodeOperation(node_extracted, scene_root))
            op.addOperation(AddSceneNodeOperation(node_remainder, scene_root))
            op.addOperation(RemoveSceneNodeOperation(node))
            op.push()
            completed = True

            CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node_extracted)
            self._updateProgress("Done!", 100)
            Logger.log(
                "i",
                "Path isolate complete: created '%s' and '%s'",
                node_extracted.getName(),
                node_remainder.getName(),
            )

        except Exception as e:
            Logger.log("e", "Error during path isolate: %s", str(e))
            import traceback
            Logger.log("e", traceback.format_exc())
        finally:
            if not completed and undo_state is not None:
                Logger.log("i", "Path isolate did not complete; restoring path editing state")
                self._restorePathToolState(undo_state)
            self._closeProgress()

    def _logPathHalfDiagnostics(self, label: str, mesh):
        """Log watertight/boundary diagnostics for one path-cut half."""
        try:
            is_watertight = bool(mesh.is_watertight)
        except Exception:
            is_watertight = False

        face_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
        vert_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
        try:
            euler = int(mesh.euler_number)
        except Exception:
            euler = -999

        boundary_edges = 0
        boundary_vertices = 0
        try:
            inv = numpy.asarray(mesh.edges_unique_inverse, dtype=numpy.int64)
            if len(inv) > 0:
                edge_use = numpy.bincount(inv, minlength=len(mesh.edges_unique))
                boundary_mask = edge_use == 1
                boundary_edges = int(numpy.sum(boundary_mask))
                if boundary_edges > 0:
                    bidx = numpy.where(boundary_mask)[0]
                    bedges = numpy.asarray(mesh.edges_unique[bidx], dtype=numpy.int64)
                    boundary_vertices = int(len(numpy.unique(bedges.reshape(-1))))
        except Exception:
            pass

        loop_lengths = []
        loop_points = []
        try:
            outline = mesh.outline()
            loops = outline.discrete if (outline is not None and hasattr(outline, "discrete")) else []
            for loop in loops:
                pts = numpy.asarray(loop, dtype=numpy.float64)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                seg_len = float(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1).sum())
                if numpy.linalg.norm(pts[0] - pts[-1]) > 1e-8:
                    seg_len += float(numpy.linalg.norm(pts[-1] - pts[0]))
                loop_lengths.append(seg_len)
                loop_points.append(int(len(pts)))
        except Exception:
            pass

        Logger.log(
            "i",
            "Path %s diagnostics: watertight=%s, faces=%d, verts=%d, euler=%d, "
            "boundary_edges=%d, boundary_vertices=%d, loops=%d",
            label,
            str(is_watertight),
            face_count,
            vert_count,
            euler,
            boundary_edges,
            boundary_vertices,
            len(loop_lengths),
        )

        if loop_lengths:
            ranked = sorted(
                zip(loop_lengths, loop_points),
                key=lambda t: t[0],
                reverse=True,
            )
            top = ranked[:4]
            summary = ", ".join([f"{lp}pts/{ll:.2f}mm" for ll, lp in top])
            Logger.log("i", "Path %s boundary loops (largest): %s", label, summary)

    @staticmethod
    def _safeVertexNormals(tm) -> Optional[numpy.ndarray]:
        """trimesh vertex_normals, or None if they cannot be computed."""
        try:
            vn = numpy.asarray(tm.vertex_normals, dtype=numpy.float32)
            if vn.shape == (len(tm.vertices), 3):
                return vn
        except Exception as e:
            Logger.log("d", "vertex_normals unavailable, will recompute: %s", e)
        return None

    def _createMeshNode(self, vertices: numpy.ndarray, faces: numpy.ndarray, name: str,
                        normals: Optional[numpy.ndarray] = None) -> CuraSceneNode:
        """Create a new CuraSceneNode from vertices and faces.

        If per-vertex ``normals`` are supplied (e.g. trimesh's vertex_normals)
        they are used directly. Cura's MeshBuilder.calculateNormals() recomputes
        normals in pure Python from indexed vertices, which is the dominant cost
        of a cut on large meshes (~11s on a 666k-face half); trimesh computes the
        same normals vectorised in a fraction of a second.
        """
        verts32 = vertices.astype(numpy.float32)
        faces32 = faces.astype(numpy.int32)

        normals32 = None
        if normals is not None:
            n = numpy.asarray(normals, dtype=numpy.float32)
            if n.shape == verts32.shape:
                normals32 = n

        if normals32 is not None:
            mesh_data = MeshData(vertices=verts32, indices=faces32, normals=normals32)
        else:
            mesh_builder = MeshBuilder()
            mesh_builder.setVertices(verts32)
            mesh_builder.setIndices(faces32)
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
