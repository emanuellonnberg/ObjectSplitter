// Copyright (c) 2024 Emanuel Lönnberg.
// This tool is released under the terms of the LGPLv3 or higher.

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import UM 1.5 as UM
import Cura 1.0 as Cura

Item {
    id: base
    width: childrenRect.width
    height: childrenRect.height
    implicitWidth: mainColumn.width
    implicitHeight: mainColumn.height

    UM.I18nCatalog { id: catalog; name: "objectsplitter" }

    Component.onCompleted: {
        console.log("Object Splitter QML loaded")
        if (UM.ActiveTool) {
            console.log("Tool active, trimesh available:", UM.ActiveTool.properties.getValue("TrimeshAvailable"))
        }
    }

    Column {
        id: mainColumn
        spacing: UM.Theme.getSize("default_margin").height
        width: 250

        // Title
        Label {
            text: catalog.i18nc("@label", "Object Splitter")
            font: UM.Theme.getFont("medium_bold")
            color: UM.Theme.getColor("text")
            renderType: Text.NativeRendering
        }

        // Warning if trimesh not available
        Rectangle {
            width: parent.width
            height: warningText.height + UM.Theme.getSize("default_margin").height
            color: "#FFEEEE"
            border.color: "#FF6666"
            border.width: 1
            radius: 4
            visible: UM.ActiveTool && !UM.ActiveTool.properties.getValue("TrimeshAvailable")

            Label {
                id: warningText
                anchors.centerIn: parent
                width: parent.width - UM.Theme.getSize("default_margin").width
                text: catalog.i18nc("@label", "trimesh library not installed.\nInstall with: pip install trimesh")
                font: UM.Theme.getFont("default")
                color: "#CC0000"
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                renderType: Text.NativeRendering
            }
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Cut Mode Selection
        Row {
            spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

            Label {
                height: UM.Theme.getSize("setting_control").height
                text: catalog.i18nc("@label", "Cut Mode:")
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                verticalAlignment: Text.AlignVCenter
                renderType: Text.NativeRendering
                width: 70
            }

            ComboBox {
                id: cutModeComboBox
                width: 170
                height: UM.Theme.getSize("setting_control").height
                // Experimental (secondary) modes only appear in this dropdown
                // when the "Show experimental cut modes" toggle (near Debug) is
                // on. Auto-on when an experimental mode is already active.
                property var primaryValues: ["path", "path_isolate", "horizontal", "vertical"]
                property var experimentalValues: ["smallest", "shortest", "radial", "valley", "valley_seam"]
                // manualShowExperimental is the user's toggle; showExperimental
                // derives from it OR an experimental mode being active, so the
                // checkbox assignment never breaks this binding.
                property bool manualShowExperimental: false
                property bool showExperimental: manualShowExperimental || (UM.ActiveTool
                    ? experimentalValues.indexOf(UM.ActiveTool.properties.getValue("CutMode")) >= 0
                    : false)
                property var modeValues: showExperimental ? primaryValues.concat(experimentalValues) : primaryValues
                model: showExperimental
                    ? ["Multi-point", "Isolate region", "Horizontal", "Vertical", "Smallest Section", "Shortest Seam", "Radial (geodesic)", "Valley (groove)", "Valley Seam (concavity)"]
                    : ["Multi-point", "Isolate region", "Horizontal", "Vertical"]
                currentIndex: {
                    if (UM.ActiveTool) {
                        var idx = modeValues.indexOf(UM.ActiveTool.properties.getValue("CutMode"))
                        return idx >= 0 ? idx : 0
                    }
                    return 0
                }
                onActivated: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("CutMode", modeValues[currentIndex])
                    }
                }
            }
        }

        // Cut Mode Description
        Label {
            width: parent.width
            text: {
                if (UM.ActiveTool) {
                    var mode = UM.ActiveTool.properties.getValue("CutMode")
                    if (mode === "horizontal") return "Cut parallel to the build plate"
                    if (mode === "vertical") return "Cut perpendicular to the build plate"
                    if (mode === "smallest") return "Find smallest cross-section at click point"
                    if (mode === "shortest") return "Plane search + min-cut refinement"
                    if (mode === "radial") return "Geodesic distance partition from click"
                    if (mode === "path") return "Click to place points, then press Cut"
                    if (mode === "path_isolate") return "Place one or more closed loops, pick a target region, then isolate it"
                    if (mode === "valley") return "Find and follow a valley/groove near click"
                    if (mode === "valley_seam") return "Concavity-biased seam around clicked feature"
                }
                return ""
            }
            font: UM.Theme.getFont("default_italic")
            color: UM.Theme.getColor("text_inactive")
            wrapMode: Text.WordWrap
            renderType: Text.NativeRendering
        }

        // Path mode controls
        Column {
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: UM.ActiveTool && (
                UM.ActiveTool.properties.getValue("CutMode") === "path" ||
                (
                    (UM.ActiveTool.properties.getValue("CutMode") === "valley" ||
                     UM.ActiveTool.properties.getValue("CutMode") === "valley_seam") &&
                    UM.ActiveTool.properties.getValue("MultiPointAnchorsEnabled")
                )
            )

            Label {
                text: "Points placed: " + (UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathPointCount") : 0)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Label {
                // Only for anchored (valley) modes; in path mode this would
                // duplicate the cut-mode description shown under the combo.
                width: parent.width
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") !== "path"
                text: "Click to place anchor points, then Cut Using Points."
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                wrapMode: Text.WordWrap
                renderType: Text.NativeRendering
            }

            CheckBox {
                id: closeLoopCheckbox
                text: "Close loop"
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path"
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathCloseLoop") : false
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: "Close the path into a loop so the cut separates an enclosed region."
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("PathCloseLoop", checked)
                    }
                }
            }

            CheckBox {
                id: capPathEndsCheckbox
                text: "Cap Path Cut"
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path"
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathCapEnds") : true
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: "Seal the cut faces so each half is closed, independent of connectors."
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("PathCapEnds", checked)
                    }
                }
            }

            CheckBox {
                id: insertPointsCheckbox
                text: "Insert Points"
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path"
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathInsertMode") : false
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: "New clicks are inserted between the nearest existing points instead of appended to the end."
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("PathInsertMode", checked)
                    }
                }
            }

            Column {
                id: pathDisplaySection
                property bool expandedState: false
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

                Label {  // collapsible header
                    text: (pathDisplaySection.expandedState ? "▾  " : "▸  ") + "Display"
                    font: UM.Theme.getFont("default_bold")
                    color: UM.Theme.getColor("text")
                    renderType: Text.NativeRendering
                    MouseArea { anchors.fill: parent; onClicked: pathDisplaySection.expandedState = !pathDisplaySection.expandedState }
                }

                Column {  // collapsible body
                    visible: pathDisplaySection.expandedState
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

                    CheckBox {
                        id: smallMarkersCheckbox
                        text: "Small Markers"
                        checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathSmallMarkers") : false
                        hoverEnabled: true
                        ToolTip.visible: hovered
                        ToolTip.text: "Smaller dots and a smaller grab radius help place points closer together."
                        onClicked: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("PathSmallMarkers", checked)
                            }
                        }
                    }

                    Row {
                        spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                        Label {
                            height: UM.Theme.getSize("setting_control").height
                            text: "Marker Color:"
                            font: UM.Theme.getFont("default")
                            color: UM.Theme.getColor("text")
                            verticalAlignment: Text.AlignVCenter
                            renderType: Text.NativeRendering
                            width: 90
                        }

                        ComboBox {
                            id: pathMarkerColorComboBox
                            width: 140
                            height: UM.Theme.getSize("setting_control").height
                            property var colorNames: ["cyan", "yellow", "white", "black", "magenta", "green"]
                            model: ["Cyan", "Yellow", "White", "Black", "Magenta", "Green"]
                            currentIndex: {
                                if (UM.ActiveTool) {
                                    var c = UM.ActiveTool.properties.getValue("PathMarkerColor")
                                    var idx = colorNames.indexOf(c)
                                    return idx >= 0 ? idx : 0
                                }
                                return 0
                            }
                            onActivated: {
                                if (UM.ActiveTool) {
                                    UM.ActiveTool.setProperty("PathMarkerColor", colorNames[index])
                                }
                            }
                        }
                    }
                }
            }

            Label {
                visible: UM.ActiveTool &&
                         UM.ActiveTool.properties.getValue("CutMode") === "path" &&
                         UM.ActiveTool.properties.getValue("HasSelectedPathPoint")
                text: "Selected: " + (UM.ActiveTool.properties.getValue("SelectedPathPointIndex") + 1)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            // Point edit actions
            RowLayout {
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Button {
                    text: "Clear Points"
                    Layout.fillWidth: true
                    Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("ClearPathPoints", true)
                        }
                    }
                }

                Button {
                    text: "Remove Selected"
                    visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path"
                    Layout.fillWidth: true
                    Layout.preferredWidth: visible ? implicitWidth : 0
                    Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("HasSelectedPathPoint")
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("RemoveSelectedPathPoint", true)
                        }
                    }
                }
            }

            // Preview the exact geodesic seam the cut will follow (same path
            // computation as the cut itself, drawn as dots).
            Button {
                text: "Preview Path"
                width: parent.width
                height: UM.Theme.getSize("setting_control").height
                enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("PathPointCount") >= 2
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("TriggerSuggestPath", true)
                    }
                }
            }

            // Primary action
            Button {
                text: (UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path") ? "Cut Along Path" : "Cut Using Points"
                width: parent.width
                height: UM.Theme.getSize("setting_control").height
                enabled: UM.ActiveTool && (
                    (UM.ActiveTool.properties.getValue("CutMode") === "path" && UM.ActiveTool.properties.getValue("PathPointCount") >= 2) ||
                    (UM.ActiveTool.properties.getValue("CutMode") !== "path" && UM.ActiveTool.properties.getValue("PathPointCount") >= 1)
                )
                onClicked: {
                    if (UM.ActiveTool) {
                        if (UM.ActiveTool.properties.getValue("CutMode") === "path") {
                            UM.ActiveTool.setProperty("TriggerPathCut", true)
                        } else {
                            UM.ActiveTool.setProperty("TriggerAnchoredCut", true)
                        }
                    }
                }
            }
        }

        // Path isolate controls
        Column {
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path_isolate"

            Label {
                text: "Loops placed: " + (UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathLoopCount") : 0)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Label {
                text: "Points in current loop: " + (UM.ActiveTool ? UM.ActiveTool.properties.getValue("CurrentLoopPointCount") : 0)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Label {
                width: parent.width
                text: {
                    if (!UM.ActiveTool) {
                        return ""
                    }
                    if (UM.ActiveTool.properties.getValue("PathIsolateTargetPickActive")) {
                        return "Click the region you want to isolate."
                    }
                    return "Place at least 3 points, press Finish Loop, then Pick Target Region (or start another loop)."
                }
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                wrapMode: Text.WordWrap
                renderType: Text.NativeRendering
            }

            Label {
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("HasSelectedPathPoint")
                text: "Selected: " + (UM.ActiveTool.properties.getValue("SelectedPathPointIndex") + 1)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Column {
                id: isolateDisplaySection
                property bool expandedState: false
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

                Label {  // collapsible header
                    text: (isolateDisplaySection.expandedState ? "▾  " : "▸  ") + "Display"
                    font: UM.Theme.getFont("default_bold")
                    color: UM.Theme.getColor("text")
                    renderType: Text.NativeRendering
                    MouseArea { anchors.fill: parent; onClicked: isolateDisplaySection.expandedState = !isolateDisplaySection.expandedState }
                }

                Column {  // collapsible body
                    visible: isolateDisplaySection.expandedState
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

                    CheckBox {
                        id: isolateSmallMarkersCheckbox
                        text: "Small Markers"
                        checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathSmallMarkers") : false
                        hoverEnabled: true
                        ToolTip.visible: hovered
                        ToolTip.text: "Smaller dots and a smaller grab radius help place isolate-loop points closer together."
                        onClicked: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("PathSmallMarkers", checked)
                            }
                        }
                    }

                    Row {
                        spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                        Label {
                            height: UM.Theme.getSize("setting_control").height
                            text: "Marker Color:"
                            font: UM.Theme.getFont("default")
                            color: UM.Theme.getColor("text")
                            verticalAlignment: Text.AlignVCenter
                            renderType: Text.NativeRendering
                            width: 90
                        }

                        ComboBox {
                            id: isolatePathMarkerColorComboBox
                            width: 140
                            height: UM.Theme.getSize("setting_control").height
                            property var colorNames: ["cyan", "yellow", "white", "black", "magenta", "green"]
                            model: ["Cyan", "Yellow", "White", "Black", "Magenta", "Green"]
                            currentIndex: {
                                if (UM.ActiveTool) {
                                    var c = UM.ActiveTool.properties.getValue("PathMarkerColor")
                                    var idx = colorNames.indexOf(c)
                                    return idx >= 0 ? idx : 0
                                }
                                return 0
                            }
                            onActivated: {
                                if (UM.ActiveTool) {
                                    UM.ActiveTool.setProperty("PathMarkerColor", colorNames[index])
                                }
                            }
                        }
                    }
                }
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Button {
                    text: "Remove Selected"
                    width: 120
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("HasSelectedPathPoint")
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("RemoveSelectedPathPoint", true)
                        }
                    }
                }

                Button {
                    text: "Clear Current Loop"
                    width: 120
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("CurrentLoopPointCount") > 0
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("TriggerClearCurrentPathLoop", true)
                        }
                    }
                }
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Button {
                    text: "Finish Loop"
                    width: 110
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("CurrentLoopPointCount") >= 3
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("TriggerStartNewPathLoop", true)
                        }
                    }
                }

                Button {
                    text: "Clear All Loops"
                    width: 110
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && (
                        UM.ActiveTool.properties.getValue("CurrentLoopPointCount") > 0 ||
                        UM.ActiveTool.properties.getValue("PathLoopCount") > 0 ||
                        UM.ActiveTool.properties.getValue("PathIsolateTargetPicked")
                    )
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("ClearPathPoints", true)
                        }
                    }
                }
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Button {
                    text: "Pick Target Region"
                    width: 125
                    height: UM.Theme.getSize("setting_control").height
                    // Needs at least one finished loop. Stays enabled between
                    // loops (current empty) and while a complete extra loop is
                    // pending (>=3 pts, auto-finalized on click); only greys out
                    // during a half-drawn new loop (1-2 pts).
                    enabled: UM.ActiveTool &&
                             UM.ActiveTool.properties.getValue("PathLoopCount") >= 1 &&
                             (UM.ActiveTool.properties.getValue("CurrentLoopPointCount") === 0 ||
                              UM.ActiveTool.properties.getValue("CurrentLoopPointCount") >= 3)
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("TriggerPickPathIsolateTarget", true)
                        }
                    }
                }

                Button {
                    text: "Preview Loops"
                    width: 115
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && (
                        UM.ActiveTool.properties.getValue("PathLoopCount") >= 1 ||
                        UM.ActiveTool.properties.getValue("CurrentLoopPointCount") >= 3
                    )
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("TriggerSuggestPath", true)
                        }
                    }
                }
            }

            Label {
                text: {
                    if (!UM.ActiveTool) {
                        return ""
                    }
                    if (UM.ActiveTool.properties.getValue("PathIsolateTargetPickActive")) {
                        return "Picking target..."
                    }
                    if (UM.ActiveTool.properties.getValue("PathIsolateTargetPicked")) {
                        return "Target selected"
                    }
                    return "No target selected"
                }
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            CheckBox {
                id: isolateCleanupCheckBox
                text: "Remove Tiny Fragments"
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathIsolatePruneTinyFragments") : true
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: "Drops disconnected pieces smaller than the threshold from each output."
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("PathIsolatePruneTinyFragments", checked)
                    }
                }
            }

            Column {
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 3)
                visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("PathIsolatePruneTinyFragments")

                RowLayout {
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        Layout.preferredWidth: 70
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: "Min faces:"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }

                    Slider {
                        id: isolateFragmentSlider
                        Layout.fillWidth: true
                        from: 0
                        to: 300
                        stepSize: 10
                        value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathIsolateTinyFragmentFaceThreshold") : 80

                        onValueChanged: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("PathIsolateTinyFragmentFaceThreshold", Math.round(value))
                            }
                        }
                    }

                    Label {
                        Layout.preferredWidth: 65
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: Math.round(isolateFragmentSlider.value) + " faces"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }
                }
            }

            Label {
                width: parent.width
                text: "Connectors are disabled in this mode for now."
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                wrapMode: Text.WordWrap
                renderType: Text.NativeRendering
            }

            Button {
                text: "Isolate Region"
                width: 120
                height: UM.Theme.getSize("setting_control").height
                enabled: UM.ActiveTool &&
                         UM.ActiveTool.properties.getValue("PathLoopCount") >= 1 &&
                         UM.ActiveTool.properties.getValue("CurrentLoopPointCount") === 0 &&
                         UM.ActiveTool.properties.getValue("PathIsolateTargetPicked")
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("TriggerPathIsolate", true)
                    }
                }
            }
        }

        // Multi-point anchor toggle for valley modes
        Row {
            spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)
            visible: UM.ActiveTool && (
                UM.ActiveTool.properties.getValue("CutMode") === "valley" ||
                UM.ActiveTool.properties.getValue("CutMode") === "valley_seam"
            )

            CheckBox {
                id: multiPointAnchorsCheckBox
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("MultiPointAnchorsEnabled") : false

                onCheckedChanged: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("MultiPointAnchorsEnabled", checked)
                    }
                }
            }

            Label {
                height: multiPointAnchorsCheckBox.height
                text: "Use point anchors"
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                verticalAlignment: Text.AlignVCenter
                renderType: Text.NativeRendering

                MouseArea {
                    anchors.fill: parent
                    onClicked: multiPointAnchorsCheckBox.checked = !multiPointAnchorsCheckBox.checked
                }
            }
        }

        // Separator (only when a mode-specific parameter section follows)
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
            visible: UM.ActiveTool && ["horizontal", "smallest", "valley", "valley_seam"].indexOf(UM.ActiveTool.properties.getValue("CutMode")) >= 0
        }

        // Cut Height (for horizontal mode)
        Column {
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "horizontal"

            Label {
                text: catalog.i18nc("@label", "Cut Height:")
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Slider {
                    id: heightSlider
                    width: 170
                    from: 0
                    to: 100
                    value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("CutHeightPercent") : 50
                    stepSize: 1

                    onValueChanged: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("CutHeightPercent", value)
                        }
                    }
                }

                Label {
                    height: UM.Theme.getSize("setting_control").height
                    text: Math.round(heightSlider.value) + "%"
                    font: UM.Theme.getFont("default")
                    color: UM.Theme.getColor("text")
                    verticalAlignment: Text.AlignVCenter
                    renderType: Text.NativeRendering
                    width: 40
                }
            }
        }

        // Search Resolution (for smallest mode)
        Column {
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "smallest"

            Label {
                text: catalog.i18nc("@label", "Search Resolution:")
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Slider {
                    id: resolutionSlider
                    width: 170
                    from: 6
                    to: 36
                    value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("SearchResolution") : 18
                    stepSize: 1
                    hoverEnabled: true
                    ToolTip.visible: hovered
                    ToolTip.text: "Higher = more accurate but slower."

                    onValueChanged: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("SearchResolution", Math.round(value))
                        }
                    }
                }

                Label {
                    height: UM.Theme.getSize("setting_control").height
                    text: Math.round(resolutionSlider.value).toString()
                    font: UM.Theme.getFont("default")
                    color: UM.Theme.getColor("text")
                    verticalAlignment: Text.AlignVCenter
                    renderType: Text.NativeRendering
                    width: 30
                }
            }
        }

        // Valley SDF Bias (experimental, valley modes only)
        Column {
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: UM.ActiveTool && (
                UM.ActiveTool.properties.getValue("CutMode") === "valley" ||
                UM.ActiveTool.properties.getValue("CutMode") === "valley_seam"
            )

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                CheckBox {
                    id: valleySdfCheckBox
                    checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ValleySdfBiasEnabled") : false

                    onCheckedChanged: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("ValleySdfBiasEnabled", checked)
                        }
                    }
                }

                Label {
                    height: valleySdfCheckBox.height
                    text: "Use SDF thinness bias (experimental)"
                    font: UM.Theme.getFont("default")
                    color: UM.Theme.getColor("text")
                    verticalAlignment: Text.AlignVCenter
                    renderType: Text.NativeRendering

                    MouseArea {
                        anchors.fill: parent
                        onClicked: valleySdfCheckBox.checked = !valleySdfCheckBox.checked
                    }
                }
            }

            Label {
                width: parent.width
                text: "Favors thin groove/throat regions in valley and valley seam modes."
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                wrapMode: Text.WordWrap
                renderType: Text.NativeRendering
            }
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Preview Toggle
        Row {
            spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

            CheckBox {
                id: previewCheckBox
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ShowPreview") : true

                onCheckedChanged: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("ShowPreview", checked)
                    }
                }
            }

            Label {
                height: previewCheckBox.height
                text: catalog.i18nc("@label", "Show cut plane preview")
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                verticalAlignment: Text.AlignVCenter
                renderType: Text.NativeRendering

                MouseArea {
                    anchors.fill: parent
                    onClicked: previewCheckBox.checked = !previewCheckBox.checked
                }
            }
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Connector Section
        Column {
            id: connectorsSection
            property bool expandedState: false
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
            visible: !(UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path_isolate")

            Label {  // collapsible header
                text: (connectorsSection.expandedState ? "▾  " : "▸  ") + "Connectors"
                font: UM.Theme.getFont("default_bold")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
                MouseArea { anchors.fill: parent; onClicked: connectorsSection.expandedState = !connectorsSection.expandedState }
            }

            Column {  // collapsible body
                visible: connectorsSection.expandedState
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

            // Connector Enable Toggle
            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                CheckBox {
                    id: connectorCheckBox
                    checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ConnectorEnabled") : true

                    onCheckedChanged: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("ConnectorEnabled", checked)
                        }
                    }
                }

                Label {
                    height: connectorCheckBox.height
                    text: catalog.i18nc("@label", "Add alignment connectors")
                    font: UM.Theme.getFont("default")
                    color: UM.Theme.getColor("text")
                    verticalAlignment: Text.AlignVCenter
                    renderType: Text.NativeRendering

                    MouseArea {
                        anchors.fill: parent
                        onClicked: connectorCheckBox.checked = !connectorCheckBox.checked
                    }
                }
            }

            // Connector description
            Label {
                width: parent.width
                text: "Adds peg to smaller part, hole to larger part"
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                visible: connectorCheckBox.checked
                renderType: Text.NativeRendering
            }

            // Connector Settings (visible when enabled)
            Column {
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
                visible: connectorCheckBox.checked

                // Diameter
                RowLayout {
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        Layout.preferredWidth: 70
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Diameter:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }

                    Slider {
                        id: diameterSlider
                        Layout.fillWidth: true
                        from: 2
                        to: 10
                        value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ConnectorDiameter") : 4
                        stepSize: 0.5

                        onValueChanged: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("ConnectorDiameter", value)
                            }
                        }
                    }

                    Label {
                        Layout.preferredWidth: 50
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: diameterSlider.value.toFixed(1) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }
                }

                // Height
                RowLayout {
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        Layout.preferredWidth: 70
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Height:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }

                    Slider {
                        id: heightConnectorSlider
                        Layout.fillWidth: true
                        from: 1
                        to: 8
                        value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ConnectorHeight") : 3
                        stepSize: 0.5

                        onValueChanged: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("ConnectorHeight", value)
                            }
                        }
                    }

                    Label {
                        Layout.preferredWidth: 50
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: heightConnectorSlider.value.toFixed(1) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }
                }

                // Clearance
                RowLayout {
                    width: parent.width
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        Layout.preferredWidth: 70
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Clearance:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }

                    Slider {
                        id: clearanceSlider
                        Layout.fillWidth: true
                        from: 0.1
                        to: 0.5
                        value: UM.ActiveTool ? UM.ActiveTool.properties.getValue("ConnectorClearance") : 0.2
                        stepSize: 0.05

                        onValueChanged: {
                            if (UM.ActiveTool) {
                                UM.ActiveTool.setProperty("ConnectorClearance", value)
                            }
                        }
                    }

                    Label {
                        Layout.preferredWidth: 50
                        Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                        text: clearanceSlider.value.toFixed(2) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                    }
                }
            }
            }  // end collapsible body
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Debug Capture Toggle
        Column {
            id: debugSection
            property bool expandedState: false
            width: parent.width
            spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

            Label {  // collapsible header
                text: (debugSection.expandedState ? "▾  " : "▸  ") + "Debug"
                font: UM.Theme.getFont("default_bold")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
                MouseArea { anchors.fill: parent; onClicked: debugSection.expandedState = !debugSection.expandedState }
            }

            Column {  // collapsible body
                visible: debugSection.expandedState
                width: parent.width
                spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                CheckBox {
                    id: debugCaptureCheckBox
                    checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("DebugCaptureEnabled") : false

                    onCheckedChanged: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("DebugCaptureEnabled", checked)
                        }
                    }
                }

                Label {
                    height: debugCaptureCheckBox.height
                    text: catalog.i18nc("@label", "Debug capture")
                    font: UM.Theme.getFont("default")
                    color: UM.Theme.getColor("text")
                    verticalAlignment: Text.AlignVCenter
                    renderType: Text.NativeRendering

                    MouseArea {
                        anchors.fill: parent
                        onClicked: debugCaptureCheckBox.checked = !debugCaptureCheckBox.checked
                    }
                }
            }

            Label {
                width: parent.width
                text: "Saves mesh and parameters for each cut to the plugin captures/ folder"
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
                visible: debugCaptureCheckBox.checked
                wrapMode: Text.WordWrap
                renderType: Text.NativeRendering
            }

            CheckBox {
                id: showExperimentalCheckBox
                text: "Show experimental cut modes"
                checked: cutModeComboBox.showExperimental
                onToggled: {
                    cutModeComboBox.manualShowExperimental = checked
                    // Turning it off while an experimental mode is active would
                    // leave the combo and CutMode out of sync; snap to Multi-point.
                    if (!checked && UM.ActiveTool &&
                        cutModeComboBox.experimentalValues.indexOf(UM.ActiveTool.properties.getValue("CutMode")) >= 0) {
                        UM.ActiveTool.setProperty("CutMode", "path")
                    }
                }
            }
            }  // end collapsible body
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Instructions
        Label {
            width: parent.width
            text: catalog.i18nc("@label", "Click on an object to split it at that location.")
            font: UM.Theme.getFont("default")
            color: UM.Theme.getColor("text")
            wrapMode: Text.WordWrap
            renderType: Text.NativeRendering
        }

        Label {
            width: parent.width
            text: catalog.i18nc("@label", "Ctrl+Click to switch to Move tool.")
            font: UM.Theme.getFont("default_italic")
            color: UM.Theme.getColor("text_inactive")
            wrapMode: Text.WordWrap
            renderType: Text.NativeRendering
        }
    }
}
