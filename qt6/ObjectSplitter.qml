// Copyright (c) 2024 Emanuel Lönnberg.
// This tool is released under the terms of the LGPLv3 or higher.

import QtQuick 2.15
import QtQuick.Controls 2.15

import UM 1.6 as UM
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
                model: ["Horizontal", "Vertical", "Smallest Section", "Shortest Seam", "Radial (geodesic)", "Path (multi-point)", "Valley (groove)"]
                currentIndex: {
                    if (UM.ActiveTool) {
                        var mode = UM.ActiveTool.properties.getValue("CutMode")
                        if (mode === "horizontal") return 0
                        if (mode === "vertical") return 1
                        if (mode === "smallest") return 2
                        if (mode === "shortest") return 3
                        if (mode === "radial") return 4
                        if (mode === "path") return 5
                        if (mode === "valley") return 6
                    }
                    return 0
                }
                onActivated: {
                    if (UM.ActiveTool) {
                        var modeMap = ["horizontal", "vertical", "smallest", "shortest", "radial", "path", "valley"]
                        UM.ActiveTool.setProperty("CutMode", modeMap[currentIndex])
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
                    if (mode === "valley") return "Find and follow a valley/groove near click"
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
            visible: UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path"

            Label {
                text: "Points placed: " + (UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathPointCount") : 0)
                font: UM.Theme.getFont("default")
                color: UM.Theme.getColor("text")
                renderType: Text.NativeRendering
            }

            CheckBox {
                id: closeLoopCheckbox
                text: "Close loop"
                checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathCloseLoop") : false
                onClicked: {
                    if (UM.ActiveTool) {
                        UM.ActiveTool.setProperty("PathCloseLoop", checked)
                    }
                }
            }

            Row {
                spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                Button {
                    text: "Clear Points"
                    width: 85
                    height: UM.Theme.getSize("setting_control").height
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("CutMode", "path")  // triggers clear
                        }
                    }
                }

                Button {
                    text: "Cut Along Path"
                    width: 100
                    height: UM.Theme.getSize("setting_control").height
                    enabled: UM.ActiveTool && UM.ActiveTool.properties.getValue("PathPointCount") >= 2
                    onClicked: {
                        if (UM.ActiveTool) {
                            UM.ActiveTool.setProperty("TriggerPathCut", true)
                        }
                    }
                }
            }
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
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

            Label {
                width: parent.width
                text: "Higher = more accurate but slower"
                font: UM.Theme.getFont("default_italic")
                color: UM.Theme.getColor("text_inactive")
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
                Row {
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        height: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Diameter:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 70
                    }

                    Slider {
                        id: diameterSlider
                        width: 120
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
                        height: UM.Theme.getSize("setting_control").height
                        text: diameterSlider.value.toFixed(1) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 50
                    }
                }

                // Height
                Row {
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        height: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Height:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 70
                    }

                    Slider {
                        id: heightConnectorSlider
                        width: 120
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
                        height: UM.Theme.getSize("setting_control").height
                        text: heightConnectorSlider.value.toFixed(1) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 50
                    }
                }

                // Clearance
                Row {
                    spacing: Math.round(UM.Theme.getSize("default_margin").width / 2)

                    Label {
                        height: UM.Theme.getSize("setting_control").height
                        text: catalog.i18nc("@label", "Clearance:")
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 70
                    }

                    Slider {
                        id: clearanceSlider
                        width: 120
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
                        height: UM.Theme.getSize("setting_control").height
                        text: clearanceSlider.value.toFixed(2) + " mm"
                        font: UM.Theme.getFont("default")
                        color: UM.Theme.getColor("text")
                        verticalAlignment: Text.AlignVCenter
                        renderType: Text.NativeRendering
                        width: 50
                    }
                }
            }
        }

        // Separator
        Rectangle {
            width: parent.width
            height: 1
            color: UM.Theme.getColor("lining")
        }

        // Debug Capture Toggle
        Column {
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
