# ObjectSplitter Panel UI/UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Cura tool panel so no control overflows the panel width and the panel is much shorter, while keeping every existing behavior, property, and action identical.

**Architecture:** Pure QML layout/disclosure changes in `qml/ObjectSplitter.qml` and `qt6/ObjectSplitter.qml`. Hard-coded pixel widths become responsive `QtQuick.Layouts` rows; rarely-used controls and help text move behind collapsible sections and hover tooltips. No Python or property changes.

**Tech Stack:** QML (QtQuick 2.15, QtQuick.Controls 2.15, QtQuick.Layouts 1.15), Uranium `UM` 1.5, Cura `Cura` 1.0.

## Global Constraints

- **Two files, kept in sync.** `qml/ObjectSplitter.qml` and `qt6/ObjectSplitter.qml` are currently byte-identical. Every change is applied to both; they must differ by at most the `import UM <version>` line.
- **Layout/disclosure only.** No change to cut algorithms, exposed properties, `setProperty`/`getValue` keys, or `ObjectSplitter.py`. The Python suite (`pytest`, 279 tests) must stay green.
- **Panel width stays ~250px.** No widening. Controls must fit within `mainColumn` `width: 250` — no hard-coded control width may push a row past the panel.
- **Primary modes:** Multi-point (`path`), Isolate region (`path_isolate`), Horizontal (`horizontal`), Vertical (`vertical`).
- **Secondary modes (behind "Other modes"):** Smallest Section (`smallest`), Shortest Seam (`shortest`), Radial (geodesic) (`radial`), Valley (groove) (`valley`), Valley Seam (concavity) (`valley_seam`).
- **No automated QML tests exist.** Per-task verification = (a) sync diff, (b) `pytest` green, (c) manual Cura smoke. The manual smoke cannot be scripted; each task lists exactly what to click.

### Sync-check command (used by every task)

```bash
diff <(sed -E 's/import UM [0-9.]+/import UM/' qml/ObjectSplitter.qml) \
     <(sed -E 's/import UM [0-9.]+/import UM/' qt6/ObjectSplitter.qml) \
  && echo "SYNC OK"
```
Expected output: `SYNC OK` (no diff lines). Because the files start identical, the simplest workflow each task is: **edit `qt6/ObjectSplitter.qml`, then `cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml`**, then run the sync check.

### Manual Cura smoke (referenced by tasks)

Cura cannot be driven from this environment. After each task, the human reloads the plugin in Cura 5.12 and confirms the panel **parses and renders** (no blank panel, no QML error in `cura.log`). Task-specific clicks are listed per task. If the panel is blank, check `cura.log` for `QQmlComponent` / `QML` errors and revert.

---

### Task 1: Add QtQuick.Layouts import + establish sync baseline

**Files:**
- Modify: `qt6/ObjectSplitter.qml:1-8` (import block)
- Modify: `qml/ObjectSplitter.qml:1-8` (via copy)

**Interfaces:**
- Produces: `QtQuick.Layouts` (RowLayout, GridLayout, ColumnLayout, `Layout.*` attached props) available to all later tasks.

- [ ] **Step 1: Confirm the two files are identical to start**

Run:
```bash
diff qml/ObjectSplitter.qml qt6/ObjectSplitter.qml && echo "IDENTICAL"
```
Expected: `IDENTICAL`.

- [ ] **Step 2: Add the Layouts import to `qt6/ObjectSplitter.qml`**

Change the import block from:
```qml
import QtQuick 2.15
import QtQuick.Controls 2.15

import UM 1.5 as UM
import Cura 1.0 as Cura
```
to:
```qml
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import UM 1.5 as UM
import Cura 1.0 as Cura
```

- [ ] **Step 3: Mirror to `qml/ObjectSplitter.qml`**

Run:
```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```

- [ ] **Step 4: Verify sync + Python guard**

Run the sync-check command (Global Constraints). Expected: `SYNC OK`.
Run:
```bash
pytest -q
```
Expected: `279 passed`.

- [ ] **Step 5: Manual Cura smoke**

Reload the plugin in Cura, open the tool. The panel must render exactly as before (import-only change). If blank, check `cura.log`.

- [ ] **Step 6: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: import QtQuick.Layouts for responsive panel layout"
```

---

### Task 2: Make the overflowing button/field rows responsive

This is the concrete "buttons outside the window" fix. Convert the three over-wide rows to fill the panel width instead of hard-coded pixels.

**Files:**
- Modify: `qt6/ObjectSplitter.qml` (path buttons row, isolate min-faces row, 3 connector param rows)
- Modify: `qml/ObjectSplitter.qml` (via copy)

**Interfaces:**
- Consumes: `QtQuick.Layouts` from Task 1.
- Produces: nothing new; same controls, responsive widths.

- [ ] **Step 1: Convert the path buttons row (Clear / Remove / Cut)**

Find the path-mode `Row` containing the `Clear Points`, `Remove Selected`, and the `Cut Along Path`/`Cut Using Points` buttons (currently widths 85, 120, 100). Replace the `Row { ... }` wrapper with a `RowLayout` and replace each button's `width: NN` with `Layout.fillWidth: true`. Keep every other property (`text`, `visible`, `enabled`, `onClicked`) unchanged. Example for the first button:

```qml
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
    // Remove Selected + Cut buttons: same conversion (width: NN -> Layout.fillWidth: true,
    // height -> Layout.preferredHeight), all other properties unchanged.
}
```
Note: a hidden button (`visible: false`) in a `RowLayout` still reserves no space only if you also set `Layout.preferredWidth: visible ? implicitWidth : 0`; simplest is to keep `visible:` as-is — `RowLayout` collapses invisible items by default in Qt 5.15. If a gap appears in Cura, add `Layout.preferredWidth: 0` to the hidden button.

- [ ] **Step 2: Convert the isolate "Min faces" row**

Find the isolate `Row` with the `Min faces:` label (width 70), the number field (width 120), and the stepper (width 65). Replace with `RowLayout`: keep the label `Layout.preferredWidth: 70`, give the field `Layout.fillWidth: true`, keep the stepper at its natural width (`Layout.preferredWidth: 65`). All bindings unchanged.

- [ ] **Step 3: Convert the three connector parameter rows**

Find the three connector parameter `Row`s (each: label width 70 + field width 120 + unit/stepper width 50). Convert each to `RowLayout` with label `Layout.preferredWidth: 70`, field `Layout.fillWidth: true`, trailing element `Layout.preferredWidth: 50`. All bindings unchanged.

- [ ] **Step 4: Mirror + verify sync + Python guard**

```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```
Run the sync-check (expect `SYNC OK`) and `pytest -q` (expect `279 passed`).

- [ ] **Step 5: Manual Cura smoke**

In Cura: Multi-point mode — place ≥1 point; confirm `Clear / Remove / Cut` row fits inside the panel with no horizontal scrollbar/clipping. Isolate region — open the prune section; confirm the Min faces row fits. Enable Connectors — confirm all three parameter rows fit.

- [ ] **Step 6: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: make over-wide button/param rows responsive (fix overflow)"
```

---

### Task 3: Reusable collapsible section; wrap Connectors and Debug

**Files:**
- Modify: `qt6/ObjectSplitter.qml` (Connectors section, Debug section)
- Modify: `qml/ObjectSplitter.qml` (via copy)

**Interfaces:**
- Produces: an inline collapsible pattern (a `Column` with `property bool expanded` + clickable header) reused by Tasks 4 and 5.

- [ ] **Step 1: Wrap the Connectors section in a collapsible**

Find the global Connectors `Column` (the one whose `visible:` excludes `path_isolate`, containing the `connectorCheckBox` and parameter rows). Wrap its existing content in the collapsible pattern. The header text is "Connectors"; the existing content moves inside the inner `Column { visible: section.expanded }`:

```qml
Column {
    id: connectorsSection
    width: parent.width
    spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
    visible: !(UM.ActiveTool && UM.ActiveTool.properties.getValue("CutMode") === "path_isolate")

    Row {
        spacing: Math.round(UM.Theme.getSize("default_margin").width / 4)
        Label {
            text: (connectorsSection.expandedState ? "▾  " : "▸  ") + "Connectors"
            font: UM.Theme.getFont("default_bold")
            color: UM.Theme.getColor("text")
            renderType: Text.NativeRendering
        }
        MouseArea { anchors.fill: parent; onClicked: connectorsSection.expandedState = !connectorsSection.expandedState }
    }

    property bool expandedState: false

    Column {
        visible: connectorsSection.expandedState
        width: parent.width
        spacing: Math.round(UM.Theme.getSize("default_margin").height / 2)
        // ... existing Connectors content (connectorCheckBox + param rows) verbatim ...
    }
}
```
Use `expandedState` (not `expanded`) to avoid clashing with any built-in. Keep all inner bindings verbatim.

- [ ] **Step 2: Wrap the Debug section in a collapsible**

Find the Debug `Column` (containing `debugCaptureCheckBox`). Apply the same pattern with header text "Debug" and `id: debugSection` / `property bool expandedState: false`. Inner content verbatim.

- [ ] **Step 3: Mirror + verify sync + Python guard**

```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```
Sync-check → `SYNC OK`; `pytest -q` → `279 passed`.

- [ ] **Step 4: Manual Cura smoke**

In Cura: confirm "▸ Connectors" and "▸ Debug" appear collapsed; clicking each toggles the body open/closed; the enable checkboxes and params still work when expanded.

- [ ] **Step 5: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: collapsible Connectors and Debug sections"
```

---

### Task 4: Primary/secondary mode disclosure

**Files:**
- Modify: `qt6/ObjectSplitter.qml` (Cut Mode row, lines around the existing `cutModeComboBox`)
- Modify: `qml/ObjectSplitter.qml` (via copy)

**Interfaces:**
- Consumes: collapsible pattern from Task 3.
- Produces: writes the same `CutMode` string property; no new properties.

- [ ] **Step 1: Replace the single mode combo with a primary combo**

Replace the existing `ComboBox { id: cutModeComboBox ... }` (the 9-item one) with a primary combo listing only the four primary modes, mapping to the same `CutMode` strings:

```qml
ComboBox {
    id: cutModeComboBox
    width: 170
    height: UM.Theme.getSize("setting_control").height
    property var modeValues: ["path", "path_isolate", "horizontal", "vertical"]
    model: ["Multi-point", "Isolate region", "Horizontal", "Vertical"]
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
```

- [ ] **Step 2: Add the "Other modes" collapsible with the secondary combo**

Directly below the Cut Mode row, add a collapsible (Task 3 pattern) whose body holds a second combo for the five secondary modes. It auto-expands when the active mode is secondary:

```qml
Column {
    id: otherModesSection
    width: parent.width
    property var secondaryValues: ["smallest", "shortest", "radial", "valley", "valley_seam"]
    property bool expandedState: UM.ActiveTool
        ? secondaryValues.indexOf(UM.ActiveTool.properties.getValue("CutMode")) >= 0
        : false

    Row {
        spacing: Math.round(UM.Theme.getSize("default_margin").width / 4)
        Label {
            text: (otherModesSection.expandedState ? "▾  " : "▸  ") + "Other modes"
            font: UM.Theme.getFont("default")
            color: UM.Theme.getColor("text_inactive")
            renderType: Text.NativeRendering
        }
        MouseArea { anchors.fill: parent; onClicked: otherModesSection.expandedState = !otherModesSection.expandedState }
    }

    ComboBox {
        visible: otherModesSection.expandedState
        width: 170
        height: UM.Theme.getSize("setting_control").height
        model: ["Smallest Section", "Shortest Seam", "Radial (geodesic)", "Valley (groove)", "Valley Seam (concavity)"]
        currentIndex: {
            if (UM.ActiveTool) {
                var idx = otherModesSection.secondaryValues.indexOf(UM.ActiveTool.properties.getValue("CutMode"))
                return idx >= 0 ? idx : 0
            }
            return 0
        }
        onActivated: {
            if (UM.ActiveTool) {
                UM.ActiveTool.setProperty("CutMode", otherModesSection.secondaryValues[currentIndex])
            }
        }
    }
}
```

- [ ] **Step 3: Verify the mode-description Label still covers all 9 modes**

The existing description `Label` (the `if (mode === ...)` chain) is unchanged and already handles all nine `CutMode` strings. Confirm it is left intact.

- [ ] **Step 4: Mirror + verify sync + Python guard**

```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```
Sync-check → `SYNC OK`; `pytest -q` → `279 passed`.

- [ ] **Step 5: Manual Cura smoke**

In Cura: the main combo lists exactly the 4 primary modes; selecting each updates the controls below. Expand "Other modes", pick e.g. Valley — its controls/description appear, and re-opening the tool with Valley active shows "Other modes" already expanded. Switch back to Multi-point via the main combo.

- [ ] **Step 6: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: primary mode combo + Other modes disclosure"
```

---

### Task 5: Per-mode "Display" collapsible (markers + color)

**Files:**
- Modify: `qt6/ObjectSplitter.qml` (path-mode controls, isolate-mode controls)
- Modify: `qml/ObjectSplitter.qml` (via copy)

**Interfaces:**
- Consumes: collapsible pattern from Task 3.

- [ ] **Step 1: Move path-mode "Small Markers" + "Marker Color" into a Display collapsible**

In the path-mode `Column`, take the existing `smallMarkersCheckbox` `CheckBox` and the "Marker Color" `Row` (label + `pathMarkerColorComboBox`) and move them into a collapsible with header "Display" (`id: pathDisplaySection`, `property bool expandedState: false`). All bindings verbatim.

- [ ] **Step 2: Move isolate-mode "Small Markers" + "Marker Color" into a Display collapsible**

Do the same in the isolate-mode `Column` for `isolateSmallMarkersCheckbox` and the isolate "Marker Color" `Row` (`isolatePathMarkerColorComboBox`). Header "Display", `id: isolateDisplaySection`.

- [ ] **Step 3: Mirror + verify sync + Python guard**

```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```
Sync-check → `SYNC OK`; `pytest -q` → `279 passed`.

- [ ] **Step 4: Manual Cura smoke**

In Cura: Multi-point and Isolate modes each show a collapsed "▸ Display"; expanding it reveals Small Markers + Marker Color, and both still work (toggling markers, changing color updates the scene).

- [ ] **Step 5: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: tuck marker display controls into a Display section"
```

---

### Task 6: Replace help paragraphs with tooltips

**Files:**
- Modify: `qt6/ObjectSplitter.qml` (help-text Labels across modes)
- Modify: `qml/ObjectSplitter.qml` (via copy)

**Interfaces:**
- Consumes: nothing new.

- [ ] **Step 1: Add a hover tooltip to each toggle, delete its help paragraph**

For each control that today is followed by an explanatory `Label` (e.g. Close loop, Cap Path Cut, Small Markers, Insert mode, Remove Tiny Fragments), add a tooltip to the control and delete the standalone help `Label`. Tooltip pattern on a `CheckBox`:

```qml
CheckBox {
    id: closeLoopCheckbox
    text: "Close loop"
    hoverEnabled: true
    ToolTip.visible: hovered
    ToolTip.text: "Close the path into a loop so the cut separates an enclosed region."
    checked: UM.ActiveTool ? UM.ActiveTool.properties.getValue("PathCloseLoop") : false
    onClicked: { if (UM.ActiveTool) { UM.ActiveTool.setProperty("PathCloseLoop", checked) } }
}
```
Map each deleted help paragraph's text into the corresponding control's `ToolTip.text` (condensed to one or two sentences). Delete only the pure explanatory `Label`s — keep status `Label`s that show live values (e.g. "Points placed:", "Selected:", "Loops placed:").

- [ ] **Step 2: Keep the mode-description Label**

Leave the single cut-mode description `Label` under the combo (it is the one-line summary per mode). Do not convert it to a tooltip.

- [ ] **Step 3: Mirror + verify sync + Python guard**

```bash
cp qt6/ObjectSplitter.qml qml/ObjectSplitter.qml
```
Sync-check → `SYNC OK`; `pytest -q` → `279 passed`.

- [ ] **Step 4: Manual Cura smoke**

In Cura: across Multi-point, Isolate, Horizontal, Vertical — the long help paragraphs are gone; hovering each toggle shows its tooltip; the panel is visibly shorter; live status labels still update.

- [ ] **Step 5: Commit**

```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: replace help paragraphs with hover tooltips"
```

---

### Task 7: Final verification sweep

**Files:**
- No code changes expected (cleanup only if a check fails).

- [ ] **Step 1: Full sync diff**

Run the sync-check command. Expected: `SYNC OK`. If it fails, re-copy `qt6` → `qml` and re-commit.

- [ ] **Step 2: Python suite**

```bash
pytest -q
```
Expected: `279 passed`.

- [ ] **Step 3: Per-mode Cura smoke checklist**

For each of the 9 modes (4 primary via main combo, 5 via Other modes): open the mode, confirm (a) no control overflows the panel horizontally, (b) collapsibles expand/collapse, (c) tooltips show, (d) the mode's primary action still runs and produces the same result as before the redesign. Note any mode that misbehaves.

- [ ] **Step 4: Commit any fixes**

If Step 3 surfaced fixes, apply to `qt6`, copy to `qml`, re-run Steps 1-2, and:
```bash
git add qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
git commit -m "ui: panel redesign fixes from per-mode smoke"
```

---

## Self-Review

**Spec coverage:**
- Responsive widths / overflow → Tasks 1, 2. ✓
- Primary/secondary mode disclosure (4/5 split) → Task 4. ✓
- Collapsible sections (Other modes, Display, Connectors, Debug) → Tasks 3, 4, 5. ✓
- Tooltip help → Task 6. ✓
- Two-file sync → Global Constraints + every task's copy+sync-check. ✓
- ~250px width, no widening → Global Constraints; responsive conversions keep within parent width. ✓
- Behavior unchanged → every task keeps `setProperty`/`getValue` bindings verbatim; `pytest` guard each task. ✓
- Testing = manual Cura smoke + pytest → each task Steps. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Tooltip texts are condensed from existing help labels at implementation time (Task 6 Step 1 states the rule and shows a full example). Repetitive conversions (Task 2, 5, 6) give a complete worked example plus the explicit list of targets and the transformation rule — acceptable because the change is mechanical and the example is complete.

**Type/name consistency:** Collapsible property is `expandedState` in every section (Connectors, Debug, otherModesSection, pathDisplaySection, isolateDisplaySection). `CutMode` strings match the spec's mode lists. Combo ids: `cutModeComboBox` (primary) reused; secondary combo is anonymous inside `otherModesSection`.

**Risk note:** `RowLayout` invisible-item spacing (Task 2 Step 1) and `ToolTip` hover behavior (Task 6) are the two spots most likely to need a Cura-side tweak; both have a stated fallback / are isolated to their task.
