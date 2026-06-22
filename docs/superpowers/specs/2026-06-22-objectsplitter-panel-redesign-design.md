# ObjectSplitter Panel UI/UX Redesign — Design

Date: 2026-06-22

## Problem

The tool panel (`qml/ObjectSplitter.qml` and `qt6/ObjectSplitter.qml`, ~1141
lines each) has two concrete usability problems:

1. **Horizontal overflow** — several button/field rows use hard-coded pixel
   widths that exceed the fixed `width: 250` panel, so controls spill outside
   the window. Known offenders:
   - Path buttons row: `Clear Points` (85) + `Remove Selected` (120) +
     `Cut` (100) = 305.
   - Isolate "Min faces" row: 70 + 120 + 65 = 255.
   - Connector parameter rows: 70 + 120 + 50 + spacing ≈ 250+ (×3).
2. **Excessive length** — each mode stacks many secondary toggles (markers,
   marker color, cap-ends, prune threshold, connector params, debug capture)
   plus multi-line help-text labels, making the panel very tall.

## Goal

A fuller redesign of the panel that:

- Makes every control responsive so nothing overflows the panel width.
- Centers the layout on the modes actually used, tucking the rarely-used ones.
- Cuts vertical length via collapsible sections and tooltip-based help.
- Changes **layout and disclosure only** — no behavior, property, or
  algorithm changes. Every `setProperty` / `getValue` binding stays identical.

## Decisions (agreed)

- **Scope:** fuller redesign (not just an overflow patch).
- **Modes:**
  - Primary (always in the main combo): **Multi-point, Isolate region,
    Horizontal, Vertical**.
  - Other (behind an "Other modes" disclosure): **Smallest Section,
    Shortest Seam, Radial, Valley, Valley Seam**.
  - Selecting any mode sets the existing `CutMode` property. If the current
    mode is a secondary one, the "Other modes" disclosure auto-expands
    (sticky), so users are not forced to re-expand each time.
- **Width:** stay ~250px but fully responsive — no hard-coded control widths;
  controls share the available width and wrap rather than spill.
- **Help text:** replace always-visible help paragraphs with hover tooltips
  (and a small `ⓘ` affordance only where a label alone is unclear).

## New panel structure (multi-point active)

```
Object Splitter
─────────────────────────
Cut: [ Multi-point      ▾ ]      ← 4 primary modes
     ▸ Other modes               ← expander → 5 secondary modes

Points placed: 5
[######  Cut Along Path  ######] ← primary action, full width
[  Clear  ][ Remove ][ Suggest ] ← share width evenly, wrap if tight

Close loop ⓘ      Cap ends ⓘ      ← tooltips replace paragraphs

▾ Display                         ← collapsible (default collapsed)
   Small markers ⓘ
   Marker color [ Cyan ▾ ]

▸ Connectors                      ← collapsible
▸ Debug                           ← collapsible
```

The same disclosure/responsive treatment applies to every mode's controls and
to the global Connectors and Debug sections.

## Mechanics

### 1. Responsive widths

- Import `QtQuick.Layouts`.
- Button rows become `RowLayout { Button { Layout.fillWidth: true } ... }` so
  buttons split the panel width and never exceed it.
- Label + field + stepper parameter rows become `GridLayout` (or a
  `RowLayout` with `Layout.fillWidth` on the field) sized to fit 250px.
- Remove every hard-coded `width: NN` on buttons and input fields. Fixed
  widths remain only where intentional and known-safe (e.g. a short label
  column), and must keep the row total under the panel width.

### 2. Collapsible section (reusable inline pattern)

A clickable header toggles `visible` on its content. No new files; the pattern
is repeated inline (and kept identical across both QML files):

```qml
Column {
    id: section
    property bool expanded: false
    width: parent.width

    Row {                                  // clickable header
        Label { text: (section.expanded ? "▾ " : "▸ ") + "Connectors" }
        MouseArea { anchors.fill: parent; onClicked: section.expanded = !section.expanded }
    }
    Column {
        visible: section.expanded
        width: parent.width
        // section body
    }
}
```

Sections: **Other modes**, **Display**, **Connectors**, **Debug** — all
default collapsed. (The `expanded` default for "Other modes" is overridden to
`true` when the active mode is a secondary one.)

### 3. Mode disclosure

- Primary `ComboBox` model = the 4 primary mode labels.
- An "Other modes" collapsible reveals a second `ComboBox` with the 5
  secondary mode labels.
- Both combos write the existing `CutMode` string via `setProperty` exactly as
  today; the mode→string maps are split across the two combos but otherwise
  unchanged.
- `currentIndex` for each combo is derived from `CutMode` as today.

### 4. Tooltips

- Use `QtQuick.Controls` `ToolTip` on each toggle/control:
  `ToolTip.visible: hovered; ToolTip.text: "…"`.
- Delete the standalone help-paragraph `Label`s.
- Add a small `ⓘ` (a themed icon or a unicode glyph with a `MouseArea`/hover)
  only where the control's own label does not make the meaning obvious.

## Two-file sync

`qml/ObjectSplitter.qml` (UM 1.5) and `qt6/ObjectSplitter.qml` (UM 1.6) must
stay in sync; only the `import UM x.y` line differs. Every change is applied
identically to both. After editing, diff the two files and confirm the only
difference is the `import UM` version (plus any version-specific component
quirk, which must be called out explicitly).

## Out of scope

- No change to cut algorithms, properties, or `ObjectSplitter.py` behavior.
- No new cut modes; no removal of existing modes (only reordering/disclosure).
- No panel widening (stays ~250px; revisit only if testing shows a hard need).
- No automated QML test harness (none exists; see Testing).

## Testing / verification

- No QML unit tests exist. Verification is a **manual Cura smoke per mode**:
  load a model, select each mode, confirm controls render within the panel
  (no horizontal overflow), disclosures expand/collapse, tooltips show, and
  each action still triggers the same behavior as before.
- The Python test suite (`pytest`, 279 tests) must remain green — it does not
  cover QML, but guards against accidental backend edits.
- A `/sync-qml`-style diff check: the two QML files differ only by the
  `import UM` version.

## Risks

- **Two-file drift** — mitigated by the post-edit diff check.
- **Cura component availability** — `QtQuick.Layouts`, `QtQuick.Controls`
  `ToolTip`, and `MouseArea`-based disclosure are all standard and used
  elsewhere in Cura; confirm exact import versions during implementation.
- **No visual regression net** — mitigated by keeping changes mechanical and
  reversible, and by the per-mode manual smoke.
