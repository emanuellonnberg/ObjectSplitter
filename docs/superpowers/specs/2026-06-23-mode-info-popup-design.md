# Per-Mode Info Popup — Design

Date: 2026-06-23

## Goal

Give each cut mode a short, discoverable explanation (what it does, when to use
it, how to use it) without permanently growing the already-tight tool panel.
Prompted by the isolate-region workflow being non-obvious.

## Approach

An info **ⓘ** glyph next to the existing one-line mode description opens a small
popup with a fuller explanation. Pure QML — no Python / property changes.

### Mechanism

- A clickable `ⓘ` rendered as a link-colored `Label` inside a `MouseArea`,
  placed in a `Row` beside the mode-description `Label`. Using a glyph + MouseArea
  (not a themed icon) keeps the markup identical across `qml/` (UM 1.5) and
  `qt6/` (UM 1.6), avoiding the `RecolorImage` vs `ColorImage` split.
- Clicking opens a `QtQuick.Controls` `Popup` (modal, `CloseOnPressOutside`)
  containing a bold title, a short paragraph, and a numbered step list.
- Help content comes from an inline JS helper `getModeHelp(mode)` returning
  `{ title, body, steps }`. The icon is always visible; every mode has help.
- Implemented identically in both QML files (kept in sync).

### Content

Concise, user-facing, derived from `docs/cut-techniques.md`:

- **Multi-point (`path`)** — Draw the cut yourself by clicking points across the
  surface; the cut follows the surface point to point and can close into a loop.
  Steps: click points → (optional) Close Loop → Cut.
- **Isolate region (`path_isolate`)** — Extract a region bounded by one or more
  closed loops, then choose which side to keep.
  Steps: place ≥3 points → Finish Loop → (optional more loops) → Pick Target
  Region → click the region → Isolate Region.
- **Horizontal** — Flat cut parallel to the bed at a chosen height.
  Steps: set Height % → Cut.
- **Vertical** — Flat cut perpendicular to the bed through your click.
  Steps: click where to cut → Cut.
- **Smallest Section (`smallest`)** — Searches many plane angles at the click and
  picks the smallest cross-section; finds necks and joints.
  Steps: click on/near the narrow feature → Cut. Higher resolution = finer, slower.
- **Shortest Seam (`shortest`)** — Surface-following cut finding the shortest seam
  around the clicked region, refined for a clean edge; organic protrusions.
  Steps: click the region → Cut. Runs with a 10 s timeout.
- **Radial (`radial`)** — Like Shortest Seam without the refinement step; faster,
  rougher boundary. Quick geodesic preview.
  Steps: click the region → Cut.
- **Valley (`valley`)** — Smallest-section search plus a slide along the cut
  direction, so it finds the narrowest spot even if you click slightly off.
  Steps: click near the groove/neck → Cut.
- **Valley Seam (`valley_seam`)** — Surface-following seam preferring concave
  regions near the click; rounded throat geometry.
  Steps: click the feature → Cut.

## Out of scope

- Help for future modes (the mechanism is generic; add entries as modes land).
- Localization of help strings.

## Verification

No automated tests (declarative, Cura-coupled QML). Verify QML paren/brace
balance and confirm in Cura that the icon shows, the popup opens for each mode,
and closes on outside click.
