---
name: sync-qml
description: Keep the two ObjectSplitter QML panels (qml/ and qt6/) in sync and balanced. Use after editing either QML file, or before committing any UI change.
---

# Sync the QML panels

`qml/ObjectSplitter.qml` (Qt5/UM 1.5) and `qt6/ObjectSplitter.qml` (Qt6/UM 1.6)
must stay in sync -- every UI change goes in **both**. The only intended
difference is the `import UM <ver>` line. It is easy to edit one and forget the
other, so check before committing.

## 1. Diff the two files

```bash
diff qml/ObjectSplitter.qml qt6/ObjectSplitter.qml
```

Expected: no output, or only the `import UM` version line. Anything else is
drift -- one file got an edit the other did not. Reconcile by applying the same
edit to both (do **not** blindly overwrite one with the other; the `import UM`
line must stay correct per file).

## 2. Check delimiter balance (both files)

A mismatched `(`/`)`, `{`/`}`, or `[`/`]` from a bad edit will break the panel
in Cura with no Python error. Verify each file is balanced:

```bash
for f in qml/ObjectSplitter.qml qt6/ObjectSplitter.qml; do
  python -c "s=open('$f',encoding='utf-8').read();print('$f',s.count('(')-s.count(')'),s.count('{')-s.count('}'),s.count('[')-s.count(']'))"
done
```

All three numbers must be `0 0 0` for both files.

## 3. Notes

- QML is Cura-coupled and cannot be unit-tested headless -- balance + diff are
  the only offline checks. Real verification is in Cura.
- New QML-exposed properties also need backend wiring: instance var,
  `setExposedProperties()` entry, `getFoo`/`setFoo` (see CLAUDE.md "Add a new
  QML-exposed property").
- Combobox modes, mode descriptions, and `getModeHelp` entries all live in both
  files -- update each in both.
