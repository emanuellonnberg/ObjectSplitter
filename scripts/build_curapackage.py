# Copyright (c) 2026 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Build a ``.curapackage`` for ObjectSplitter.

A ``.curapackage`` is an OPC (Open Packaging Conventions) zip -- the same
container Cura uses for ``.3mf``/material packages -- with this layout::

    package.json                       # marketplace metadata (zip root)
    [Content_Types].xml                # OPC plumbing
    _rels/.rels                        # OPC: points at package.json
    _rels/package.json.rels            # OPC: marks /plugins as a plugin
    files/plugins/ObjectSplitter/...   # the plugin payload (single nesting)

The plugin version, SDK level and description are read from ``plugin.json`` so
that file stays the single source of truth. Marketplace author/website fields
live in ``AUTHOR``/``WEBSITE`` below -- edit them before a public submission.

Usage::

    python scripts/build_curapackage.py            # -> dist/ObjectSplitter-<ver>.curapackage
    python scripts/build_curapackage.py -o out.curapackage
"""

import argparse
import json
import os
import zipfile

PLUGIN_ID = "ObjectSplitter"

# Marketplace author block. Edit before a public Marketplace submission.
AUTHOR = {
    "author_id": "emanuellonnberg",
    "display_name": "Emanuel Lönnberg",
    "email": "emanuel@lonnberg.net",
    "website": "https://github.com/emanuellonnberg/ObjectSplitter",
}
WEBSITE = "https://github.com/emanuellonnberg/ObjectSplitter"

# Top-level entries copied into files/plugins/ObjectSplitter/. Everything else
# (tests, scripts, docs, captures, debug logs, .venv, .git, viz) is dev-only and
# deliberately left out. `lib/` carries the bundled runtime deps and must ship.
INCLUDE_TOP_LEVEL = [
    "__init__.py",
    "ObjectSplitter.py",
    "plugin.json",
    "README.md",
    "LICENSE",
    "icon.svg",
    "core",
    "qml",
    "qt6",
    "lib",
]

# Skipped anywhere in the tree while walking the included directories.
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="json" ContentType="text/json" />
</Types>
"""

RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/package.json" Type="http://schemas.ultimaker.org/package/2018/relationships/opc_metadata" Id="rel0" />
</Relationships>
"""

PACKAGE_RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/plugins" Type="plugin" Id="rel0" />
</Relationships>
"""


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_plugin_meta(root: str) -> dict:
    with open(os.path.join(root, "plugin.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _build_package_metadata(plugin_meta: dict) -> dict:
    """Translate plugin.json into the package.json the Marketplace expects."""
    sdk_versions = plugin_meta.get("supported_sdk_versions") or ["8.0.0"]
    return {
        "author": AUTHOR,
        "description": plugin_meta["description"],
        "display_name": plugin_meta["name"],
        "package_id": PLUGIN_ID,
        "package_type": "plugin",
        "package_version": plugin_meta["version"],
        "sdk_version": plugin_meta["api"],
        # Lowest supported SDK; Marketplace uses this as the floor.
        "sdk_version_semver": min(sdk_versions),
        "website": WEBSITE,
    }


def _iter_payload_files(root: str):
    """Yield (absolute_path, arc_relative_path) for every shipped plugin file."""
    for entry in INCLUDE_TOP_LEVEL:
        abs_entry = os.path.join(root, entry)
        if not os.path.exists(abs_entry):
            print(f"  warning: skipping missing entry '{entry}'")
            continue
        if os.path.isfile(abs_entry):
            yield abs_entry, entry
            continue
        for dirpath, dirnames, filenames in os.walk(abs_entry):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            for name in filenames:
                if name.endswith(EXCLUDE_SUFFIXES):
                    continue
                abs_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
                yield abs_path, rel_path


def build(output_path: str) -> str:
    root = _project_root()
    plugin_meta = _load_plugin_meta(root)
    package_meta = _build_package_metadata(plugin_meta)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # OPC metadata first.
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", RELS_XML)
        zf.writestr("_rels/package.json.rels", PACKAGE_RELS_XML)
        zf.writestr("package.json", json.dumps(package_meta, ensure_ascii=False))
        # Plugin payload under files/plugins/<id>/.
        for abs_path, rel_path in _iter_payload_files(root):
            zf.write(abs_path, f"files/plugins/{PLUGIN_ID}/{rel_path}")
            count += 1

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(
        f"Built {output_path}\n"
        f"  version {package_meta['package_version']}  "
        f"sdk {package_meta['sdk_version']} ({package_meta['sdk_version_semver']}+)  "
        f"{count} files, {size_mb:.1f} MB"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ObjectSplitter .curapackage")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output path (default: dist/ObjectSplitter-<version>.curapackage)",
    )
    args = parser.parse_args()

    root = _project_root()
    version = _load_plugin_meta(root)["version"]
    output = args.output or os.path.join(root, "dist", f"{PLUGIN_ID}-{version}.curapackage")
    build(output)


if __name__ == "__main__":
    main()
