# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

# Add the plugin's lib/ (bundled non-Cura deps) to sys.path before other imports.
#
# Only packages Cura does not ship are bundled in lib/: trimesh 4.x, networkx
# and rtree. numpy, scipy and shapely come from Cura itself, so the plugin
# stays ABI-correct across Cura updates instead of carrying Python-version-
# locked copies. See requirements-bundle.txt / scripts/bundle_deps.py.
#
# Cura ships an old trimesh (3.9.36) whose section() requires rtree and lacks
# the to_2D() API, so we load our bundled trimesh 4.x ahead of it (replacing
# the cached 3.x module). trimesh 4.x is backward compatible for the operations
# Cura uses (mesh loading via trimesh.load()).
import sys
import os
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_plugin_dir, "lib")
if os.path.isdir(_lib_dir):
    if _lib_dir not in sys.path:
        sys.path.insert(0, _lib_dir)

    # Replace Cura's old trimesh 3.x with our bundled 4.x
    if 'trimesh' in sys.modules:
        _old_version = getattr(sys.modules['trimesh'], '__version__', '0')
        if _old_version.startswith('3.'):
            _to_remove = [k for k in sys.modules
                          if k == 'trimesh' or k.startswith('trimesh.')]
            for k in _to_remove:
                del sys.modules[k]
            import trimesh  # loads 4.x from lib/

# Cura-specific imports are deferred so that the core/ and viz/ subpackages
# can be imported and tested outside of the Cura environment.
try:
    from . import ObjectSplitter as _ObjectSplitterModule
    from UM.Logger import Logger
    from UM.i18n import i18nCatalog

    _CURA_AVAILABLE = True
    i18n_catalog = i18nCatalog("objectsplitter")
except ImportError:
    _CURA_AVAILABLE = False


def getMetaData():
    if not _CURA_AVAILABLE:
        return {}
    metadata = {
        "tool": {
            "name": i18n_catalog.i18nc("@label", "Object Splitter"),
            "description": i18n_catalog.i18nc("@info:tooltip", "Split objects into multiple parts by cutting along planes."),
            "icon": "icon.svg",
            "tool_panel": "qml/ObjectSplitter.qml",
            "weight": 5
        }
    }
    Logger.log("d", "ObjectSplitter.getMetaData: returning %s", metadata)
    return metadata


def register(app):
    if not _CURA_AVAILABLE:
        return {}
    Logger.log("d", "ObjectSplitter: Registering tool")
    tool = _ObjectSplitterModule.ObjectSplitter()
    tool.setPluginId("ObjectSplitter")
    Logger.log("d", "ObjectSplitter: Tool instance created with ID: ObjectSplitter")
    return { "tool": tool }
