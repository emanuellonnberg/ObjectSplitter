# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

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
