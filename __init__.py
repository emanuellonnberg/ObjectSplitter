# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

from . import ObjectSplitter
from UM.Logger import Logger

from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog("objectsplitter")

def getMetaData():
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
    Logger.log("d", "ObjectSplitter: Registering tool")
    tool = ObjectSplitter.ObjectSplitter()
    tool.setPluginId("ObjectSplitter")
    Logger.log("d", "ObjectSplitter: Tool instance created with ID: ObjectSplitter")
    return { "tool": tool }
