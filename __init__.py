# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

from . import ObjectSplitter

from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog("objectsplitter")

def getMetaData():
    return {
        "tool": {
            "id": "ObjectSplitter",
            "name": i18n_catalog.i18nc("@label", "Object Splitter"),
            "description": i18n_catalog.i18nc("@info:tooltip", "Split objects into multiple parts by cutting along planes."),
            "icon": "icon.svg",
            "tool_panel": "qml/ObjectSplitter.qml",
            "weight": 5,
            "button_style": "tool_button",
            "visible": True,
            "version": 1
        }
    }

def register(app):
    tool = ObjectSplitter.ObjectSplitter()
    return { "tool": tool }
