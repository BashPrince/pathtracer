from editor.ui.view import View
from editor.scene.updatetype import UpdateType

from PySide2.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QInputDialog
from PySide2 import QtCore, Qt
from PySide2.QtCore import Qt

class SceneListView(View, QListWidget):
    def __init__(self, scene, parent = None):
        View.__init__(self, scene)
        QListWidget.__init__(self, parent)
        self.scene.registerView(self, [UpdateType.OBJECT_CREATE, UpdateType.OBJECT_DELETE, UpdateType.OBJECT_NAME_CHANGE, UpdateType.MATERIAL_SELECTION])
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.itemSelectionChanged.connect(self.selectionSlot)
        self.itemChanged.connect(self.itemEditSlot)
        self.notify(UpdateType.OBJECT_CREATE)
    
    def notify(self, updateType):
        self.itemSelectionChanged.disconnect(self.selectionSlot)
        if updateType == UpdateType.MATERIAL_SELECTION:
            self.clearSelection()
        else:
            self.itemChanged.disconnect(self.itemEditSlot)
            self.clear()
            for o in self.scene.getObjectIterator():
                item = QListWidgetItem(o.name, self)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                item.setData(QtCore.Qt.UserRole, o)
            self.itemChanged.connect(self.itemEditSlot)
        self.itemSelectionChanged.connect(self.selectionSlot)
    

    def selectionSlot(self):
        selectedObjects = []
        for item in self.selectedItems():
            selectedObjects.append(item.data(QtCore.Qt.UserRole))
        
        self.scene.setSelectedObjects(selectedObjects)
    
    def itemEditSlot(self, item):
        item.data(QtCore.Qt.UserRole).setName(item.text())