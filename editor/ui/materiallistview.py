from editor.ui.view import View
from editor.scene.updatetype import UpdateType

from PySide2.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PySide2 import QtCore
from PySide2.QtCore import Qt

class MaterialListView(View, QListWidget):
    def __init__(self, scene, parent = None):
        View.__init__(self, scene)
        QListWidget.__init__(self, parent)
        self.scene.registerView(self, [UpdateType.MATERIAL_CREATE, UpdateType.MATERIAL_DELETE, UpdateType.MATERIAL_NAME_CHANGE, UpdateType.OBJECT_SELECTION])
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.itemSelectionChanged.connect(self.selectionSlot)
        self.itemChanged.connect(self.itemEditSlot)
        self.notify(UpdateType.MATERIAL_CREATE)
    
    def notify(self, updateType):
        self.itemSelectionChanged.disconnect(self.selectionSlot)
        if updateType == UpdateType.OBJECT_SELECTION:
            self.clearSelection()
        else:
            self.itemChanged.disconnect(self.itemEditSlot)
            self.clear()
            for o in self.scene.getMaterialIterator():
                item = QListWidgetItem(o.name, self)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                item.setData(QtCore.Qt.UserRole, o)
            self.itemChanged.connect(self.itemEditSlot)
        self.itemSelectionChanged.connect(self.selectionSlot)

    def selectionSlot(self):
        selectedMaterials = []
        for item in self.selectedItems():
            selectedMaterials.append(item.data(QtCore.Qt.UserRole))
        
        self.scene.setSelectedMaterials(selectedMaterials)
    
    def itemEditSlot(self, item):
        item.data(QtCore.Qt.UserRole).setName(item.text())