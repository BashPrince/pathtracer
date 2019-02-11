from editor.scene.updatetype import UpdateType
from editor.scene.material import Material
from PySide2.QtGui import QVector4D

class DiffuseMaterial(Material):
    def __init__(self, name, scene, color = QVector4D(0.5, 0.5, 0.5, 1.0)):
        super().__init__(name, scene, color)
        self.color = color
    
    def setColor(self, color):
        self.color = color
        self.editorColor = color
        if self.scene:
            self.scene.update(UpdateType.MATERIAL_CHANGE)
    
    def getColor(self):
        return self.color