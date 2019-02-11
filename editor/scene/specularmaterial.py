from editor.scene.updatetype import UpdateType
from editor.scene.material import Material
from PySide2.QtGui import QVector4D

class SpecularMaterial(Material):
    def __init__(self, name, scene, reflectionColor = QVector4D(1.0, 1.0, 1.0, 1.0), transmissionColor = QVector4D(1.0, 1.0, 1.0, 1.0), IOR = 1.44):
        super().__init__(name, scene, reflectionColor)
        self.reflectionColor = reflectionColor
        self.transmissionColor = transmissionColor
        self.IOR = IOR
    
    def setReflectionColor(self, reflectionColor):
        self.reflectionColor = reflectionColor
        self.editorColor = reflectionColor
        if self.scene:
            self.scene.update(UpdateType.MATERIAL_CHANGE)
    
    def getReflectionColor(self):
        return self.reflectionColor

    def setTransmissionColor(self, transmissionColor):
        self.transmissionColor = transmissionColor
        if self.scene:
            self.scene.update(UpdateType.MATERIAL_CHANGE)
    
    def getTransmissionColor(self):
        return self.transmissionColor

    def setIOR(self, IOR):
        self.IOR = IOR
        if self.scene:
            self.scene.update(UpdateType.MATERIAL_CHANGE)
    
    def getIOR(self):
        return self.IOR