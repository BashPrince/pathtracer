"""Objects that can be placed in a scene"""
from editor.scene.sceneobject import SceneObject
from editor.scene.updatetype import UpdateType

from PySide2.QtGui import QMatrix4x4, QVector3D

class TransformableObject(SceneObject):
    def __init__(self, name, scene = None, material = None, translation = QVector3D(), rotation = QVector3D(), scale = QVector3D()):
        super().__init__(name, scene)
        self.translationVec = translation
        self.rotationVec = rotation
        self.scaleVec = scale
        self.model = self.calculateModel()
        self.hasFocus = False
        if not material is None:
            self.material = material
        else:
            self.material = scene.getDefaultMaterial()

    def setMaterial(self, material):
        self.material = material
        if self.scene:
            self.scene.update(UpdateType.OBJECT_TRANSFORM)
    
    def deleteMaterial(self, material):
        if self.material is material:
            self.material = self.scene.getDefaultMaterial()

    def setTranslation(self, translationVec):
        self.translationVec = translationVec
        self.model = self.calculateModel()
        if self.scene:
            self.scene.update(UpdateType.OBJECT_TRANSFORM)
    
    def setRotation(self, rotationVec):
        self.rotationVec = rotationVec
        self.model = self.calculateModel()
        if self.scene:
            self.scene.update(UpdateType.OBJECT_TRANSFORM)


    def setScale(self, scaleVec):
        self.scaleVec = scaleVec
        self.model = self.calculateModel()
        if self.scene:
            self.scene.update(UpdateType.OBJECT_TRANSFORM)
    
    def getTranslation(self):
        return self.translationVec
    
    def getRotation(self):
        return self.rotationVec
    
    def getScale(self):
        return self.scaleVec

    def calculateModel(self):
        translation = QMatrix4x4()
        translation.translate(self.translationVec)

        scale = QMatrix4x4()
        scale.scale(self.scaleVec)

        rotation = QMatrix4x4()
        rotation.rotate(self.rotationVec.y(), QVector3D(0, 1, 0))
        rotation.rotate(self.rotationVec.x(), QVector3D(1, 0, 0))
        rotation.rotate(self.rotationVec.z(), QVector3D(0, 0, 1))

        return translation * rotation * scale
    
    def isFocusObject(self):
        return self.hasFocus
    
    def setFocus(self, focus):
        self.hasFocus = focus
    
    def userSetFocus(self, focus):
        self.hasFocus = focus
        
        if focus:
            self.scene.setFocusObject(self)
        else:
            self.scene.unsetFocusObject()
    
    def area(self):
        return 0.0