from editor.scene.transformableobject import TransformableObject
from editor.scene.cube import Cube
from editor.scene.plane import Plane
from editor.scene.disc import Disc
from editor.scene.sphere import Sphere
from editor.scene.diffusematerial import DiffuseMaterial
from editor.scene.updatetype import UpdateType
from editor.scene import SceneObjectType


from PySide2.QtGui import QMatrix4x4, QVector3D, QVector4D, QOpenGLBuffer, QOpenGLVertexArrayObject
from OpenGL import GL
from PySide2.support import VoidPtr
from enum import Enum



class Light(TransformableObject):

    def __init__(self, name, scene = None, material = None, translation = QVector3D(), rotation = QVector3D(), scale = QVector3D(1.0, 1.0, 1.0), emitterShape = SceneObjectType.SPHERE, intensity = 1.0, color = QVector4D(1.0, 1.0, 1.0, 1.0), isVisible = True):
        super().__init__(name, scene, material, translation, rotation, scale)
        self.emitterShape = emitterShape
        self.intensity = intensity
        self.color = color
        self.isVisible = isVisible
    
    def setEmitterShape(self, emitterShape):
        self.emitterShape = emitterShape
        self.scene.update(UpdateType.OBJECT_CREATE)
    
    def setIntensity(self, intensity):
        self.intensity = intensity
        self.scene.update(UpdateType.OBJECT_CREATE)
    
    def setColor(self, color):
        self.color = color
        self.scene.update(UpdateType.LIGHT_CHANGE)

    def getEmitterShape(self):
        return self.emitterShape
    
    def getIntensity(self):
        return self.intensity
    
    def getColor(self):
        return self.color

    def setVisibility(self, isVisible):
        self.isVisible = isVisible
        self.scene.update(UpdateType.LIGHT_CHANGE)
    
    def getVisibility(self):
        return self.isVisible
    
    def draw(self, shaderProgram, glFunctions):
        obj = None
        if self.emitterShape == SceneObjectType.SPHERE:
            obj = Sphere(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec)
        if self.emitterShape == SceneObjectType.PLANE:
            obj = Plane(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec)
        if self.emitterShape == SceneObjectType.DISC:
            obj = Disc(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec)
        if self.emitterShape == SceneObjectType.CUBE:
            obj = Cube(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec)
        
        obj.draw(shaderProgram, glFunctions, self.scene.getCamera())
    
    def area(self):
        if self.emitterShape == SceneObjectType.SPHERE:
            return Sphere(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec).area()
        if self.emitterShape == SceneObjectType.PLANE:
            return Plane(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec).area()
        if self.emitterShape == SceneObjectType.DISC:
            return Disc(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec).area()
        if self.emitterShape == SceneObjectType.CUBE:
            return Cube(self.name, material=DiffuseMaterial('TempMaterial', None, self.color * self.intensity), translation=self.translationVec, rotation=self.rotationVec, scale=self.scaleVec).area()