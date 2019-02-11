from editor.scene.transformableobject import TransformableObject
from editor.scene.updatetype import UpdateType

from PySide2.QtGui import QMatrix4x4, QVector3D

class Camera(TransformableObject):
    def __init__(self, name, scene, width = 1, height = 1, translation = QVector3D(), rotation = QVector3D(), scale = QVector3D()):
        super().__init__(name, scene, translation, rotation, scale)
        self.fovY = 45.0
        self.fovYMin = 10.0
        self.fovYMax = 120.0
        self.eyeLoc = QVector3D(0.0, 0.0, 10.0)
        self.focusDistance = self.eyeLoc.z()
        self.fStop = 5.6
        self.width = width
        self.height = height
        self.horRot = 0.0
        self.vertRot = 0.0
        self.stratificationLevel = 1
    
    def getViewMatrix(self):
        view = QMatrix4x4()
        center = self.scene.getFocusObject().getTranslation()
        up = QVector3D(0.0, 1.0, 0.0)
        right = QVector3D(1.0, 0.0, 0.0)

        r = QMatrix4x4()
        r.rotate(self.vertRot, QVector3D(1.0, 0.0, 0.0))
        r.rotate(self.horRot, QVector3D(0.0, 1.0, 0.0))
        view.lookAt(self.eyeLoc * r + center, center, up)
        return view
    
    def getProjectionMatrix(self):
        projection = QMatrix4x4()
        projection.setToIdentity()
        projection.perspective(self.fovY, self.width / self.height, 0.1, 100.0)
        return projection

    def setAspect(self, width, height):
        self.width = width
        self.height = height
        self.scene.update(UpdateType.OBJECT_TRANSFORM)
    
    def rotate(self, horRot, vertRot):
        self.horRot += horRot
        self.horRot %= 360
        self.vertRot += vertRot
        if self.vertRot > 89.5:
            self.vertRot = 89.5
        if self.vertRot < -89.5:
            self.vertRot = -89.5
    
        self.scene.update(UpdateType.CAMERA_CHANGE)
    
    def getTranslation(self):
        r = QMatrix4x4()
        r.rotate(self.vertRot, QVector3D(1.0, 0.0, 0.0))
        r.rotate(self.horRot, QVector3D(0.0, 1.0, 0.0))
        return self.eyeLoc * r + self.scene.getFocusObject().getTranslation()

    
    def draw(self, shaderProgram, glFunctions):
        pass
    
    def track(self, zDelta):
        self.eyeLoc += QVector3D(0.0, 0.0, zDelta)
        self.scene.update(UpdateType.CAMERA_CHANGE)

    def zoom(self, angleDelta):
        self.fovY += angleDelta
        self.fovY = self.fovYMin if self.fovY < self.fovYMin else self.fovY
        self.fovY = self.fovYMax if self.fovY > self.fovYMax else self.fovY
        self.scene.update(UpdateType.CAMERA_CHANGE)

    def setFocusDistance(self, focusDistance):
        if(focusDistance < 0.01):
            return
        self.focusDistance = focusDistance
        self.scene.update(UpdateType.CAMERA_CHANGE)

    def setFstop(self, fStop):
        if(fStop < 0.01):
            return
        self.fStop = fStop
        self.scene.update(UpdateType.CAMERA_CHANGE)
    
    def setStratificationLevel(self, level):
        if level < 1 or level > 5:
            return
        self.stratificationLevel = level
        self.scene.update(UpdateType.CAMERA_CHANGE)

    def getFocusDistance(self):
        return self.focusDistance

    def getFstop(self):
        return self.fStop
    
    def getStratificationLevel(self):
        return self.stratificationLevel