from editor.ui.view import View
from editor.ui.rendersettings import Rendersettings
from editor.scene.updatetype import UpdateType
from editor.scene import drawableClasses
from editor.scene import SceneObjectType
from editor.scene.camera import Camera
from editor.scene.cube import Cube
from editor.scene.sphere import Sphere
from editor.scene.plane import Plane
from editor.scene.disc import Disc
from editor.scene.light import Light
from editor.scene.viewportplane import ViewportPlane
from editor.scene.diffusematerial import DiffuseMaterial
from editor.scene.specularmaterial import SpecularMaterial

from PyRenderer import PyRenderer

from PySide2.QtWidgets import QOpenGLWidget, QApplication, QFileDialog
from PySide2.QtGui import QImage, QColor, QOpenGLShader, QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShaderProgram, QOpenGLTexture, QMatrix4x4, QMouseEvent, QSurfaceFormat, QOpenGLVersionProfile, QVector3D, QVector2D
from PySide2.QtCore import Qt, QByteArray, QTimer
from PySide2.support import VoidPtr
from OpenGL import GL
import numpy as np
from ctypes import *
import time
from enum import IntEnum


class MaterialType(IntEnum):
    DIFFUSE = 0

class Viewport(View, QOpenGLWidget):
    def __init__(self, scene, updateFpsDisplay, renderSettings = Rendersettings(), parent = None):
        View.__init__(self, scene)
        QOpenGLWidget.__init__(self, parent)
        self.renderImage = QImage()
        self.renderSettings = renderSettings
        self.updateFpsDisplay = updateFpsDisplay
        self.shaderProgram = QOpenGLShaderProgram()
        self.viewPortShaderProgram = QOpenGLShaderProgram()
        self.lightShaderProgram = QOpenGLShaderProgram()
        self.eyeLoc = QVector3D(0.0, 0.0, 5.0)
        self.pressLoc = QVector2D()
        self.isRendering = False
        self.viewportPlane = ViewportPlane()
        self.viewportTexture = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.checkRender)
        self.scene.registerView(self, [UpdateType.MATERIAL_CHANGE, UpdateType.MATERIAL_CREATE, UpdateType.MATERIAL_DELETE, UpdateType.OBJECT_CREATE, UpdateType.OBJECT_DELETE, UpdateType.OBJECT_TRANSFORM, UpdateType.CAMERA_CHANGE, UpdateType.LIGHT_CHANGE, UpdateType.SCENE_LOAD])
        self.timestamps = None
        self.fpsWindow = 10
        self.renderer = None
        self.renderStartTime = None

    def initializeGL(self):
        self.glFunctions = self.context().functions()
        self.glFunctions.glEnable(GL.GL_DEPTH_TEST)
        self.glFunctions.glEnable(GL.GL_CULL_FACE)
        self.glFunctions.glClearColor(0.15, 0.25, 0.3, 1.0)

        vertSuccess = self.shaderProgram.addShaderFromSourceFile(QOpenGLShader.Vertex, "editor/ui/shaders/vertex.vert")
        fragSuccess = self.shaderProgram.addShaderFromSourceFile(QOpenGLShader.Fragment, "editor/ui/shaders/fragment.frag")

        print("Vertex shader compilation {}".format("successful" if vertSuccess else "failed"))
        print("Fragment shader compilation {}".format("successful" if fragSuccess else "failed"))
        print("Program linking {}".format("successful" if self.shaderProgram.link() else "failed"))

        vertSuccess = self.viewPortShaderProgram.addShaderFromSourceFile(QOpenGLShader.Vertex, "editor/ui/shaders/viewportvertex.vert")
        fragSuccess = self.viewPortShaderProgram.addShaderFromSourceFile(QOpenGLShader.Fragment, "editor/ui/shaders/viewportfragment.frag")

        print("Viewport vertex shader compilation {}".format("successful" if vertSuccess else "failed"))
        print("Viewport fragment shader compilation {}".format("successful" if fragSuccess else "failed"))
        print("Viewport program linking {}".format("successful" if self.viewPortShaderProgram.link() else "failed"))

        vertSuccess = self.lightShaderProgram.addShaderFromSourceFile(QOpenGLShader.Vertex, "editor/ui/shaders/vertex.vert")
        fragSuccess = self.lightShaderProgram.addShaderFromSourceFile(QOpenGLShader.Fragment, "editor/ui/shaders/lightfragment.frag")

        print("Light vertex shader compilation {}".format("successful" if vertSuccess else "failed"))
        print("Light fragment shader compilation {}".format("successful" if fragSuccess else "failed"))
        print("Light program linking {}".format("successful" if self.viewPortShaderProgram.link() else "failed"))

        self.shaderProgram.release()
        self.viewPortShaderProgram.release()

        # Call all drawable Objects initGL
        for c in drawableClasses:
            if isinstance(c, ViewportPlane):
                c.initGL(self.viewPortShaderProgram)
            else:
                c.initGL(self.shaderProgram)

    def resizeGL(self, w, h):
        self.scene.getCamera().setAspect(w, h)
        if self.viewportTexture and self.context().isValid():
            self.viewportTexture.destroy()
        if self.context().isValid():
            self.renderImage = QImage(w, h, QImage.Format_RGB32)
            self.renderImage.fill(Qt.black)
            self.viewportTexture = QOpenGLTexture(QOpenGLTexture.Target2D)
            self.viewportTexture.setSize(w, h)
            self.viewportTexture.setBorderColor(0, 0, 0, 255)
            self.viewportTexture.setWrapMode(QOpenGLTexture.ClampToBorder)
            self.viewportTexture.setMinificationFilter(QOpenGLTexture.Nearest)
            self.viewportTexture.setMagnificationFilter(QOpenGLTexture.Nearest)
            self.viewportTexture.setData(self.renderImage, QOpenGLTexture.DontGenerateMipMaps)
            self.updateRenderResolution(w, h)
        

    def paintGL(self):
        self.glFunctions.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.isRendering:
            self.shaderProgram.bind()
            self.shaderProgram.setUniformValue("eyePos", self.scene.getCamera().getTranslation())
            self.shaderProgram.release()

            for obj in self.scene.getObjectIterator():
                if isinstance(obj, Light):
                    obj.draw(self.lightShaderProgram, self.glFunctions)
                else:
                    obj.draw(self.shaderProgram, self.glFunctions)
        else:
            self.viewportTexture.bind()
            self.viewportPlane.draw(self.viewPortShaderProgram, self.glFunctions)
            self.viewportTexture.release()

    def mousePressEvent(self, e):
        self.pressLoc = QVector2D(e.localPos())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            currPos = QVector2D(e.localPos())
            diff = currPos - self.pressLoc
            diff *= 0.5
            self.pressLoc = currPos
            self.scene.getCamera().rotate(diff.x(), diff.y())
    
    def wheelEvent(self, e):
        if e.modifiers() == Qt.ShiftModifier:
            self.scene.getCamera().zoom(-e.angleDelta().y() / 45.0)
        else:
            self.scene.getCamera().track(-e.angleDelta().y() / 600.0)
    
    def notify(self, updateType):
        if updateType == UpdateType.SCENE_LOAD:
            self.scene.getCamera().setAspect(self.width(), self.height())
        if self.isRendering:
            if updateType == UpdateType.CAMERA_CHANGE:
                self.cameraToRenderer()
            elif updateType in [UpdateType.OBJECT_TRANSFORM, UpdateType.OBJECT_CREATE, UpdateType.OBJECT_DELETE, UpdateType.MATERIAL_CREATE, UpdateType.MATERIAL_DELETE, UpdateType.MATERIAL_CHANGE, UpdateType.LIGHT_CHANGE]:
                self.renderer.stop()
                self.renderer.clearMaterials()
                self.renderer.clearObjects()
                for mat in self.scene.getMaterialIterator():
                    self.materialToRenderer(mat)
                for obj in self.scene.getObjectIterator():
                    self.sceneObjectToRenderer(obj)
                self.cameraToRenderer()
                self.renderer.start()
            self.timestamps = [[time.time(), 0]]
            self.renderStartTime = time.time()
        
        self.update()
    
    def startRender(self):
        if self.isRendering:
            return
        self.isRendering = True
        self.renderer = PyRenderer(self.width(), self.height())
        self.renderer.setRenderSettings(self.renderSettings)
        self.timestamps = [[time.time(), 0]]
        self.cameraToRenderer()
        self.update()
        for mat in self.scene.getMaterialIterator():
            self.materialToRenderer(mat)
        for obj in self.scene.getObjectIterator():
            self.sceneObjectToRenderer(obj)
        self.renderer.start()
        self.timer.setInterval(20)
        self.timer.start()
        self.renderStartTime = time.time()
    
    def stopRender(self):
        if not self.isRendering:
            return
        self.timer.stop()
        self.isRendering = False
        self.renderer = None
        self.resizeGL(self.width(), self.height())
        self.update()
        self.updateFpsDisplay('')
    
    def updateRenderResolution(self, width, height):
        if not self.isRendering:
            return
        self.renderer.setResolution(width, height)
        self.timestamps = [[time.time(), 0]]

    def checkRender(self):
        if not self.isRendering:
            return
        imageArray, numIterations = self.renderer.getData(self.width(), self.height())
        if numIterations < 0 or imageArray.shape != (self.height(), self.width()):
            self.timer.start()
            return
        if self.viewportTexture and self.context().isValid():
            self.viewportTexture.destroy()
        if self.context().isValid():
            self.renderImage = QImage(self.width(), self.height(), QImage.Format_RGB32)
            copyArray = np.ndarray(shape = (self.height(), self.width()), dtype = np.uint32, buffer = self.renderImage.bits())
            copyArray[0:,0:] = imageArray[0:,0:]
            self.viewportTexture = QOpenGLTexture(QOpenGLTexture.Target2D)
            self.viewportTexture.setSize(self.width(), self.height())
            self.viewportTexture.setBorderColor(0, 0, 0, 255)
            self.viewportTexture.setWrapMode(QOpenGLTexture.ClampToBorder)
            self.viewportTexture.setMinificationFilter(QOpenGLTexture.Nearest)
            self.viewportTexture.setMagnificationFilter(QOpenGLTexture.Nearest)
            self.viewportTexture.setData(self.renderImage, QOpenGLTexture.DontGenerateMipMaps)
            if len(self.timestamps) == self.fpsWindow:
                self.timestamps.pop()
            self.timestamps.insert(0, [time.time(), numIterations])
            diffs = [(self.timestamps[i][1] - self.timestamps[i + 1][1]) / (self.timestamps[i][0] - self.timestamps[i + 1][0]) for i in range(len(self.timestamps) - 1)]
            secs = int(time.time() - self.renderStartTime)
            self.updateFpsDisplay('{: 4.1f} iterations/sec     {:02.0f}:{:02.0f} min    {:7d} iterations    {:4d} x {:4d} pixel'.format(sum(diffs) / len(diffs), secs // 60, secs % 60, numIterations, self.width(), self.height()))
        self.update()
        self.timer.start()

    def closeViewport(self):
        self.stopRender()
        if self.viewportTexture and self.context().isValid():
            self.viewportTexture.destroy()
    
    def cameraToRenderer(self):
        cam = self.scene.getCamera()
        worldPos = cam.getTranslation()
        targetPos = self.scene.getFocusObject().getTranslation()
        self.renderer.updateCamera([worldPos.x(), worldPos.y(), worldPos.z()], [targetPos.x(), targetPos.y(), targetPos.z()], cam.fovY, cam.fStop, cam.focusDistance, cam.stratificationLevel)
    
    def sceneObjectToRenderer(self, sceneObject):
        objType = None
        materialId = None
        lightIntensity = 0.0
        lightColor = [0.0, 0.0, 0.0]
        isVisible = True
        if isinstance(sceneObject, Camera):
            objType = SceneObjectType.CAMERA
            materialId = -1
        if isinstance(sceneObject, Sphere):
            objType = SceneObjectType.SPHERE
            materialId = sceneObject.material.id
        if isinstance(sceneObject, Cube):
            objType = SceneObjectType.CUBE
            materialId = sceneObject.material.id
        if isinstance(sceneObject, Plane):
            objType = SceneObjectType.PLANE
            materialId = sceneObject.material.id
        if isinstance(sceneObject, Disc):
            objType = SceneObjectType.DISC
            materialId = sceneObject.material.id
        if isinstance(sceneObject, Light):
            objType = sceneObject.getEmitterShape()
            materialId = sceneObject.material.id
            lightIntensity = sceneObject.getIntensity()
            colVec = sceneObject.getColor()
            lightColor = [colVec.x(), colVec.y(), colVec.z()]
            isVisible = sceneObject.getVisibility()

        matrixArray = sceneObject.calculateModel().transposed().copyDataTo()
        self.renderer.addObject(sceneObject.id, materialId, objType, matrixArray, lightIntensity, lightColor, sceneObject.area(), isVisible)

    def materialToRenderer(self, material):
        if type(material) == DiffuseMaterial:
            col = material.getEditorColor()
            self.renderer.addDiffuseMaterial(material.id, [col.x(), col.y(), col.z()])
        if type(material) == SpecularMaterial:
            reflectionColor = material.getReflectionColor()
            transmissionColor = material.getTransmissionColor()
            self.renderer.addSpecularMaterial(material.id, [reflectionColor.x(), reflectionColor.y(), reflectionColor.z()],
                                                          [transmissionColor.x(), transmissionColor.y(), transmissionColor.z()],
                                                          material.getIOR())
    
    def setRenderSettings(self, renderSettings):
        self.renderSettings = renderSettings
        if self.isRendering:
            self.renderer.setRenderSettings(self.renderSettings)
    
    def getRenderSettings(self):
        return self.renderSettings
    
    def saveImage(self):
        if not self.isRendering:
            return
        
        saveImage = self.renderImage.mirrored(horizontal=True)
        fileTuple = QFileDialog.getSaveFileName(self, 'Save destination', dir="./images", filter='Images (*.png *.jpg);;All files (*)')
        if fileTuple[0]:
            filename = fileTuple[0]
            if '.' not in filename:
                filename += '.png'
            saveImage.save(filename, quality=100)
