from editor.scene.transformableobject import TransformableObject

from PySide2.QtGui import QMatrix4x4, QVector3D, QOpenGLBuffer, QOpenGLVertexArrayObject
from OpenGL import GL
import numpy as np
from ctypes import *
from PySide2.support import VoidPtr
import math


class Disc(TransformableObject):

    vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    vao = QOpenGLVertexArrayObject()

    vertices = []
    
    @classmethod
    def createDiscArray(cls):
        numSegments = 200
        arr = []
        for i in range(numSegments, 0, -1):
            x = math.cos((i / numSegments) * 2 * math.pi)
            z = math.sin((i / numSegments) * 2 * math.pi)
            next_x = math.cos(((i - 1) / numSegments) * 2 * math.pi)
            next_z = math.sin(((i - 1) / numSegments) * 2 * math.pi)
            arr[len(arr):] = 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
            arr[len(arr):] = x, 0.0, z, 0.0, 1.0, 0.0
            arr[len(arr):] = next_x, 0.0, next_z, 0.0, 1.0, 0.0
            arr[len(arr):] = 0.0, 0.0, 0.0, 0.0, -1.0, 0.0
            arr[len(arr):] = next_x, 0.0, next_z, 0.0, -1.0, 0.0
            arr[len(arr):] = x, 0.0, z, 0.0, -1.0, 0.0
        return arr

    @classmethod
    def initGL(cls, shaderProgram):
        cls.vertices = np.array(cls.createDiscArray(), dtype = c_float)
        cls.vbo.create()
        cls.vao.create()
        cls.vbo.bind()
        cls.vbo.allocate(cls.vertices.tobytes(), cls.vertices.size * sizeof(c_float))

        cls.vao.bind()
        shaderProgram.setAttributeBuffer(0, GL.GL_FLOAT, 0, 3, 6 * sizeof(c_float))
        shaderProgram.setAttributeBuffer(1, GL.GL_FLOAT, 3 * sizeof(c_float), 3, 6 * sizeof(c_float))
        shaderProgram.enableAttributeArray(0)
        shaderProgram.enableAttributeArray(1)
        cls.vao.release()

    def __init__(self, name, scene = None, material = None, translation = QVector3D(), rotation = QVector3D(), scale = QVector3D(1.0, 1.0, 1.0)):
        super().__init__(name, scene, material, translation, rotation, scale)
    
    def draw(self, shaderProgram, glFunctions, camera = None):
        self.__class__.vao.bind()
        shaderProgram.bind()

        #calculate view
        if not camera:
            camera = self.scene.getCamera()
        view = camera.getViewMatrix()
        projection = camera.getProjectionMatrix()

        shaderProgram.setUniformValue("model", self.model)
        shaderProgram.setUniformValue("view", view)
        shaderProgram.setUniformValue("projection", projection)
        shaderProgram.setUniformValue("inColor", self.material.getEditorColor())

        glFunctions.glDrawArrays(GL.GL_TRIANGLES, 0, self.__class__.vertices.size / 2)

        self.__class__.vao.release()
        shaderProgram.release()
    
    def area(self):
        return np.pi * self.scaleVec.x() * self.scaleVec.z()