from editor.scene.transformableobject import TransformableObject

from PySide2.QtGui import QMatrix4x4, QVector3D, QOpenGLBuffer, QOpenGLVertexArrayObject
from OpenGL import GL
import numpy as np
from ctypes import *
from PySide2.support import VoidPtr
import math


class Sphere(TransformableObject):

    vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
    vao = QOpenGLVertexArrayObject()
    
    @classmethod
    def initGL(cls, shaderProgram):
        cls.vbo.create()
        cls.vao.create()
        cls.ebo.create()
        cls.vbo.bind()

        vertices = []
        elements = []

        cls.createCircleArrays(50, vertices, elements)

        cls.vertices = np.array(vertices, dtype = c_float)
        cls.elements = np.array(elements, dtype = c_ushort)


        cls.vbo.allocate(cls.vertices.tobytes(), cls.vertices.size * sizeof(c_float))

        cls.vao.bind()
        cls.ebo.bind()
        cls.ebo.allocate(cls.elements.tobytes(), cls.elements.size * sizeof(c_ushort))
        shaderProgram.setAttributeBuffer(0, GL.GL_FLOAT, 0, 3, 3 * sizeof(c_float))
        shaderProgram.setAttributeBuffer(1, GL.GL_FLOAT, 0, 3, 3 * sizeof(c_float))
        shaderProgram.enableAttributeArray(0)
        shaderProgram.enableAttributeArray(1)
        cls.vao.release()
        cls.ebo.release()
    
    @classmethod
    def createCircleArrays(cls, phiSubdivisions, vertices, elements):
        phiSubdivisions -= phiSubdivisions % 2
        vertices[0:3] = 0.0, 1.0, 0.0
        delta = 360.0 / phiSubdivisions
        
        for th in range(1, phiSubdivisions // 2):
            for ph in range(0, phiSubdivisions):
                v = cls.sphericalToCartesian(th * delta, -ph * delta)
                vertices[len(vertices):] = v.x(), v.y(), v.z()

        vertices[len(vertices):] = 0.0, -1.0, 0.0

        offset = 1
        for colIdx in range(0, phiSubdivisions):
            nextColIdx = (colIdx + 1) % phiSubdivisions
            elements[len(elements):] = offset + colIdx, offset + nextColIdx, 0

        
        for row in range(1, phiSubdivisions // 2 - 1):
            rowIdx = row * phiSubdivisions + 1
            prevRowIdx = rowIdx - phiSubdivisions
            for colIdx in range(0, phiSubdivisions):
                nextColIdx = (colIdx + 1) % phiSubdivisions
                elements[len(elements):] = rowIdx + colIdx, rowIdx + nextColIdx, prevRowIdx + colIdx
                elements[len(elements):] = rowIdx + nextColIdx, prevRowIdx + nextColIdx, prevRowIdx + colIdx
        
        offset = 1 + (phiSubdivisions // 2 - 2) * phiSubdivisions 
        for colIdx in range(0, phiSubdivisions):
            nextColIdx = (colIdx + 1) % phiSubdivisions
            elements[len(elements):] = offset + colIdx, offset + phiSubdivisions, offset + nextColIdx
    
    @staticmethod
    def sphericalToCartesian(theta, phi):
        thetaRot = QMatrix4x4()
        thetaRot.rotate(theta, QVector3D(1.0, 0.0, 0.0))
        phiRot = QMatrix4x4()
        phiRot.rotate(phi, QVector3D(0.0, 1.0, 0.0))
        v = QVector3D(0.0, 1.0, 0.0)
        return (v * thetaRot) * phiRot

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

        glFunctions.glDrawElements(GL.GL_TRIANGLES, self.__class__.elements.size, GL.GL_UNSIGNED_SHORT, VoidPtr(0))

        self.__class__.vao.release()
        shaderProgram.release()

    def area(self):
        return 4 * np.pi * pow(((pow(self.scaleVec.x() * self.scaleVec.y(), 1.6) + pow(self.scaleVec.x() * self.scaleVec.z(), 1.6) + pow(self.scaleVec.y() * self.scaleVec.z(), 1.6)) / 3), 1 / 1.6)
