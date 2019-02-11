from editor.scene.transformableobject import TransformableObject

from PySide2.QtGui import QMatrix4x4, QVector3D, QOpenGLBuffer, QOpenGLVertexArrayObject
from OpenGL import GL
import numpy as np
from ctypes import *
from PySide2.support import VoidPtr


class Cube(TransformableObject):

    vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    vao = QOpenGLVertexArrayObject()

    vertices = np.array([
         1.0, -1.0, -1.0,  0.0,  0.0, -1.0,
        -1.0, -1.0, -1.0,  0.0,  0.0, -1.0,
         1.0,  1.0, -1.0,  0.0,  0.0, -1.0,
        -1.0,  1.0, -1.0,  0.0,  0.0, -1.0,
         1.0,  1.0, -1.0,  0.0,  0.0, -1.0,
        -1.0, -1.0, -1.0,  0.0,  0.0, -1.0,

        -1.0, -1.0,  1.0,  0.0,  0.0,  1.0,
         1.0, -1.0,  1.0,  0.0,  0.0,  1.0,
         1.0,  1.0,  1.0,  0.0,  0.0,  1.0,
         1.0,  1.0,  1.0,  0.0,  0.0,  1.0,
        -1.0,  1.0,  1.0,  0.0,  0.0,  1.0,
        -1.0, -1.0,  1.0,  0.0,  0.0,  1.0,

        -1.0,  1.0,  1.0, -1.0,  0.0,  0.0,
        -1.0,  1.0, -1.0, -1.0,  0.0,  0.0,
        -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,
        -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,
        -1.0, -1.0,  1.0, -1.0,  0.0,  0.0,
        -1.0,  1.0,  1.0, -1.0,  0.0,  0.0,

         1.0,  1.0, -1.0,  1.0,  0.0,  0.0,
         1.0,  1.0,  1.0,  1.0,  0.0,  0.0,
         1.0, -1.0, -1.0,  1.0,  0.0,  0.0,
         1.0, -1.0,  1.0,  1.0,  0.0,  0.0,
         1.0, -1.0, -1.0,  1.0,  0.0,  0.0,
         1.0,  1.0,  1.0,  1.0,  0.0,  0.0,

        -1.0, -1.0, -1.0,  0.0, -1.0,  0.0,
         1.0, -1.0, -1.0,  0.0, -1.0,  0.0,
         1.0, -1.0,  1.0,  0.0, -1.0,  0.0,
         1.0, -1.0,  1.0,  0.0, -1.0,  0.0,
        -1.0, -1.0,  1.0,  0.0, -1.0,  0.0,
        -1.0, -1.0, -1.0,  0.0, -1.0,  0.0,

        -1.0,  1.0, -1.0,  0.0,  1.0,  0.0,
         1.0,  1.0,  1.0,  0.0,  1.0,  0.0,
         1.0,  1.0, -1.0,  0.0,  1.0,  0.0,
        -1.0,  1.0,  1.0,  0.0,  1.0,  0.0,
         1.0,  1.0,  1.0,  0.0,  1.0,  0.0,
        -1.0,  1.0, -1.0,  0.0,  1.0,  0.0
        ], dtype = c_float)
    
    @classmethod
    def initGL(cls, shaderProgram):
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
        return 2 * 2 * self.scaleVec.x() * 2 * self.scaleVec.y() + 2 * 2 * self.scaleVec.x() * 2 * self.scaleVec.z() + 2 * 2 * self.scaleVec.y() * 2 * self.scaleVec.z()