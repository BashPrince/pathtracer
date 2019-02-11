from PySide2.QtGui import QMatrix4x4, QVector3D, QOpenGLBuffer, QOpenGLVertexArrayObject
from OpenGL import GL
import numpy as np
from ctypes import *
from PySide2.support import VoidPtr


class ViewportPlane():

    vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    vao = QOpenGLVertexArrayObject()

    vertices = np.array([
        -1.0, -1.0, 0.0, 0.0, 0.0,
        1.0, -1.0, 0.0, 1.0, 0.0,
        1.0, 1.0, 0.0, 1.0, 1.0,

        1.0, 1.0, 0.0, 1.0, 1.0,
        -1.0, 1.0, 0.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0, 0.0
        ], dtype = c_float)
    
    @classmethod
    def initGL(cls, shaderProgram):
        cls.vbo.create()
        cls.vao.create()
        cls.vbo.bind()
        cls.vbo.allocate(cls.vertices.tobytes(), cls.vertices.size * sizeof(c_float))

        cls.vao.bind()
        shaderProgram.setAttributeBuffer(0, GL.GL_FLOAT, 0, 3, 5 * sizeof(c_float))
        shaderProgram.setAttributeBuffer(1, GL.GL_FLOAT, 3 * sizeof(c_float), 2, 5 * sizeof(c_float))
        shaderProgram.enableAttributeArray(0)
        shaderProgram.enableAttributeArray(1)
        cls.vao.release()

    def __init__(self):
        pass

    def draw(self, shaderProgram, glFunctions):
        self.__class__.vao.bind()
        shaderProgram.bind()

        glFunctions.glDrawArrays(GL.GL_TRIANGLES, 0, self.__class__.vertices.size / 2)

        self.__class__.vao.release()
        shaderProgram.release()