# distutils: language = c++

from Renderer cimport Renderer
from libc.stdlib cimport malloc, free
import numpy as np

cdef class PyRenderer:
    cdef Renderer *c_renderer

    def __cinit__(self, int width, int height):
        self.c_renderer = new Renderer(width, height)
    
    def __dealloc__(self):
        del self.c_renderer
    
    def setResolution(self, width, height):
        self.c_renderer.setResolution(width, height)
    
    def start(self):
        self.c_renderer.start()
    
    def stop(self):
        self.c_renderer.stop()
    
    def updateCamera(self, worldPos, target, fovY, fStop, focusDistance, stratificationLevel):
        cdef float c_world[3]
        cdef float c_target[3]
        for i in range(3):
            c_world[i] = worldPos[i]
            c_target[i] = target[i]
        self.c_renderer.updateCamera(c_world, c_target, fovY, fStop, focusDistance, stratificationLevel)
    
    def addObject(self, id, materialId, sceneObjectType, matrixArray, lightIntensity, lightColor, area, isVisible):
        cdef float c_arr[16]
        cdef float c_color[3]
        for i in range(16):
            c_arr[i] = matrixArray[i]
        for i in range(3):
            c_color[i] = lightColor[i]
        self.c_renderer.addObject(id, materialId, sceneObjectType, c_arr, lightIntensity, c_color, area, isVisible)
    
    def addDiffuseMaterial(self, id, colorArray):
        cdef float c_arr[3]
        for i in range(3):
            c_arr[i] = colorArray[i]
        self.c_renderer.addDiffuseMaterial(id, c_arr)

    def addSpecularMaterial(self, id, reflectionColorArray, transmissionColorArray, IOR):
        cdef float r_arr[3]
        cdef float t_arr[3]
        for i in range(3):
            r_arr[i] = reflectionColorArray[i]
            t_arr[i] = transmissionColorArray[i]
        self.c_renderer.addSpecularMaterial(id, r_arr, t_arr, IOR)
    
    def clearObjects(self):
        self.c_renderer.clearObjects()
    
    def clearMaterials(self):
        self.c_renderer.clearMaterials()
    
    def getData(self, width, height):
        cdef i = 0, j = 0
        cdef unsigned int pixVal
        cdef unsigned int *data = <unsigned int *> malloc(width * height * sizeof(unsigned int))
        ret = np.ndarray(shape = (height, width), dtype = np.uint32)
        numIterations = self.c_renderer.getData(data, width, height)
        if numIterations < 0:
            free(data)
            return(ret, numIterations)

        cdef unsigned int[:, :] retView = ret

        for i in range(0, height):
            for j in range(0, width):
                ret[i,j] = data[i * width + j]

        free(data)
        return (ret, numIterations)
    
    def setRenderSettings(self, renderSettings):
        self.c_renderer.setRenderSettings(renderSettings.useLightSampling, renderSettings.renderCaustics, renderSettings.bounces)