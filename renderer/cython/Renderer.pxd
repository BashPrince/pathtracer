cdef extern from "../cpp/src/core/Renderer.hpp":
    cdef cppclass Renderer:
            Renderer(int, int) except +
            void setResolution(int, int)
            void updateCamera(float[3], float[3], float, float, float, int)
            void start()
            void stop()
            void addObject(int, int, int, float[16], float, float[3], float, bint)
            void addDiffuseMaterial(int, float[3])
            void addSpecularMaterial(int, float[3], float[3], float)
            void clearObjects()
            void clearMaterials()
            void setRenderSettings(bint, bint, int)
            int getWidth()
            int getHeight()
            int getData(unsigned int*, int, int)