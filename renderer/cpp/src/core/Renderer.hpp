#include <cuda.h>
#include <curand_kernel.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "shape.hpp"
#include "camera.hpp"
#include "pathstate.hpp"

enum SceneObjectType {sphere = 0, cube, plane, disc, camera};

class Renderer
{
  public:
    Renderer(int width, int height);
    ~Renderer();
    void setResolution(int width, int height);
    void updateCamera(float worldPos[3], float target[3], float fov_y, float fStop, float focusDistance, int stratificationLevel);
    void addObject(int id, int materialId, int sceneObjectType, float arr[16], float lightIntensity, float lightColor[3], float area, bool isVisible);
    void addDiffuseMaterial(int id, float colorArray[3]);
    void addSpecularMaterial(int id, float reflectionColorArray[3], float transmissionColorArray[3], float IOR);
    void setRenderSettings(bool useLightSampling, bool renderCaustics, int numBounces);
    void clearObjects();
    void clearMaterials();
    void start();
    void stop();
    int getData(unsigned int *outImage, int width, int height);
    int getWidth();
    int getHeight();


private:
    void setup();
    void renderLoop();
    void clearImage();
    void copyShapes();
    void copyLights();
    void freeArrays();

    std::thread *renderThread;
    int width, height, r_width, r_height, iterationsDone, destIterationsDone, destImageWidth, destImageHeight, numBounces, r_numBounces;
    glm::vec3 *image;
    unsigned int *destImage;
    Camera camera;
    bool resetImage, useLightSampling, renderCaustics, r_useLightSampling, r_renderCaustics;
    Shape *shapes;
    Shape *lights;
    int shapeLen;
    int lightLen;
    Pathstate *pathstates;
    Ray *rays;
    Ray *shadowRays;
    int *pathRequests;
    int *raycastRequests;
    int *shadowRaycastRequests;
    int *diffuseMaterialRequests;
    int *specularReflectionRequests;
    int *specularTransmissionRequests;
    int *filmIndex;
    int *numPathRequests;
    int *numRaycastRequests;
    int *numShadowRaycastRequests;
    int *numDiffuseMaterialRequests;
    int *numSpecularReflectionRequests;
    int *numSpecularTransmissionRequests;
    std::vector<Shape> shapeVec;
    std::vector<Shape> lightVec;
    std::unordered_map<int, Material> materialMap;
    curandState *randState;
    PathstateSoA pathstateSoA;

    // Synchronisation
    std::atomic_bool stopRendering;
    std::atomic_bool resolutionChanged;
    std::atomic_bool sceneChanged;
    std::atomic_bool imageUpdated;
    std::atomic_bool cameraChanged;

    std::mutex destImageMutex;
    std::mutex cameraMutex;
    std::mutex sceneMutex;
    std::mutex resolutionMutex;
    std::mutex stopMutex;
};
