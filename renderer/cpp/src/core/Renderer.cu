#include "Renderer.hpp"
#include "kernels.hpp"
#include "shading.hpp"
#include "../util/util.hpp"
#include <iostream>
#include "math.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstring>

const int queueSize = 200000;
const int blockDimensionX = 512;
const float gamma22 = 2.2;

Renderer::Renderer(int width, int height)
:   width(width),
    height(height),
    r_width(width),
    r_height(height),
    iterationsDone(0),
    destIterationsDone(0),
    numBounces(30),
    r_numBounces(30),
    image(nullptr),
    destImage(nullptr),
    randState(nullptr),
    pathstates(nullptr),
    rays(nullptr),
    shadowRays(nullptr),
    pathRequests(nullptr),
    raycastRequests(nullptr),
    shadowRaycastRequests(nullptr),
    diffuseMaterialRequests(nullptr),
    specularReflectionRequests(nullptr),
    specularTransmissionRequests(nullptr),
    filmIndex(nullptr),
    numPathRequests(nullptr),
    numRaycastRequests(nullptr),
    numShadowRaycastRequests(nullptr),
    numDiffuseMaterialRequests(nullptr),
    numSpecularReflectionRequests(nullptr),
    numSpecularTransmissionRequests(nullptr),
    shapes(nullptr),
    lights(nullptr),
    shapeLen(0),
    lightLen(0),
    camera(width, height, glm::vec3(0.0, 0.0, 10.0), glm::vec3(0.0)),
    stopRendering(false),
    sceneChanged(false),
    resolutionChanged(false),
    imageUpdated(false),
    cameraChanged(false),
    renderThread(nullptr),
    useLightSampling(true),
    renderCaustics(true),
    r_useLightSampling(true),
    r_renderCaustics(true)
{
    setup();
}

void Renderer::start() {
    stopRendering = false;
    renderThread = new std::thread(&Renderer::renderLoop, this);
}

Renderer::~Renderer() {
    stop();
    freeArrays();
}

int Renderer::getWidth() {
    return width;
}

int Renderer::getHeight() {
    return height;
}

void Renderer::copyShapes() {
    cudaMallocManaged(&shapes, shapeVec.size() * sizeof(Shape));
    waitAndCheckError("setup::cudaMallocManaged(shapes)");

    std::memcpy(shapes, shapeVec.data(), shapeVec.size() * sizeof(Shape));
}

void Renderer::copyLights() {
    cudaMallocManaged(&lights, lightVec.size() * sizeof(Shape));
    waitAndCheckError("setup::cudaMallocManaged(lights)");

    std::memcpy(lights, lightVec.data(), lightVec.size() * sizeof(Shape));
}

void Renderer::setup() {
    freeArrays();

    r_width = width;
    r_height = height;
    r_useLightSampling = useLightSampling;
    r_renderCaustics = renderCaustics;
    r_numBounces = numBounces;
    camera.update(r_width, r_height);
    
    int imgLen = r_width * r_height;
    int imgLenStratified = imgLen * camera.getStratificationLevel() * camera.getStratificationLevel();

    
    cudaMallocManaged(&destImage, imgLen * sizeof(float));
    waitAndCheckError("setup::cudaMallocManaged(destImage)");
    cudaMallocManaged(&image, imgLenStratified * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(image)");
    cudaMallocManaged(&randState, queueSize * sizeof(curandState));
    waitAndCheckError("setup::cudaMallocManaged(randState)");
    if(shapeVec.size() > 0) {
        copyShapes();
    }
    if(lightVec.size() > 0) {
        copyLights();
    }
    shapeLen = shapeVec.size();
    lightLen = lightVec.size();
    cudaMallocManaged(&pathstates, queueSize * sizeof(Pathstate));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&rays, queueSize * sizeof(Ray));
    waitAndCheckError("setup::cudaMallocManaged(rays)");
    cudaMallocManaged(&shadowRays, queueSize * sizeof(Ray));
    waitAndCheckError("setup::cudaMallocManaged(shadowRays)");
    cudaMallocManaged(&pathRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(pathRequests)");
    cudaMallocManaged(&raycastRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(raycastRequests)");
    cudaMallocManaged(&shadowRaycastRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(shadowRaycastRequests)");
    cudaMallocManaged(&diffuseMaterialRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(diffuseMaterialRequests)");
    cudaMallocManaged(&specularReflectionRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(specularReflectionRequests)");
    cudaMallocManaged(&specularTransmissionRequests, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(specularTransmissionRequests)");
    cudaMallocManaged(&filmIndex, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(filmIndex)");
    cudaMallocManaged(&numRaycastRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numRaycastRequests)");
    cudaMallocManaged(&numShadowRaycastRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numShadowRaycastRequests)");
    cudaMallocManaged(&numPathRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numPathRequests)");
    cudaMallocManaged(&numDiffuseMaterialRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numDiffuseMaterialRequests)");
    cudaMallocManaged(&numSpecularReflectionRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numSpecularReflectionRequests)");
    cudaMallocManaged(&numSpecularTransmissionRequests, sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(numSpecularTransmissionRequests)");
    cudaMemset(image, 0, imgLenStratified * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(image)");
    cudaMemset(destImage, 0, imgLen * sizeof(unsigned int));
    waitAndCheckError("setup::cudaMallocManaged(destImage)");


    cudaMallocManaged(&pathstateSoA.currentIsect, queueSize * sizeof(Interaction));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.extensionIsect, queueSize * sizeof(Interaction));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.foundExtensionIsect, queueSize * sizeof(bool));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.isActive, queueSize * sizeof(bool));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.pdf, queueSize * sizeof(float));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.lightPdf, queueSize * sizeof(float));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.bounces, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.beta, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.L, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.wi_L, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.Lemit, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.filmIndex, queueSize * sizeof(int));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.material, queueSize * sizeof(Material));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.f, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.light_f, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.Lsample, queueSize * sizeof(glm::vec3));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.LsampleOccluded, queueSize * sizeof(bool));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.isLightSample, queueSize * sizeof(bool));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");
    cudaMallocManaged(&pathstateSoA.diffuseBounce, queueSize * sizeof(bool));
    waitAndCheckError("setup::cudaMallocManaged(pathstates)");

    
    iterationsDone = 0;
    destIterationsDone = 0;
    *filmIndex = 0;
    *numPathRequests = 0;
    *numRaycastRequests = 0;
    *numShadowRaycastRequests = 0;
    *numDiffuseMaterialRequests = 0;
    *numSpecularReflectionRequests = 0;
    *numSpecularTransmissionRequests = 0;
    camera.setImage(image);
    camera.setFilmIndexPointer(filmIndex);
    
    dim3 blockDim(blockDimensionX);
    dim3 gridDim(queueSize / blockDim.x + (queueSize % blockDim.x ? 1 : 0));
    
    initRand<<<gridDim, blockDim>>>(randState, queueSize);
    waitAndCheckError("setup::initRand<<<>>>");
    
    resolutionChanged = false;
    sceneChanged = false;
}

void Renderer::clearImage() {
    cudaMemset(image, 0, r_width * r_height * camera.getStratificationLevel() * camera.getStratificationLevel() * sizeof(glm::vec3));
    waitAndCheckError("clearImage::cudaMemset");
    iterationsDone = 0;
    destIterationsDone = 0;    
}

void Renderer::renderLoop() {
    while(!stopRendering) {
        // if resolutionChanged bool lock mutex and copy resolution -> setup
        cameraMutex.lock();
        resolutionMutex.lock();
        if(resolutionChanged) {
            destImageMutex.lock();
            sceneMutex.lock();
            setup();
            destImageMutex.unlock();
            sceneMutex.unlock();
        }
        // if cameraChanged bool lock mutex and copy camera
        Camera renderCamera = camera;
        renderCamera.resetFilmIndex();
        if(cameraChanged) {
            clearImage();
            cameraChanged = false;
        }
        cameraMutex.unlock();
        resolutionMutex.unlock();

        // if sceneChanged bool lock mutex and copy scene -> setup
        if(sceneChanged) {
            std::lock_guard<std::mutex> sceneLock(sceneMutex);
            if(shapes) {
                cudaFree(shapes);
                waitAndCheckError("start::cudaFree(shapes)");
                shapes = nullptr;
            }
            if(shapeVec.size() > 0) {
                copyShapes();
            }
            if(lights) {
                cudaFree(lights);
                waitAndCheckError("start::cudaFree(lights)");
                lights = nullptr;
            }
            if(lightVec.size() > 0) {
                copyLights();
            }
            shapeLen = shapeVec.size();
            lightLen = lightVec.size();
            clearImage();
            r_useLightSampling = useLightSampling;
            r_renderCaustics = renderCaustics;
            r_numBounces = numBounces;
            sceneChanged = false;
        }

        dim3 blockDim(blockDimensionX);
        dim3 gridDim(queueSize / blockDim.x + (queueSize % blockDim.x ? 1 : 0));

        init<<<gridDim, blockDim>>>(pathstateSoA, queueSize, pathRequests, raycastRequests, shadowRaycastRequests, diffuseMaterialRequests, specularReflectionRequests, specularTransmissionRequests);
        waitAndCheckError("renderLoop::init<<<>>>");
        do {
            *numPathRequests = 0;
            *numRaycastRequests = 0;
            *numShadowRaycastRequests = 0;
            *numDiffuseMaterialRequests = 0;
            *numSpecularReflectionRequests = 0;
            *numSpecularTransmissionRequests = 0;
            render<<<gridDim, blockDim>>>(renderCamera, pathstateSoA, queueSize, pathRequests, numPathRequests, diffuseMaterialRequests, numDiffuseMaterialRequests, specularReflectionRequests, numSpecularReflectionRequests, specularTransmissionRequests, numSpecularTransmissionRequests, randState, r_useLightSampling, r_renderCaustics, r_numBounces);
            waitAndCheckError("renderLoop::render<<<>>>(loop)");
            newPath<<<gridDim, blockDim>>>(pathRequests, pathstateSoA, numPathRequests, raycastRequests, rays, numRaycastRequests, renderCamera, randState);
            waitAndCheckError("renderLoop::newPath<<<>>>");
            diffuseKernel<<<gridDim, blockDim>>>(diffuseMaterialRequests, numDiffuseMaterialRequests, pathstateSoA, queueSize, raycastRequests, rays, numRaycastRequests, shadowRaycastRequests, shadowRays, numShadowRaycastRequests, lights, lightLen, randState, r_useLightSampling);
            waitAndCheckError("renderLoop::diffuseKernel<<<>>>");
            specularReflectionKernel<<<gridDim, blockDim>>>(specularReflectionRequests, numSpecularReflectionRequests, pathstateSoA, queueSize, raycastRequests, rays, numRaycastRequests, randState);
            waitAndCheckError("renderLoop::specularReflectionKernel<<<>>>");
            specularTransmissionKernel<<<gridDim, blockDim>>>(specularTransmissionRequests, numSpecularTransmissionRequests, pathstateSoA, queueSize, raycastRequests, rays, numRaycastRequests, randState);
            waitAndCheckError("renderLoop::specularTransmissionKernel<<<>>>");
            raycast<<<gridDim, blockDim>>>(raycastRequests, rays, numRaycastRequests, pathstateSoA, queueSize, shapes, shapeLen);
            waitAndCheckError("renderLoop::raycast<<<>>>");
            shadowRaycast<<<gridDim, blockDim>>>(shadowRaycastRequests, shadowRays, numShadowRaycastRequests, pathstateSoA, queueSize, shapes, shapeLen);
            waitAndCheckError("renderLoop::shadowRaycast<<<>>>");

            cudaDeviceSynchronize();
        } while(!(*numRaycastRequests == 0 && *numShadowRaycastRequests == 0 && *numDiffuseMaterialRequests == 0 && *numSpecularReflectionRequests == 0 && *numSpecularTransmissionRequests == 0));

        ++iterationsDone;
        if(destImageMutex.try_lock()) {
            dim3 filterBlockDim(32, 32, 1);
            dim3 filterGridDim(r_width / filterBlockDim.x + (r_width % filterBlockDim.x ? 1 : 0), r_height / filterBlockDim.y + (r_height % filterBlockDim.y ? 1 : 0));
            filterImage<<<filterGridDim, filterBlockDim>>>(image, destImage, r_width, r_height, renderCamera.getStratificationLevel(), iterationsDone, 1 / gamma22);
            waitAndCheckError("renderLoop::filterImage<<<>>>");
            imageUpdated = true;
            destImageWidth = r_width;
            destImageHeight = r_height;
            destIterationsDone = iterationsDone;
            cudaDeviceSynchronize(); // need to wait for kernel before unlocking mutex
            destImageMutex.unlock();
        }
    }
}

int Renderer::getData(unsigned int *outImage, int width, int height) {
    if(!imageUpdated) {
        return -1;
    }

    std::lock_guard<std::mutex> destImageLock(destImageMutex);
    imageUpdated = false;

    if(width != destImageWidth || height != destImageHeight || destImage == nullptr) {
        return -1;
    }
    std::memcpy(outImage, destImage, width * height * sizeof(unsigned int));
    return destIterationsDone;
}

void Renderer::stop() {
    if(!stopRendering) {
        stopRendering = true;
        renderThread->join();
        delete renderThread;
        renderThread = nullptr;
    }
}

void Renderer::addObject(int id, int materialId, int sceneObjectType, float arr[16], float lightIntensity, float lightColor[3], float area, bool isVisible) {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    glm::mat4 objectToWorld = glm::make_mat4(arr);
    glm::vec3 color = glm::make_vec3(lightColor);
    bool isLight  = !isZeroVec(color) && lightIntensity != 0;
    switch(sceneObjectType) {
        case SceneObjectType::sphere:
        shapeVec.push_back(Shape(ShapeType::sphere, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22), isVisible));
        if(isLight)
            lightVec.push_back(Shape(ShapeType::sphere, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22),isVisible));
        break;
        case SceneObjectType::cube:
        shapeVec.push_back(Shape(ShapeType::cube, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22), isVisible));
        if(isLight)
            lightVec.push_back(Shape(ShapeType::cube, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22),isVisible));
        break;
        case SceneObjectType::plane:
        shapeVec.push_back(Shape(ShapeType::plane, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22), isVisible));
        if(isLight)
            lightVec.push_back(Shape(ShapeType::plane, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22),isVisible));
        break;
        case SceneObjectType::disc:
        shapeVec.push_back(Shape(ShapeType::disc, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22), isVisible));
        if(isLight)
            lightVec.push_back(Shape(ShapeType::disc, materialMap[materialId], area, objectToWorld, lightIntensity * pow(color, 1.0 / gamma22),isVisible));
        break;
        default:
        return;
    }
    sceneChanged = true;
}

void Renderer::addDiffuseMaterial(int id, float colorArray[3]) {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    glm::vec3 color = glm::make_vec3(colorArray);
    color = pow(color, gamma22);
    Material m;
    m.materialType = Materialtype::diffuse;
    m.diffuse = DiffuseMaterial{color};
    materialMap.insert(std::pair<int, Material>(id, m));
}

void Renderer::addSpecularMaterial(int id, float reflectionColorArray[3], float transmissionColorArray[3], float IOR) {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    glm::vec3 reflectionColor = glm::make_vec3(reflectionColorArray);
    glm::vec3 transmissionColor = glm::make_vec3(transmissionColorArray);
    reflectionColor = pow(reflectionColor, gamma22);
    transmissionColor = pow(transmissionColor, gamma22);
    Material m;
    m.materialType = Materialtype::specular;
    m.specular = SpecularMaterial{reflectionColor, transmissionColor, 1.0, IOR};
    materialMap.insert(std::pair<int, Material>(id, m));
}

void Renderer::clearObjects() {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    shapeVec.clear();
    lightVec.clear();
    sceneChanged = true;
}

void Renderer::clearMaterials() {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    materialMap.clear();
    sceneChanged = true;
}

void Renderer::setResolution(int width, int height) {
    std::lock_guard<std::mutex> resolutionLock(resolutionMutex);
    this->width = width;
    this->height = height;
    resolutionChanged = true;
}

void Renderer::updateCamera(float worldPos[3], float target[3], float fov_y, float fStop, float focusDistance, int stratificationLevel) {
    std::lock_guard<std::mutex> cameraLock(cameraMutex);
    std::lock_guard<std::mutex> resolutionLock(resolutionMutex);
    glm::vec3 worldVec = glm::make_vec3(worldPos);
    glm::vec3 targetVec = glm::make_vec3(target);
    if(stratificationLevel != camera.getStratificationLevel())
        resolutionChanged = true;
    camera.update(worldVec, targetVec, fov_y, fStop, focusDistance, stratificationLevel);
    cameraChanged = true;
}

void Renderer::freeArrays() {
    if(image) {
        cudaFree(image);
        waitAndCheckError("setup::cudaFree(image)");
        image = nullptr;
    }
    if(destImage) {
        cudaFree(destImage);
        waitAndCheckError("setup::cudaFree(destImage)");
        destImage = nullptr;
    }
    if(randState) {
        cudaFree(randState);
        waitAndCheckError("setup::cudaFree(randstate)");
        randState = nullptr;
    }
    if(shapes) {
        cudaFree(shapes);
        waitAndCheckError("setup::cudaFree(shapes)");
        shapes = nullptr;
    }
    if(pathstates) {
        cudaFree(pathstates);
        waitAndCheckError("setup::cudaFree(pathstates)");
        pathstates = nullptr;
    }
    if(rays) {
        cudaFree(rays);
        waitAndCheckError("setup::cudaFree(rays)");
        rays = nullptr;
    }
    if(shadowRays) {
        cudaFree(shadowRays);
        waitAndCheckError("setup::cudaFree(shadowRays)");
        shadowRays = nullptr;
    }
    if(pathRequests) {
        cudaFree(pathRequests);
        waitAndCheckError("setup::cudaFree(pathRequests)");
        pathRequests = nullptr;
    }
    if(filmIndex) {
        cudaFree(filmIndex);
        waitAndCheckError("setup::cudaFree(filmIndex)");
        filmIndex = nullptr;
    }
    if(raycastRequests) {
        cudaFree(raycastRequests);
        waitAndCheckError("setup::cudaFree(raycastRequests)");
        raycastRequests = nullptr;
    }
    if(numRaycastRequests) {
        cudaFree(numRaycastRequests);
        waitAndCheckError("setup::cudaFree(numRaycastRequests)");
        numRaycastRequests = nullptr;
    }
    if(shadowRaycastRequests) {
        cudaFree(shadowRaycastRequests);
        waitAndCheckError("setup::cudaFree(shadowRaycastRequests)");
        shadowRaycastRequests = nullptr;
    }
    if(numShadowRaycastRequests) {
        cudaFree(numShadowRaycastRequests);
        waitAndCheckError("setup::cudaFree(numShadowRaycastRequests)");
        numShadowRaycastRequests = nullptr;
    }
    if(numPathRequests) {
        cudaFree(numPathRequests);
        waitAndCheckError("setup::cudaFree(numPathRequests)");
        numPathRequests = nullptr;
    }
    if(diffuseMaterialRequests) {
        cudaFree(diffuseMaterialRequests);
        waitAndCheckError("setup::cudaFree(diffuseMaterialRequests)");
        diffuseMaterialRequests = nullptr;
    }
    if(numDiffuseMaterialRequests) {
        cudaFree(numDiffuseMaterialRequests);
        waitAndCheckError("setup::cudaFree(numDiffuseMaterialRequests)");
        numDiffuseMaterialRequests = nullptr;
    }
    if(specularReflectionRequests) {
        cudaFree(specularReflectionRequests);
        waitAndCheckError("setup::cudaFree(specularReflectionRequests)");
        specularReflectionRequests = nullptr;
    }
    if(specularTransmissionRequests) {
        cudaFree(specularTransmissionRequests);
        waitAndCheckError("setup::cudaFree(specularTransmissionRequests)");
        specularTransmissionRequests = nullptr;
    }
    if(numSpecularReflectionRequests) {
        cudaFree(numSpecularReflectionRequests);
        waitAndCheckError("setup::cudaFree(numSpecularReflectionRequests)");
        numSpecularReflectionRequests = nullptr;
    }
    if(numSpecularTransmissionRequests) {
        cudaFree(numSpecularTransmissionRequests);
        waitAndCheckError("setup::cudaFree(numSpecularTransmissionRequests)");
        numSpecularTransmissionRequests = nullptr;
    }

    pathstateSoA.freeArrays();
}

void Renderer::setRenderSettings(bool useLightSampling, bool renderCaustics, int numBounces) {
    std::lock_guard<std::mutex> sceneLock(sceneMutex);
    this->useLightSampling = useLightSampling;
    this->renderCaustics = renderCaustics;
    this->numBounces = numBounces;
    sceneChanged = true;
}