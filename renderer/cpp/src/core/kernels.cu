#include "kernels.hpp"
#include "../util/util.hpp"

__global__ void initRand(curandState *randState, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= size)
        return;
    curand_init((254 << 20) + index, 0, 0, &randState[index]);
}

__global__
void init(PathstateSoA pathstates, const int lenStates, int *pathRequests, int *raycastRequests, int *shadowRaycastRequests, int *diffuseMaterialRequests, int *specularReflectionRequests, int *specularTransmissionRequests) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < lenStates) {
        pathstates.isActive[index] = true;
        pathstates.bounces[index] = -1;
        pathstates.foundExtensionIsect[index] = false;
        pathstates.diffuseBounce[index] = false;
        pathRequests[index] = -1;
        raycastRequests[index] = -1;
        shadowRaycastRequests[index] = -1;
        diffuseMaterialRequests[index] = -1;
        specularReflectionRequests[index] = -1;
        specularTransmissionRequests[index] = -1;
    }
}

__global__
void render(Camera camera, PathstateSoA pathstates, const int lenPathStates, int *pathRequests, int *numPathRequests, int *diffuseMaterialRequests, int *numDiffuseMaterialRequests, int *specularReflectionRequests, int *numSpecularReflectionRequests, int *specularTransmissionRequests, int *numSpecularTransmissionRequests, curandState *randState, const bool useLightSampling, const bool renderCaustics, const int maxBounce) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index >= lenPathStates)
        return;
    
    Pathstate state;
    state.currentIsect = pathstates.currentIsect[index];
    state.extensionIsect = pathstates.extensionIsect[index];
    state.foundExtensionIsect = pathstates.foundExtensionIsect[index];
    state.material = pathstates.material[index];
    state.isActive = pathstates.isActive[index];
    state.pdf = pathstates.pdf[index];
    state.bounces = pathstates.bounces[index];
    state.beta = pathstates.beta[index];
    state.L = pathstates.L[index];
    state.Lemit = pathstates.Lemit[index];
    state.filmIndex = pathstates.filmIndex[index];
    state.f = pathstates.f[index];
    state.isLightSample = pathstates.isLightSample[index];
    state.light_f = pathstates.light_f[index];
    state.Lsample = pathstates.Lsample[index];
    state.lightPdf = pathstates.lightPdf[index];
    state.LsampleOccluded = pathstates.LsampleOccluded[index];
    state.wi_L = pathstates.wi_L[index];
    state.diffuseBounce = pathstates.diffuseBounce[index];
    curandState localRandState = randState[index];


    if(!state.isActive)
        return;
    
    bool zeroBeta = glm::dot(state.beta, state.beta) < 0.00001;
    bool zero_f = isZeroVec(state.f);
    bool hitLight = !isZeroVec(state.Lemit);
    bool endPath = false;
    bool causticStop = !renderCaustics && state.diffuseBounce && state.material.materialType == Materialtype::specular;
    if(state.bounces == -1 || state.bounces >= maxBounce || zeroBeta || !state.foundExtensionIsect || zero_f || hitLight || causticStop) {
        endPath = true;

        int reqIdx = atomicAdd(numPathRequests, 1);
        pathRequests[reqIdx] = index;
    }

    if(useLightSampling && state.isLightSample && !isZeroVec(state.Lsample) && !state.LsampleOccluded) {
        state.L += state.Lsample * state.beta * state.light_f * abs(glm::dot(state.wi_L, state.currentIsect.n)) / state.lightPdf;
    }

    if(state.foundExtensionIsect)
    {
        if(state.bounces == 0) {
            state.L += state.Lemit;
        }
        else {
            state.beta *= state.f * abs(glm::dot(-state.extensionIsect.wo, state.currentIsect.n)) / state.pdf;
            if(!useLightSampling || !state.isLightSample)
                state.L += state.Lemit * state.beta;
        }
        ++state.bounces;
        if(!endPath) {
            int reqIdx;
            if(state.material.materialType == Materialtype::diffuse) {
                reqIdx = atomicAdd(numDiffuseMaterialRequests, 1);
                diffuseMaterialRequests[reqIdx] = index;
                state.diffuseBounce = true;
            }
            else if(state.material.materialType == Materialtype::specular) {
                if(curand_uniform(&localRandState) > 0.5f) {
                    reqIdx = atomicAdd(numSpecularReflectionRequests, 1);
                    specularReflectionRequests[reqIdx] = index;
                } else {
                    reqIdx = atomicAdd(numSpecularTransmissionRequests, 1);
                    specularTransmissionRequests[reqIdx] = index;
                }
                state.diffuseBounce = false;
            }
        }
    }

    if(endPath && state.bounces != -1) {
        if(!state.foundExtensionIsect && state.bounces == 0) {
            camera.addPixelVal(state.filmIndex, glm::vec3(0.015));
        } else {
            #ifdef CudaDebug
                bool error = false;
                if(state.pdf == 0 && state.bounces - 1 > 0) {
                    camera.addPixelVal(state.filmIndex, glm::vec3(0.0, 0.0, 1.0));
                    error = true;
                }
                if(isnan(state.extensionIsect.wo.x) || isnan(state.extensionIsect.wo.y) || isnan(state.extensionIsect.wo.z))
                {
                    camera.addPixelVal(state.filmIndex, glm::vec3(1.0, 0.0, 0.0));
                    error = true;
                }
                if(!error)
                    camera.addPixelVal(state.filmIndex, state.L);
            #else
                camera.addPixelVal(state.filmIndex, state.L);
            #endif            
        }
        state.beta = glm::vec3(0.0);
    }

    pathstates.currentIsect[index] = state.extensionIsect;
    pathstates.foundExtensionIsect[index] = false;
    pathstates.bounces[index] = state.bounces;
    pathstates.beta[index] = state.beta;
    pathstates.L[index] = state.L;
    pathstates.diffuseBounce[index] = state.diffuseBounce;
    pathstates.isLightSample[index] = false;
    randState[index] = localRandState;
}

__global__
void newPath(int *pathRequests, PathstateSoA states, const int *numPathRequests, int *raycastRequests, Ray *rays, int *numRaycastRequests, Camera camera, curandState *randState) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numPathRequests)
        return;
    
    int pathIdx = pathRequests[index];
    if(pathIdx >= 0) {

        int filmIndex = camera.getFilmIndex();
        if(filmIndex < 0) {
            states.isActive[pathIdx] = false;
            return;
        }

        Pathstate state;
        state.bounces = 0;
        state.foundExtensionIsect = false;
        state.isActive = true;
        state.beta = glm::vec3(1.0);
        state.f = glm::vec3(1.0);
        state.L = glm::vec3(0.0);
        state.Lemit = glm::vec3(0.0);
        state.filmIndex = filmIndex;
        state.isLightSample = false;
        state.diffuseBounce = false;
        Ray ray;
        curandState localRandState = randState[pathIdx];
        camera.getRay(filmIndex, &ray,
                        glm::vec2(curand_uniform(&localRandState), curand_uniform(&localRandState)),
                        glm::vec2(curand_uniform(&localRandState), curand_uniform(&localRandState)));
        randState[pathIdx] = localRandState;
        int reqIdx = atomicAdd(numRaycastRequests, 1);
        rays[reqIdx] = ray;
        raycastRequests[reqIdx] = pathIdx;
        pathRequests[index] = -1;
        states.foundExtensionIsect[pathIdx] = state.foundExtensionIsect;
        states.isActive[pathIdx] = state.isActive;
        states.bounces[pathIdx] = state.bounces;
        states.beta[pathIdx] = state.beta;
        states.L[pathIdx] = state.L;
        states.Lemit[pathIdx] = state.Lemit;
        states.filmIndex[pathIdx] = state.filmIndex;
        states.f[pathIdx] = state.f;
        states.isLightSample[pathIdx] = state.isLightSample;
        states.diffuseBounce[pathIdx] = state.diffuseBounce;
    }
}

__global__
void raycast(int *raycastRequests, Ray *rays, int *numRaycastRequests, PathstateSoA states, const int lenPathStates, const Shape *shapes, const int numShapes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numRaycastRequests)
        return;
    
    int pathIdx = raycastRequests[index];
    if(pathIdx >= 0) {
        Ray ray = rays[index];
        Pathstate state;
        state.bounces = states.bounces[pathIdx];
        Interaction isect;
        Material material;
        glm::vec3 Lemit;
        bool hit = false;
        for(int i = 0; i < numShapes; ++i) {
            if(shapes[i].Intersect(ray, &isect, &material, &Lemit, state.bounces == 0)) {
                hit = true;
            }
        }
        if(hit) {
            isect.n = glm::normalize(isect.n);
            isect.wo = glm::normalize(isect.wo);
            state.extensionIsect = isect;
            state.foundExtensionIsect = true;
            state.material = material;
            state.Lemit = Lemit;

            states.extensionIsect[pathIdx] = state.extensionIsect;
            states.foundExtensionIsect[pathIdx] = state.foundExtensionIsect;
            states.Lemit[pathIdx] = state.Lemit;
            states.material[pathIdx] = state.material;
        }
        raycastRequests[index] = -1;
    }
}

__global__
void shadowRaycast(int *raycastRequests, Ray *rays, int *numRaycastRequests, PathstateSoA states, const int lenPathStates, const Shape *shapes, const int numShapes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numRaycastRequests)
        return;
    
    int pathIdx = raycastRequests[index];
    if(pathIdx >= 0) {
        Ray ray = rays[index];
        bool hit = false;
        for(int i = 0; i < numShapes; ++i) {
            if(shapes[i].OccludesRay(ray)) {
                hit = true;
            }
        }

        if(hit) {
            states.LsampleOccluded[pathIdx] = true;
        } else {
            states.wi_L[pathIdx] = ray.d;
        }
        raycastRequests[index] = -1;
    }
}

__global__
void filterImage(const glm::vec3 *srcImage, unsigned int *destImage, const int widthFinal, const int heightFinal, const int stratificationLevel, const int numIterations, const float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * widthFinal + x;

    if(index >= widthFinal * heightFinal)
        return;
    
    glm::vec3 val(0.0);
    float weight = 1.0 / numIterations;
    for(int j = 0; j < stratificationLevel; ++j) {
        for(int i = 0; i < stratificationLevel; ++i) {
            int srcX = stratificationLevel * x + j;
            int srcY = stratificationLevel * y + i;
            int srcIndex = srcY * widthFinal * stratificationLevel + srcX;
            val += srcImage[srcIndex] * weight;
        }
    }
    val /= stratificationLevel * stratificationLevel;
    val = pow(val, gamma);
    val *= 255;
    unsigned int pixR = lrintf(val.r);
    unsigned int pixG = lrintf(val.g);
    unsigned int pixB = lrintf(val.b);
    pixR = pixR > 255 ? 255 : pixR;
    pixG = pixG > 255 ? 255 : pixG;
    pixB = pixB > 255 ? 255 : pixB;

    destImage[index] = (pixR << 16) + (pixG << 8) + pixB;
}