#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include "pathstate.hpp"
#include "geometry.hpp"
#include "camera.hpp"
#include "shape.hpp"

__global__ void initRand(curandState *randState, int size);

__global__ void init(PathstateSoA pathstates, const int lenStates, int *pathRequests, int *raycastRequests, int *shadowRaycastRequests, int *diffuseMaterialRequests, int *specularReflectionRequests, int *specularTransmissionRequests);

__global__ void render(Camera camera, PathstateSoA pathstates, const int lenPathStates, int *pathRequests, int *numPathRequests, int *diffuseMaterialRequests, int *numDiffuseMaterialRequests, int *specularReflectionRequests, int *numSpecularReflectionRequests, int *specularTransmissionRequests, int *numSpecularTransmissionRequests, curandState *randState, const bool useLightSampling, const bool renderCaustics, const int maxBounce);

__global__ void newPath(int *pathRequests, PathstateSoA states, const int *numPathRequests, int *raycastRequests, Ray *rays, int *numRaycastRequests, Camera camera, curandState *randState);

__global__ void raycast(int *raycastRequests, Ray *rays, int *numRaycastRequests, PathstateSoA states, const int lenPathStates, const Shape *shapes, const int numShapes);

__global__ void shadowRaycast(int *raycastRequests, Ray *rays, int *numRaycastRequests, PathstateSoA states, const int lenPathStates, const Shape *shapes, const int numShapes);

__global__ void filterImage(const glm::vec3 *srcImage, unsigned int *destImage, const int widthFinal, const int heightFinal, const int stratificationLevel, const int numIterations, const float gamma);