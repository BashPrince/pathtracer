#pragma once

#include <curand_kernel.h>
#include "pathstate.hpp"
#include "shape.hpp"

__global__ void diffuseKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, int *shadowRaycastRequests, Ray *shadowRays, int *numShadowRaycastRequests, const Shape *lights, const int numLights, curandState *randState, bool useLightSampling);
__global__ void specularReflectionKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, curandState *randState);
__global__ void specularTransmissionKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, curandState *randState);
