/*
    This source file makes use of modified code from pbrt-v3

    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "shading.hpp"
#include "sampling.hpp"
#include "../util/util.hpp"
#include <cmath>
#include <algorithm>

const float InvPi = 1.0 / M_PI;
__device__ float CosTheta(const glm::vec3 &v) { return v.z; }
__device__ float AbsCosTheta(const glm::vec3 &v) { return abs(v.z); }
__device__ bool SameHemisphere(const glm::vec3 &v1, const glm::vec3 &v2) { return v1.z * v2.z > 0; }

__device__
void NormalToCoordinateSystem(const glm::vec3 &n, glm::mat3 *localToWorld, glm::mat3 *worldToLocal) {
    glm::vec3 s, t;
    if (abs(n.x) > abs(n.y)) {
        s = glm::vec3(-n.z, 0, n.x) /
        sqrt(n.x * n.x + n.z * n.z);
    }
    else {
        s = glm::vec3(0, n.z, -n.y) /
        sqrt(n.y * n.y + n.z * n.z);
    }
    t = glm::cross(n, s);

    *localToWorld = glm::mat3(s, t, n);
    *worldToLocal = glm::inverse(*localToWorld);
}

__device__
float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(fmaxf((float)0, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    float cosThetaT = sqrt(fmaxf((float)0, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__device__
bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta, glm::vec3 *wt) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI = fmaxf(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

__device__
glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

__device__
glm::vec3 diffuse_f(const Material &material, const glm::vec3 &wo, const glm::vec3 &wi) {
    return material.diffuse.color * InvPi;
}

__device__
float diffuse_Pdf(const Material &material, const glm::vec3 &wo, const glm::vec3 &wi) {
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0.0;
}

__device__
glm::vec3 diffuse_Sample_f(const Material &material, const glm::vec3 &wo, glm::vec3 *wi, const glm::vec2 &sample, float *pdf) {
    *wi = CosineSampleHemisphere(sample);
    if (wo.z < 0) wi->z *= -1;
    *pdf = diffuse_Pdf(material, wo, *wi);
    return diffuse_f(material, wo, *wi);
}

__device__
glm::vec3 specularReflection_Sample_f(const Material &material, const glm::vec3 &wo, glm::vec3 *wi, float *pdf) {
    float F = FrDielectric(CosTheta(wo), material.specular.etaA, material.specular.etaB);
    // Compute specular reflection for _FresnelSpecular_

    // Compute perfect specular reflection direction
    *wi = glm::vec3(-wo.x, -wo.y, wo.z);
    *pdf = 0.5;
    return F * material.specular.R / AbsCosTheta(*wi);
}

__device__
glm::vec3 specularTransmission_Sample_f(const Material &material, const glm::vec3 &wo, glm::vec3 *wi, float *pdf) {
    float F = FrDielectric(CosTheta(wo), material.specular.etaA, material.specular.etaB);
    // Compute specular transmission for _FresnelSpecular_

    // Figure out which $\eta$ is incident and which is transmitted
    bool entering = CosTheta(wo) > 0;
    float etaI = entering ? material.specular.etaA : material.specular.etaB;
    float etaT = entering ? material.specular.etaB : material.specular.etaA;

    // Compute ray direction for specular transmission
    if (!Refract(wo, Faceforward(glm::vec3(0.0, 0.0, 1.0), wo), etaI / etaT, wi))
        return glm::vec3(0.0);
    glm::vec3 ft = material.specular.T * (1 - F);

    *pdf = 0.5;
    return ft / AbsCosTheta(*wi);
}

__global__
void diffuseKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, int *shadowRaycastRequests, Ray *shadowRays, int *numShadowRaycastRequests, const Shape *lights, const int numLights, curandState *randState, bool useLightSampling) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numMaterialRequests)
        return;
    
    int pathIdx = materialRequests[index];
    if(pathIdx >= 0) {
        Pathstate state;
        state.currentIsect = states.currentIsect[pathIdx];
        state.material = states.material[pathIdx];
        bool inside = glm::dot(state.currentIsect.n, state.currentIsect.wo) < 0.0f;

        glm::mat3 localToWorld, worldToLocal;
        Ray ray(state.currentIsect.p, glm::vec3(0.0));
        curandState localRandState = randState[pathIdx];
        glm::vec2 sample(curand_uniform(&localRandState), curand_uniform(&localRandState));
        NormalToCoordinateSystem(state.currentIsect.n, &localToWorld, &worldToLocal);

        state.f = diffuse_Sample_f(state.material, worldToLocal * state.currentIsect.wo, &ray.d, sample, &state.pdf);

        ray.d = glm::normalize(localToWorld * ray.d);
        ray.o = ray.o + 0.00005f * (inside ? -state.currentIsect.n : state.currentIsect.n);
        int reqIdx = atomicAdd(numRaycastRequests, 1);
        rays[reqIdx] = ray;
        raycastRequests[reqIdx] = pathIdx;

        // Light Sampling
        if(useLightSampling) {
            state.Lsample = glm::vec3(0.0);
            if (numLights > 0) {
                int lightIndex = min(int(numLights * curand_uniform(&localRandState)), numLights - 1);
                sample = glm::vec2(curand_uniform(&localRandState), curand_uniform(&localRandState));
                Ray shadowRay(ray.o, glm::vec3(0.0));
                state.lightPdf = lights[lightIndex].Sample(shadowRay, state.Lsample, sample);
                if(glm::dot(shadowRay.d, (inside ? -state.currentIsect.n : state.currentIsect.n)) > 0.0001) {
                    reqIdx = atomicAdd(numShadowRaycastRequests, 1);
                    shadowRays[reqIdx] = shadowRay;
                    shadowRaycastRequests[reqIdx] = pathIdx;
                    states.lightPdf[pathIdx] = float(1.0 / numLights) * state.lightPdf;
                    states.LsampleOccluded[pathIdx] = false;
                    states.light_f[pathIdx] = diffuse_f(state.material, worldToLocal * state.currentIsect.wo, worldToLocal * shadowRay.d);
                } else {
                    state.Lsample = glm::vec3(0.0);
                }
            }
            states.Lsample[pathIdx] = state.Lsample;
            states.isLightSample[pathIdx] = true;
        }
        states.pdf[pathIdx] = state.pdf;
        states.f[pathIdx] = state.f;
        materialRequests[index] = -1;
        randState[pathIdx] = localRandState;
    }
}

__global__
void specularReflectionKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, curandState *randState) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numMaterialRequests)
        return;
    
    int pathIdx = materialRequests[index];
    if(pathIdx >= 0) {
        Pathstate state;
        state.currentIsect = states.currentIsect[pathIdx];
        state.material = states.material[pathIdx];

        glm::mat3 localToWorld, worldToLocal;
        Ray ray(state.currentIsect.p, glm::vec3(0.0));
        NormalToCoordinateSystem(state.currentIsect.n, &localToWorld, &worldToLocal);

        state.f = specularReflection_Sample_f(state.material, worldToLocal * state.currentIsect.wo, &ray.d, &state.pdf);

        ray.d = glm::normalize(localToWorld * ray.d);
        ray.o = ray.o + 0.0001f * ray.d;
        int reqIdx = atomicAdd(numRaycastRequests, 1);
        rays[reqIdx] = ray;
        raycastRequests[reqIdx] = pathIdx;
        states.pdf[pathIdx] = state.pdf;
        states.f[pathIdx] = state.f;
        materialRequests[index] = -1;
    }
}

__global__
void specularTransmissionKernel(int *materialRequests, const int *numMaterialRequests, PathstateSoA states, const int lenPathStates, int *raycastRequests, Ray *rays, int *numRaycastRequests, curandState *randState) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= *numMaterialRequests)
        return;
    
    int pathIdx = materialRequests[index];
    if(pathIdx >= 0) {
        Pathstate state;
        state.currentIsect = states.currentIsect[pathIdx];
        state.material = states.material[pathIdx];

        glm::mat3 localToWorld, worldToLocal;
        Ray ray(state.currentIsect.p, glm::vec3(0.0));
        NormalToCoordinateSystem(state.currentIsect.n, &localToWorld, &worldToLocal);

        state.f = specularTransmission_Sample_f(state.material, worldToLocal * state.currentIsect.wo, &ray.d, &state.pdf);

        if(!isZeroVec(state.f)) {
            ray.d = glm::normalize(localToWorld * ray.d);
            ray.o = ray.o + 0.0001f * ray.d;
            int reqIdx = atomicAdd(numRaycastRequests, 1);
            rays[reqIdx] = ray;
            raycastRequests[reqIdx] = pathIdx;
        }
        states.pdf[pathIdx] = state.pdf;
        states.f[pathIdx] = state.f;
        materialRequests[index] = -1;
    }
}