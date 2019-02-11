#pragma once
#include <glm/glm.hpp>
#include "geometry.hpp"
#include "material.hpp"

struct Pathstate
{
    Interaction currentIsect;
    Interaction extensionIsect;
    Material material;
    glm::vec3 beta;
    glm::vec3 L;
    glm::vec3 Lemit;
    glm::vec3 Lsample;
    glm::vec3 f;
    glm::vec3 light_f;
    glm::vec3 wi_L;
    bool foundExtensionIsect;
    bool isActive;
    bool LsampleOccluded;
    bool isLightSample;
    bool diffuseBounce;
    int bounces;
    int filmIndex;
    float pdf;
    float lightPdf;
};

struct PathstateSoA
{
    PathstateSoA();

    Interaction *currentIsect;
    Interaction *extensionIsect;
    bool *foundExtensionIsect;
    bool *isActive;
    float *pdf;
    float *lightPdf;
    int *bounces;
    glm::vec3 *beta;
    glm::vec3 *L;
    glm::vec3 *wi_L;
    glm::vec3 *Lemit;
    glm::vec3 *Lsample;
    bool *LsampleOccluded;
    bool *isLightSample;
    int *filmIndex;
    Material *material;
    glm::vec3 *f;
    glm::vec3 *light_f;
    bool *diffuseBounce;

    void freeArrays();
};