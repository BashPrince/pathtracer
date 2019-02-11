#pragma once

#include <glm/glm.hpp>

enum class Materialtype
{
    nullMaterial = 0,
    diffuse = 1,
    specular = 2
};

struct DiffuseMaterial
{
    glm::vec3 color;
};

struct SpecularMaterial
{
    glm::vec3 R, T;
    float etaA, etaB;
};

struct Material
{
    Materialtype materialType;
    union {
        DiffuseMaterial diffuse;
        SpecularMaterial specular;
    };
};