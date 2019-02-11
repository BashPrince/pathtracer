#pragma once
#include <glm/glm.hpp>
#include <math.h>
#include "geometry.hpp"
#include "material.hpp"

enum class ShapeType {sphere = 0, cube, plane, disc};

class Shape {
public:
    __device__ __host__ Shape(const ShapeType shapeType, Material material, float area, const glm::mat4 &objectToWorld = glm::mat4(1.0), const glm::vec3 &Lemit = glm::vec3(0.0), const bool isVisible = true);
    __device__  bool Intersect(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit, bool firstRay) const;
    __device__  bool OccludesRay(const Ray &ray) const;
    __device__ float Sample(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const;
    const glm::mat4 objectToWorld, worldToObject;
    Material material;
    const ShapeType shapeType;
    glm::vec3 Lemit;
    float area;
    bool isVisible;

private:
    __device__ bool IntersectSphere(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const;
    __device__ bool IntersectPlane(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const;
    __device__ bool IntersectDisc(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const;
    __device__ bool IntersectCube(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const;
    __device__ bool SphereOccludesRay(const Ray &ray) const;
    __device__ bool PlaneOccludesRay(const Ray &ray) const;
    __device__ bool DiscOccludesRay(const Ray &ray) const;
    __device__ bool CubeOccludesRay(const Ray &ray) const;
    __device__ float SampleSphere(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const;
    __device__ float SamplePlane(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const;
    __device__ float SampleDisc(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const;
    __device__ float SampleCube(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const;
};