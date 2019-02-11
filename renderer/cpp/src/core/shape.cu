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

#include "shape.hpp"
#include "sampling.hpp"
#include <cmath>
#include "../util/util.hpp"

__device__ __host__
Shape::Shape(const ShapeType shapeType, Material material, float area, const glm::mat4 &objectToWorld, const glm::vec3 &Lemit, const bool isVisible)
    :shapeType(shapeType), material(material), area(area), objectToWorld(objectToWorld), worldToObject(glm::inverse(objectToWorld)), Lemit(Lemit), isVisible(isVisible)
{}

__device__
bool Shape::Intersect(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit, bool firstRay) const {
    if(!isVisible && firstRay)
        return false;
    switch(shapeType) {
        case ShapeType::sphere: return IntersectSphere(ray, isect, mat, Lemit);
        case ShapeType::plane: return IntersectPlane(ray, isect, mat, Lemit);
        case ShapeType::disc: return IntersectDisc(ray, isect, mat, Lemit);
        case ShapeType::cube: return IntersectCube(ray, isect, mat, Lemit);
        default: return false;
    }
}

__device__
bool Shape::OccludesRay(const Ray &ray) const {
    switch(shapeType) {
        case ShapeType::sphere: return SphereOccludesRay(ray);
        case ShapeType::plane: return PlaneOccludesRay(ray);
        case ShapeType::disc: return DiscOccludesRay(ray);
        case ShapeType::cube: return CubeOccludesRay(ray);
        default: return false;
    }
}

__device__
float Shape::Sample(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const {
    switch(shapeType) {
        case ShapeType::sphere: return SampleSphere(ray, Lsample, sample);
        case ShapeType::plane: return SamplePlane(ray, Lsample, sample);
        case ShapeType::disc: return SampleDisc(ray, Lsample, sample);
        case ShapeType::cube: return SampleCube(ray, Lsample, sample);
        default: return 0.0;
    }
}

__device__
bool Shape::IntersectSphere(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const {
    glm::vec3 pHit;
    Ray objRay = worldToObject * ray;

    float a = objRay.d.x * objRay.d.x + objRay.d.y * objRay.d.y + objRay.d.z * objRay.d.z;
    float b = 2 * (objRay.d.x * objRay.o.x + objRay.d.y * objRay.o.y + objRay.d.z * objRay.o.z);
    float c = objRay.o.x * objRay.o.x + objRay.o.y * objRay.o.y + objRay.o.z * objRay.o.z - 1.0;

    float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) {
        return false;
    }

    if(t0 > objRay.t || t1 <= 0.0) {
        return false;
    }
    float tShapeHit = t0;
    if(tShapeHit <= 0) {
        tShapeHit = t1;
        if(tShapeHit > objRay.t) {
            return false;
        }
    }

    pHit = objRay(tShapeHit);
    *isect = objectToWorld * Interaction(pHit, pHit, -objRay.d);
    *mat = material;
    *Lemit = this->Lemit;
    ray.t = tShapeHit;
    
    return true;
}

__device__
bool Shape::IntersectPlane(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const {
    glm::vec3 pHit;
    glm::vec3 pMin(-1.0, 0.0, -1.0);
    glm::vec3 pMax(1.0, 1.0, 1.0);
    glm::vec3 n;
    Ray objRay = worldToObject * ray;

    float t0 = 0.0, t1 = objRay.t;
    int indexNear = -1;
    int indexFar = -1;
    for(int i = 0; i < 3; ++i) {
        float invRayDir = 1.0 / objRay.d[i];
        float tNear = (pMin[i] - objRay.o[i]) * invRayDir;
        float tFar = (pMax[i] - objRay.o[i]) * invRayDir;

        if(tNear > tFar) {
            float temp = tNear;
            tNear = tFar;
            tFar = temp;
        }

        if(tNear > t0) {
            t0 = tNear;
            indexNear = i;
        }
        if(tFar < t1) {
            t1 = tFar;
            indexFar = i;
        }
        if(t0 > t1) return false;
    }

    n = glm::vec3(0.0, 1.0, 0.0);
    if(indexNear == 1 && indexFar == 1) {
        if(objRay(t0).y <= objRay(t1).y) {
            pHit = objRay(t0);
            ray.t = t0;
        } else {
            pHit = objRay(t1);
            ray.t = t1;
        }
    } else if(indexNear == 1 && glm::dot(n, objRay.d) >= 0.0) {
        pHit = objRay(t0);
        ray.t = t0;
    } else if(indexFar == 1 && glm::dot(n, objRay.d) < 0.0) {
        pHit = objRay(t1);
        ray.t = t1;
    } else {
        return false;
    }

    if(glm::dot(objRay.d, n) > 0.0)
        n *= -1.0;
    *isect = objectToWorld * Interaction(pHit, n, -objRay.d);
    *mat = material;
    *Lemit = this->Lemit;
    
    return true;
}

__device__
bool Shape::IntersectCube(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const {
    glm::vec3 pHit;
    glm::vec3 pMin(-1.0, -1.0, -1.0);
    glm::vec3 pMax(1.0, 1.0, 1.0);
    glm::vec3 n(0.0);
    Ray objRay = worldToObject * ray;


    float t0 = 0.0, t1 = objRay.t;
    int it0 = -1, it1 = -1;
    for(int i = 0; i < 3; ++i) {
        float invRayDir = 1.0 / objRay.d[i];
        float tNear = (pMin[i] - objRay.o[i]) * invRayDir;
        float tFar = (pMax[i] - objRay.o[i]) * invRayDir;

        if(tNear > tFar) {
            float temp = tNear;
            tNear = tFar;
            tFar = temp;
        }

        if(tNear > t0) {
            t0 = tNear;
            it0 = i;
        }
        if(tFar < t1) {
            t1 = tFar;
            it1 = i;
        }
        if(t0 > t1) return false;
    }

    if (it0 >= 0){
        pHit = objRay(t0);
        n[it0] = 1.0;
        ray.t = t0;
    }
    else if (t1 < objRay.t && t1 > 0.0)
    {
        pHit = objRay(t1);
        n[it1] = 1.0;
        ray.t = t1;
    }

    if(glm::dot(objRay.d, n) > 0.0)
        n *= -1.0;
    *isect = objectToWorld * Interaction(pHit, n, -objRay.d);
    *mat = material;
    *Lemit = this->Lemit;
    
    return true;
}

__device__
bool Shape::IntersectDisc(const Ray &ray, Interaction *isect, Material *mat, glm::vec3 *Lemit) const {
    glm::vec3 pHit;
    glm::vec3 n = glm::vec3(glm::transpose(glm::inverse(objectToWorld)) * glm::vec4(0.0, 1.0, 0.0, 0.0));
    glm::vec3 worldPos = objectToWorld * glm::vec4(0.0, 0.0, 0.0, 1.0);
    n = glm::normalize(n);
    float denom = glm::dot(n, ray.d);
    float t;

    if(abs(denom) > 1e-6) {
        glm::vec3 pOriginTorOrigin = worldPos - ray.o;
        t = glm::dot(pOriginTorOrigin, n) / denom;
    } else {
        return false;
    }
    if(t < 0.0 || t >= ray.t) {
        return false;
    }

    pHit = ray(t);
    glm::vec3 pHitObj = worldToObject * pHit;
    if(glm::dot(pHitObj, pHitObj) > 1.0) {
        return false;
    }
    *isect = Interaction(pHit, glm::dot(ray.d, n) >= 0.0 ? -n : n, -ray.d);
    *mat = material;
    *Lemit = this->Lemit;
    ray.t = t;
    
    return true;
}

__device__
bool Shape::SphereOccludesRay(const Ray &ray) const {
    Ray objRay = worldToObject * ray;

    float a = objRay.d.x * objRay.d.x + objRay.d.y * objRay.d.y + objRay.d.z * objRay.d.z;
    float b = 2 * (objRay.d.x * objRay.o.x + objRay.d.y * objRay.o.y + objRay.d.z * objRay.o.z);
    float c = objRay.o.x * objRay.o.x + objRay.o.y * objRay.o.y + objRay.o.z * objRay.o.z - 1.0;

    float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) {
        return false;
    }

    if(t0 > objRay.t || t1 <= 0.0) {
        return false;
    }
    float tShapeHit = t0;
    if(tShapeHit <= 0) {
        tShapeHit = t1;
        if(tShapeHit > objRay.t) {
            return false;
        }
    }
    
    return true;
}

__device__
bool Shape::PlaneOccludesRay(const Ray &ray) const {
    glm::vec3 n;
    glm::vec3 pMin(-1.0, 0.0, -1.0);
    glm::vec3 pMax(1.0, 1.0, 1.0);
    Ray objRay = worldToObject * ray;

    float t0 = 0.0, t1 = objRay.t;
    int indexNear = -1;
    int indexFar = -1;
    for(int i = 0; i < 3; ++i) {
        float invRayDir = 1.0 / objRay.d[i];
        float tNear = (pMin[i] - objRay.o[i]) * invRayDir;
        float tFar = (pMax[i] - objRay.o[i]) * invRayDir;

        if(tNear > tFar) {
            float temp = tNear;
            tNear = tFar;
            tFar = temp;
        }

        if(tNear > t0) {
            t0 = tNear;
            indexNear = i;
        }
        if(tFar < t1) {
            t1 = tFar;
            indexFar = i;
        }
        if(t0 > t1) return false;
    }

    n = glm::vec3(0.0, 1.0, 0.0);
    if(indexNear == 1 && indexFar == 1) {

    } else if(indexNear == 1 && glm::dot(n, objRay.d) >= 0.0) {

    } else if(indexFar == 1 && glm::dot(n, objRay.d) < 0.0) {

    } else {
        return false;
    }
    
    return true;
}

__device__
bool Shape::CubeOccludesRay(const Ray &ray) const {
    glm::vec3 pMin(-1.0, -1.0, -1.0);
    glm::vec3 pMax(1.0, 1.0, 1.0);
    Ray objRay = worldToObject * ray;

    float t0 = 0.0, t1 = objRay.t;
    for(int i = 0; i < 3; ++i) {
        float invRayDir = 1.0 / objRay.d[i];
        float tNear = (pMin[i] - objRay.o[i]) * invRayDir;
        float tFar = (pMax[i] - objRay.o[i]) * invRayDir;

        if(tNear > tFar) {
            float temp = tNear;
            tNear = tFar;
            tFar = temp;
        }

        if(tNear > t0) {
            t0 = tNear;
        }
        if(tFar < t1) {
            t1 = tFar;
        }
        if(t0 > t1) return false;
    }
    
    return true;
}

__device__
bool Shape::DiscOccludesRay(const Ray &ray) const {
    glm::vec3 pHit;
    glm::vec3 n = glm::vec3(glm::transpose(glm::inverse(objectToWorld)) * glm::vec4(0.0, 1.0, 0.0, 0.0));
    glm::vec3 worldPos = objectToWorld * glm::vec4(0.0, 0.0, 0.0, 1.0);
    n = glm::normalize(n);
    float denom = glm::dot(n, ray.d);
    float t;

    if(abs(denom) > 1e-6) {
        glm::vec3 pOriginTorOrigin = worldPos - ray.o;
        t = glm::dot(pOriginTorOrigin, n) / denom;
    } else {
        return false;
    }
    if(t < 0.0 || t >= ray.t) {
        return false;
    }

    pHit = ray(t);
    glm::vec3 pHitObj = worldToObject * pHit;
    if(glm::dot(pHitObj, pHitObj) > 1.0) {
        return false;
    }

    return true;
}

__device__
float Shape::SampleSphere(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const {
    glm::vec3 p = UniformSampleSphere(sample);
    glm::vec3 p_world = objectToWorld * p;
    glm::mat3 invTransp = glm::transpose(glm::inverse(glm::mat3(objectToWorld)));
    glm::vec3 n_world = glm::normalize(invTransp * p);
    ray.d = p_world - ray.o;
    ray.t = 0.9999f * glm::length(ray.d);
    ray.d = glm::normalize(ray.d);
    Lsample = Lemit;
    return ray.t * ray.t / (abs(glm::dot(-ray.d, n_world)) * area);
}

__device__
float Shape::SamplePlane(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const {
    glm:: vec3 n(0.0, 1.0, 0.0);
    glm::vec3 p = glm::vec3(-1.0, 0.0, -1.0) + 2.0f * glm::vec3(sample.x, 0.0, sample.y);
    glm::vec3 p_world = objectToWorld * p;
    glm::mat3 invTransp = glm::transpose(glm::inverse(glm::mat3(objectToWorld)));
    glm::vec3 n_world = glm::normalize(invTransp * n);
    ray.d = p_world - ray.o;
    ray.t = 0.9999f * glm::length(ray.d);
    ray.d = glm::normalize(ray.d);
    Lsample = Lemit;
    return ray.t * ray.t / (abs(glm::dot(-ray.d, n_world)) * area);
}

__device__
float Shape::SampleDisc(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const {
    glm:: vec3 n(0.0, 1.0, 0.0);
    glm::vec2 s = UniformSampleDisk(sample);
    glm::vec3 p = glm::vec3(s.x, 0.0, s.y);
    glm::vec3 p_world = objectToWorld * p;
    glm::mat3 invTransp = glm::transpose(glm::inverse(glm::mat3(objectToWorld)));
    glm::vec3 n_world = glm::normalize(invTransp * n);
    ray.d = p_world - ray.o;
    ray.t = 0.9999f * glm::length(ray.d);
    ray.d = glm::normalize(ray.d);
    Lsample = Lemit;
    return ray.t * ray.t / (abs(glm::dot(-ray.d, n_world)) * area);
}

__device__
float Shape::SampleCube(Ray &ray, glm::vec3 &Lsample, const glm::vec2 &sample) const {
    glm::vec2 s = sample;
    int i = min(int(3 * s.x), 2);
    glm::vec3 o(1.0, 1.0, 1.0);
    glm::vec3 axis1;
    glm::vec3 axis2;
    glm::vec3 n;
    if(i == 0) {
        axis1 = glm::vec3(-1.0, 0.0, 0.0);
        axis2 = glm::vec3(0.0, -1.0, 0.0);
        n = glm::vec3(0.0, 0.0, 1.0);
        s.x *= 3.0;
    }
    else if(i == 1) {
        axis1 = glm::vec3(0.0, 0.0, -1.0);
        axis2 = glm::vec3(0.0, -1.0, 0.0);
        n = glm::vec3(1.0, 0.0, 0.0);
        s.x = s.x * 3.0 - 1.0;
    }
    else if(i == 2) {
        axis1 = glm::vec3(-1.0, 0.0, 0.0);
        axis2 = glm::vec3(0.0, 0.0, -1.0);
        n = glm::vec3(0.0, 1.0, 0.0);
        s.x = s.x * 3.0 - 2.0;
    }
    if(s.y > 0.5) {
        o *= -1.0f;
        axis1 *= -1.0f;
        axis2 *= -1.0f;
        n *= -1.0f;
        s.y = s.y * 2.0 - 1.0;
    } else {
        s.y *= 2.0;
    }

    glm::vec3 p = o + 2.0f * s.x * axis1 + 2.0f * s.y * axis2;
    glm::vec3 p_world = objectToWorld * p;
    glm::mat3 invTransp = glm::transpose(glm::inverse(glm::mat3(objectToWorld)));
    glm::vec3 n_world = glm::normalize(invTransp * n);
    ray.d = p_world - ray.o;
    ray.t = 0.9999f * glm::length(ray.d);
    ray.d = glm::normalize(ray.d);
    Lsample = Lemit;
    return ray.t * ray.t / (abs(glm::dot(-ray.d, n_world)) * area);
}
