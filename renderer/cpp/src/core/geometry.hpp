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

#pragma once
#include <glm/glm.hpp>
#include <math.h>

struct Ray;
struct Interaction;

__device__ __host__
glm::vec3 operator*(const glm::mat4 &m, const glm::vec3 &v);

__device__ __host__
bool isZeroVec(const glm::vec3 &v);

__device__
Ray operator*(const glm::mat4 &m, const Ray &ray);

__device__
Interaction operator*(const glm::mat4 &m, const Interaction &i);

struct Ray {
    __device__  Ray();
    __device__  Ray(const glm::vec3 &o, const glm::vec3 &d, const float t = INFINITY);
    __device__ glm::vec3 operator()(const float t) const;
    glm::vec3 o;
    glm::vec3 d;
    mutable float t;
};

struct Interaction {
    __device__ Interaction();
    __device__ Interaction(const glm::vec3 &p, const glm::vec3 &n, const glm::vec3 &wo);
    glm::vec3 p;
    glm::vec3 n;
    glm::vec3 wo;
};