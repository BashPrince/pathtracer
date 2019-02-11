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
 
#include "geometry.hpp"

__device__ __host__
glm::vec3 operator*(const glm::mat4 &m, const glm::vec3 &v) {
    return glm::vec3(m * glm::vec4(v, 1.0));
}

__device__ __host__
bool isZeroVec(const glm::vec3 &v) {
    return v.x == 0.0 && v.y == 0.0 && v.z == 0.0;
}

__device__
Ray operator*(const glm::mat4 &m, const Ray &ray) {
    glm::vec3 o = m * ray.o;
    glm::vec3 d = m * glm::vec4(ray.d, 0.0);
    return Ray(o, d, ray.t);
}

__device__
Interaction operator*(const glm::mat4 &m, const Interaction &i) {
    glm::mat3 invTrans = glm::transpose(glm::inverse(glm::mat3(m)));
    return Interaction(m * i.p, invTrans * i.n, m * glm::vec4(i.wo, 0.0));
}

__device__
Ray::Ray()
    : t(INFINITY)
{}

__device__
Ray::Ray(const glm::vec3 &o, const glm::vec3 &d, float t)
    :o(o), d(d), t(t)
{}

__device__
glm::vec3 Ray::operator()(const float t) const {
    return o + d * t;
}

__device__
Interaction::Interaction()
    :p(glm::vec3(0.0)), n(glm::vec3(0.0)), wo(glm::vec3(0.0))
{}

__device__
Interaction::Interaction(const glm::vec3 &p, const glm::vec3 &n, const glm::vec3 &wo)
    :p(p), n(n), wo(wo)
{}