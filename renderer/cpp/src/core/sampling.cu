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

#include <cmath>
#include "sampling.hpp"

__device__
glm::vec2 UniformSampleDisk(const glm::vec2 &sample) {
    float r = sqrt(sample.x);
    float theta = 2 * M_PI * sample.y;
    return glm::vec2(r * cos(theta), r * sin(theta));
}

__device__
glm::vec3 CosineSampleHemisphere(const glm::vec2 &sample) {
    glm::vec2 d = UniformSampleDisk(sample);
    float z = sqrt(fmaxf(0.00001f, 1.0f - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z);
}

__device__
glm::vec3 UniformSampleSphere(const glm::vec2 &sample)
{
    float z = 1 - 2 * sample.x;
    float r = sqrt(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2 * M_PI * sample.y;
    return glm::vec3(r * cos(phi), r * sin(phi), z);
}