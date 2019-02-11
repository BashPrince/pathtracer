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
 
#include "util.hpp"
#include <iostream>

__device__
bool Quadratic(const float a, const float b, const float c, float *t0, float *t1) {
    float dis = b * b - 4.0 * a * c;
    if(dis < 0.0 || a == 0.0) {
        return false;
    }
    dis = sqrt(dis);
    float x0 = (-b + dis) / (2 * a);
    float x1 = (-b - dis) / (2 * a);
    if(x0 <= x1) {
        *t0 = x0;
        *t1 = x1;
    } else {
        *t0 = x1;
        *t1 = x0;
    }
    return true;
}

__host__ __device__
glm::vec3 pow(const glm::vec3 &v, const float exponent) {
    return glm::vec3(pow(v.r, exponent), pow(v.g, exponent), pow(v.b, exponent));
}

__device__
float clamp(const float &v, const float &min, const float &max) {
    return fmaxf(min, fminf(v, max));
}

void waitAndCheckError(const char* location) {
#ifdef CudaDebug
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        std::cout << location << ": " << cudaGetErrorString(err) << std::endl;
#endif
}