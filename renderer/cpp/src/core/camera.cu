#include "camera.hpp"
#include "sampling.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

__host__
Camera::Camera(const int resX, const int resY, const glm::vec3 &worldPos, const glm::vec3 &target, const float fov_y, const float fStop, const float focusDistance, const int stratificationLevel)
    : resX(resX), resY(resY), worldPos(worldPos), target(target), fov_y(fov_y), pFilmIndex(nullptr), pImage(nullptr), focusDistance(focusDistance), stratificationLevel(stratificationLevel)
{
    height = 2 * tan(glm::radians(fov_y * 0.5));
    width = height * ((float)resX) / resY;
    this->apertureRadius = fStopToApertureSize(fStop);
}

__host__
void Camera::update(const glm::vec3 &worldPos, const glm::vec3 &target, const float fov_y, const float fStop, const float focusDistance, const int stratificationLevel) {
    this->worldPos = worldPos;
    this->target = target;
    this->fov_y = fov_y;
    this->apertureRadius = fStopToApertureSize(fStop);
    this->focusDistance = focusDistance;
    this->stratificationLevel = stratificationLevel;

    height = 2 * tan(glm::radians(fov_y * 0.5));
    width = height * ((float)resX) / resY;
}

__host__
int Camera::getStratificationLevel() const {
    return stratificationLevel;
}

__host__
void Camera::update(const int resX, const int resY) {
    this->resX = resX * stratificationLevel;
    this->resY = resY * stratificationLevel;

    height = 2 * tan(glm::radians(fov_y * 0.5));
    width = height * ((float)this->resX) / this->resY;
}

__device__
void Camera::getRay(const int filmIndex, Ray *ray, const glm::vec2 &pixelSample, const glm::vec2 &lensSample) const {
    if(filmIndex < resX * resY) {
        long y = filmIndex / resX;
        long x = filmIndex % resX;
        getRay(x, y, ray, pixelSample, lensSample);
    }
}

__device__
void Camera::getRay(const int x, const int y, Ray *ray, const glm::vec2 &pixelSample, const glm::vec2 &lensSample) const {
    float scaleX = (x + pixelSample.x) / resX;
    float scaleY = (y + pixelSample.y) / resY;
    glm::vec3 imgPoint(-width * 0.5 + scaleX * width, -height * 0.5 + scaleY * height, -1.0);
    glm::vec3 focusPoint(imgPoint.x * focusDistance, imgPoint.y * focusDistance, -focusDistance);
    glm::vec3 lensPoint = apertureRadius * glm::vec3(UniformSampleDisk(lensSample), 0.0);

    Ray r(lensPoint, focusPoint - lensPoint);
    glm::mat4 lookAt = glm::inverse(glm::lookAt(worldPos, target, glm::vec3(0.0, 1.0, 0.0)));
    *ray = lookAt * r;
    ray->d = glm::normalize(ray->d);
}

__device__
int Camera::getFilmIndex() {
    int idx = atomicAdd(pFilmIndex, 1);
    return idx < resX * resY ? idx : -1;
}

__host__
bool Camera::isDone() const {
    return *pFilmIndex >= resX * resY;
}

__host__
void Camera::resetFilmIndex() {
    *pFilmIndex = 0;
}

__host__
void Camera::setImage(glm::vec3 *pImage) {
    this->pImage = pImage;
}

__host__ void Camera::setFilmIndexPointer(int *pFilmIndex) {
    this->pFilmIndex = pFilmIndex;
}

__device__
void Camera::addPixelVal(const int filmindex, const glm::vec3 &val) {
    if(filmindex < resX * resY) {
        pImage[filmindex] += val;
    }
}

__host__
float Camera::fStopToApertureSize(const float fStop) const {
    float f = 1 / tan(glm::radians(fov_y * 0.5)) / 2;
    return f * 0.5 / fStop;
}