#pragma once
#include "geometry.hpp"

class Camera {
public:
    __host__ Camera(const int resX, const int resY, const glm::vec3 &worldPos = glm::vec3(), const glm::vec3 &target = glm::vec3(0.0, 0.0, 1.0), const float fov_y = 45.0, const float fStop = 5.6, const float focusDistance = 10.0, const int stratificationLevel = 1);
    __host__ void update(const glm::vec3 &worldPos, const glm::vec3 &target, const float fov_y, const float fStop, const float focusDistance, const int stratificationLevel);
    __host__ void update(const int resX, const int resY);
    __device__ void getRay(const int filmIndex, Ray *ray, const glm::vec2 &pixelSample, const glm::vec2 &lensSample) const;
    __device__ int getFilmIndex();
    __host__ bool isDone() const;
    __host__ void resetFilmIndex();
    __host__ void setImage(glm::vec3 *pImage);
    __host__ void setFilmIndexPointer(int *pFilmIndex);
    __device__ void addPixelVal(const int filmIndex, const glm::vec3 &val);
    __host__ int getStratificationLevel() const;

private:
    __device__ void getRay(const int x, const int y, Ray *ray, const glm::vec2 &pixelSample, const glm::vec2 &lensSample) const;
    __host__ float fStopToApertureSize(const float fStop) const;

    glm::vec3 worldPos;
    glm::vec3 target;
    float fov_y;
    int resX;
    int resY;
    float width;
    float height;
    glm::vec3 *pImage;
    int *pFilmIndex;
    float focusDistance;
    float apertureRadius;
    int stratificationLevel;
};