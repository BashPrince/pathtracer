#include "pathstate.hpp"
#include "../util/util.hpp"

PathstateSoA::PathstateSoA()
    :
    currentIsect(nullptr),
    extensionIsect(nullptr),
    foundExtensionIsect(nullptr),
    isActive(nullptr),
    pdf(nullptr),
    lightPdf(nullptr),
    bounces(nullptr),
    beta(nullptr),
    L(nullptr),
    wi_L(nullptr),
    Lemit(nullptr),
    Lsample(nullptr),
    LsampleOccluded(nullptr),
    isLightSample(nullptr),
    filmIndex(nullptr),
    material(nullptr),
    f(nullptr),
    light_f(nullptr),
    diffuseBounce(nullptr)
{}

void PathstateSoA::freeArrays() {
        if(currentIsect) {
            cudaFree(currentIsect);
            waitAndCheckError("PathstateSoA::freeArrays(currentIsect)");
            currentIsect = nullptr;
        }
        if(extensionIsect) {
            cudaFree(extensionIsect);
            waitAndCheckError("PathstateSoA::freeArrays(extensionIsect)");
            extensionIsect = nullptr;
        }
        if(foundExtensionIsect) {
            cudaFree(foundExtensionIsect);
            waitAndCheckError("PathstateSoA::freeArrays(foundExtensionIsect)");
            foundExtensionIsect = nullptr;
        }
        if(isActive) {
            cudaFree(isActive);
            waitAndCheckError("PathstateSoA::freeArrays(isActive)");
            isActive = nullptr;
        }
        if(pdf) {
            cudaFree(pdf);
            waitAndCheckError("PathstateSoA::freeArrays(pdf)");
            pdf = nullptr;
        }
        if(lightPdf) {
            cudaFree(lightPdf);
            waitAndCheckError("PathstateSoA::freeArrays(lightPdf)");
            lightPdf = nullptr;
        }
        if(bounces) {
            cudaFree(bounces);
            waitAndCheckError("PathstateSoA::freeArrays(bounces)");
            bounces = nullptr;
        }
        if(beta) {
            cudaFree(beta);
            waitAndCheckError("PathstateSoA::freeArrays(beta)");
            beta = nullptr;
        }
        if(L) {
            cudaFree(L);
            waitAndCheckError("PathstateSoA::freeArrays(L)");
            L = nullptr;
        }
        if(wi_L) {
            cudaFree(wi_L);
            waitAndCheckError("PathstateSoA::freeArrays(wi_L)");
            wi_L = nullptr;
        }
        if(Lemit) {
            cudaFree(Lemit);
            waitAndCheckError("PathstateSoA::freeArrays(Lemit)");
            Lemit = nullptr;
        }
        if(filmIndex) {
            cudaFree(filmIndex);
            waitAndCheckError("PathstateSoA::freeArrays(filmIndex)");
            filmIndex = nullptr;
        }
        if(material) {
            cudaFree(material);
            waitAndCheckError("PathstateSoA::freeArrays(material)");
            material = nullptr;
        }
        if(f) {
            cudaFree(f);
            waitAndCheckError("PathstateSoA::freeArrays(f)");
            f = nullptr;
        }
        if(light_f) {
            cudaFree(light_f);
            waitAndCheckError("PathstateSoA::freeArrays(light_f)");
            light_f = nullptr;
        }
        if(Lsample) {
            cudaFree(Lsample);
            waitAndCheckError("PathstateSoA::freeArrays(Lsample)");
            Lsample = nullptr;
        }
        if(LsampleOccluded) {
            cudaFree(LsampleOccluded);
            waitAndCheckError("PathstateSoA::freeArrays(LsampleOccluded)");
            LsampleOccluded = nullptr;
        }
        if(isLightSample) {
            cudaFree(isLightSample);
            waitAndCheckError("PathstateSoA::freeArrays(isLightSample)");
            isLightSample = nullptr;
        }
        if(diffuseBounce) {
            cudaFree(diffuseBounce);
            waitAndCheckError("PathstateSoA::freeArrays(diffuseBounce)");
            diffuseBounce = nullptr;
        }
    }