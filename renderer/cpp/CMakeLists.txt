cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(renderer LANGUAGES CXX CUDA)

add_library(renderer SHARED
            src/core/Renderer.hpp
            src/core/Renderer.cu
            src/core/geometry.hpp
            src/core/geometry.cu
            src/core/shape.hpp
            src/core/shape.cu
            src/core/camera.hpp
            src/core/camera.cu
            src/util/util.hpp
            src/util/util.cu
            src/core/sampling.hpp
            src/core/sampling.cu
            src/core/shading.hpp
            src/core/shading.cu
            src/core/kernels.hpp
            src/core/kernels.cu
            src/core/pathstate.hpp
            src/core/pathstate.cu)


target_compile_features(renderer PUBLIC cxx_std_11)
target_link_libraries(renderer pthread)
set_target_properties(renderer PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      ARCHIVE_OUTPUT_DIRECTORY "../lib"
                      LIBRARY_OUTPUT_DIRECTORY "../lib")

#target_compile_definitions(renderer PUBLIC CudaDebug)