cmake_minimum_required(VERSION 3.18)
project(janus-p3m LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Project options
option(JANUS_USE_NVTX "Enable NVIDIA Tools Extension profiling" OFF)
option(JANUS_FAST_MATH "Enable CUDA fast math optimizations" OFF)
option(JANUS_DETERMINISTIC "Enable deterministic execution" ON)

# CUDA compiler flags
if(JANUS_FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()

if(JANUS_DETERMINISTIC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
endif()

# Include directories for vendor libraries
include_directories(
    vendor/cli11/include
    vendor/spdlog/include
    vendor/tomlpp
    vendor/nlohmann
    vendor/gtest/googletest/include
)

# Find required packages
find_package(CUDAToolkit REQUIRED)
if(JANUS_USE_NVTX)
    find_package(NVTX REQUIRED)
endif()

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)