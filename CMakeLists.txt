cmake_minimum_required(VERSION 3.18)
project(janus
    LANGUAGES CXX CUDA
    VERSION 0.1.0
    DESCRIPTION "Janus is a...") # TODO: Complete the description.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_compile_definitions(
    JANUS_VERSION="${CMAKE_PROJECT_VERSION}"
    JANUS_VERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR}
    JANUS_VERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR}
    JANUS_VERSION_PATCH=${CMAKE_PROJECT_VERSION_PATCH}
    JANUS_NAME="${PROJECT_NAME}"
    JANUS_DESCRIPTION="${PROJECT_DESCRIPTION}"
)

if(${CMAKE_BUILD_TYPE} STREQUAL Release)
    add_compile_definitions(JANUS_DEBUG=0)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    add_compile_definitions(JANUS_DEBUG=1)
else()
    message(FATAL_ERROR "Build type `${CMAKE_BUILD_TYPE}` not supported")
endif()

enable_testing()

add_subdirectory(lib)
add_subdirectory(solver)
