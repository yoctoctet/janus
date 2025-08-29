cmake_minimum_required(VERSION 3.18)
project(janus
        LANGUAGES CXX CUDA
        VERSION 0.1.0
        DESCRIPTION "Janus is a n-body solver to evaluate the JCM by J.P. Petit") # TODO: Complete the description.

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

include_directories(src)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

enable_testing()

#========== MAIN PROGRAM ==========#

add_executable(janus
    src/main.cpp
    src/buffer.cpp
    src/buffer_kernels.cu)

target_link_libraries(janus PRIVATE spdlog::spdlog)
target_link_libraries(janus PRIVATE CLI11::CLI11)
target_link_libraries(janus PRIVATE CUDA::cudart)

#========== UNIT TESTS ==========#

add_executable(test_buffer
        tests/test_buffer.cpp
        src/buffer.cpp
        src/buffer_kernels.cu)

target_link_libraries(test_buffer PRIVATE GTest::gtest_main)
target_link_libraries(test_buffer PRIVATE spdlog::spdlog)
target_link_libraries(test_buffer PRIVATE CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(test_buffer)