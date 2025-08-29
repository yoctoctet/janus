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

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

enable_testing()

#========== LIBRARY ==========#

add_library(janus-lib
        src/buffer.cpp
        src/buffer_kernels.cu
        src/direct_sum_force_cpu.cpp
        src/direct_sum_force_cuda.cu)

target_include_directories(janus-lib PUBLIC src)

target_link_libraries(janus-lib PUBLIC spdlog::spdlog)
target_link_libraries(janus-lib PUBLIC CUDA::cudart)

#========== MAIN PROGRAM ==========#

add_executable(janus src/main.cpp)

target_link_libraries(janus PRIVATE CLI11::CLI11)
target_link_libraries(janus PRIVATE janus-lib)

#========== UNIT TESTS ==========#

add_executable(test_buffer tests/test_buffer.cpp)
target_link_libraries(test_buffer PRIVATE GTest::gtest_main)
target_link_libraries(test_buffer PRIVATE janus-lib)

add_executable(test_direct_sum_force tests/test_direct_sum_force.cpp)
target_link_libraries(test_direct_sum_force PRIVATE GTest::gtest_main)
target_link_libraries(test_direct_sum_force PRIVATE janus-lib)

add_executable(test_direct_sum_force_cpu tests/test_direct_sum_force_cpu.cpp)
target_link_libraries(test_direct_sum_force_cpu PRIVATE GTest::gtest_main)
target_link_libraries(test_direct_sum_force_cpu PRIVATE janus-lib)

add_executable(test_direct_sum_force_cuda tests/test_direct_sum_force_cuda.cpp)
target_link_libraries(test_direct_sum_force_cuda PRIVATE GTest::gtest_main)
target_link_libraries(test_direct_sum_force_cuda PRIVATE janus-lib)

include(GoogleTest)
gtest_discover_tests(test_buffer)
gtest_discover_tests(test_direct_sum_force)
gtest_discover_tests(test_direct_sum_force_cpu)
gtest_discover_tests(test_direct_sum_force_cuda)