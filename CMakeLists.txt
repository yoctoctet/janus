cmake_minimum_required(VERSION 3.10)
project(gravitation
        LANGUAGES CUDA CXX
        VERSION 0.1.0)

set(CMAKE_CUDA_STANDARD 26)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable testing
enable_testing()

# Try to find GTest, but make it optional
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(WARNING "GTest not found. Unit tests will not be built.")
    message("To install GTest on Ubuntu/Debian: sudo apt-get install libgtest-dev")
    message("Then build and install: cd /usr/src/gtest && sudo cmake CMakeLists.txt && sudo make && sudo cp *.a /usr/lib")
    set(BUILD_TESTS OFF)
else()
    set(BUILD_TESTS ON)
endif()

find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Main executable
add_executable(gravitation
        sources/main.cu)

target_link_libraries(gravitation PRIVATE CUDA::curand)
target_link_libraries(gravitation PRIVATE CUDA::cufft)

# Build tests only if GTest is available
if(BUILD_TESTS)
    # Test executable
    add_executable(gravitation_tests
            # Source files
            sources/configuration.cpp
            sources/cuda_manager.cpp
            sources/particle_system.cpp
            sources/mesh.cpp
            sources/force_calculator.cpp
            # Test files
            tests/test_configuration.cpp
            tests/test_cuda_manager.cpp
            tests/test_particle_system.cpp
            tests/test_mesh.cpp
            tests/test_force_calculator.cpp)
    
    # Benchmark executable
    add_executable(p3m_benchmark
            sources/benchmark_p3m.cpp
            sources/configuration.cpp
            sources/cuda_manager.cpp
            sources/particle_system.cpp
            sources/mesh.cpp
            sources/force_calculator.cpp)
    
    # Link CUDA libraries for benchmark
    target_link_libraries(p3m_benchmark PRIVATE CUDA::cudart CUDA::curand CUDA::cufft)

    # Set C++ standard for tests
    set_target_properties(gravitation_tests PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)

    # Link libraries for tests
    target_link_libraries(gravitation_tests PRIVATE
            GTest::GTest
            GTest::Main
            CUDA::cudart
            CUDA::curand
            CUDA::cufft)

    # Add tests
    add_test(NAME ConfigurationTest COMMAND gravitation_tests --gtest_filter=ConfigurationTest.*)
    add_test(NAME CUDAManagerTest COMMAND gravitation_tests --gtest_filter=CUDAManagerTest.*)
    add_test(NAME ParticleSystemTest COMMAND gravitation_tests --gtest_filter=ParticleSystemTest.*)
    add_test(NAME MeshTest COMMAND gravitation_tests --gtest_filter=MeshTest.*)
    add_test(NAME ForceCalculatorTest COMMAND gravitation_tests --gtest_filter=ForceCalculatorTest.*)

    message(STATUS "Unit tests will be built (GTest found)")
else()
    message(STATUS "Unit tests will NOT be built (GTest not found)")
endif()

# Create test data directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test_data)
