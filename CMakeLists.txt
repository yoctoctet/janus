cmake_minimum_required(VERSION 3.10)
project(janus
        LANGUAGES CUDA CXX
        VERSION 0.1.0)

set(CMAKE_CUDA_ARCHITECTURES 86)

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
add_executable(janus sources/main.cu)

# Build tests only if GTest is available
if(BUILD_TESTS)
    message(STATUS "Unit tests will be built (GTest found)")
else()
    message(STATUS "Unit tests will NOT be built (GTest not found)")
endif()
