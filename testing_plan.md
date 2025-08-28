# Testing Plan and CMake Integration

## CMakeLists.txt Updates for Testing

```cmake
cmake_minimum_required(VERSION 3.10)
project(gravitation
        LANGUAGES CUDA CXX
        VERSION 0.1.0)

set(CMAKE_CUDA_STANDARD 26)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

# Enable testing
enable_testing()
find_package(GTest REQUIRED)

# Main executable
add_executable(gravitation
        sources/main.cu)

target_link_libraries(gravitation PRIVATE CUDA::curand)
target_link_libraries(gravitation PRIVATE CUDA::cufft)

# Test executable
add_executable(gravitation_tests
        tests/test_configuration.cpp
        tests/test_cuda_manager.cpp
        tests/test_particle_system.cpp
        tests/test_mesh.cpp
        tests/test_force_calculator.cpp
        sources/configuration.cpp
        sources/cuda_manager.cpp
        sources/particle_system.cpp
        sources/mesh.cpp
        sources/force_calculator.cpp)

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

# Test data directory
configure_file(tests/test_config.json ${CMAKE_BINARY_DIR}/test_config.json COPYONLY)
```

## Test File Structure

```
project/
├── CMakeLists.txt
├── sources/
│   ├── main.cu
│   ├── configuration.cpp
│   ├── cuda_manager.cpp
│   ├── particle_system.cpp
│   ├── mesh.cpp
│   └── force_calculator.cpp
├── tests/
│   ├── test_config.json
│   ├── test_configuration.cpp
│   ├── test_cuda_manager.cpp
│   ├── test_particle_system.cpp
│   ├── test_mesh.cpp
│   └── test_force_calculator.cpp
└── include/
    ├── configuration.h
    ├── cuda_manager.h
    ├── particle_system.h
    ├── mesh.h
    └── force_calculator.h
```

## Test Execution Commands

```bash
# Build and run all tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
ctest --verbose

# Run specific test suite
ctest -R ConfigurationTest

# Run with memory checking (if CUDA Memcheck available)
cuda-memcheck ctest --verbose

# Run performance tests
ctest -R PerformanceTest
```

## GTest Installation (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install libgtest-dev cmake build-essential

# Build and install GTest
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib
```

## Test Data Files

### tests/test_config.json
```json
{
    "simulation": {
        "numParticles": 100,
        "maxIterations": 10,
        "timeStep": 0.01,
        "softeningParameter": 0.001
    },
    "p3m": {
        "gridResolution": 16,
        "cutoffRadius": 2.0,
        "ewaldAlpha": 0.35,
        "interpolationScheme": "cic"
    },
    "cuda": {
        "deviceId": 0,
        "pinnedMemory": true,
        "unifiedMemory": false
    }
}
```

## Continuous Integration Setup

### GitHub Actions Workflow (.github/workflows/ci.yml)

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Install CUDA
      run: |
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
        sudo sh cuda_11.8.0_520.61.05_linux.run --no-opengl-libs --no-man-page --noexe

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libgtest-dev cmake build-essential

    - name: Build
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Debug
        make -j$(nproc)

    - name: Run tests
      run: |
        cd build
        ctest --verbose --output-on-failure
```

## Test Categories and Priorities

### Phase 1: Foundation Tests (Implement First)
1. **Configuration Tests** - Validate parameter loading and validation
2. **CUDA Manager Tests** - Ensure GPU device management works
3. **Particle System Tests** - Verify particle data structures

### Phase 2: Algorithm Tests (Implement Second)
4. **Mesh Tests** - Validate grid operations and CIC interpolation
5. **Force Calculator Tests** - Test P3M algorithm components

### Phase 3: Integration Tests (Implement Last)
6. **Full Simulation Tests** - End-to-end simulation validation
7. **Performance Tests** - Benchmarking and optimization validation

## Mock Objects Strategy

For testing components with heavy dependencies:

```cpp
// Mock CUDA Manager for CPU-only testing
class MockCUDAManager {
public:
    bool initializeDevice() { return true; }
    bool allocateDeviceMemory(void** ptr, size_t size) {
        *ptr = malloc(size); // Use CPU memory
        return *ptr != nullptr;
    }
    // ... other mock methods
};
```

## Performance Benchmarking

```cpp
// Performance test fixture
class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup large test case
        config.numParticles = 10000;
        cudaMgr.initializeDevice();
    }

    Configuration config;
    CUDAManager cudaMgr;
};

// Benchmark force calculation
TEST_F(PerformanceTest, ForceCalculationPerformance) {
    // Setup particles and mesh
    // Measure execution time
    // Assert performance meets requirements
}
```

This testing plan ensures robust, maintainable code with comprehensive validation at each development stage.