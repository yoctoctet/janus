# Janus P3M - GPU-Accelerated Particle Simulation

A C++20 + CUDA project for particle simulation with GPU acceleration using NVIDIA CUDA.

## Features

- **GPU Acceleration**: CUDA kernels for high-performance particle simulation
- **Flexible Configuration**: TOML-based configuration system
- **Command-Line Interface**: Rich CLI with CLI11
- **Logging**: Structured logging with spdlog
- **JSON Metadata**: JSON serialization support
- **Testing**: Unit tests with gtest framework
- **Build Options**:
  - `JANUS_USE_NVTX`: Enable NVIDIA Tools Extension profiling (default: OFF)
  - `JANUS_FAST_MATH`: Enable CUDA fast math optimizations (default: OFF)
  - `JANUS_DETERMINISTIC`: Enable deterministic execution (default: ON)

## Prerequisites

- **CMake** 3.18 or later
- **CUDA Toolkit** 11.0 or later
- **C++20 compatible compiler** (GCC 10+, Clang 10+, MSVC 2019+)
- **NVIDIA GPU** with CUDA support (optional, can run in CPU-only mode)

## Building

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd janus-p3m

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
ctest

# Run the simulator
./janus-sim --help
```

### Build Options

```bash
# Enable NVTX profiling
cmake .. -DJANUS_USE_NVTX=ON

# Enable fast math optimizations
cmake .. -DJANUS_FAST_MATH=ON

# Disable deterministic execution
cmake .. -DJANUS_DETERMINISTIC=OFF

# CPU-only build (no CUDA required)
cmake .. -DCMAKE_CUDA_COMPILER=""  # This will disable CUDA features
```

### Configuration

Create a `config.toml` file in the project root:

```toml
[simulation]
name = "my_simulation"
num_particles = 10000
time_step = 0.001
max_steps = 100000
use_gpu = true

[gpu]
block_size = 256
```

## Usage

### Command Line Options

```bash
./janus-sim [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -c,--config TEXT            Configuration file path
  -o,--output TEXT            Output directory
  -v,--verbose                Enable verbose logging
  -q,--quiet                  Disable logging
  --particles INT             Number of particles
  --time-step FLOAT           Simulation time step
  --max-steps INT             Maximum simulation steps
  --cpu-only                  Run simulation on CPU only
```

### Examples

```bash
# Run with default settings
./janus-sim

# Run with custom configuration
./janus-sim -c my_config.toml -o results/

# Run small simulation for testing
./janus-sim --particles 1000 --max-steps 1000 --cpu-only

# Run with verbose logging
./janus-sim -v
```

## Project Structure

```
janus-p3m/
├── CMakeLists.txt           # Main CMake configuration
├── src/                     # Source code
│   ├── CMakeLists.txt       # Source CMake configuration
│   ├── main.cpp            # Application entry point
│   ├── simulator.hpp/cpp   # Main simulator class
│   ├── config.hpp/cpp      # Configuration management
│   └── gpu_kernels.cu/cuh  # CUDA kernels
├── tests/                   # Unit tests
│   ├── CMakeLists.txt      # Tests CMake configuration
│   ├── test_main.cpp       # Test entry point
│   ├── test_config.cpp     # Configuration tests
│   └── test_simulator.cpp  # Simulator tests
├── vendor/                  # Vendored dependencies
│   ├── cli11/              # CLI11 library
│   ├── spdlog/             # spdlog library
│   ├── tomlpp/             # toml++ library
│   ├── nlohmann/           # nlohmann/json library
│   └── gtest/              # gtest framework
└── README.md               # This file
```

## Testing

Run the test suite:

```bash
# From build directory
ctest

# Or run tests directly
./janus-tests
```

## Dependencies

This project vendors all dependencies as header-only libraries:

- **toml++**: TOML configuration file parsing
- **CLI11**: Command-line argument parsing
- **spdlog**: Logging framework
- **nlohmann/json**: JSON serialization
- **gtest**: Unit testing framework

## Architecture

The project uses a modular architecture:

- **Simulator**: Main simulation orchestrator
- **Config**: Configuration management with TOML/JSON support
- **GPU Kernels**: CUDA kernels for GPU acceleration
- **CLI**: Command-line interface handling

## Performance Notes

- GPU acceleration provides significant speedup for large particle counts
- Use `JANUS_FAST_MATH=ON` for maximum performance (may reduce precision)
- `JANUS_DETERMINISTIC=ON` ensures reproducible results
- NVTX profiling helps identify performance bottlenecks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Add your license information here]