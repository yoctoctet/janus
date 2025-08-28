#pragma once

#include <cuda_runtime.h>

namespace janus
{

    // CUDA kernel declarations
    __global__ void hello_gpu_kernel(float *data, int size);
    __global__ void update_positions_kernel(float *positions, float *velocities,
                                            float time_step, int num_particles);

    // Host functions for GPU operations
    cudaError_t allocate_device_memory(void **device_ptr, size_t size);
    cudaError_t free_device_memory(void *device_ptr);
    cudaError_t copy_to_device(void *device_ptr, const void *host_ptr, size_t size);
    cudaError_t copy_from_device(void *host_ptr, const void *device_ptr, size_t size);

    // GPU utility functions
    int get_optimal_block_size(int data_size);
    int get_optimal_grid_size(int data_size, int block_size);

} // namespace janus