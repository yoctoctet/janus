#include "gpu_kernels.cuh"
#include <cstdio>

namespace janus
{

    // CUDA kernels
    __global__ void hello_gpu_kernel(float *data, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = data[idx] * 2.0f + 1.0f;
        }
    }

    __global__ void update_positions_kernel(float *positions, float *velocities,
                                            float time_step, int num_particles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_particles)
        {
            // Simple Euler integration
            positions[idx] += velocities[idx] * time_step;
        }
    }

    // Host functions for GPU operations
    cudaError_t allocate_device_memory(void **device_ptr, size_t size)
    {
        return cudaMalloc(device_ptr, size);
    }

    cudaError_t free_device_memory(void *device_ptr)
    {
        return cudaFree(device_ptr);
    }

    cudaError_t copy_to_device(void *device_ptr, const void *host_ptr, size_t size)
    {
        return cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    }

    cudaError_t copy_from_device(void *host_ptr, const void *device_ptr, size_t size)
    {
        return cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
    }

    // GPU utility functions
    int get_optimal_block_size(int data_size)
    {
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        // Use a reasonable block size, not exceeding max threads per block
        int block_size = 256;
        if (block_size > prop.maxThreadsPerBlock)
        {
            block_size = prop.maxThreadsPerBlock;
        }
        return block_size;
    }

    int get_optimal_grid_size(int data_size, int block_size)
    {
        return (data_size + block_size - 1) / block_size;
    }

} // namespace janus