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

    __global__ void update_positions_kernel(double *x, double *y, double *vx, double *vy,
                                            double *ax, double *ay, double time_step, int num_particles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_particles)
        {
            // Velocity Verlet integration for 2D
            // Half kick: v += 0.5*dt*a
            vx[idx] += 0.5 * time_step * ax[idx];
            vy[idx] += 0.5 * time_step * ay[idx];

            // Drift: x += dt*v
            x[idx] += vx[idx] * time_step;
            y[idx] += vy[idx] * time_step;

            // Note: Second half kick and acceleration update would be done here
            // For now, keeping accelerations as-is (zero forces)
            // vx[idx] += 0.5 * time_step * ax[idx];
            // vy[idx] += 0.5 * time_step * ay[idx];
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

    // High-level GPU kernel launcher
    cudaError_t launch_update_positions_kernel(double *d_x, double *d_y, double *d_vx, double *d_vy,
                                               double *d_ax, double *d_ay, double time_step, int num_particles)
    {
        // Compute grid and block dimensions
        int block_size = get_optimal_block_size(num_particles);
        int grid_size = get_optimal_grid_size(num_particles, block_size);

        // Launch GPU kernel
        update_positions_kernel<<<grid_size, block_size>>>(d_x, d_y, d_vx, d_vy, d_ax, d_ay, time_step, num_particles);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return err;
        }

        // Synchronize to ensure kernel completion
        err = cudaDeviceSynchronize();
        return err;
    }

} // namespace janus