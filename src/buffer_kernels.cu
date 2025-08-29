// CUDA kernels for Buffer operations
#include <cuda_runtime.h>

// Define for CUDA kernel thread count
#ifndef JANUS_FILL_THREADS_PER_BLOCK
#define JANUS_FILL_THREADS_PER_BLOCK 256
#endif

// CUDA kernel for filling device memory
template <typename T>
__global__ void fillKernel(T *data, T value, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = value;
    }
}

// Explicit template instantiations for CUDA kernels
template __global__ void fillKernel<double>(double *, double, size_t);
template __global__ void fillKernel<float>(float *, float, size_t);
template __global__ void fillKernel<int>(int *, int, size_t);

// C wrapper functions for kernel launches
extern "C"
{

    cudaError_t launchFillKernelDouble(double *data, double value, size_t size)
    {
        int threadsPerBlock = JANUS_FILL_THREADS_PER_BLOCK;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        fillKernel<double><<<blocksPerGrid, threadsPerBlock>>>(data, value, size);
        return cudaGetLastError();
    }

    cudaError_t launchFillKernelFloat(float *data, float value, size_t size)
    {
        int threadsPerBlock = JANUS_FILL_THREADS_PER_BLOCK;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        fillKernel<float><<<blocksPerGrid, threadsPerBlock>>>(data, value, size);
        return cudaGetLastError();
    }

    cudaError_t launchFillKernelInt(int *data, int value, size_t size)
    {
        int threadsPerBlock = JANUS_FILL_THREADS_PER_BLOCK;
        int blocksPerBlock = (size + threadsPerBlock - 1) / threadsPerBlock;
        fillKernel<int><<<blocksPerBlock, threadsPerBlock>>>(data, value, size);
        return cudaGetLastError();
    }
}