#include "buffer.h"

#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

// Define for CUDA kernel thread count
#ifndef JANUS_FILL_THREADS_PER_BLOCK
#define JANUS_FILL_THREADS_PER_BLOCK 256
#endif

namespace janus
{
    // Helper function to check CUDA errors
    inline void checkCudaError(cudaError_t error, const std::string &operation)
    {
        if (error != cudaSuccess)
        {
            std::string message = "CUDA error during " + operation + ": " + cudaGetErrorString(error);
            spdlog::error(message);
            throw BufferException(message);
        }
    }

    // Forward declarations for CUDA kernel wrapper functions
    extern "C"
    {
        cudaError_t launchFillKernelDouble(double *data, double value, size_t size);
        cudaError_t launchFillKernelFloat(float *data, float value, size_t size);
        cudaError_t launchFillKernelInt(int *data, int value, size_t size);
    }

    // Host specialization implementation
    template <typename T>
    Buffer<T, Space::Host>::Buffer(size_t count) : data_(nullptr), size_(count)
    {
        if (count == 0)
        {
            spdlog::warn("Creating buffer with zero size");
            return;
        }

        try
        {
            data_ = new T[count]();
            spdlog::debug("Allocated host buffer with {} elements ({} bytes)", count, count * sizeof(T));
        }
        catch (const std::bad_alloc &e)
        {
            std::string message = "Failed to allocate host memory: " + std::string(e.what());
            spdlog::error(message);
            throw BufferException(message);
        }
    }

    template <typename T>
    Buffer<T, Space::Host>::~Buffer()
    {
        if (data_)
        {
            delete[] data_;
            spdlog::debug("Deallocated host buffer");
        }
    }

    template <typename T>
    Buffer<T, Space::Host>::Buffer(Buffer &&other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        spdlog::debug("Moved host buffer");
    }

    template <typename T>
    Buffer<T, Space::Host> &Buffer<T, Space::Host>::operator=(Buffer &&other) noexcept
    {
        if (this != &other)
        {
            if (data_)
            {
                delete[] data_;
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            spdlog::debug("Move assigned host buffer");
        }
        return *this;
    }

    template <typename T>
    template <Space O>
    void Buffer<T, Space::Host>::copy_from(const Buffer<T, O> &other)
    {
        if (size_ != other.size())
        {
            std::string message = "Buffer size mismatch: " + std::to_string(size_) + " vs " + std::to_string(other.size());
            spdlog::error(message);
            throw BufferException(message);
        }

        if constexpr (O == Space::Host)
        {
            // Host to Host copy
            std::memcpy(data_, other.data(), bytes());
            spdlog::debug("Copied {} bytes from host to host", bytes());
        }
        else
        {
            // Device to Host copy
            checkCudaError(cudaMemcpy(data_, other.data(), bytes(), cudaMemcpyDeviceToHost),
                           "copy from device to host");
            spdlog::debug("Copied {} bytes from device to host", bytes());
        }
    }

    template <typename T>
    void Buffer<T, Space::Host>::fill(const T &value)
    {
        if (data_)
        {
            std::fill(data_, data_ + size_, value);
            spdlog::debug("Filled host buffer with value");
        }
    }

    template <typename T>
    void Buffer<T, Space::Host>::zero()
    {
        if (data_)
        {
            std::memset(data_, 0, bytes());
            spdlog::debug("Zeroed host buffer");
        }
    }

    // Device specialization implementation
    template <typename T>
    Buffer<T, Space::Device>::Buffer(size_t count) : data_(nullptr), size_(count)
    {
        if (count == 0)
        {
            spdlog::warn("Creating device buffer with zero size");
            return;
        }

        cudaError_t error = cudaMalloc(&data_, count * sizeof(T));
        if (error != cudaSuccess)
        {
            std::string message = "Failed to allocate device memory: " + std::string(cudaGetErrorString(error));
            spdlog::error(message);
            throw BufferException(message);
        }
        spdlog::debug("Allocated device buffer with {} elements ({} bytes)", count, count * sizeof(T));
    }

    template <typename T>
    Buffer<T, Space::Device>::~Buffer()
    {
        if (data_)
        {
            cudaError_t error = cudaFree(data_);
            if (error != cudaSuccess)
            {
                spdlog::error("Failed to deallocate device memory: {}", cudaGetErrorString(error));
            }
            else
            {
                spdlog::debug("Deallocated device buffer");
            }
        }
    }

    template <typename T>
    Buffer<T, Space::Device>::Buffer(Buffer &&other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        spdlog::debug("Moved device buffer");
    }

    template <typename T>
    Buffer<T, Space::Device> &Buffer<T, Space::Device>::operator=(Buffer &&other) noexcept
    {
        if (this != &other)
        {
            if (data_)
            {
                cudaFree(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            spdlog::debug("Move assigned device buffer");
        }
        return *this;
    }

    template <typename T>
    template <Space O>
    void Buffer<T, Space::Device>::copy_from(const Buffer<T, O> &other)
    {
        if (size_ != other.size())
        {
            std::string message = "Buffer size mismatch: " + std::to_string(size_) + " vs " + std::to_string(other.size());
            spdlog::error(message);
            throw BufferException(message);
        }

        if constexpr (O == Space::Device)
        {
            // Device to Device copy
            checkCudaError(cudaMemcpy(data_, other.data(), bytes(), cudaMemcpyDeviceToDevice),
                           "copy from device to device");
            spdlog::debug("Copied {} bytes from device to device", bytes());
        }
        else
        {
            // Host to Device copy
            checkCudaError(cudaMemcpy(data_, other.data(), bytes(), cudaMemcpyHostToDevice),
                           "copy from host to device");
            spdlog::debug("Copied {} bytes from host to device", bytes());
        }
    }

    template <typename T>
    void Buffer<T, Space::Device>::fill(const T &value)
    {
        if (data_)
        {
            cudaError_t error;

            // Launch appropriate kernel based on type
            if constexpr (std::is_same_v<T, double>)
            {
                error = launchFillKernelDouble(data_, value, size_);
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                error = launchFillKernelFloat(data_, value, size_);
            }
            else if constexpr (std::is_same_v<T, int>)
            {
                error = launchFillKernelInt(data_, value, size_);
            }
            else
            {
                // Fallback to memcpy for unsupported types
                std::vector<T> host_data(size_, value);
                error = cudaMemcpy(data_, host_data.data(), bytes(), cudaMemcpyHostToDevice);
            }

            // Check for kernel launch errors
            checkCudaError(error, "kernel launch for fill");

            // Synchronize to ensure kernel completion
            checkCudaError(cudaDeviceSynchronize(), "kernel synchronization for fill");

            spdlog::debug("Filled device buffer with value using CUDA kernel");
        }
    }

    template <typename T>
    void Buffer<T, Space::Device>::zero()
    {
        if (data_)
        {
            checkCudaError(cudaMemset(data_, 0, bytes()), "zero device buffer");
            spdlog::debug("Zeroed device buffer");
        }
    }

    // Explicit template instantiations for common types
    template class Buffer<double, Space::Host>;
    template class Buffer<float, Space::Host>;
    template class Buffer<int, Space::Host>;
    template class Buffer<double, Space::Device>;
    template class Buffer<float, Space::Device>;
    template class Buffer<int, Space::Device>;

    // Explicit template instantiations for copy_from methods
    template void Buffer<double, Space::Host>::copy_from<Space::Host>(const Buffer<double, Space::Host> &);
    template void Buffer<double, Space::Host>::copy_from<Space::Device>(const Buffer<double, Space::Device> &);
    template void Buffer<double, Space::Device>::copy_from<Space::Host>(const Buffer<double, Space::Host> &);
    template void Buffer<double, Space::Device>::copy_from<Space::Device>(const Buffer<double, Space::Device> &);

    template void Buffer<float, Space::Host>::copy_from<Space::Host>(const Buffer<float, Space::Host> &);
    template void Buffer<float, Space::Host>::copy_from<Space::Device>(const Buffer<float, Space::Device> &);
    template void Buffer<float, Space::Device>::copy_from<Space::Host>(const Buffer<float, Space::Host> &);
    template void Buffer<float, Space::Device>::copy_from<Space::Device>(const Buffer<float, Space::Device> &);

    template void Buffer<int, Space::Host>::copy_from<Space::Host>(const Buffer<int, Space::Host> &);
    template void Buffer<int, Space::Host>::copy_from<Space::Device>(const Buffer<int, Space::Device> &);
    template void Buffer<int, Space::Device>::copy_from<Space::Host>(const Buffer<int, Space::Host> &);
    template void Buffer<int, Space::Device>::copy_from<Space::Device>(const Buffer<int, Space::Device> &);

    // Additional template instantiations for test types
    template class Buffer<uint32_t, Space::Host>;
    template class Buffer<uint32_t, Space::Device>;
    template class Buffer<char, Space::Host>;
    template class Buffer<char, Space::Device>;

    // Copy instantiations for test types
    template void Buffer<uint32_t, Space::Host>::copy_from<Space::Host>(const Buffer<uint32_t, Space::Host> &);
    template void Buffer<uint32_t, Space::Host>::copy_from<Space::Device>(const Buffer<uint32_t, Space::Device> &);
    template void Buffer<uint32_t, Space::Device>::copy_from<Space::Host>(const Buffer<uint32_t, Space::Host> &);
    template void Buffer<uint32_t, Space::Device>::copy_from<Space::Device>(const Buffer<uint32_t, Space::Device> &);

    template void Buffer<char, Space::Host>::copy_from<Space::Host>(const Buffer<char, Space::Host> &);
    template void Buffer<char, Space::Host>::copy_from<Space::Device>(const Buffer<char, Space::Device> &);
    template void Buffer<char, Space::Device>::copy_from<Space::Host>(const Buffer<char, Space::Host> &);
    template void Buffer<char, Space::Device>::copy_from<Space::Device>(const Buffer<char, Space::Device> &);

}