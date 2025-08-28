#ifndef CUDA_MANAGER_H
#define CUDA_MANAGER_H

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

// Custom exception for CUDA errors
class CUDAException : public std::runtime_error
{
public:
    explicit CUDAException(const std::string &message, cudaError_t error = cudaSuccess)
        : std::runtime_error(message + " (CUDA error: " + cudaGetErrorString(error) + ")"),
          errorCode(error) {}

    cudaError_t getErrorCode() const { return errorCode; }

private:
    cudaError_t errorCode;
};

// CUDA device manager class
class CUDAManager
{
public:
    // Constructor and destructor
    CUDAManager();
    ~CUDAManager();

    // Device management
    bool initializeDevice(int deviceId = 0);
    bool resetDevice();
    int getDeviceCount() const;
    int getCurrentDeviceId() const;

    // Device properties
    cudaDeviceProp getDeviceProperties() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    std::string getDeviceName() const;
    int getComputeCapability() const;

    // Memory management
    template <typename T>
    bool allocateDeviceMemory(T **ptr, size_t count)
    {
        return allocateDeviceMemoryImpl(reinterpret_cast<void **>(ptr), count * sizeof(T));
    }

    template <typename T>
    bool allocateHostMemory(T **ptr, size_t count, bool pinned = true)
    {
        return allocateHostMemoryImpl(reinterpret_cast<void **>(ptr), count * sizeof(T), pinned);
    }

    bool freeDeviceMemory(void *ptr);
    bool freeHostMemory(void *ptr);

    // Data transfer
    bool copyHostToDevice(const void *hostPtr, void *devicePtr, size_t size);
    bool copyDeviceToHost(const void *devicePtr, void *hostPtr, size_t size);
    bool copyDeviceToDevice(const void *srcPtr, void *dstPtr, size_t size);

    // Asynchronous operations
    bool copyHostToDeviceAsync(const void *hostPtr, void *devicePtr, size_t size, cudaStream_t stream = 0);
    bool copyDeviceToHostAsync(const void *devicePtr, void *hostPtr, size_t size, cudaStream_t stream = 0);

    // Synchronization
    bool synchronize();
    bool synchronizeStream(cudaStream_t stream);

    // Memory information
    bool getMemoryInfo(size_t &free, size_t &total) const;

    // Error handling
    cudaError_t getLastError() const;
    std::string getErrorString(cudaError_t error) const;
    void checkCudaError(cudaError_t error, const std::string &message = "") const;

    // Performance monitoring
    bool setDeviceFlags(unsigned int flags);
    bool getDeviceAttribute(int &value, cudaDeviceAttr attr, int device = -1) const;

    // Utility functions
    void printDeviceInfo() const;
    bool isInitialized() const { return initialized; }

private:
    bool initialized;
    int currentDeviceId;
    cudaDeviceProp deviceProps;

    // Implementation functions for templates
    bool allocateDeviceMemoryImpl(void **ptr, size_t size);
    bool allocateHostMemoryImpl(void **ptr, size_t size, bool pinned);

    // Helper methods
    void checkInitialization() const;
};

#endif // CUDA_MANAGER_H