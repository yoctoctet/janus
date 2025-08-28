#include "cuda_manager.h"
#include <iostream>
#include <sstream>

// Constructor
CUDAManager::CUDAManager()
    : initialized(false), currentDeviceId(-1)
{
}

// Destructor
CUDAManager::~CUDAManager()
{
    if (initialized)
    {
        resetDevice();
    }
}

// Initialize CUDA device
bool CUDAManager::initializeDevice(int deviceId)
{
    try
    {
        // Check if CUDA is available
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess)
        {
            throw CUDAException("Failed to get CUDA device count", error);
        }

        if (deviceCount == 0)
        {
            throw CUDAException("No CUDA devices found");
        }

        if (deviceId >= deviceCount)
        {
            throw CUDAException("Invalid device ID: " + std::to_string(deviceId) +
                                " (available devices: " + std::to_string(deviceCount) + ")");
        }

        // Set the device
        error = cudaSetDevice(deviceId);
        if (error != cudaSuccess)
        {
            throw CUDAException("Failed to set CUDA device", error);
        }

        // Get device properties
        error = cudaGetDeviceProperties(&deviceProps, deviceId);
        if (error != cudaSuccess)
        {
            throw CUDAException("Failed to get device properties", error);
        }

        currentDeviceId = deviceId;
        initialized = true;

        return true;
    }
    catch (const CUDAException &e)
    {
        std::cerr << "CUDA Manager initialization failed: " << e.what() << std::endl;
        initialized = false;
        return false;
    }
}

// Reset CUDA device
bool CUDAManager::resetDevice()
{
    if (!initialized)
    {
        return true; // Already reset
    }

    cudaError_t error = cudaDeviceReset();
    if (error != cudaSuccess)
    {
        std::cerr << "Warning: Failed to reset CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    initialized = false;
    currentDeviceId = -1;
    return true;
}

// Get device count
int CUDAManager::getDeviceCount() const
{
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess)
    {
        std::cerr << "Warning: Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return 0;
    }
    return count;
}

// Get current device ID
int CUDAManager::getCurrentDeviceId() const
{
    checkInitialization();

    int deviceId = -1;
    cudaError_t error = cudaGetDevice(&deviceId);
    if (error != cudaSuccess)
    {
        throw CUDAException("Failed to get current device", error);
    }
    return deviceId;
}

// Get device properties
cudaDeviceProp CUDAManager::getDeviceProperties() const
{
    checkInitialization();
    return deviceProps;
}

// Get total memory
size_t CUDAManager::getTotalMemory() const
{
    checkInitialization();
    return deviceProps.totalGlobalMem;
}

// Get free memory
size_t CUDAManager::getFreeMemory() const
{
    size_t free = 0, total = 0;
    if (getMemoryInfo(free, total))
    {
        return free;
    }
    return 0;
}

// Get device name
std::string CUDAManager::getDeviceName() const
{
    checkInitialization();
    return std::string(deviceProps.name);
}

// Get compute capability
int CUDAManager::getComputeCapability() const
{
    checkInitialization();
    return deviceProps.major * 10 + deviceProps.minor;
}

// Allocate device memory (implementation)
bool CUDAManager::allocateDeviceMemoryImpl(void **ptr, size_t size)
{
    checkInitialization();

    if (ptr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMalloc(ptr, size);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Allocate host memory (implementation)
bool CUDAManager::allocateHostMemoryImpl(void **ptr, size_t size, bool pinned)
{
    checkInitialization();

    if (ptr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error;
    if (pinned)
    {
        error = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
    }
    else
    {
        *ptr = malloc(size);
        error = (*ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
    }

    if (error != cudaSuccess)
    {
        std::cerr << "Failed to allocate host memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Free device memory
bool CUDAManager::freeDeviceMemory(void *ptr)
{
    if (ptr == nullptr)
    {
        return true;
    }

    cudaError_t error = cudaFree(ptr);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to free device memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Free host memory
bool CUDAManager::freeHostMemory(void *ptr)
{
    if (ptr == nullptr)
    {
        return true;
    }

    cudaError_t error = cudaFreeHost(ptr);
    if (error != cudaSuccess)
    {
        // If cudaFreeHost fails, try regular free
        free(ptr);
    }

    return true;
}

// Copy host to device
bool CUDAManager::copyHostToDevice(const void *hostPtr, void *devicePtr, size_t size)
{
    checkInitialization();

    if (hostPtr == nullptr || devicePtr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy host to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Copy device to host
bool CUDAManager::copyDeviceToHost(const void *devicePtr, void *hostPtr, size_t size)
{
    checkInitialization();

    if (devicePtr == nullptr || hostPtr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy device to host: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Copy device to device
bool CUDAManager::copyDeviceToDevice(const void *srcPtr, void *dstPtr, size_t size)
{
    checkInitialization();

    if (srcPtr == nullptr || dstPtr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMemcpy(dstPtr, srcPtr, size, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy device to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Asynchronous copy host to device
bool CUDAManager::copyHostToDeviceAsync(const void *hostPtr, void *devicePtr, size_t size, cudaStream_t stream)
{
    checkInitialization();

    if (hostPtr == nullptr || devicePtr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMemcpyAsync(devicePtr, hostPtr, size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy host to device async: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Asynchronous copy device to host
bool CUDAManager::copyDeviceToHostAsync(const void *devicePtr, void *hostPtr, size_t size, cudaStream_t stream)
{
    checkInitialization();

    if (devicePtr == nullptr || hostPtr == nullptr || size == 0)
    {
        return false;
    }

    cudaError_t error = cudaMemcpyAsync(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy device to host async: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Synchronize device
bool CUDAManager::synchronize()
{
    checkInitialization();

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to synchronize device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Synchronize stream
bool CUDAManager::synchronizeStream(cudaStream_t stream)
{
    checkInitialization();

    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to synchronize stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Get memory information
bool CUDAManager::getMemoryInfo(size_t &free, size_t &total) const
{
    checkInitialization();

    cudaError_t error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Get last error
cudaError_t CUDAManager::getLastError() const
{
    return cudaGetLastError();
}

// Get error string
std::string CUDAManager::getErrorString(cudaError_t error) const
{
    return std::string(cudaGetErrorString(error));
}

// Check CUDA error
void CUDAManager::checkCudaError(cudaError_t error, const std::string &message) const
{
    if (error != cudaSuccess)
    {
        std::string fullMessage = message.empty() ? "CUDA error" : message;
        throw CUDAException(fullMessage, error);
    }
}

// Set device flags
bool CUDAManager::setDeviceFlags(unsigned int flags)
{
    cudaError_t error = cudaSetDeviceFlags(flags);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to set device flags: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// Get device attribute
bool CUDAManager::getDeviceAttribute(int &value, cudaDeviceAttr attr, int device) const
{
    if (device == -1)
    {
        device = currentDeviceId;
    }

    cudaError_t error = cudaDeviceGetAttribute(&value, attr, device);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to get device attribute: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

// Print device information
void CUDAManager::printDeviceInfo() const
{
    if (!initialized)
    {
        std::cout << "CUDA Manager not initialized" << std::endl;
        return;
    }

    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device ID: " << currentDeviceId << std::endl;
    std::cout << "Device Name: " << deviceProps.name << std::endl;
    std::cout << "Compute Capability: " << deviceProps.major << "." << deviceProps.minor << std::endl;
    std::cout << "Total Global Memory: " << deviceProps.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << deviceProps.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << deviceProps.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << deviceProps.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Warp Size: " << deviceProps.warpSize << std::endl;
    std::cout << "Max Grid Size: [" << deviceProps.maxGridSize[0] << ", "
              << deviceProps.maxGridSize[1] << ", " << deviceProps.maxGridSize[2] << "]" << std::endl;
    std::cout << "Max Block Size: [" << deviceProps.maxThreadsDim[0] << ", "
              << deviceProps.maxThreadsDim[1] << ", " << deviceProps.maxThreadsDim[2] << "]" << std::endl;

    size_t freeMem, totalMem;
    if (getMemoryInfo(freeMem, totalMem))
    {
        std::cout << "Free Memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Used Memory: " << (totalMem - freeMem) / (1024 * 1024) << " MB" << std::endl;
    }

    std::cout << "=================================" << std::endl;
}

// Check initialization
void CUDAManager::checkInitialization() const
{
    if (!initialized)
    {
        throw CUDAException("CUDA Manager not initialized. Call initializeDevice() first.");
    }
}