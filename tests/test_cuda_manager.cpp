#include <gtest/gtest.h>
#include "cuda_manager.h"
#include <vector>
#include <iostream>

// Test fixture for CUDA Manager tests
class CUDAManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cudaMgr = new CUDAManager();
    }

    void TearDown() override
    {
        delete cudaMgr;
        cudaMgr = nullptr;
    }

    CUDAManager *cudaMgr;
};

// Test device count
TEST_F(CUDAManagerTest, GetDeviceCount)
{
    int count = cudaMgr->getDeviceCount();
    // At least one device should be available (can be 0 if no GPU)
    EXPECT_GE(count, 0);
}

// Test device initialization with valid device
TEST_F(CUDAManagerTest, InitializeValidDevice)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));
        EXPECT_TRUE(cudaMgr->isInitialized());
        EXPECT_EQ(cudaMgr->getCurrentDeviceId(), 0);
    }
    else
    {
        // Skip test if no CUDA devices available
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test device initialization with invalid device ID
TEST_F(CUDAManagerTest, InitializeInvalidDevice)
{
    int deviceCount = cudaMgr->getDeviceCount();

    // Try to initialize with an invalid device ID
    EXPECT_FALSE(cudaMgr->initializeDevice(deviceCount));
    EXPECT_FALSE(cudaMgr->isInitialized());
}

// Test device reset
TEST_F(CUDAManagerTest, ResetDevice)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));
        EXPECT_TRUE(cudaMgr->isInitialized());

        EXPECT_TRUE(cudaMgr->resetDevice());
        EXPECT_FALSE(cudaMgr->isInitialized());
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test device properties after initialization
TEST_F(CUDAManagerTest, DeviceProperties)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Test device properties
        cudaDeviceProp props = cudaMgr->getDeviceProperties();
        EXPECT_GT(props.totalGlobalMem, 0u);
        EXPECT_GE(props.multiProcessorCount, 1);
        EXPECT_GE(props.warpSize, 1);

        // Test convenience methods
        EXPECT_GT(cudaMgr->getTotalMemory(), 0u);
        EXPECT_FALSE(cudaMgr->getDeviceName().empty());
        EXPECT_GE(cudaMgr->getComputeCapability(), 10); // At least 1.0
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test memory allocation and deallocation
TEST_F(CUDAManagerTest, MemoryAllocation)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Test device memory allocation
        void *d_ptr = nullptr;
        size_t size = 1024 * sizeof(float);

        EXPECT_TRUE(cudaMgr->allocateDeviceMemory(&d_ptr, size));
        EXPECT_NE(d_ptr, nullptr);

        EXPECT_TRUE(cudaMgr->freeDeviceMemory(d_ptr));

        // Test host memory allocation (pinned)
        void *h_ptr = nullptr;
        EXPECT_TRUE(cudaMgr->allocateHostMemory(&h_ptr, size, true));
        EXPECT_NE(h_ptr, nullptr);

        EXPECT_TRUE(cudaMgr->freeHostMemory(h_ptr));

        // Test host memory allocation (unpinned)
        h_ptr = nullptr;
        EXPECT_TRUE(cudaMgr->allocateHostMemory(&h_ptr, size, false));
        EXPECT_NE(h_ptr, nullptr);

        EXPECT_TRUE(cudaMgr->freeHostMemory(h_ptr));
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test data transfer between host and device
TEST_F(CUDAManagerTest, DataTransfer)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Prepare test data
        const size_t dataSize = 1000;
        std::vector<float> hostData(dataSize, 3.14159f);
        std::vector<float> hostResult(dataSize, 0.0f);

        // Allocate device memory
        float *d_data = nullptr;
        EXPECT_TRUE(cudaMgr->allocateDeviceMemory(&d_data, dataSize * sizeof(float)));

        // Copy host to device
        EXPECT_TRUE(cudaMgr->copyHostToDevice(hostData.data(), d_data, dataSize * sizeof(float)));

        // Copy device to host
        EXPECT_TRUE(cudaMgr->copyDeviceToHost(d_data, hostResult.data(), dataSize * sizeof(float)));

        // Verify data integrity
        for (size_t i = 0; i < dataSize; ++i)
        {
            EXPECT_FLOAT_EQ(hostData[i], hostResult[i]);
        }

        // Clean up
        EXPECT_TRUE(cudaMgr->freeDeviceMemory(d_data));
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test device to device transfer
TEST_F(CUDAManagerTest, DeviceToDeviceTransfer)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Prepare test data
        const size_t dataSize = 500;
        std::vector<double> hostData(dataSize, 2.71828);

        // Allocate device memory
        double *d_src = nullptr;
        double *d_dst = nullptr;
        EXPECT_TRUE(cudaMgr->allocateDeviceMemory(&d_src, dataSize * sizeof(double)));
        EXPECT_TRUE(cudaMgr->allocateDeviceMemory(&d_dst, dataSize * sizeof(double)));

        // Copy host to device (source)
        EXPECT_TRUE(cudaMgr->copyHostToDevice(hostData.data(), d_src, dataSize * sizeof(double)));

        // Copy device to device
        EXPECT_TRUE(cudaMgr->copyDeviceToDevice(d_src, d_dst, dataSize * sizeof(double)));

        // Copy back to host and verify
        std::vector<double> hostResult(dataSize, 0.0);
        EXPECT_TRUE(cudaMgr->copyDeviceToHost(d_dst, hostResult.data(), dataSize * sizeof(double)));

        for (size_t i = 0; i < dataSize; ++i)
        {
            EXPECT_DOUBLE_EQ(hostData[i], hostResult[i]);
        }

        // Clean up
        EXPECT_TRUE(cudaMgr->freeDeviceMemory(d_src));
        EXPECT_TRUE(cudaMgr->freeDeviceMemory(d_dst));
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test memory information
TEST_F(CUDAManagerTest, MemoryInfo)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        size_t freeMem, totalMem;
        EXPECT_TRUE(cudaMgr->getMemoryInfo(freeMem, totalMem));

        EXPECT_GT(totalMem, 0u);
        EXPECT_GE(freeMem, 0u);
        EXPECT_LE(freeMem, totalMem);

        // Test convenience method
        EXPECT_EQ(cudaMgr->getFreeMemory(), freeMem);
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test synchronization
TEST_F(CUDAManagerTest, Synchronization)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Device synchronization should succeed
        EXPECT_TRUE(cudaMgr->synchronize());

        // Default stream synchronization
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        EXPECT_TRUE(cudaMgr->synchronizeStream(stream));
        cudaStreamDestroy(stream);
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test error handling
TEST_F(CUDAManagerTest, ErrorHandling)
{
    // Test with uninitialized manager
    EXPECT_THROW(cudaMgr->getDeviceProperties(), CUDAException);
    EXPECT_THROW(cudaMgr->getCurrentDeviceId(), CUDAException);

    // Test CUDA error string
    cudaError_t error = cudaErrorInvalidDevice;
    std::string errorStr = cudaMgr->getErrorString(error);
    EXPECT_FALSE(errorStr.empty());
}

// Test device attributes
TEST_F(CUDAManagerTest, DeviceAttributes)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        int value;
        // Test getting a common device attribute
        EXPECT_TRUE(cudaMgr->getDeviceAttribute(value, cudaDevAttrMaxThreadsPerBlock));
        EXPECT_GT(value, 0);
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test null pointer handling
TEST_F(CUDAManagerTest, NullPointerHandling)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Test with zero size (template functions can't handle nullptr directly)
        float *ptr = nullptr;
        EXPECT_FALSE(cudaMgr->allocateDeviceMemory(&ptr, 0));
        EXPECT_FALSE(cudaMgr->allocateHostMemory(&ptr, 0));

        // Test data transfer with null pointers
        float dummyData = 1.0f;
        float *dummyPtr = &dummyData;
        EXPECT_FALSE(cudaMgr->copyHostToDevice(nullptr, dummyPtr, sizeof(float)));
        EXPECT_FALSE(cudaMgr->copyHostToDevice(dummyPtr, nullptr, sizeof(float)));
        EXPECT_FALSE(cudaMgr->copyDeviceToHost(nullptr, dummyPtr, sizeof(float)));
        EXPECT_FALSE(cudaMgr->copyDeviceToHost(dummyPtr, nullptr, sizeof(float)));
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test multiple device initialization
TEST_F(CUDAManagerTest, MultipleInitialization)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        // Initialize multiple times
        EXPECT_TRUE(cudaMgr->initializeDevice(0));
        EXPECT_TRUE(cudaMgr->isInitialized());

        // Re-initializing should work
        EXPECT_TRUE(cudaMgr->initializeDevice(0));
        EXPECT_TRUE(cudaMgr->isInitialized());

        // Reset and re-initialize
        EXPECT_TRUE(cudaMgr->resetDevice());
        EXPECT_FALSE(cudaMgr->isInitialized());

        EXPECT_TRUE(cudaMgr->initializeDevice(0));
        EXPECT_TRUE(cudaMgr->isInitialized());
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}

// Test print device info (doesn't throw)
TEST_F(CUDAManagerTest, PrintDeviceInfo)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        EXPECT_TRUE(cudaMgr->initializeDevice(0));

        // Should not throw any exception
        EXPECT_NO_THROW(cudaMgr->printDeviceInfo());
    }
    else
    {
        // Even without devices, should not throw
        EXPECT_NO_THROW(cudaMgr->printDeviceInfo());
    }
}

// Test destructor cleanup
TEST_F(CUDAManagerTest, DestructorCleanup)
{
    int deviceCount = cudaMgr->getDeviceCount();

    if (deviceCount > 0)
    {
        CUDAManager *tempMgr = new CUDAManager();
        EXPECT_TRUE(tempMgr->initializeDevice(0));

        // Allocate some memory
        void *d_ptr = nullptr;
        EXPECT_TRUE(tempMgr->allocateDeviceMemory(&d_ptr, 1024));

        // Delete manager (should cleanup automatically)
        delete tempMgr;

        // Note: We can't test the actual cleanup, but at least ensure no exceptions
    }
    else
    {
        GTEST_SKIP() << "No CUDA devices available for testing";
    }
}