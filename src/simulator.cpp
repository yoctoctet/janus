#include "simulator.hpp"
#include "gpu_kernels.cuh"
#include <iostream>
#include <spdlog/spdlog.h>
#include <algorithm>

#ifdef JANUS_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace janus
{

    Simulator::Simulator(const SimulationConfig &config)
        : config_(config)
    {
        spdlog::info("Initializing Janus Simulator with {} particles",
                     config_.num_particles);
    }

    Simulator::~Simulator()
    {
        if (initialized_)
        {
            finalize();
        }
    }

    void Simulator::initialize()
    {
        if (initialized_)
        {
            spdlog::warn("Simulator already initialized");
            return;
        }

        spdlog::info("Initializing simulation data...");

        // Initialize simulation data
        positions_.resize(config_.num_particles * 3, 0.0f);
        velocities_.resize(config_.num_particles * 3, 0.0f);

        // Initialize with some test data
        for (int i = 0; i < config_.num_particles; ++i)
        {
            positions_[i * 3] = static_cast<float>(i) * 0.1f;
            positions_[i * 3 + 1] = static_cast<float>(i) * 0.1f;
            positions_[i * 3 + 2] = static_cast<float>(i) * 0.1f;

            velocities_[i * 3] = 0.01f;
            velocities_[i * 3 + 1] = 0.01f;
            velocities_[i * 3 + 2] = 0.01f;
        }

        if (config_.use_gpu)
        {
            initialize_gpu();
        }

        initialized_ = true;
        spdlog::info("Simulator initialized successfully");
    }

    void Simulator::run()
    {
        if (!initialized_)
        {
            spdlog::error("Simulator not initialized");
            return;
        }

        spdlog::info("Starting simulation with {} steps", config_.max_steps);

#ifdef JANUS_USE_NVTX
        nvtxRangePush("Simulation Loop");
#endif

        for (int step_idx = 0; step_idx < config_.max_steps; ++step_idx)
        {
            current_step_ = step_idx;
            this->step();

            if (step_idx % 1000 == 0)
            {
                spdlog::info("Completed step {}/{}", step_idx, config_.max_steps);
            }
        }

#ifdef JANUS_USE_NVTX
        nvtxRangePop();
#endif

        spdlog::info("Simulation completed");
    }

    void Simulator::step()
    {
#ifdef JANUS_USE_NVTX
        nvtxRangePush("Simulation Step");
#endif

        if (config_.use_gpu)
        {
            run_gpu_kernel();
        }
        else
        {
            // CPU fallback - simple position update
            for (size_t i = 0; i < positions_.size(); ++i)
            {
                positions_[i] += velocities_[i] * static_cast<float>(config_.time_step);
            }
        }

#ifdef JANUS_USE_NVTX
        nvtxRangePop();
#endif
    }

    void Simulator::finalize()
    {
        if (!initialized_)
        {
            return;
        }

        spdlog::info("Finalizing simulator...");

        if (config_.use_gpu)
        {
            cleanup_gpu();
        }

        positions_.clear();
        velocities_.clear();

        initialized_ = false;
        spdlog::info("Simulator finalized");
    }

    void Simulator::initialize_gpu()
    {
        spdlog::info("Initializing GPU resources...");

        data_size_ = positions_.size() * sizeof(float);

        // Allocate device memory
        cudaError_t err;
        err = allocate_device_memory(&device_data_, data_size_ * 2); // positions + velocities
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to allocate GPU memory: {}", cudaGetErrorString(err));
            return;
        }

        // Copy data to device
        err = copy_to_device(device_data_, positions_.data(), data_size_);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy positions to GPU: {}", cudaGetErrorString(err));
            return;
        }

        err = copy_to_device(static_cast<char *>(device_data_) + data_size_,
                             velocities_.data(), data_size_);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy velocities to GPU: {}", cudaGetErrorString(err));
            return;
        }

        spdlog::info("GPU resources initialized");
    }

    void Simulator::run_gpu_kernel()
    {
        if (!device_data_)
        {
            spdlog::error("GPU not initialized");
            return;
        }

#ifdef JANUS_USE_NVTX
        nvtxRangePush("GPU Kernel");
#endif

        // For now, just perform CPU computation on GPU-allocated data
        // In a real implementation, this would launch CUDA kernels
        float *positions = static_cast<float *>(device_data_);
        float *velocities = static_cast<float *>(device_data_) + config_.num_particles;

        for (int i = 0; i < config_.num_particles; ++i)
        {
            positions[i] += velocities[i] * static_cast<float>(config_.time_step);
        }

        spdlog::info("GPU kernel simulation completed (CPU fallback for now)");

#ifdef JANUS_USE_NVTX
        nvtxRangePop();
#endif
    }

    void Simulator::cleanup_gpu()
    {
        if (device_data_)
        {
            free_device_memory(device_data_);
            device_data_ = nullptr;
            spdlog::info("GPU resources cleaned up");
        }
    }

} // namespace janus