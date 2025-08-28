#include "simulator.hpp"
#include "gpu_kernels.cuh"
#include <iostream>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

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

        // Initialize simulation data - SoA format for coalesced access
        x_.resize(config_.num_particles, 0.0);
        y_.resize(config_.num_particles, 0.0);
        vx_.resize(config_.num_particles, 0.0);
        vy_.resize(config_.num_particles, 0.0);
        m_.resize(config_.num_particles, 1.0);
        sgn_.resize(config_.num_particles, 1);

        // Initialize with some test data (2D only)
        for (int i = 0; i < config_.num_particles; ++i)
        {
            x_[i] = static_cast<double>(i) * 0.1;
            y_[i] = static_cast<double>(i) * 0.1;

            vx_[i] = 0.01;
            vy_[i] = 0.01;

            m_[i] = 1.0;
            sgn_[i] = 1;
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
            // CPU implementation - Velocity Verlet integration (2D only)
            const double dt = config_.time_step;
            const double L = 1.0; // Box size, assuming unit box for now

            for (size_t i = 0; i < x_.size(); ++i)
            {
                // Half kick: v += 0.5*dt*a(x)
                // For now, a(x) = 0 (no forces), so velocity unchanged
                // vx_[i] += 0.5 * dt * ax[i];
                // vy_[i] += 0.5 * dt * ay[i];

                // Drift + wrap: x = wrap(x + dt*v, L)
                x_[i] = wrap(x_[i] + dt * vx_[i], L);
                y_[i] = wrap(y_[i] + dt * vy_[i], L);

                // Recompute a(x) - stub implementation (zero forces)
                // ax[i] = 0.0;
                // ay[i] = 0.0;

                // Half kick: v += 0.5*dt*a(x)
                // For now, a(x) = 0, so velocity unchanged
                // vx_[i] += 0.5 * dt * ax[i];
                // vy_[i] += 0.5 * dt * ay[i];

                // Placeholder for dt limiter logging (next task)
                // Log which dt limiter would fire based on current conditions
                // This will be implemented in the next task
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

        x_.clear();
        y_.clear();
        vx_.clear();
        vy_.clear();
        m_.clear();
        sgn_.clear();

        initialized_ = false;
        spdlog::info("Simulator finalized");
    }

    void Simulator::initialize_gpu()
    {
        spdlog::info("Initializing GPU resources...");

        size_t data_size = x_.size() * sizeof(double);

        // Allocate separate device memory for coalesced access
        cudaError_t err;
        err = allocate_device_memory(reinterpret_cast<void **>(&d_x), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to allocate GPU memory for x: {}", cudaGetErrorString(err));
            return;
        }

        err = allocate_device_memory(reinterpret_cast<void **>(&d_y), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to allocate GPU memory for y: {}", cudaGetErrorString(err));
            return;
        }

        err = allocate_device_memory(reinterpret_cast<void **>(&d_vx), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to allocate GPU memory for vx: {}", cudaGetErrorString(err));
            return;
        }

        err = allocate_device_memory(reinterpret_cast<void **>(&d_vy), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to allocate GPU memory for vy: {}", cudaGetErrorString(err));
            return;
        }

        // Copy data to device
        err = copy_to_device(d_x, x_.data(), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy x to GPU: {}", cudaGetErrorString(err));
            return;
        }

        err = copy_to_device(d_y, y_.data(), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy y to GPU: {}", cudaGetErrorString(err));
            return;
        }

        err = copy_to_device(d_vx, vx_.data(), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy vx to GPU: {}", cudaGetErrorString(err));
            return;
        }

        err = copy_to_device(d_vy, vy_.data(), data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy vy to GPU: {}", cudaGetErrorString(err));
            return;
        }

        spdlog::info("GPU resources initialized");
    }

    void Simulator::run_gpu_kernel()
    {
        if (!d_x || !d_y || !d_vx || !d_vy)
        {
            spdlog::error("GPU not initialized");
            return;
        }

#ifdef JANUS_USE_NVTX
        nvtxRangePush("GPU Kernel");
#endif

        // Launch GPU kernel using the wrapper function
        cudaError_t err = launch_update_positions_kernel(d_x, d_y, d_vx, d_vy,
                                                         config_.time_step, config_.num_particles);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to launch GPU kernel: {}", cudaGetErrorString(err));
            return;
        }

        // Copy updated positions back from device to host
        size_t data_size = config_.num_particles * sizeof(double);
        err = copy_from_device(x_.data(), d_x, data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy x from GPU: {}", cudaGetErrorString(err));
            return;
        }

        err = copy_from_device(y_.data(), d_y, data_size);
        if (err != cudaSuccess)
        {
            spdlog::error("Failed to copy y from GPU: {}", cudaGetErrorString(err));
            return;
        }

        spdlog::info("GPU kernel simulation completed");

#ifdef JANUS_USE_NVTX
        nvtxRangePop();
#endif
    }

    void Simulator::cleanup_gpu()
    {
        if (d_x)
        {
            free_device_memory(d_x);
            d_x = nullptr;
        }
        if (d_y)
        {
            free_device_memory(d_y);
            d_y = nullptr;
        }
        if (d_vx)
        {
            free_device_memory(d_vx);
            d_vx = nullptr;
        }
        if (d_vy)
        {
            free_device_memory(d_vy);
            d_vy = nullptr;
        }
        spdlog::info("GPU resources cleaned up");
    }

    // Utility function implementations
    double Simulator::wrap(double u, double L)
    {
        return u - L * std::floor(u / L);
    }

    double Simulator::min_image(double dx, double L)
    {
        return dx - L * std::round(dx / L);
    }

} // namespace janus