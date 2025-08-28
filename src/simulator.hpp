#pragma once

#include "config.hpp"
#include <memory>
#include <vector>

namespace janus
{

    class Simulator
    {
    public:
        explicit Simulator(const SimulationConfig &config);
        ~Simulator();

        // Delete copy operations
        Simulator(const Simulator &) = delete;
        Simulator &operator=(const Simulator &) = delete;

        // Allow move operations
        Simulator(Simulator &&) noexcept = default;
        Simulator &operator=(Simulator &&) noexcept = default;

        void initialize();
        void run();
        void step();
        void finalize();

        // GPU operations
        void initialize_gpu();
        void run_gpu_kernel();
        void cleanup_gpu();

        // Getters
        bool is_initialized() const { return initialized_; }
        int get_current_step() const { return current_step_; }
        const SimulationConfig &get_config() const { return config_; }

    private:
        SimulationConfig config_;
        bool initialized_ = false;
        int current_step_ = 0;

        // GPU resources
        void *device_data_ = nullptr;
        size_t data_size_ = 0;

        // Simulation data
        std::vector<float> positions_;
        std::vector<float> velocities_;
    };

} // namespace janus