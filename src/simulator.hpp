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

        // Testing utilities
        const std::vector<double> &get_positions_x() const { return x_; }
        const std::vector<double> &get_positions_y() const { return y_; }
        const std::vector<double> &get_velocities_x() const { return vx_; }
        const std::vector<double> &get_velocities_y() const { return vy_; }

    private:
        SimulationConfig config_;
        bool initialized_ = false;
        int current_step_ = 0;

        // GPU resources
        void *device_data_ = nullptr;
        size_t data_size_ = 0;
        double *d_x = nullptr, *d_y = nullptr, *d_vx = nullptr, *d_vy = nullptr;

        // Simulation data - SoA (Struct of Arrays) for coalesced access
        std::vector<double> x_, y_;   // positions (2D)
        std::vector<double> vx_, vy_; // velocities (2D)
        std::vector<double> m_;       // masses
        std::vector<int8_t> sgn_;     // signs
    };

} // namespace janus