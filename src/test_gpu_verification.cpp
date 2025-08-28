#include "simulator.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <cstring>
#include <vector>

namespace janus
{

    // Test function to verify GPU implementation matches CPU byte-for-byte
    bool test_gpu_cpu_consistency()
    {
        const int N = 100000; // 100k particles as specified
        const double time_step = 0.001;

        std::cout << "Testing GPU vs CPU consistency with N=" << N << " particles..." << std::endl;

        // Create configurations
        SimulationConfig cpu_config;
        cpu_config.num_particles = N;
        cpu_config.time_step = time_step;
        cpu_config.max_steps = 1; // Just one step
        cpu_config.use_gpu = false;

        SimulationConfig gpu_config = cpu_config;
        gpu_config.use_gpu = true;

        // Create simulators
        Simulator cpu_simulator(cpu_config);
        Simulator gpu_simulator(gpu_config);

        // Initialize simulators
        cpu_simulator.initialize();
        gpu_simulator.initialize();

        // Run one step on CPU
        std::cout << "Running CPU step..." << std::endl;
        cpu_simulator.step();

        // Run one step on GPU
        std::cout << "Running GPU step..." << std::endl;
        gpu_simulator.step();

        // Get position data for comparison
        const auto &cpu_x = cpu_simulator.get_positions_x();
        const auto &cpu_y = cpu_simulator.get_positions_y();
        const auto &gpu_x = gpu_simulator.get_positions_x();
        const auto &gpu_y = gpu_simulator.get_positions_y();

        // Compare results byte-for-byte
        bool results_match = true;
        int first_mismatch_idx = -1;

        std::cout << "Comparing results..." << std::endl;

        for (int i = 0; i < N; ++i)
        {
            // Compare X positions
            if (std::memcmp(&cpu_x[i], &gpu_x[i], sizeof(double)) != 0)
            {
                results_match = false;
                if (first_mismatch_idx == -1)
                {
                    first_mismatch_idx = i;
                    std::cout << std::fixed << std::setprecision(15);
                    std::cout << "First mismatch at particle " << i << ":" << std::endl;
                    std::cout << "  CPU X: " << cpu_x[i] << std::endl;
                    std::cout << "  GPU X: " << gpu_x[i] << std::endl;
                    std::cout << "  Difference: " << std::abs(cpu_x[i] - gpu_x[i]) << std::endl;
                }
            }

            // Compare Y positions
            if (std::memcmp(&cpu_y[i], &gpu_y[i], sizeof(double)) != 0)
            {
                results_match = false;
                if (first_mismatch_idx == -1)
                {
                    first_mismatch_idx = i;
                    std::cout << std::fixed << std::setprecision(15);
                    std::cout << "First mismatch at particle " << i << ":" << std::endl;
                    std::cout << "  CPU Y: " << cpu_y[i] << std::endl;
                    std::cout << "  GPU Y: " << gpu_y[i] << std::endl;
                    std::cout << "  Difference: " << std::abs(cpu_y[i] - gpu_y[i]) << std::endl;
                }
            }
        }

        cpu_simulator.finalize();
        gpu_simulator.finalize();

        if (results_match)
        {
            std::cout << "✓ SUCCESS: CPU and GPU results match byte-for-byte!" << std::endl;
            return true;
        }
        else
        {
            std::cout << "✗ FAILURE: CPU and GPU results differ!" << std::endl;
            return false;
        }
    }

} // namespace janus

// Main function for standalone test execution
int run_gpu_verification_test()
{
    std::cout << "GPU Verification Test" << std::endl;
    std::cout << "====================" << std::endl;

    if (janus::test_gpu_cpu_consistency())
    {
        std::cout << "✓ GPU verification completed successfully" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "✗ GPU verification failed" << std::endl;
        return 1;
    }
}