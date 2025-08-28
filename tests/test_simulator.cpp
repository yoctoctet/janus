#include "simulator.hpp"
#include <iostream>
#include <vector>
#include <cmath>

void test_simulator_construction()
{
    janus::SimulationConfig config;
    config.num_particles = 100;
    config.use_gpu = false; // CPU-only test

    janus::Simulator simulator(config);
    if (simulator.is_initialized())
    {
        std::cerr << "FAILED: Simulator should not be initialized" << std::endl;
        return;
    }
    if (simulator.get_current_step() != 0)
    {
        std::cerr << "FAILED: Initial step should be 0" << std::endl;
        return;
    }
    std::cout << "PASSED: Simulator construction" << std::endl;
}

void test_simulator_initialization()
{
    janus::SimulationConfig config;
    config.num_particles = 50;
    config.use_gpu = false;

    janus::Simulator simulator(config);
    simulator.initialize();

    if (!simulator.is_initialized())
    {
        std::cerr << "FAILED: Simulator should be initialized" << std::endl;
        return;
    }
    if (simulator.get_current_step() != 0)
    {
        std::cerr << "FAILED: Initial step should be 0" << std::endl;
        return;
    }
    std::cout << "PASSED: Simulator initialization" << std::endl;
}

void test_trivial_math()
{
    // Another trivial CPU-only test
    double x = 3.14;
    double y = 2.71;
    if (!(x > y))
    {
        std::cerr << "FAILED: 3.14 is not > 2.71" << std::endl;
        return;
    }
    if (x + y != 5.85)
    {
        std::cerr << "FAILED: 3.14 + 2.71 != 5.85" << std::endl;
        return;
    }
    std::cout << "PASSED: Trivial math" << std::endl;
}

void test_gpu_hello_concept()
{
    // Test that demonstrates the GPU "hello" functionality conceptually
    int data[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; ++i)
    {
        data[i] = data[i] * 2 + 1;
    }

    if (data[0] != 3)
    {
        std::cerr << "FAILED: data[0] != 3" << std::endl;
        return;
    }
    if (data[1] != 5)
    {
        std::cerr << "FAILED: data[1] != 5" << std::endl;
        return;
    }
    if (data[2] != 7)
    {
        std::cerr << "FAILED: data[2] != 7" << std::endl;
        return;
    }
    if (data[3] != 9)
    {
        std::cerr << "FAILED: data[3] != 9" << std::endl;
        return;
    }
    if (data[4] != 11)
    {
        std::cerr << "FAILED: data[4] != 11" << std::endl;
        return;
    }
    std::cout << "PASSED: GPU hello concept" << std::endl;
}

void test_velocity_verlet_integration()
{
    // Test velocity Verlet integration with a=0 (no forces)
    // Positions should advance linearly and remain in [0, L)
    janus::SimulationConfig config;
    config.num_particles = 10;
    config.time_step = 0.01;
    config.use_gpu = false;
    config.max_steps = 1000;

    janus::Simulator simulator(config);
    simulator.initialize();

    // Store initial positions and velocities
    std::vector<double> initial_x = simulator.get_positions_x();
    std::vector<double> initial_y = simulator.get_positions_y();
    std::vector<double> initial_vx = simulator.get_velocities_x();
    std::vector<double> initial_vy = simulator.get_velocities_y();

    // Run simulation for several steps
    for (int step = 0; step < 100; ++step)
    {
        simulator.step();

        // Check that positions remain in [0, 1) (assuming L=1)
        const auto &current_x = simulator.get_positions_x();
        const auto &current_y = simulator.get_positions_y();

        for (size_t i = 0; i < current_x.size(); ++i)
        {
            if (current_x[i] < 0.0 || current_x[i] >= 1.0)
            {
                std::cerr << "FAILED: Position x[" << i << "] = " << current_x[i]
                          << " is not in [0, 1)" << std::endl;
                return;
            }
            if (current_y[i] < 0.0 || current_y[i] >= 1.0)
            {
                std::cerr << "FAILED: Position y[" << i << "] = " << current_y[i]
                          << " is not in [0, 1)" << std::endl;
                return;
            }
        }
    }

    // Check that positions advanced linearly (with wrapping)
    const auto &final_x = simulator.get_positions_x();
    const auto &final_y = simulator.get_positions_y();

    for (size_t i = 0; i < final_x.size(); ++i)
    {
        // Expected final position (accounting for wrapping)
        double expected_x = janus::Simulator::wrap(initial_x[i] + 100 * config.time_step * initial_vx[i], 1.0);
        double expected_y = janus::Simulator::wrap(initial_y[i] + 100 * config.time_step * initial_vy[i], 1.0);

        // Allow small numerical tolerance
        const double tolerance = 1e-10;
        if (std::abs(final_x[i] - expected_x) > tolerance)
        {
            std::cerr << "FAILED: Final x[" << i << "] = " << final_x[i]
                      << ", expected " << expected_x << std::endl;
            return;
        }
        if (std::abs(final_y[i] - expected_y) > tolerance)
        {
            std::cerr << "FAILED: Final y[" << i << "] = " << final_y[i]
                      << ", expected " << expected_y << std::endl;
            return;
        }
    }

    std::cout << "PASSED: Velocity Verlet integration with a=0" << std::endl;
}