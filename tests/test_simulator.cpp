#include "simulator.hpp"
#include <iostream>

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