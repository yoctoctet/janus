#include "config.hpp"
#include <iostream>

void test_config_default_construction()
{
    janus::SimulationConfig config;
    if (config.name != "default_simulation")
    {
        std::cerr << "FAILED: Default name incorrect" << std::endl;
        return;
    }
    if (config.num_particles != 1000)
    {
        std::cerr << "FAILED: Default particles incorrect" << std::endl;
        return;
    }
    if (config.time_step != 0.001)
    {
        std::cerr << "FAILED: Default time_step incorrect" << std::endl;
        return;
    }
    if (config.dt_max != 1e-3)
    {
        std::cerr << "FAILED: Default dt_max incorrect" << std::endl;
        return;
    }
    if (config.dt_min != 1e-6)
    {
        std::cerr << "FAILED: Default dt_min incorrect" << std::endl;
        return;
    }
    if (config.theta != 0.5)
    {
        std::cerr << "FAILED: Default theta incorrect" << std::endl;
        return;
    }
    if (config.eta_v != 0.4)
    {
        std::cerr << "FAILED: Default eta_v incorrect" << std::endl;
        return;
    }
    if (config.eta_a != 0.4)
    {
        std::cerr << "FAILED: Default eta_a incorrect" << std::endl;
        return;
    }
    if (config.softening != 1e-3)
    {
        std::cerr << "FAILED: Default softening incorrect" << std::endl;
        return;
    }
    if (config.G != 1.0)
    {
        std::cerr << "FAILED: Default G incorrect" << std::endl;
        return;
    }
    if (config.sign_matrix != std::vector<int>{1, -1, -1, 1})
    {
        std::cerr << "FAILED: Default sign_matrix incorrect" << std::endl;
        return;
    }
    if (config.max_steps != 10000)
    {
        std::cerr << "FAILED: Default max_steps incorrect" << std::endl;
        return;
    }
    if (!config.use_gpu)
    {
        std::cerr << "FAILED: Default use_gpu incorrect" << std::endl;
        return;
    }
    std::cout << "PASSED: Default construction" << std::endl;
}

void test_config_manager_construction()
{
    janus::ConfigManager manager;
    const auto &config = manager.get_config();
    if (config.name != "default_simulation")
    {
        std::cerr << "FAILED: ConfigManager name incorrect" << std::endl;
        return;
    }
    if (config.num_particles != 1000)
    {
        std::cerr << "FAILED: ConfigManager particles incorrect" << std::endl;
        return;
    }
    if (config.dt_max != 1e-3)
    {
        std::cerr << "FAILED: ConfigManager dt_max incorrect" << std::endl;
        return;
    }
    if (config.G != 1.0)
    {
        std::cerr << "FAILED: ConfigManager G incorrect" << std::endl;
        return;
    }
    std::cout << "PASSED: ConfigManager construction" << std::endl;
}

void test_basic_arithmetic()
{
    // Trivial CPU-only test as required
    int a = 5;
    int b = 3;
    if (a + b != 8)
    {
        std::cerr << "FAILED: 5 + 3 != 8" << std::endl;
        return;
    }
    if (a * b != 15)
    {
        std::cerr << "FAILED: 5 * 3 != 15" << std::endl;
        return;
    }
    if (!(a > b))
    {
        std::cerr << "FAILED: 5 is not > 3" << std::endl;
        return;
    }
    if (a < b)
    {
        std::cerr << "FAILED: 5 is not < 3" << std::endl;
        return;
    }
    std::cout << "PASSED: Basic arithmetic" << std::endl;
}