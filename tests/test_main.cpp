#include <iostream>

// Forward declarations of test functions
void test_config_default_construction();
void test_config_manager_construction();
void test_basic_arithmetic();
void test_simulator_construction();
void test_simulator_initialization();
void test_trivial_math();
void test_gpu_hello_concept();
void test_velocity_verlet_integration();
void test_adaptive_timestep_calculation();
void test_dt_limiter_bounds_verification();
void test_spiked_velocity_acceleration_scenarios();

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::cout << "Running Janus P3M Tests..." << std::endl;
    std::cout << "========================" << std::endl;

    // Run all tests
    test_config_default_construction();
    test_config_manager_construction();
    test_basic_arithmetic();
    test_simulator_construction();
    test_simulator_initialization();
    test_trivial_math();
    test_gpu_hello_concept();
    test_velocity_verlet_integration();
    test_adaptive_timestep_calculation();
    test_dt_limiter_bounds_verification();
    test_spiked_velocity_acceleration_scenarios();

    std::cout << "========================" << std::endl;
    std::cout << "All tests completed!" << std::endl;

    return 0;
}