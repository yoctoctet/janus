#include "simulator.hpp"
#include "config.hpp"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sstream>

// Forward declaration for GPU test function
int run_gpu_verification_test();

// Function to apply override values to config
void apply_override(janus::SimulationConfig &config, const std::string &override_str)
{
    std::istringstream iss(override_str);
    std::string path, value_str;
    if (std::getline(iss, path, '=') && std::getline(iss, value_str))
    {
        // Parse dotted path (e.g., "integrator.dt_max")
        std::istringstream path_iss(path);
        std::string section, key;
        if (std::getline(path_iss, section, '.') && std::getline(path_iss, key))
        {
            if (section == "integrator")
            {
                if (key == "dt_max")
                    config.dt_max = std::stod(value_str);
                else if (key == "dt_min")
                    config.dt_min = std::stod(value_str);
                else if (key == "theta")
                    config.theta = std::stod(value_str);
                else if (key == "eta_v")
                    config.eta_v = std::stod(value_str);
                else if (key == "eta_a")
                    config.eta_a = std::stod(value_str);
                else if (key == "softening")
                    config.softening = std::stod(value_str);
            }
            else if (section == "physics")
            {
                if (key == "G")
                    config.G = std::stod(value_str);
                // Note: sign_matrix override not implemented for simplicity
            }
            else if (section == "simulation")
            {
                if (key == "name")
                    config.name = value_str;
                else if (key == "num_particles")
                    config.num_particles = std::stoi(value_str);
                else if (key == "time_step")
                    config.time_step = std::stod(value_str);
                else if (key == "max_steps")
                    config.max_steps = std::stoi(value_str);
                else if (key == "use_gpu")
                    config.use_gpu = (value_str == "true");
            }
            else if (section == "gpu")
            {
                if (key == "block_size")
                    config.block_size = std::stoi(value_str);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        // Initialize logging
        spdlog::set_level(spdlog::level::info);
        spdlog::info("Starting Janus P3M Simulator");

        // CLI setup
        CLI::App app{"Janus P3M - GPU-accelerated particle simulation"};

        std::string config_file = "config.toml";
        std::string output_dir = "output";
        bool verbose = false;
        bool quiet = false;

        app.add_option("-c,--config", config_file, "Configuration file path")
            ->check(CLI::ExistingFile);
        app.add_option("-o,--output", output_dir, "Output directory");
        app.add_flag("-v,--verbose", verbose, "Enable verbose logging");
        app.add_flag("-q,--quiet", quiet, "Disable logging");

        // Simulation parameters
        int num_particles = 1000;
        double time_step = 0.001;
        int max_steps = 10000;
        bool use_gpu = true;

        // Integrator parameters
        double dt_max = 1e-3;
        double dt_min = 1e-6;
        double theta = 0.5;
        double eta_v = 0.4;
        double eta_a = 0.4;
        double softening = 1e-3;

        // Physics parameters
        double G = 1.0;
        std::vector<int> sign_matrix = {1, -1, -1, 1};

        app.add_option("--particles", num_particles, "Number of particles")
            ->check(CLI::PositiveNumber);
        app.add_option("--time-step", time_step, "Simulation time step")
            ->check(CLI::PositiveNumber);
        app.add_option("--max-steps", max_steps, "Maximum simulation steps")
            ->check(CLI::PositiveNumber);
        app.add_flag("--cpu-only", [&use_gpu](bool)
                     { use_gpu = false; }, "Run simulation on CPU only");

        // Integrator options
        app.add_option("--dt-max", dt_max, "Maximum time step")
            ->check(CLI::PositiveNumber);
        app.add_option("--dt-min", dt_min, "Minimum time step")
            ->check(CLI::PositiveNumber);
        app.add_option("--theta", theta, "Theta parameter");
        app.add_option("--eta-v", eta_v, "Velocity damping factor");
        app.add_option("--eta-a", eta_a, "Acceleration damping factor");
        app.add_option("--softening", softening, "Softening parameter")
            ->check(CLI::PositiveNumber);

        // Physics options
        app.add_option("--G", G, "Gravitational constant");
        app.add_option("--sign-matrix", sign_matrix, "Sign matrix for physics");

        // Generic override option
        std::vector<std::string> overrides;
        app.add_option("--override", overrides, "Override config values (e.g., integrator.dt_max=5e-4)")
            ->allow_extra_args(true);

        bool run_gpu_test = false;
        app.add_flag("--gpu-test", run_gpu_test, "Run GPU verification test");

        CLI11_PARSE(app, argc, argv);

        // Handle GPU test mode
        if (run_gpu_test)
        {
            return run_gpu_verification_test();
        }

        // Configure logging
        if (verbose)
        {
            spdlog::set_level(spdlog::level::debug);
        }
        else if (quiet)
        {
            spdlog::set_level(spdlog::level::off);
        }

        // Load or create configuration
        janus::ConfigManager config_manager(config_file);

        // Apply generic overrides
        auto config = config_manager.get_config();
        for (const auto &override_str : overrides)
        {
            apply_override(config, override_str);
        }

        // Override config with CLI parameters (only if explicitly provided)
        // Note: CLI11 doesn't provide easy way to check if option was provided,
        // so we'll use a different approach - check against defaults
        if (num_particles != 1000)
            config.num_particles = num_particles;
        if (time_step != 0.001)
            config.time_step = time_step;
        if (max_steps != 10000)
            config.max_steps = max_steps;
        if (!use_gpu)
            config.use_gpu = use_gpu; // use_gpu default is true, so only override if false

        // Override integrator parameters (only if different from defaults)
        if (dt_max != 1e-3)
            config.dt_max = dt_max;
        if (dt_min != 1e-6)
            config.dt_min = dt_min;
        if (theta != 0.5)
            config.theta = theta;
        if (eta_v != 0.4)
            config.eta_v = eta_v;
        if (eta_a != 0.4)
            config.eta_a = eta_a;
        if (softening != 1e-3)
            config.softening = softening;

        // Override physics parameters (only if different from defaults)
        if (G != 1.0)
            config.G = G;
        if (sign_matrix != std::vector<int>{1, -1, -1, 1})
            config.sign_matrix = sign_matrix;

        spdlog::info("Configuration loaded:");
        spdlog::info("  Particles: {}", config.num_particles);
        spdlog::info("  Time step: {}", config.time_step);
        spdlog::info("  Max steps: {}", config.max_steps);
        spdlog::info("  Use GPU: {}", config.use_gpu ? "Yes" : "No");

#ifdef JANUS_USE_NVTX
        spdlog::info("  NVTX profiling: Enabled");
#endif

#ifdef JANUS_FAST_MATH
        spdlog::info("  Fast math: Enabled");
#endif

#ifdef JANUS_DETERMINISTIC
        spdlog::info("  Deterministic: Enabled");
#endif

        // Create output directory
        std::filesystem::create_directories(output_dir);

        // Create and run simulator
        janus::Simulator simulator(config);
        simulator.initialize();
        simulator.run();
        simulator.finalize();

        // Save final configuration
        config_manager.set_config(config);
        config_manager.save_to_file(std::filesystem::path(output_dir) / "final_config.toml");

        spdlog::info("Simulation completed successfully");
        return 0;
    }
    catch (const std::exception &e)
    {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}