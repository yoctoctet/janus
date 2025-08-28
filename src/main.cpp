#include "simulator.hpp"
#include "config.hpp"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <filesystem>

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

        app.add_option("--particles", num_particles, "Number of particles")
            ->check(CLI::PositiveNumber);
        app.add_option("--time-step", time_step, "Simulation time step")
            ->check(CLI::PositiveNumber);
        app.add_option("--max-steps", max_steps, "Maximum simulation steps")
            ->check(CLI::PositiveNumber);
        app.add_flag("--cpu-only", [&use_gpu](bool)
                     { use_gpu = false; }, "Run simulation on CPU only");

        CLI11_PARSE(app, argc, argv);

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

        // Override config with CLI parameters
        auto config = config_manager.get_config();
        config.num_particles = num_particles;
        config.time_step = time_step;
        config.max_steps = max_steps;
        config.use_gpu = use_gpu;

        // Calculate grid size if not set
        if (config.grid_size == 0)
        {
            config.grid_size = (config.num_particles + config.block_size - 1) / config.block_size;
        }

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