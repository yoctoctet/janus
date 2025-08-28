#include "config.hpp"
#include <fstream>
#include <iostream>

namespace janus
{

    ConfigManager::ConfigManager(const std::filesystem::path &config_path)
        : config_path_(config_path)
    {
        // Load default config if file exists
        if (std::filesystem::exists(config_path_))
        {
            load_from_file(config_path_);
        }
    }

    bool ConfigManager::load_from_file(const std::filesystem::path &path)
    {
        try
        {
            auto config_file = toml::parse_file(path.string());

            if (auto name = config_file["simulation"]["name"])
            {
                config_.name = name.value<std::string>().value();
            }
            if (auto particles = config_file["simulation"]["num_particles"])
            {
                config_.num_particles = particles.value<int>().value();
            }
            if (auto time_step = config_file["simulation"]["time_step"])
            {
                config_.time_step = time_step.value<double>().value();
            }
            if (auto max_steps = config_file["simulation"]["max_steps"])
            {
                config_.max_steps = max_steps.value<int>().value();
            }
            if (auto use_gpu = config_file["simulation"]["use_gpu"])
            {
                config_.use_gpu = use_gpu.value<bool>().value();
            }

            if (auto block_size = config_file["gpu"]["block_size"])
            {
                config_.block_size = block_size.value<int>().value();
            }

            config_path_ = path;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading config file: " << e.what() << std::endl;
            return false;
        }
    }

    bool ConfigManager::save_to_file(const std::filesystem::path &path) const
    {
        try
        {
            std::ofstream file(path);
            if (!file.is_open())
            {
                return false;
            }

            file << "# Janus P3M Simulation Configuration\n\n";
            file << "[simulation]\n";
            file << "name = \"" << config_.name << "\"\n";
            file << "num_particles = " << config_.num_particles << "\n";
            file << "time_step = " << config_.time_step << "\n";
            file << "max_steps = " << config_.max_steps << "\n";
            file << "use_gpu = " << (config_.use_gpu ? "true" : "false") << "\n\n";
            file << "[gpu]\n";
            file << "block_size = " << config_.block_size << "\n";

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving config file: " << e.what() << std::endl;
            return false;
        }
    }

    nlohmann::json ConfigManager::to_json() const
    {
        return {
            {"simulation", {{"name", config_.name}, {"num_particles", config_.num_particles}, {"time_step", config_.time_step}, {"max_steps", config_.max_steps}, {"use_gpu", config_.use_gpu}}},
            {"gpu", {{"block_size", config_.block_size}}}};
    }

    void ConfigManager::from_json(const nlohmann::json &j)
    {
        if (j.contains("simulation"))
        {
            const auto &sim = j["simulation"];
            if (sim.contains("name"))
                config_.name = sim["name"];
            if (sim.contains("num_particles"))
                config_.num_particles = sim["num_particles"];
            if (sim.contains("time_step"))
                config_.time_step = sim["time_step"];
            if (sim.contains("max_steps"))
                config_.max_steps = sim["max_steps"];
            if (sim.contains("use_gpu"))
                config_.use_gpu = sim["use_gpu"];
        }

        if (j.contains("gpu"))
        {
            const auto &gpu = j["gpu"];
            if (gpu.contains("block_size"))
                config_.block_size = gpu["block_size"];
        }
    }

} // namespace janus