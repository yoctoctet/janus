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

            // Parse integrator settings
            if (auto integrator_table = config_file["integrator"])
            {
                if (auto dt_max = integrator_table["dt_max"])
                {
                    config_.dt_max = dt_max.value<double>().value();
                }
                if (auto dt_min = integrator_table["dt_min"])
                {
                    config_.dt_min = dt_min.value<double>().value();
                }
                if (auto theta = integrator_table["theta"])
                {
                    config_.theta = theta.value<double>().value();
                }
                if (auto eta_v = integrator_table["eta_v"])
                {
                    config_.eta_v = eta_v.value<double>().value();
                }
                if (auto eta_a = integrator_table["eta_a"])
                {
                    config_.eta_a = eta_a.value<double>().value();
                }
                if (auto softening = integrator_table["softening"])
                {
                    config_.softening = softening.value<double>().value();
                }
            }

            // Parse physics settings
            if (auto physics_table = config_file["physics"])
            {
                if (auto G = physics_table["G"])
                {
                    config_.G = G.value<double>().value();
                }
                if (auto sign_matrix_array = physics_table["sign_matrix"])
                {
                    if (sign_matrix_array.is_array())
                    {
                        config_.sign_matrix.clear();
                        for (auto &elem : *sign_matrix_array.as_array())
                        {
                            if (elem.is_integer())
                            {
                                config_.sign_matrix.push_back(elem.as_integer()->get());
                            }
                        }
                    }
                }
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
            file << "block_size = " << config_.block_size << "\n\n";

            file << "[integrator]\n";
            file << "dt_max = " << config_.dt_max << "\n";
            file << "dt_min = " << config_.dt_min << "\n";
            file << "theta = " << config_.theta << "\n";
            file << "eta_v = " << config_.eta_v << "\n";
            file << "eta_a = " << config_.eta_a << "\n";
            file << "softening = " << config_.softening << "\n\n";

            file << "[physics]\n";
            file << "G = " << config_.G << "\n";
            file << "sign_matrix = [";
            for (size_t i = 0; i < config_.sign_matrix.size(); ++i)
            {
                if (i > 0)
                    file << ", ";
                file << config_.sign_matrix[i];
            }
            file << "]\n";

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
            {"gpu", {{"block_size", config_.block_size}}},
            {"integrator", {{"dt_max", config_.dt_max}, {"dt_min", config_.dt_min}, {"theta", config_.theta}, {"eta_v", config_.eta_v}, {"eta_a", config_.eta_a}, {"softening", config_.softening}}},
            {"physics", {{"G", config_.G}, {"sign_matrix", config_.sign_matrix}}}};
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

        if (j.contains("integrator"))
        {
            const auto &integrator = j["integrator"];
            if (integrator.contains("dt_max"))
                config_.dt_max = integrator["dt_max"];
            if (integrator.contains("dt_min"))
                config_.dt_min = integrator["dt_min"];
            if (integrator.contains("theta"))
                config_.theta = integrator["theta"];
            if (integrator.contains("eta_v"))
                config_.eta_v = integrator["eta_v"];
            if (integrator.contains("eta_a"))
                config_.eta_a = integrator["eta_a"];
            if (integrator.contains("softening"))
                config_.softening = integrator["softening"];
        }

        if (j.contains("physics"))
        {
            const auto &physics = j["physics"];
            if (physics.contains("G"))
                config_.G = physics["G"];
            if (physics.contains("sign_matrix"))
                config_.sign_matrix = physics["sign_matrix"].get<std::vector<int>>();
        }
    }

} // namespace janus