#pragma once

#include <toml++/toml.h>
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <vector>

namespace janus
{

    struct SimulationConfig
    {
        std::string name = "default_simulation";
        int num_particles = 1000;
        double time_step = 0.001;
        int max_steps = 10000;
        bool use_gpu = true;

        // GPU specific settings
        int block_size = 256;

        // Integrator settings
        double dt_max = 1e-3;
        double dt_min = 1e-6;
        double theta = 0.5;
        double eta_v = 0.4;
        double eta_a = 0.4;
        double softening = 1e-3;

        // Physics settings
        double G = 1.0;
        std::vector<int> sign_matrix = {1, -1, -1, 1};
    };

    class ConfigManager
    {
    public:
        explicit ConfigManager(const std::filesystem::path &config_path = "config.toml");

        bool load_from_file(const std::filesystem::path &path);
        bool save_to_file(const std::filesystem::path &path) const;

        const SimulationConfig &get_config() const { return config_; }
        void set_config(const SimulationConfig &config) { config_ = config; }

        nlohmann::json to_json() const;
        void from_json(const nlohmann::json &j);

    private:
        SimulationConfig config_;
        std::filesystem::path config_path_;
    };

} // namespace janus