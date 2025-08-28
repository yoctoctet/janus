#include "configuration.h"
#include <algorithm>
#include <cctype>

// Constructor with default values
Configuration::Configuration()
{
    setDefaults();
}

// Load configuration from file
bool Configuration::loadFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw ConfigurationException("Cannot open configuration file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    return loadFromString(buffer.str());
}

// Load configuration from string (simple key-value format for now)
bool Configuration::loadFromString(const std::string &content)
{
    try
    {
        std::istringstream iss(content);
        std::string line;
        std::unordered_map<std::string, std::string> configMap;

        // Parse simple key=value format
        while (std::getline(iss, line))
        {
            // Remove comments
            size_t commentPos = line.find('#');
            if (commentPos != std::string::npos)
            {
                line = line.substr(0, commentPos);
            }

            // Trim whitespace
            line.erase(line.begin(), std::find_if(line.begin(), line.end(),
                                                  [](unsigned char ch)
                                                  { return !std::isspace(ch); }));
            line.erase(std::find_if(line.rbegin(), line.rend(),
                                    [](unsigned char ch)
                                    { return !std::isspace(ch); })
                           .base(),
                       line.end());

            if (line.empty())
                continue;

            // Parse key=value
            size_t equalsPos = line.find('=');
            if (equalsPos != std::string::npos)
            {
                std::string key = line.substr(0, equalsPos);
                std::string value = line.substr(equalsPos + 1);

                // Trim key and value
                key.erase(key.begin(), std::find_if(key.begin(), key.end(),
                                                    [](unsigned char ch)
                                                    { return !std::isspace(ch); }));
                key.erase(std::find_if(key.rbegin(), key.rend(),
                                       [](unsigned char ch)
                                       { return !std::isspace(ch); })
                              .base(),
                          key.end());

                value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                                        [](unsigned char ch)
                                                        { return !std::isspace(ch); }));
                value.erase(std::find_if(value.rbegin(), value.rend(),
                                         [](unsigned char ch)
                                         { return !std::isspace(ch); })
                                .base(),
                            value.end());

                configMap[key] = value;
            }
        }

        // Parse values and set configuration
        if (configMap.count("numParticles"))
            numParticles = std::stoull(configMap["numParticles"]);
        if (configMap.count("maxIterations"))
            maxIterations = std::stoull(configMap["maxIterations"]);
        if (configMap.count("timeStep"))
            timeStep = std::stod(configMap["timeStep"]);
        if (configMap.count("softeningParameter"))
            softeningParameter = std::stod(configMap["softeningParameter"]);
        if (configMap.count("integrationScheme"))
            integrationScheme = parseIntegrationScheme(configMap["integrationScheme"]);

        if (configMap.count("gridResolution"))
            gridResolution = std::stoull(configMap["gridResolution"]);
        if (configMap.count("cutoffRadius"))
            cutoffRadius = std::stod(configMap["cutoffRadius"]);
        if (configMap.count("ewaldAlpha"))
            ewaldAlpha = std::stod(configMap["ewaldAlpha"]);
        if (configMap.count("interpolationScheme"))
            interpolationScheme = configMap["interpolationScheme"];

        if (configMap.count("deviceId"))
            deviceId = std::stoi(configMap["deviceId"]);
        if (configMap.count("pinnedMemory"))
            pinnedMemory = configMap["pinnedMemory"] == "true";
        if (configMap.count("unifiedMemory"))
            unifiedMemory = configMap["unifiedMemory"] == "true";

        if (configMap.count("outputFrequency"))
            outputFrequency = std::stoull(configMap["outputFrequency"]);
        if (configMap.count("outputPath"))
            outputPath = configMap["outputPath"];
        if (configMap.count("outputFormat"))
            outputFormat = configMap["outputFormat"];

        if (configMap.count("profilingEnabled"))
            profilingEnabled = configMap["profilingEnabled"] == "true";
        if (configMap.count("logFile"))
            logFile = configMap["logFile"];

        validate();
        return true;
    }
    catch (const std::exception &e)
    {
        throw ConfigurationException("Error parsing configuration: " + std::string(e.what()));
    }
}

// Validate configuration parameters
void Configuration::validate() const
{
    if (numParticles == 0)
    {
        throw ConfigurationException("numParticles must be greater than 0");
    }
    if (maxIterations == 0)
    {
        throw ConfigurationException("maxIterations must be greater than 0");
    }
    if (timeStep <= 0.0)
    {
        throw ConfigurationException("timeStep must be greater than 0");
    }
    if (softeningParameter < 0.0)
    {
        throw ConfigurationException("softeningParameter must be non-negative");
    }
    if (gridResolution == 0)
    {
        throw ConfigurationException("gridResolution must be greater than 0");
    }
    if (cutoffRadius <= 0.0)
    {
        throw ConfigurationException("cutoffRadius must be greater than 0");
    }
    if (ewaldAlpha <= 0.0)
    {
        throw ConfigurationException("ewaldAlpha must be greater than 0");
    }
    if (interpolationScheme != "cic" && interpolationScheme != "ngp")
    {
        throw ConfigurationException("interpolationScheme must be 'cic' or 'ngp'");
    }
    if (deviceId < 0)
    {
        throw ConfigurationException("deviceId must be non-negative");
    }
    if (outputFrequency == 0)
    {
        throw ConfigurationException("outputFrequency must be greater than 0");
    }
}

// Save configuration to file
bool Configuration::saveToFile(const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        return false;
    }

    file << "# N-Body Gravitational Simulation Configuration\n";
    file << "# Generated automatically\n\n";

    file << "# Simulation Parameters\n";
    file << "numParticles = " << numParticles << "\n";
    file << "maxIterations = " << maxIterations << "\n";
    file << "timeStep = " << timeStep << "\n";
    file << "softeningParameter = " << softeningParameter << "\n";
    file << "integrationScheme = " << integrationSchemeToString(integrationScheme) << "\n\n";

    file << "# P3M Parameters\n";
    file << "gridResolution = " << gridResolution << "\n";
    file << "cutoffRadius = " << cutoffRadius << "\n";
    file << "ewaldAlpha = " << ewaldAlpha << "\n";
    file << "interpolationScheme = " << interpolationScheme << "\n\n";

    file << "# CUDA Parameters\n";
    file << "deviceId = " << deviceId << "\n";
    file << "pinnedMemory = " << (pinnedMemory ? "true" : "false") << "\n";
    file << "unifiedMemory = " << (unifiedMemory ? "true" : "false") << "\n\n";

    file << "# Output Parameters\n";
    file << "outputFrequency = " << outputFrequency << "\n";
    file << "outputPath = " << outputPath << "\n";
    file << "outputFormat = " << outputFormat << "\n\n";

    file << "# Profiling Parameters\n";
    file << "profilingEnabled = " << (profilingEnabled ? "true" : "false") << "\n";
    file << "logFile = " << logFile << "\n";

    file.close();
    return true;
}

// Get configuration as formatted string
std::string Configuration::toJsonString() const
{
    // For now, return a simple formatted string
    // In a real implementation, this would return JSON
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"numParticles\": " << numParticles << ",\n";
    ss << "  \"maxIterations\": " << maxIterations << ",\n";
    ss << "  \"timeStep\": " << timeStep << ",\n";
    ss << "  \"ewaldAlpha\": " << ewaldAlpha << ",\n";
    ss << "  \"gridResolution\": " << gridResolution << "\n";
    ss << "}\n";
    return ss.str();
}

// Print configuration summary
void Configuration::printSummary() const
{
    std::cout << "=== N-Body Simulation Configuration ===\n";
    std::cout << "Particles: " << numParticles << "\n";
    std::cout << "Max Iterations: " << maxIterations << "\n";
    std::cout << "Time Step: " << timeStep << "\n";
    std::cout << "Grid Resolution: " << gridResolution << "x" << gridResolution << "x" << gridResolution << "\n";
    std::cout << "Ewald Alpha: " << ewaldAlpha << "\n";
    std::cout << "Interpolation: " << interpolationScheme << "\n";
    std::cout << "CUDA Device: " << deviceId << "\n";
    std::cout << "=======================================\n";
}

// Set default values
void Configuration::setDefaults()
{
    numParticles = 1000;
    maxIterations = 10000;
    timeStep = 0.001;
    softeningParameter = 0.01;
    integrationScheme = IntegrationScheme::LEAPFROG;

    gridResolution = 64;
    cutoffRadius = 2.5;
    ewaldAlpha = 0.5;
    interpolationScheme = "cic";

    deviceId = 0;
    pinnedMemory = true;
    unifiedMemory = false;

    outputFrequency = 100;
    outputPath = "./output/";
    outputFormat = "binary";

    profilingEnabled = true;
    logFile = "performance.log";
}

// Parse integration scheme from string
IntegrationScheme Configuration::parseIntegrationScheme(const std::string &scheme) const
{
    std::string lowerScheme = scheme;
    std::transform(lowerScheme.begin(), lowerScheme.end(), lowerScheme.begin(), ::tolower);

    if (lowerScheme == "leapfrog")
    {
        return IntegrationScheme::LEAPFROG;
    }
    else if (lowerScheme == "velocity_verlet" || lowerScheme == "verlet")
    {
        return IntegrationScheme::VELOCITY_VERLET;
    }
    else if (lowerScheme == "runge_kutta_4" || lowerScheme == "rk4")
    {
        return IntegrationScheme::RUNGE_KUTTA_4;
    }
    else
    {
        throw ConfigurationException("Unknown integration scheme: " + scheme);
    }
}

// Convert integration scheme to string
std::string Configuration::integrationSchemeToString(IntegrationScheme scheme) const
{
    switch (scheme)
    {
    case IntegrationScheme::LEAPFROG:
        return "leapfrog";
    case IntegrationScheme::VELOCITY_VERLET:
        return "velocity_verlet";
    case IntegrationScheme::RUNGE_KUTTA_4:
        return "runge_kutta_4";
    default:
        return "unknown";
    }
}