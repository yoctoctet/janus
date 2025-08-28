#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

// Custom exception for configuration errors
class ConfigurationException : public std::runtime_error
{
public:
    explicit ConfigurationException(const std::string &message)
        : std::runtime_error(message) {}
};

// Integration scheme enumeration
enum class IntegrationScheme
{
    LEAPFROG,
    VELOCITY_VERLET,
    RUNGE_KUTTA_4
};

// Main configuration class for the N-Body simulation
class Configuration
{
public:
    // Simulation parameters
    size_t numParticles;
    size_t maxIterations;
    double timeStep;
    double softeningParameter;
    IntegrationScheme integrationScheme;

    // P3M parameters
    size_t gridResolution;
    double cutoffRadius;
    double ewaldAlpha;
    std::string interpolationScheme;

    // CUDA parameters
    int deviceId;
    bool pinnedMemory;
    bool unifiedMemory;

    // Output parameters
    size_t outputFrequency;
    std::string outputPath;
    std::string outputFormat;

    // Profiling parameters
    bool profilingEnabled;
    std::string logFile;

    // Constructor with default values
    Configuration();

    // Load configuration from JSON file
    bool loadFromFile(const std::string &filename);

    // Load configuration from JSON string
    bool loadFromString(const std::string &jsonString);

    // Validate configuration parameters
    void validate() const;

    // Save configuration to JSON file
    bool saveToFile(const std::string &filename) const;

    // Get configuration as JSON string
    std::string toJsonString() const;

    // Print configuration summary
    void printSummary() const;

public:
    // Helper method to parse integration scheme from string (public for testing)
    IntegrationScheme parseIntegrationScheme(const std::string &scheme) const;

private:
    // Helper method to set default values
    void setDefaults();

    // Helper method to convert integration scheme to string
    std::string integrationSchemeToString(IntegrationScheme scheme) const;
};

#endif // CONFIGURATION_H