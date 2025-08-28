#include <gtest/gtest.h>
#include "configuration.h"
#include <fstream>
#include <cstdio>

// Test fixture for Configuration tests
class ConfigurationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a temporary config file for testing
        configFileName = "test_config.txt";
        createTestConfigFile();
    }

    void TearDown() override
    {
        // Clean up temporary file
        std::remove(configFileName.c_str());
    }

    void createTestConfigFile()
    {
        std::ofstream file(configFileName);
        file << "# Test configuration file\n";
        file << "numParticles = 1000\n";
        file << "maxIterations = 5000\n";
        file << "timeStep = 0.01\n";
        file << "softeningParameter = 0.001\n";
        file << "integrationScheme = leapfrog\n";
        file << "gridResolution = 32\n";
        file << "cutoffRadius = 2.0\n";
        file << "ewaldAlpha = 0.35\n";
        file << "interpolationScheme = cic\n";
        file << "deviceId = 0\n";
        file << "pinnedMemory = true\n";
        file << "unifiedMemory = false\n";
        file << "outputFrequency = 50\n";
        file << "outputPath = ./test_output/\n";
        file << "outputFormat = binary\n";
        file << "profilingEnabled = true\n";
        file << "logFile = test.log\n";
        file.close();
    }

    std::string configFileName;
};

// Test default constructor
TEST_F(ConfigurationTest, DefaultConstructor)
{
    Configuration config;

    EXPECT_EQ(config.numParticles, 1000u);
    EXPECT_EQ(config.maxIterations, 10000u);
    EXPECT_DOUBLE_EQ(config.timeStep, 0.001);
    EXPECT_DOUBLE_EQ(config.softeningParameter, 0.01);
    EXPECT_EQ(config.integrationScheme, IntegrationScheme::LEAPFROG);
    EXPECT_EQ(config.gridResolution, 64u);
    EXPECT_DOUBLE_EQ(config.cutoffRadius, 2.5);
    EXPECT_DOUBLE_EQ(config.ewaldAlpha, 0.5);
    EXPECT_EQ(config.interpolationScheme, "cic");
    EXPECT_EQ(config.deviceId, 0);
    EXPECT_TRUE(config.pinnedMemory);
    EXPECT_FALSE(config.unifiedMemory);
    EXPECT_EQ(config.outputFrequency, 100u);
    EXPECT_EQ(config.outputPath, "./output/");
    EXPECT_EQ(config.outputFormat, "binary");
    EXPECT_TRUE(config.profilingEnabled);
    EXPECT_EQ(config.logFile, "performance.log");
}

// Test loading valid configuration file
TEST_F(ConfigurationTest, LoadValidConfigFile)
{
    Configuration config;

    EXPECT_TRUE(config.loadFromFile(configFileName));

    EXPECT_EQ(config.numParticles, 1000u);
    EXPECT_EQ(config.maxIterations, 5000u);
    EXPECT_DOUBLE_EQ(config.timeStep, 0.01);
    EXPECT_DOUBLE_EQ(config.softeningParameter, 0.001);
    EXPECT_EQ(config.integrationScheme, IntegrationScheme::LEAPFROG);
    EXPECT_EQ(config.gridResolution, 32u);
    EXPECT_DOUBLE_EQ(config.cutoffRadius, 2.0);
    EXPECT_DOUBLE_EQ(config.ewaldAlpha, 0.35);
    EXPECT_EQ(config.interpolationScheme, "cic");
    EXPECT_EQ(config.deviceId, 0);
    EXPECT_TRUE(config.pinnedMemory);
    EXPECT_FALSE(config.unifiedMemory);
    EXPECT_EQ(config.outputFrequency, 50u);
    EXPECT_EQ(config.outputPath, "./test_output/");
    EXPECT_EQ(config.outputFormat, "binary");
    EXPECT_TRUE(config.profilingEnabled);
    EXPECT_EQ(config.logFile, "test.log");
}

// Test loading non-existent file
TEST_F(ConfigurationTest, LoadNonExistentFile)
{
    Configuration config;

    EXPECT_THROW(config.loadFromFile("nonexistent_file.txt"), ConfigurationException);
}

// Test parameter validation - valid parameters
TEST_F(ConfigurationTest, ValidateValidParameters)
{
    Configuration config;

    // Should not throw any exception
    EXPECT_NO_THROW(config.validate());
}

// Test parameter validation - invalid numParticles
TEST_F(ConfigurationTest, ValidateInvalidNumParticles)
{
    Configuration config;
    config.numParticles = 0;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid timeStep
TEST_F(ConfigurationTest, ValidateInvalidTimeStep)
{
    Configuration config;
    config.timeStep = 0.0;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid timeStep negative
TEST_F(ConfigurationTest, ValidateNegativeTimeStep)
{
    Configuration config;
    config.timeStep = -1.0;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid gridResolution
TEST_F(ConfigurationTest, ValidateInvalidGridResolution)
{
    Configuration config;
    config.gridResolution = 0;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid ewaldAlpha
TEST_F(ConfigurationTest, ValidateInvalidEwaldAlpha)
{
    Configuration config;
    config.ewaldAlpha = 0.0;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid interpolationScheme
TEST_F(ConfigurationTest, ValidateInvalidInterpolationScheme)
{
    Configuration config;
    config.interpolationScheme = "invalid";

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test parameter validation - invalid deviceId
TEST_F(ConfigurationTest, ValidateInvalidDeviceId)
{
    Configuration config;
    config.deviceId = -1;

    EXPECT_THROW(config.validate(), ConfigurationException);
}

// Test saving configuration to file
TEST_F(ConfigurationTest, SaveToFile)
{
    Configuration config;
    std::string saveFileName = "save_test.txt";

    // Modify some parameters
    config.numParticles = 2000;
    config.timeStep = 0.005;
    config.ewaldAlpha = 0.4;

    EXPECT_TRUE(config.saveToFile(saveFileName));

    // Load the saved file and verify
    Configuration loadedConfig;
    EXPECT_TRUE(loadedConfig.loadFromFile(saveFileName));

    EXPECT_EQ(loadedConfig.numParticles, 2000u);
    EXPECT_DOUBLE_EQ(loadedConfig.timeStep, 0.005);
    EXPECT_DOUBLE_EQ(loadedConfig.ewaldAlpha, 0.4);

    // Clean up
    std::remove(saveFileName.c_str());
}

// Test integration scheme parsing
TEST_F(ConfigurationTest, ParseIntegrationSchemeLeapfrog)
{
    Configuration config;
    EXPECT_EQ(config.parseIntegrationScheme("leapfrog"), IntegrationScheme::LEAPFROG);
    EXPECT_EQ(config.parseIntegrationScheme("LEAPFROG"), IntegrationScheme::LEAPFROG);
}

TEST_F(ConfigurationTest, ParseIntegrationSchemeVelocityVerlet)
{
    Configuration config;
    EXPECT_EQ(config.parseIntegrationScheme("velocity_verlet"), IntegrationScheme::VELOCITY_VERLET);
    EXPECT_EQ(config.parseIntegrationScheme("verlet"), IntegrationScheme::VELOCITY_VERLET);
}

TEST_F(ConfigurationTest, ParseIntegrationSchemeRK4)
{
    Configuration config;
    EXPECT_EQ(config.parseIntegrationScheme("runge_kutta_4"), IntegrationScheme::RUNGE_KUTTA_4);
    EXPECT_EQ(config.parseIntegrationScheme("rk4"), IntegrationScheme::RUNGE_KUTTA_4);
}

TEST_F(ConfigurationTest, ParseInvalidIntegrationScheme)
{
    Configuration config;
    EXPECT_THROW(config.parseIntegrationScheme("invalid"), ConfigurationException);
}

// Test configuration summary printing (doesn't throw)
TEST_F(ConfigurationTest, PrintSummary)
{
    Configuration config;

    // This should not throw any exception
    EXPECT_NO_THROW(config.printSummary());
}

// Test toJsonString (basic functionality)
TEST_F(ConfigurationTest, ToJsonString)
{
    Configuration config;
    config.numParticles = 500;
    config.timeStep = 0.002;

    std::string json = config.toJsonString();

    // Basic checks that JSON contains expected values
    EXPECT_NE(json.find("500"), std::string::npos);
    EXPECT_NE(json.find("0.002"), std::string::npos);
}

// Test loading configuration with comments and empty lines
TEST_F(ConfigurationTest, LoadConfigWithComments)
{
    std::string testFileName = "test_comments.txt";
    std::ofstream file(testFileName);
    file << "# This is a comment\n";
    file << "\n"; // Empty line
    file << "numParticles = 1500  # Inline comment\n";
    file << "timeStep = 0.02\n";
    file << "# Another comment\n";
    file << "ewaldAlpha = 0.6\n";
    file.close();

    Configuration config;
    EXPECT_TRUE(config.loadFromFile(testFileName));

    EXPECT_EQ(config.numParticles, 1500u);
    EXPECT_DOUBLE_EQ(config.timeStep, 0.02);
    EXPECT_DOUBLE_EQ(config.ewaldAlpha, 0.6);

    // Clean up
    std::remove(testFileName.c_str());
}

// Test loading configuration with malformed content
TEST_F(ConfigurationTest, LoadMalformedConfig)
{
    std::string testFileName = "test_malformed.txt";
    std::ofstream file(testFileName);
    file << "invalid line without equals\n";
    file << "numParticles = not_a_number\n";
    file.close();

    Configuration config;
    // Should throw an exception for malformed content
    EXPECT_THROW(config.loadFromFile(testFileName), ConfigurationException);

    // Clean up
    std::remove(testFileName.c_str());
}