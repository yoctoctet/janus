#include <gtest/gtest.h>
#include "particle_system.h"
#include "cuda_manager.h"
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdio>

// Test fixture for ParticleSystem tests
class ParticleSystemTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cudaMgr = new CUDAManager();
        // Try to initialize CUDA, but don't fail if not available
        try
        {
            cudaMgr->initializeDevice(0);
        }
        catch (const CUDAException &)
        {
            // CUDA not available, tests will skip GPU operations
        }
    }

    void TearDown() override
    {
        delete cudaMgr;
        cudaMgr = nullptr;

        // Clean up test files
        std::remove("test_particles.txt");
        std::remove("test_particles_invalid.txt");
    }

    CUDAManager *cudaMgr;

    // Helper method to create test particle data
    void createTestParticleData(size_t numParticles,
                                std::vector<double> &positions,
                                std::vector<double> &velocities,
                                std::vector<double> &masses)
    {
        positions.resize(numParticles * 3);
        velocities.resize(numParticles * 3);
        masses.resize(numParticles);

        for (size_t i = 0; i < numParticles; ++i)
        {
            // Simple test data
            positions[i * 3] = static_cast<double>(i);
            positions[i * 3 + 1] = static_cast<double>(i) * 2.0;
            positions[i * 3 + 2] = static_cast<double>(i) * 3.0;

            velocities[i * 3] = static_cast<double>(i) * 0.1;
            velocities[i * 3 + 1] = static_cast<double>(i) * 0.2;
            velocities[i * 3 + 2] = static_cast<double>(i) * 0.3;

            masses[i] = 1.0 + static_cast<double>(i) * 0.5;
        }
    }
};

// Test constructor
TEST_F(ParticleSystemTest, Constructor)
{
    const size_t numParticles = 100;
    ParticleSystem ps(numParticles);

    EXPECT_EQ(ps.getNumParticles(), numParticles);
    EXPECT_EQ(ps.getHostPositions().size(), numParticles * 3);
    EXPECT_EQ(ps.getHostVelocities().size(), numParticles * 3);
    EXPECT_EQ(ps.getHostMasses().size(), numParticles);
    EXPECT_EQ(ps.getHostForces().size(), numParticles * 3);
    EXPECT_EQ(ps.getHostPotentials().size(), numParticles);
}

// Test initialization with random data
TEST_F(ParticleSystemTest, InitializeRandom)
{
    const size_t numParticles = 50;
    ParticleSystem ps(numParticles);

    EXPECT_TRUE(ps.initializeRandom(*cudaMgr, 10.0));

    // Check that data is initialized
    const auto &positions = ps.getHostPositions();
    const auto &velocities = ps.getHostVelocities();
    const auto &masses = ps.getHostMasses();

    EXPECT_EQ(positions.size(), numParticles * 3);
    EXPECT_EQ(velocities.size(), numParticles * 3);
    EXPECT_EQ(masses.size(), numParticles);

    // Check that masses are uniform (default 1.0)
    for (double mass : masses)
    {
        EXPECT_DOUBLE_EQ(mass, 1.0);
    }

    // Check that forces and potentials are zero
    const auto &forces = ps.getHostForces();
    const auto &potentials = ps.getHostPotentials();

    for (double force : forces)
    {
        EXPECT_DOUBLE_EQ(force, 0.0);
    }
    for (double potential : potentials)
    {
        EXPECT_DOUBLE_EQ(potential, 0.0);
    }
}

// Test initialization with uniform grid
TEST_F(ParticleSystemTest, InitializeUniformGrid)
{
    const size_t numParticles = 8; // 2x2x2 grid
    ParticleSystem ps(numParticles);

    EXPECT_TRUE(ps.initializeUniformGrid(*cudaMgr, 1.0));

    const auto &positions = ps.getHostPositions();
    const auto &velocities = ps.getHostVelocities();
    const auto &masses = ps.getHostMasses();

    // Check that all arrays have correct size
    EXPECT_EQ(positions.size(), numParticles * 3);
    EXPECT_EQ(velocities.size(), numParticles * 3);
    EXPECT_EQ(masses.size(), numParticles);

    // Check that masses are uniform
    for (double mass : masses)
    {
        EXPECT_DOUBLE_EQ(mass, 1.0);
    }

    // Check that forces and potentials are zero
    const auto &forces = ps.getHostForces();
    const auto &potentials = ps.getHostPotentials();
    for (double force : forces)
    {
        EXPECT_DOUBLE_EQ(force, 0.0);
    }
    for (double potential : potentials)
    {
        EXPECT_DOUBLE_EQ(potential, 0.0);
    }

    // Check that positions are within reasonable bounds
    for (double pos : positions)
    {
        EXPECT_GE(pos, -1.0);
        EXPECT_LE(pos, 1.0);
    }
}

// Test data setters
TEST_F(ParticleSystemTest, SetData)
{
    const size_t numParticles = 10;
    ParticleSystem ps(numParticles);

    // Create test data
    std::vector<double> testPositions(numParticles * 3, 1.0);
    std::vector<double> testVelocities(numParticles * 3, 2.0);
    std::vector<double> testMasses(numParticles, 3.0);
    std::vector<double> testForces(numParticles * 3, 4.0);
    std::vector<double> testPotentials(numParticles, 5.0);

    // Set data
    EXPECT_TRUE(ps.setPositions(testPositions));
    EXPECT_TRUE(ps.setVelocities(testVelocities));
    EXPECT_TRUE(ps.setMasses(testMasses));
    EXPECT_TRUE(ps.setForces(testForces));
    EXPECT_TRUE(ps.setPotentials(testPotentials));

    // Verify data
    const auto &positions = ps.getHostPositions();
    const auto &velocities = ps.getHostVelocities();
    const auto &masses = ps.getHostMasses();
    const auto &forces = ps.getHostForces();
    const auto &potentials = ps.getHostPotentials();

    for (double pos : positions)
        EXPECT_DOUBLE_EQ(pos, 1.0);
    for (double vel : velocities)
        EXPECT_DOUBLE_EQ(vel, 2.0);
    for (double mass : masses)
        EXPECT_DOUBLE_EQ(mass, 3.0);
    for (double force : forces)
        EXPECT_DOUBLE_EQ(force, 4.0);
    for (double pot : potentials)
        EXPECT_DOUBLE_EQ(pot, 5.0);
}

// Test data setters with wrong size
TEST_F(ParticleSystemTest, SetDataWrongSize)
{
    const size_t numParticles = 10;
    ParticleSystem ps(numParticles);

    // Wrong size arrays
    std::vector<double> wrongPositions(numParticles * 2, 1.0); // Should be * 3
    std::vector<double> wrongMasses(numParticles * 3, 1.0);    // Should be * 1

    EXPECT_FALSE(ps.setPositions(wrongPositions));
    EXPECT_FALSE(ps.setMasses(wrongMasses));
}

// Test reset functions
TEST_F(ParticleSystemTest, ResetFunctions)
{
    const size_t numParticles = 5;
    ParticleSystem ps(numParticles);

    // Set some non-zero forces and potentials
    std::vector<double> forces(numParticles * 3, 1.0);
    std::vector<double> potentials(numParticles, 2.0);
    ps.setForces(forces);
    ps.setPotentials(potentials);

    // Reset
    ps.resetForces();
    ps.resetPotentials();

    // Check they are zero
    const auto &hostForces = ps.getHostForces();
    const auto &hostPotentials = ps.getHostPotentials();

    for (double force : hostForces)
    {
        EXPECT_DOUBLE_EQ(force, 0.0);
    }
    for (double potential : hostPotentials)
    {
        EXPECT_DOUBLE_EQ(potential, 0.0);
    }
}

// Test physical property calculations
TEST_F(ParticleSystemTest, PhysicalProperties)
{
    const size_t numParticles = 2;
    ParticleSystem ps(numParticles);

    // Set up simple test case
    std::vector<double> positions = {0, 0, 0, 1, 0, 0};  // Two particles at origin and (1,0,0)
    std::vector<double> velocities = {1, 0, 0, 0, 1, 0}; // Velocities (1,0,0) and (0,1,0)
    std::vector<double> masses = {2.0, 3.0};             // Masses 2.0 and 3.0

    ps.setPositions(positions);
    ps.setVelocities(velocities);
    ps.setMasses(masses);

    // Test total mass
    EXPECT_DOUBLE_EQ(ps.getTotalMass(), 5.0);

    // Test center of mass
    double com[3];
    ps.getCenterOfMass(com);
    EXPECT_DOUBLE_EQ(com[0], 0.6); // (2*0 + 3*1) / 5 = 0.6
    EXPECT_DOUBLE_EQ(com[1], 0.0);
    EXPECT_DOUBLE_EQ(com[2], 0.0);

    // Test total momentum
    double momentum[3];
    ps.getTotalMomentum(momentum);
    EXPECT_DOUBLE_EQ(momentum[0], 2.0); // 2*1 + 3*0 = 2.0
    EXPECT_DOUBLE_EQ(momentum[1], 3.0); // 2*0 + 3*1 = 3.0
    EXPECT_DOUBLE_EQ(momentum[2], 0.0);

    // Test kinetic energy
    double expectedKE = 0.5 * 2.0 * (1 * 1 + 0 * 0 + 0 * 0) + 0.5 * 3.0 * (0 * 0 + 1 * 1 + 0 * 0);
    EXPECT_DOUBLE_EQ(ps.getKineticEnergy(), expectedKE);
}

// Test data validation
TEST_F(ParticleSystemTest, DataValidation)
{
    const size_t numParticles = 3;
    ParticleSystem ps(numParticles);

    // Valid data should pass
    EXPECT_TRUE(ps.validateData());

    // Set invalid mass (negative)
    std::vector<double> badMasses = {1.0, -1.0, 1.0};
    ps.setMasses(badMasses);
    EXPECT_FALSE(ps.validateData());

    // Reset to valid state
    std::vector<double> goodMasses = {1.0, 1.0, 1.0};
    ps.setMasses(goodMasses);
    EXPECT_TRUE(ps.validateData());
}

// Test file I/O
TEST_F(ParticleSystemTest, FileIO)
{
    const size_t numParticles = 3;
    ParticleSystem ps(numParticles);

    // Set up test data
    std::vector<double> positions = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> velocities = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<double> masses = {1.0, 2.0, 3.0};
    std::vector<double> forces = {0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3};
    std::vector<double> potentials = {10.0, 20.0, 30.0};

    ps.setPositions(positions);
    ps.setVelocities(velocities);
    ps.setMasses(masses);
    ps.setForces(forces);
    ps.setPotentials(potentials);

    // Save to file
    EXPECT_TRUE(ps.saveToFile("test_particles.txt"));

    // Create new particle system and load
    ParticleSystem ps2(numParticles);
    EXPECT_TRUE(ps2.loadFromFile("test_particles.txt"));

    // Verify data matches
    const auto &loadedPositions = ps2.getHostPositions();
    const auto &loadedVelocities = ps2.getHostVelocities();
    const auto &loadedMasses = ps2.getHostMasses();
    const auto &loadedForces = ps2.getHostForces();
    const auto &loadedPotentials = ps2.getHostPotentials();

    for (size_t i = 0; i < positions.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(loadedPositions[i], positions[i]);
    }
    for (size_t i = 0; i < velocities.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(loadedVelocities[i], velocities[i]);
    }
    for (size_t i = 0; i < masses.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(loadedMasses[i], masses[i]);
        EXPECT_DOUBLE_EQ(loadedForces[i], forces[i]);
        EXPECT_DOUBLE_EQ(loadedPotentials[i], potentials[i]);
    }
}

// Test loading from non-existent file
TEST_F(ParticleSystemTest, LoadNonExistentFile)
{
    ParticleSystem ps(10);
    EXPECT_FALSE(ps.loadFromFile("nonexistent_file.txt"));
}

// Test loading with wrong particle count
TEST_F(ParticleSystemTest, LoadWrongParticleCount)
{
    // Create file with different particle count
    std::ofstream file("test_particles_invalid.txt");
    file << "5\n"; // Wrong count
    file << "1 2 3 0.1 0.2 0.3 1.0 0.1 0.2 0.3 10.0\n";
    file.close();

    ParticleSystem ps(3); // Expected 3 particles
    EXPECT_FALSE(ps.loadFromFile("test_particles_invalid.txt"));
}

// Test device memory allocation (requires CUDA)
TEST_F(ParticleSystemTest, DeviceMemoryAllocation)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for device memory tests";
    }

    const size_t numParticles = 10;
    ParticleSystem ps(numParticles);

    // Should not have device memory initially
    EXPECT_FALSE(ps.isDeviceMemoryAllocated());

    // Allocate device memory
    EXPECT_TRUE(ps.allocateDeviceMemory(*cudaMgr));
    EXPECT_TRUE(ps.isDeviceMemoryAllocated());

    // Check that device pointers are not null
    EXPECT_NE(ps.getDevicePositionsX(), nullptr);
    EXPECT_NE(ps.getDevicePositionsY(), nullptr);
    EXPECT_NE(ps.getDevicePositionsZ(), nullptr);
    EXPECT_NE(ps.getDeviceMasses(), nullptr);

    // Free device memory
    EXPECT_TRUE(ps.freeDeviceMemory(*cudaMgr));
    EXPECT_FALSE(ps.isDeviceMemoryAllocated());
}

// Test data transfer to/from device (requires CUDA)
TEST_F(ParticleSystemTest, DeviceDataTransfer)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for device transfer tests";
    }

    const size_t numParticles = 5;
    ParticleSystem ps(numParticles);

    // Set up test data
    std::vector<double> testPositions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<double> testMasses = {1.0, 2.0, 3.0, 4.0, 5.0};
    ps.setPositions(testPositions);
    ps.setMasses(testMasses);

    // Allocate device memory
    EXPECT_TRUE(ps.allocateDeviceMemory(*cudaMgr));

    // Copy to device
    EXPECT_TRUE(ps.copyToDevice(*cudaMgr));

    // Modify host data
    std::vector<double> modifiedPositions = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
    ps.setPositions(modifiedPositions);

    // Copy back from device (should restore original data)
    EXPECT_TRUE(ps.copyToHost(*cudaMgr));

    // Verify data was restored
    const auto &restoredPositions = ps.getHostPositions();
    for (size_t i = 0; i < testPositions.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(restoredPositions[i], testPositions[i]);
    }

    // Clean up
    EXPECT_TRUE(ps.freeDeviceMemory(*cudaMgr));
}

// Test print summary (should not throw)
TEST_F(ParticleSystemTest, PrintSummary)
{
    const size_t numParticles = 3;
    ParticleSystem ps(numParticles);

    // Should not throw any exception
    EXPECT_NO_THROW(ps.printSummary());
}

// Test destructor cleanup
TEST_F(ParticleSystemTest, DestructorCleanup)
{
    // Create particle system in a scope
    {
        ParticleSystem *ps = new ParticleSystem(10);
        ps->initializeRandom(*cudaMgr, 1.0);

        if (cudaMgr->isInitialized())
        {
            ps->allocateDeviceMemory(*cudaMgr);
        }

        // Delete (should cleanup properly)
        delete ps;
    }

    // If we get here without crashing, the test passes
    EXPECT_TRUE(true);
}

// Test edge case: zero particles
TEST_F(ParticleSystemTest, ZeroParticles)
{
    ParticleSystem ps(0);

    EXPECT_EQ(ps.getNumParticles(), 0u);
    EXPECT_EQ(ps.getHostPositions().size(), 0u);
    EXPECT_EQ(ps.getHostMasses().size(), 0u);
    EXPECT_DOUBLE_EQ(ps.getTotalMass(), 0.0);
    EXPECT_TRUE(ps.validateData());
}

// Test edge case: single particle
TEST_F(ParticleSystemTest, SingleParticle)
{
    ParticleSystem ps(1);

    EXPECT_EQ(ps.getNumParticles(), 1u);

    // Test center of mass with single particle
    std::vector<double> positions = {1.0, 2.0, 3.0};
    std::vector<double> masses = {5.0};
    ps.setPositions(positions);
    ps.setMasses(masses);

    double com[3];
    ps.getCenterOfMass(com);
    EXPECT_DOUBLE_EQ(com[0], 1.0);
    EXPECT_DOUBLE_EQ(com[1], 2.0);
    EXPECT_DOUBLE_EQ(com[2], 3.0);

    EXPECT_DOUBLE_EQ(ps.getTotalMass(), 5.0);
}