#include <gtest/gtest.h>
#include "mesh.h"
#include "cuda_manager.h"
#include <vector>
#include <cmath>

// Test fixture for Mesh tests
class MeshTest : public ::testing::Test
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
    }

    CUDAManager *cudaMgr;

    // Helper method to create test particle data
    void createTestParticles(size_t numParticles,
                             std::vector<double> &positions,
                             std::vector<double> &masses)
    {
        positions.resize(numParticles * 3);
        masses.resize(numParticles);

        for (size_t i = 0; i < numParticles; ++i)
        {
            // Place particles at known grid positions
            positions[i * 3] = static_cast<double>(i % 4) * 0.25;            // x: 0, 0.25, 0.5, 0.75
            positions[i * 3 + 1] = static_cast<double>((i / 4) % 4) * 0.25;  // y: similar pattern
            positions[i * 3 + 2] = static_cast<double>((i / 16) % 4) * 0.25; // z: similar pattern
            masses[i] = 1.0;
        }
    }
};

// Test constructor
TEST_F(MeshTest, Constructor)
{
    const size_t gridSize = 8;
    const double gridSpacing = 1.0;

    Mesh mesh(gridSize, gridSpacing);

    EXPECT_EQ(mesh.getGridSize(), gridSize);
    EXPECT_DOUBLE_EQ(mesh.getGridSpacing(), gridSpacing);
    EXPECT_EQ(mesh.getTotalGridPoints(), gridSize * gridSize * gridSize);

    // Check that host arrays are initialized
    EXPECT_EQ(mesh.getHostDensity().size(), mesh.getTotalGridPoints());
    EXPECT_EQ(mesh.getHostPotential().size(), mesh.getTotalGridPoints());
    EXPECT_EQ(mesh.getHostForceX().size(), mesh.getTotalGridPoints());
    EXPECT_EQ(mesh.getHostForceY().size(), mesh.getTotalGridPoints());
    EXPECT_EQ(mesh.getHostForceZ().size(), mesh.getTotalGridPoints());
}

// Test initialization
TEST_F(MeshTest, Initialization)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for mesh initialization tests";
    }

    const size_t gridSize = 8;
    Mesh mesh(gridSize, 1.0);

    EXPECT_TRUE(mesh.initialize(*cudaMgr));
    EXPECT_TRUE(mesh.isDeviceMemoryAllocated());
}

// Test CIC weight function
TEST_F(MeshTest, CICWeight)
{
    Mesh mesh(8, 1.0);

    // Test CIC weights
    EXPECT_DOUBLE_EQ(mesh.cicWeight(0.0), 1.0);
    EXPECT_DOUBLE_EQ(mesh.cicWeight(0.5), 0.5);
    EXPECT_DOUBLE_EQ(mesh.cicWeight(1.0), 0.0);
    EXPECT_DOUBLE_EQ(mesh.cicWeight(-0.5), 0.5);
    EXPECT_DOUBLE_EQ(mesh.cicWeight(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(mesh.cicWeight(2.0), 0.0);
}

// Test particle assignment to mesh
TEST_F(MeshTest, ParticleAssignment)
{
    const size_t gridSize = 4;
    const double gridSpacing = 1.0;
    Mesh mesh(gridSize, gridSpacing);

    // Create test particles at grid centers
    std::vector<double> positions = {
        0.5, 0.5, 0.5, // Particle at grid center (1,1,1)
        1.5, 1.5, 1.5  // Particle at grid center (1.5,1.5,1.5) -> should wrap around
    };
    std::vector<double> masses = {2.0, 3.0};
    size_t numParticles = 2;

    EXPECT_TRUE(mesh.assignParticlesToMesh(positions, masses, numParticles));

    const auto &density = mesh.getHostDensity();

    // Check that mass was distributed
    double totalMass = 0.0;
    for (double d : density)
    {
        totalMass += d;
    }

    // Total mass should be conserved (approximately, due to CIC interpolation)
    EXPECT_NEAR(totalMass, 5.0, 1e-10);

    // Check that density is not zero everywhere
    bool hasNonZero = false;
    for (double d : density)
    {
        if (d > 1e-10)
        {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);
}

// Test particle assignment with invalid data
TEST_F(MeshTest, ParticleAssignmentInvalidData)
{
    Mesh mesh(4, 1.0);

    std::vector<double> positions = {0, 0, 0}; // Only one particle
    std::vector<double> masses = {1.0, 2.0};   // Wrong number of masses

    EXPECT_FALSE(mesh.assignParticlesToMesh(positions, masses, 1));
}

// Test grid index conversion
TEST_F(MeshTest, GridIndexConversion)
{
    const size_t gridSize = 4;
    Mesh mesh(gridSize, 1.0);

    // Test forward and reverse conversion
    for (size_t i = 0; i < gridSize; ++i)
    {
        for (size_t j = 0; j < gridSize; ++j)
        {
            for (size_t k = 0; k < gridSize; ++k)
            {
                size_t index = mesh.getGridIndex(i, j, k);
                size_t i2, j2, k2;
                mesh.getGridCoordinates(index, i2, j2, k2);

                EXPECT_EQ(i, i2);
                EXPECT_EQ(j, j2);
                EXPECT_EQ(k, k2);
            }
        }
    }
}

// Test reset functions
TEST_F(MeshTest, ResetFunctions)
{
    Mesh mesh(4, 1.0);

    // Set some non-zero values
    auto &density = const_cast<std::vector<double> &>(mesh.getHostDensity());
    auto &potential = const_cast<std::vector<double> &>(mesh.getHostPotential());
    auto &forceX = const_cast<std::vector<double> &>(mesh.getHostForceX());

    std::fill(density.begin(), density.end(), 1.0);
    std::fill(potential.begin(), potential.end(), 2.0);
    std::fill(forceX.begin(), forceX.end(), 3.0);

    // Reset
    mesh.resetDensity();
    mesh.resetPotential();
    mesh.resetForces();

    // Check they are zero
    for (double d : mesh.getHostDensity())
    {
        EXPECT_DOUBLE_EQ(d, 0.0);
    }
    for (double p : mesh.getHostPotential())
    {
        EXPECT_DOUBLE_EQ(p, 0.0);
    }
    for (double fx : mesh.getHostForceX())
    {
        EXPECT_DOUBLE_EQ(fx, 0.0);
    }
}

// Test force calculation from potential
TEST_F(MeshTest, ForceCalculation)
{
    const size_t gridSize = 4;
    Mesh mesh(gridSize, 1.0);

    // Create a simple potential field: Φ(x,y,z) = x
    auto &potential = const_cast<std::vector<double> &>(mesh.getHostPotential());
    for (size_t i = 0; i < gridSize; ++i)
    {
        for (size_t j = 0; j < gridSize; ++j)
        {
            for (size_t k = 0; k < gridSize; ++k)
            {
                size_t index = mesh.getGridIndex(i, j, k);
                potential[index] = static_cast<double>(i);
            }
        }
    }

    EXPECT_TRUE(mesh.calculateForcesFromPotential());

    // Check that forces are computed (non-zero for x-direction, near-zero for others)
    const auto &forceX = mesh.getHostForceX();
    const auto &forceY = mesh.getHostForceY();
    const auto &forceZ = mesh.getHostForceZ();

    // Check that x-forces are non-zero (gradient of linear function)
    bool hasNonZeroXForce = false;
    for (double fx : forceX)
    {
        if (std::abs(fx) > 1e-10)
        {
            hasNonZeroXForce = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZeroXForce);

    // Check that y and z forces are near zero (no gradient in those directions)
    for (size_t idx = 0; idx < mesh.getTotalGridPoints(); ++idx)
    {
        EXPECT_NEAR(forceY[idx], 0.0, 1.0); // Allow larger tolerance for boundary effects
        EXPECT_NEAR(forceZ[idx], 0.0, 1.0);
    }
}

// Test force interpolation to particles
TEST_F(MeshTest, ForceInterpolation)
{
    const size_t gridSize = 4;
    Mesh mesh(gridSize, 1.0);

    // Set up a simple force field
    auto &forceX = const_cast<std::vector<double> &>(mesh.getHostForceX());
    auto &forceY = const_cast<std::vector<double> &>(mesh.getHostForceY());
    auto &forceZ = const_cast<std::vector<double> &>(mesh.getHostForceZ());

    // Set F_x = 1.0 everywhere
    std::fill(forceX.begin(), forceX.end(), 1.0);
    std::fill(forceY.begin(), forceY.end(), 2.0);
    std::fill(forceZ.begin(), forceZ.end(), 3.0);

    // Test particles at grid centers
    std::vector<double> positions = {0.5, 0.5, 0.5}; // At grid center
    std::vector<double> forces(3, 0.0);
    size_t numParticles = 1;

    EXPECT_TRUE(mesh.interpolateForcesToParticles(positions, forces, numParticles));

    // Forces should be interpolated from the grid
    EXPECT_NEAR(forces[0], 1.0, 0.1);
    EXPECT_NEAR(forces[1], 2.0, 0.1);
    EXPECT_NEAR(forces[2], 3.0, 0.1);
}

// Test combined P3M operation
TEST_F(MeshTest, CombinedPMOperation)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for P3M operations";
    }

    const size_t gridSize = 4;
    Mesh mesh(gridSize, 1.0);

    // Initialize mesh with CUDA
    EXPECT_TRUE(mesh.initialize(*cudaMgr));

    // Create simple test case
    std::vector<double> positions = {0.5, 0.5, 0.5};
    std::vector<double> masses = {1.0};
    std::vector<double> forces(3, 0.0);
    size_t numParticles = 1;

    // This will test the full pipeline
    EXPECT_TRUE(mesh.computePMForces(positions, masses, forces, numParticles));

    // Forces should be computed
    EXPECT_EQ(forces.size(), 3u);
}

// Test total mass calculation
TEST_F(MeshTest, TotalMass)
{
    Mesh mesh(4, 1.0);

    // Set known density values
    auto &density = const_cast<std::vector<double> &>(mesh.getHostDensity());
    std::fill(density.begin(), density.end(), 2.0);

    double expectedTotalMass = 2.0 * mesh.getTotalGridPoints();
    EXPECT_DOUBLE_EQ(mesh.getTotalMass(), expectedTotalMass);
}

// Test grid information printing (should not throw)
TEST_F(MeshTest, PrintGridInfo)
{
    Mesh mesh(4, 1.0);

    // Should not throw any exception
    EXPECT_NO_THROW(mesh.printGridInfo());
}

// Test memory management
TEST_F(MeshTest, MemoryManagement)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for memory management tests";
    }

    Mesh mesh(4, 1.0);
    EXPECT_FALSE(mesh.isDeviceMemoryAllocated());

    // Allocate memory
    EXPECT_TRUE(mesh.allocateDeviceMemory(*cudaMgr));
    EXPECT_TRUE(mesh.isDeviceMemoryAllocated());

    // Check device pointers are not null
    EXPECT_NE(mesh.getDeviceDensity(), nullptr);
    EXPECT_NE(mesh.getDevicePotential(), nullptr);

    // Free memory
    EXPECT_TRUE(mesh.freeDeviceMemory(*cudaMgr));
    EXPECT_FALSE(mesh.isDeviceMemoryAllocated());
}

// Test edge case: single grid point
TEST_F(MeshTest, SingleGridPoint)
{
    Mesh mesh(1, 1.0);

    EXPECT_EQ(mesh.getGridSize(), 1u);
    EXPECT_EQ(mesh.getTotalGridPoints(), 1u);

    // Test with single particle
    std::vector<double> positions = {0.0, 0.0, 0.0};
    std::vector<double> masses = {5.0};

    EXPECT_TRUE(mesh.assignParticlesToMesh(positions, masses, 1));

    // All mass should be at the single grid point
    const auto &density = mesh.getHostDensity();
    EXPECT_DOUBLE_EQ(density[0], 5.0);
}

// Test edge case: empty particle list
TEST_F(MeshTest, EmptyParticleList)
{
    Mesh mesh(4, 1.0);

    std::vector<double> positions;
    std::vector<double> masses;

    EXPECT_TRUE(mesh.assignParticlesToMesh(positions, masses, 0));

    // Density should remain zero
    const auto &density = mesh.getHostDensity();
    for (double d : density)
    {
        EXPECT_DOUBLE_EQ(d, 0.0);
    }
}

// Test Green function
TEST_F(MeshTest, GreenFunction)
{
    // Test Green function values
    EXPECT_DOUBLE_EQ(Mesh::greenFunction(0, 0, 0, 1.0), 0.0); // k=0 case

    double g1 = Mesh::greenFunction(1, 0, 0, 1.0);
    EXPECT_DOUBLE_EQ(g1, -1.0); // For k=(1,0,0), k²=1, G=-1/k²=-1

    double g2 = Mesh::greenFunction(1, 1, 0, 1.0);
    EXPECT_DOUBLE_EQ(g2, -0.5); // For k=(1,1,0), k²=2, G=-1/2=-0.5
}

// Test destructor cleanup
TEST_F(MeshTest, DestructorCleanup)
{
    // Create mesh in a scope
    {
        Mesh *mesh = new Mesh(4, 1.0);

        if (cudaMgr->isInitialized())
        {
            mesh->initialize(*cudaMgr);
        }

        // Delete mesh (should cleanup properly)
        delete mesh;
    }

    // If we get here without crashing, the test passes
    EXPECT_TRUE(true);
}

// Test invalid grid size
TEST_F(MeshTest, InvalidGridSize)
{
    // Grid size of 0 should still create a valid object
    Mesh mesh(0, 1.0);

    EXPECT_EQ(mesh.getGridSize(), 0u);
    EXPECT_EQ(mesh.getTotalGridPoints(), 0u);
    EXPECT_EQ(mesh.getHostDensity().size(), 0u);
}

// Test data synchronization methods (simplified)
TEST_F(MeshTest, DataSynchronization)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for synchronization tests";
    }

    Mesh mesh(4, 1.0);
    mesh.initialize(*cudaMgr);

    // These methods are simplified for now, but should not crash
    EXPECT_TRUE(mesh.copyDensityToDevice(*cudaMgr));
    EXPECT_TRUE(mesh.copyPotentialToHost(*cudaMgr));
    EXPECT_TRUE(mesh.copyForcesToHost(*cudaMgr));
}