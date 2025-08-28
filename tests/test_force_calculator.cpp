#include <gtest/gtest.h>
#include "force_calculator.h"
#include "particle_system.h"
#include "mesh.h"
#include "cuda_manager.h"
#include <vector>
#include <cmath>

// Test fixture for ForceCalculator tests
class ForceCalculatorTest : public ::testing::Test
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

        // Create mesh
        mesh = new Mesh(8, 1.0);
        if (cudaMgr->isInitialized())
        {
            mesh->initialize(*cudaMgr);
        }

        // Create particle system
        particles = new ParticleSystem(10);
        particles->initializeRandom(*cudaMgr, 2.0);

        // Create force calculator
        forceCalc = new ForceCalculator(particles, mesh, cudaMgr, 1.5, 0.5);
    }

    void TearDown() override
    {
        delete forceCalc;
        delete particles;
        delete mesh;
        delete cudaMgr;

        forceCalc = nullptr;
        particles = nullptr;
        mesh = nullptr;
        cudaMgr = nullptr;
    }

    // Helper method to create simple test system
    void createSimpleTestSystem()
    {
        // Create a simple 2-particle system
        ParticleSystem *simpleParticles = new ParticleSystem(2);

        std::vector<double> positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        std::vector<double> masses = {1.0, 1.0};

        simpleParticles->setPositions(positions);
        simpleParticles->setMasses(masses);

        // Replace the existing particle system
        delete particles;
        particles = simpleParticles;

        // Recreate force calculator with new particle system
        delete forceCalc;
        forceCalc = new ForceCalculator(particles, mesh, cudaMgr, 2.0, 0.35);
    }

    CUDAManager *cudaMgr;
    Mesh *mesh;
    ParticleSystem *particles;
    ForceCalculator *forceCalc;
};

// Test constructor
TEST_F(ForceCalculatorTest, Constructor)
{
    EXPECT_DOUBLE_EQ(forceCalc->getCutoffRadius(), 1.5);
    EXPECT_DOUBLE_EQ(forceCalc->getEwaldAlpha(), 0.5);
    EXPECT_EQ(forceCalc->getNumForceEvaluations(), 0u);
}

// Test parameter setters
TEST_F(ForceCalculatorTest, ParameterSetters)
{
    forceCalc->setCutoffRadius(2.0);
    forceCalc->setEwaldAlpha(0.3);
    forceCalc->setSofteningParameter(0.01);

    EXPECT_DOUBLE_EQ(forceCalc->getCutoffRadius(), 2.0);
    EXPECT_DOUBLE_EQ(forceCalc->getEwaldAlpha(), 0.3);
}

// Test invalid parameter handling
TEST_F(ForceCalculatorTest, InvalidParameters)
{
    // Invalid cutoff radius should not change
    forceCalc->setCutoffRadius(-1.0);
    EXPECT_DOUBLE_EQ(forceCalc->getCutoffRadius(), 1.5);

    // Invalid Ewald alpha should not change
    forceCalc->setEwaldAlpha(0.0);
    EXPECT_DOUBLE_EQ(forceCalc->getEwaldAlpha(), 0.5);

    // Invalid softening should not change
    forceCalc->setSofteningParameter(-1.0);
    // Note: We can't test the softening parameter directly as it's private
}

// Test direct summation calculation
TEST_F(ForceCalculatorTest, DirectSummation)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    const auto &forces = forceCalc->getTotalForces();
    EXPECT_EQ(forces.size(), 6u); // 2 particles * 3 components

    // For two particles at distance 1.0 with mass 1.0 each:
    // Force magnitude should be 1.0 (G=1 in our simplified units)
    // Forces should be equal and opposite
    EXPECT_NEAR(forces[0], 1.0, 0.1); // F_x on particle 0
    EXPECT_NEAR(forces[1], 0.0, 0.1); // F_y on particle 0
    EXPECT_NEAR(forces[2], 0.0, 0.1); // F_z on particle 0

    EXPECT_NEAR(forces[3], -1.0, 0.1); // F_x on particle 1
    EXPECT_NEAR(forces[4], 0.0, 0.1);  // F_y on particle 1
    EXPECT_NEAR(forces[5], 0.0, 0.1);  // F_z on particle 1
}

// Test particle-particle forces
TEST_F(ForceCalculatorTest, ParticleParticleForces)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::PARTICLE_PARTICLE));

    const auto &ppForces = forceCalc->getPPForces();
    EXPECT_EQ(ppForces.size(), 6u);

    // PP forces with Ewald screening (α=0.35) will be reduced at short distances
    // For particles at distance 1.0, expect significantly reduced force due to screening
    EXPECT_NEAR(std::abs(ppForces[0]), 0.27, 0.1); // Ewald screened force
    EXPECT_NEAR(std::abs(ppForces[3]), 0.27, 0.1);

    // Forces should still be non-zero and finite
    EXPECT_NE(ppForces[0], 0.0);
    EXPECT_NE(ppForces[3], 0.0);
    EXPECT_TRUE(std::isfinite(ppForces[0]));
    EXPECT_TRUE(std::isfinite(ppForces[3]));
}

// Test particle-mesh forces
TEST_F(ForceCalculatorTest, ParticleMeshForces)
{
    if (!cudaMgr->isInitialized())
    {
        GTEST_SKIP() << "CUDA not available for PM forces test";
    }

    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::PARTICLE_MESH));

    const auto &pmForces = forceCalc->getPMForces();
    EXPECT_EQ(pmForces.size(), 6u);

    // PM forces should be computed (may be small for this simple case)
    // We just check that the computation doesn't crash
}

// Test full P3M calculation
TEST_F(ForceCalculatorTest, FullP3M)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::P3M_FULL));

    const auto &totalForces = forceCalc->getTotalForces();
    EXPECT_EQ(totalForces.size(), 6u);

    // Check that forces are finite
    for (double force : totalForces)
    {
        EXPECT_TRUE(std::isfinite(force));
    }
}

// Test force combination
TEST_F(ForceCalculatorTest, ForceCombination)
{
    createSimpleTestSystem();

    // Calculate individual components
    EXPECT_TRUE(forceCalc->calculatePPForces());
    EXPECT_TRUE(forceCalc->calculatePMForces());

    // Combine forces
    EXPECT_TRUE(forceCalc->combineForces());

    const auto &ppForces = forceCalc->getPPForces();
    const auto &pmForces = forceCalc->getPMForces();
    const auto &totalForces = forceCalc->getTotalForces();

    // Check that total = PP + PM
    for (size_t i = 0; i < totalForces.size(); ++i)
    {
        EXPECT_NEAR(totalForces[i], ppForces[i] + pmForces[i], 1e-10);
    }
}

// Test energy calculations
TEST_F(ForceCalculatorTest, EnergyCalculations)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    double potentialEnergy = forceCalc->getPotentialEnergy();
    double kineticEnergy = forceCalc->getKineticEnergy();

    // Potential energy should be negative for gravitational systems
    EXPECT_LT(potentialEnergy, 0.0);

    // Kinetic energy should be non-negative
    EXPECT_GE(kineticEnergy, 0.0);

    // Both should be finite
    EXPECT_TRUE(std::isfinite(potentialEnergy));
    EXPECT_TRUE(std::isfinite(kineticEnergy));
}

// Test accuracy comparison with direct summation
TEST_F(ForceCalculatorTest, AccuracyComparison)
{
    createSimpleTestSystem();

    double maxError, rmsError;
    EXPECT_TRUE(forceCalc->compareWithDirectSummation(maxError, rmsError, 2));

    // For a simple 2-particle system, P3M should be reasonably accurate
    EXPECT_LT(maxError, 1.0); // Allow some error due to approximations
    EXPECT_LT(rmsError, 0.5);

    std::cout << "P3M vs Direct Summation - Max Error: " << maxError
              << ", RMS Error: " << rmsError << std::endl;
}

// Test accuracy comparison with larger system
TEST_F(ForceCalculatorTest, AccuracyComparisonLargeSystem)
{
    // Create a larger test system
    ParticleSystem *largeParticles = new ParticleSystem(20);
    largeParticles->initializeRandom(*cudaMgr, 3.0);

    // Replace particle system
    delete particles;
    particles = largeParticles;

    // Recreate force calculator
    delete forceCalc;
    forceCalc = new ForceCalculator(particles, mesh, cudaMgr, 2.5, 0.4);

    double maxError, rmsError;
    EXPECT_TRUE(forceCalc->compareWithDirectSummation(maxError, rmsError, 10));

    // For larger systems, P3M should still be reasonably accurate
    EXPECT_LT(maxError, 10.0);
    EXPECT_LT(rmsError, 5.0);

    std::cout << "Large System P3M vs Direct - Max Error: " << maxError
              << ", RMS Error: " << rmsError << std::endl;
}

// Test force application to particles
TEST_F(ForceCalculatorTest, ForceApplication)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));
    EXPECT_TRUE(forceCalc->applyForcesToParticles());

    // Check that forces were applied to particles
    const auto &particleForces = particles->getHostForces();
    const auto &calcForces = forceCalc->getTotalForces();

    EXPECT_EQ(particleForces.size(), calcForces.size());

    for (size_t i = 0; i < particleForces.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(particleForces[i], calcForces[i]);
    }
}

// Test multiple force evaluations
TEST_F(ForceCalculatorTest, MultipleEvaluations)
{
    createSimpleTestSystem();

    size_t initialCount = forceCalc->getNumForceEvaluations();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));
    EXPECT_EQ(forceCalc->getNumForceEvaluations(), initialCount + 1);

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::PARTICLE_PARTICLE));
    EXPECT_EQ(forceCalc->getNumForceEvaluations(), initialCount + 2);
}

// Test performance statistics
TEST_F(ForceCalculatorTest, PerformanceStats)
{
    createSimpleTestSystem();

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    // Should not throw any exception
    EXPECT_NO_THROW(forceCalc->printPerformanceStats());
}

// Test edge case: single particle
TEST_F(ForceCalculatorTest, SingleParticle)
{
    ParticleSystem *singleParticle = new ParticleSystem(1);
    std::vector<double> positions = {0.0, 0.0, 0.0};
    std::vector<double> masses = {1.0};
    singleParticle->setPositions(positions);
    singleParticle->setMasses(masses);

    delete particles;
    particles = singleParticle;

    delete forceCalc;
    forceCalc = new ForceCalculator(particles, mesh, cudaMgr, 1.0, 0.5);

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    const auto &forces = forceCalc->getTotalForces();
    EXPECT_EQ(forces.size(), 3u);

    // Single particle should have zero force
    EXPECT_DOUBLE_EQ(forces[0], 0.0);
    EXPECT_DOUBLE_EQ(forces[1], 0.0);
    EXPECT_DOUBLE_EQ(forces[2], 0.0);
}

// Test edge case: zero particles
TEST_F(ForceCalculatorTest, ZeroParticles)
{
    ParticleSystem *zeroParticles = new ParticleSystem(0);

    delete particles;
    particles = zeroParticles;

    delete forceCalc;
    forceCalc = new ForceCalculator(particles, mesh, cudaMgr, 1.0, 0.5);

    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    const auto &forces = forceCalc->getTotalForces();
    EXPECT_EQ(forces.size(), 0u);
}

// Test force reset functionality
TEST_F(ForceCalculatorTest, ForceReset)
{
    createSimpleTestSystem();

    // Calculate forces
    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));

    // Check forces are non-zero
    const auto &forces1 = forceCalc->getTotalForces();
    bool hasNonZero = false;
    for (double f : forces1)
    {
        if (std::abs(f) > 1e-10)
        {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);

    // Force reset by calculating different method
    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::PARTICLE_PARTICLE));

    // Forces should be different (reset and recalculated)
    const auto &forces2 = forceCalc->getTotalForces();
    EXPECT_EQ(forces1.size(), forces2.size());
}

// Test Ewald parameter effects
TEST_F(ForceCalculatorTest, EwaldParameterEffects)
{
    createSimpleTestSystem();

    // Test with different Ewald alpha values
    ForceCalculator fc1(particles, mesh, cudaMgr, 2.0, 0.1);
    ForceCalculator fc2(particles, mesh, cudaMgr, 2.0, 1.0);

    EXPECT_TRUE(fc1.calculateForces(ForceMethod::PARTICLE_PARTICLE));
    EXPECT_TRUE(fc2.calculateForces(ForceMethod::PARTICLE_PARTICLE));

    const auto &forces1 = fc1.getPPForces();
    const auto &forces2 = fc2.getPPForces();

    // Different alpha should give different forces
    bool forcesDiffer = false;
    for (size_t i = 0; i < forces1.size(); ++i)
    {
        if (std::abs(forces1[i] - forces2[i]) > 1e-6)
        {
            forcesDiffer = true;
            break;
        }
    }
    EXPECT_TRUE(forcesDiffer);
}

// Test cutoff radius effects
TEST_F(ForceCalculatorTest, CutoffRadiusEffects)
{
    // Create a system where cutoff matters
    ParticleSystem *testParticles = new ParticleSystem(3);

    std::vector<double> positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    std::vector<double> masses = {1.0, 1.0, 1.0};
    testParticles->setPositions(positions);
    testParticles->setMasses(masses);

    delete particles;
    particles = testParticles;

    // Test with small cutoff (only nearest neighbor)
    ForceCalculator fcSmallCutoff(particles, mesh, cudaMgr, 1.5, 0.5);
    EXPECT_TRUE(fcSmallCutoff.calculateForces(ForceMethod::PARTICLE_PARTICLE));

    // Test with large cutoff (all particles)
    ForceCalculator fcLargeCutoff(particles, mesh, cudaMgr, 5.0, 0.5);
    EXPECT_TRUE(fcLargeCutoff.calculateForces(ForceMethod::PARTICLE_PARTICLE));

    const auto &forcesSmall = fcSmallCutoff.getPPForces();
    const auto &forcesLarge = fcLargeCutoff.getPPForces();

    // Different cutoffs should give different forces
    bool forcesDiffer = false;
    for (size_t i = 0; i < forcesSmall.size(); ++i)
    {
        if (std::abs(forcesSmall[i] - forcesLarge[i]) > 1e-6)
        {
            forcesDiffer = true;
            break;
        }
    }
    EXPECT_TRUE(forcesDiffer);
}

// Test error handling
TEST_F(ForceCalculatorTest, ErrorHandling)
{
    createSimpleTestSystem();

    // Test invalid force method (this would require modifying the enum, so we skip)
    // Instead, test that valid methods work
    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::DIRECT_SUMMATION));
    EXPECT_TRUE(forceCalc->calculateForces(ForceMethod::PARTICLE_PARTICLE));
    // PM forces might fail if CUDA not available, but that's tested elsewhere
}

// Test destructor cleanup
TEST_F(ForceCalculatorTest, DestructorCleanup)
{
    // Create force calculator in a scope
    {
        ForceCalculator *tempFC = new ForceCalculator(particles, mesh, cudaMgr, 1.0, 0.5);
        tempFC->calculateForces(ForceMethod::DIRECT_SUMMATION);
        delete tempFC;
    }

    // If we get here without crashing, the test passes
    EXPECT_TRUE(true);
}