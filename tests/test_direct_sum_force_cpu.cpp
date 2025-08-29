#include <gtest/gtest.h>
#include <cmath>
#include "force/direct_sum_force_cpu.h"
#include "types.h"

namespace janus
{

    class DirectSumForceCPUTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Common setup if needed
        }

        void TearDown() override
        {
            // Common teardown if needed
        }
    };

    // Test 1: Two-body along the x-axis
    TEST_F(DirectSumForceCPUTest, TwoBodyAlongXAxis)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 2.0;
        particles.m.data()[1] = 3.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 3.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], -2.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 2: Single particle => zero acceleration
    TEST_F(DirectSumForceCPUTest, SingleParticleZeroAcceleration)
    {
        Particles<Space::Host> particles(1);
        particles.m.data()[0] = 5.0;
        particles.px.data()[0] = 123.4;
        particles.py.data()[0] = -9.9;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
    }

    // Test 3: Three equal masses in a line (−1,0,+1)
    TEST_F(DirectSumForceCPUTest, ThreeEqualMassesInLine)
    {
        Particles<Space::Host> particles(3);
        particles.m.data()[0] = 1.0;
        particles.m.data()[1] = 1.0;
        particles.m.data()[2] = 1.0;
        particles.px.data()[0] = -1.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 0.0;
        particles.py.data()[1] = 0.0;
        particles.px.data()[2] = 1.0;
        particles.py.data()[2] = 0.0;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 1.25, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[2], -1.25, 1e-13);
        EXPECT_NEAR(particles.ay.data()[2], 0.0, 1e-13);
    }

    // Test 4: Equilateral triangle, unit side, equal masses
    TEST_F(DirectSumForceCPUTest, EquilateralTriangle)
    {
        Particles<Space::Host> particles(3);
        particles.m.data()[0] = 1.0;
        particles.m.data()[1] = 1.0;
        particles.m.data()[2] = 1.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;
        particles.px.data()[2] = 0.5;
        particles.py.data()[2] = std::sqrt(3.0) / 2.0;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 1.5, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], std::sqrt(3.0) / 2.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], -1.5, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], std::sqrt(3.0) / 2.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[2], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[2], -std::sqrt(3.0), 1e-13);
    }

    // Test 5: Zero-mass source + massive neighbor
    TEST_F(DirectSumForceCPUTest, ZeroMassSource)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 0.0;
        particles.m.data()[1] = 5.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 5.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 6: Overlapping particles with softening
    TEST_F(DirectSumForceCPUTest, OverlappingWithSoftening)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 2.0;
        particles.m.data()[1] = 3.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 0.0;
        particles.py.data()[1] = 0.0;

        DirectSumForceCPU force(1e-3);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 7: Softening monotonicity
    TEST_F(DirectSumForceCPUTest, SofteningMonotonicity)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 1.0;
        particles.m.data()[1] = 1.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;

        DirectSumForceCPU force0(0.0);
        force0.compute(particles);
        double ax0 = particles.ax.data()[0];

        Particles<Space::Host> particles1(2);
        particles1.m.data()[0] = 1.0;
        particles1.m.data()[1] = 1.0;
        particles1.px.data()[0] = 0.0;
        particles1.py.data()[0] = 0.0;
        particles1.px.data()[1] = 1.0;
        particles1.py.data()[1] = 0.0;

        DirectSumForceCPU force1(1.0);
        force1.compute(particles1);
        double ax1 = particles1.ax.data()[0];

        EXPECT_NEAR(std::abs(ax0), 1.0, 1e-13);
        EXPECT_NEAR(std::abs(ax1), 1.0 / std::pow(2.0, 1.5), 1e-13);
        EXPECT_GT(std::abs(ax0), std::abs(ax1));
    }

    // Test 8: Acceleration overwrite
    TEST_F(DirectSumForceCPUTest, AccelerationOverwrite)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 2.0;
        particles.m.data()[1] = 3.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;
        particles.ax.data()[0] = 999.0;
        particles.ay.data()[0] = 999.0;
        particles.ax.data()[1] = 999.0;
        particles.ay.data()[1] = 999.0;

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 3.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ax.data()[1], -2.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 9: Non-accidental writes
    TEST_F(DirectSumForceCPUTest, NoAccidentalWrites)
    {
        Particles<Space::Host> particles(2);
        particles.m.data()[0] = 2.0;
        particles.m.data()[1] = 3.0;
        particles.px.data()[0] = 0.0;
        particles.py.data()[0] = 0.0;
        particles.px.data()[1] = 1.0;
        particles.py.data()[1] = 0.0;
        particles.vx.data()[0] = 1.0;
        particles.vy.data()[0] = 2.0;
        particles.vx.data()[1] = 3.0;
        particles.vy.data()[1] = 4.0;

        double m0 = particles.m.data()[0];
        double m1 = particles.m.data()[1];
        double px0 = particles.px.data()[0];
        double py0 = particles.py.data()[0];
        double px1 = particles.px.data()[1];
        double py1 = particles.py.data()[1];
        double vx0 = particles.vx.data()[0];
        double vy0 = particles.vy.data()[0];
        double vx1 = particles.vx.data()[1];
        double vy1 = particles.vy.data()[1];

        DirectSumForceCPU force(0.0);
        force.compute(particles);

        EXPECT_EQ(particles.m.data()[0], m0);
        EXPECT_EQ(particles.m.data()[1], m1);
        EXPECT_EQ(particles.px.data()[0], px0);
        EXPECT_EQ(particles.py.data()[0], py0);
        EXPECT_EQ(particles.px.data()[1], px1);
        EXPECT_EQ(particles.py.data()[1], py1);
        EXPECT_EQ(particles.vx.data()[0], vx0);
        EXPECT_EQ(particles.vy.data()[0], vy0);
        EXPECT_EQ(particles.vx.data()[1], vx1);
        EXPECT_EQ(particles.vy.data()[1], vy1);
    }

    // Test 10: Empty and zero-length buffers
    TEST_F(DirectSumForceCPUTest, EmptyBuffers)
    {
        Particles<Space::Host> particles(0);
        DirectSumForceCPU force(0.0);
        EXPECT_NO_THROW(force.compute(particles));
    }

    TEST_F(DirectSumForceCPUTest, SingleParticleWithEpsilon)
    {
        Particles<Space::Host> particles(1);
        particles.m.data()[0] = 5.0;
        particles.px.data()[0] = 123.4;
        particles.py.data()[0] = -9.9;

        DirectSumForceCPU force(1e-6);
        force.compute(particles);

        EXPECT_NEAR(particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(particles.ay.data()[0], 0.0, 1e-13);
    }

} // namespace janus