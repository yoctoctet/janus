#include <gtest/gtest.h>
#include <cmath>
#include "direct_sum_force_cuda.h"
#include "types.h"

namespace janus
{

    class DirectSumForceCUDATest : public ::testing::Test
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

    // Helper to copy host to device and back for ax, ay
    void copyToDeviceAndCompute(Particles<Space::Host> &host_particles, DirectSumForceCuda &force, Particles<Space::Device> &device_particles)
    {
        device_particles.m.copy_from(host_particles.m);
        device_particles.px.copy_from(host_particles.px);
        device_particles.py.copy_from(host_particles.py);
        device_particles.vx.copy_from(host_particles.vx);
        device_particles.vy.copy_from(host_particles.vy);
        device_particles.ax.copy_from(host_particles.ax);
        device_particles.ay.copy_from(host_particles.ay);

        force.compute(device_particles);

        host_particles.ax.copy_from(device_particles.ax);
        host_particles.ay.copy_from(device_particles.ay);
    }

    // Test 1: Two-body along the x-axis
    TEST_F(DirectSumForceCUDATest, TwoBodyAlongXAxis)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 2.0;
        host_particles.m.data()[1] = 3.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 3.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], -2.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 2: Single particle => zero acceleration
    TEST_F(DirectSumForceCUDATest, SingleParticleZeroAcceleration)
    {
        Particles<Space::Host> host_particles(1);
        Particles<Space::Device> device_particles(1);
        host_particles.m.data()[0] = 5.0;
        host_particles.px.data()[0] = 123.4;
        host_particles.py.data()[0] = -9.9;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
    }

    // Test 3: Three equal masses in a line (−1,0,+1)
    TEST_F(DirectSumForceCUDATest, ThreeEqualMassesInLine)
    {
        Particles<Space::Host> host_particles(3);
        Particles<Space::Device> device_particles(3);
        host_particles.m.data()[0] = 1.0;
        host_particles.m.data()[1] = 1.0;
        host_particles.m.data()[2] = 1.0;
        host_particles.px.data()[0] = -1.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 0.0;
        host_particles.py.data()[1] = 0.0;
        host_particles.px.data()[2] = 1.0;
        host_particles.py.data()[2] = 0.0;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 1.25, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[2], -1.25, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[2], 0.0, 1e-13);
    }

    // Test 4: Equilateral triangle, unit side, equal masses
    TEST_F(DirectSumForceCUDATest, EquilateralTriangle)
    {
        Particles<Space::Host> host_particles(3);
        Particles<Space::Device> device_particles(3);
        host_particles.m.data()[0] = 1.0;
        host_particles.m.data()[1] = 1.0;
        host_particles.m.data()[2] = 1.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;
        host_particles.px.data()[2] = 0.5;
        host_particles.py.data()[2] = std::sqrt(3.0) / 2.0;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 1.5, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], std::sqrt(3.0) / 2.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], -1.5, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], std::sqrt(3.0) / 2.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[2], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[2], -std::sqrt(3.0), 1e-13);
    }

    // Test 5: Zero-mass source + massive neighbor
    TEST_F(DirectSumForceCUDATest, ZeroMassSource)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 0.0;
        host_particles.m.data()[1] = 5.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 5.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 6: Overlapping particles with softening
    TEST_F(DirectSumForceCUDATest, OverlappingWithSoftening)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 2.0;
        host_particles.m.data()[1] = 3.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 0.0;
        host_particles.py.data()[1] = 0.0;

        DirectSumForceCuda force(1e-3);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 7: Softening monotonicity
    TEST_F(DirectSumForceCUDATest, SofteningMonotonicity)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 1.0;
        host_particles.m.data()[1] = 1.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;

        DirectSumForceCuda force0(0.0);
        copyToDeviceAndCompute(host_particles, force0, device_particles);
        double ax0 = host_particles.ax.data()[0];

        Particles<Space::Host> host_particles1(2);
        Particles<Space::Device> device_particles1(2);
        host_particles1.m.data()[0] = 1.0;
        host_particles1.m.data()[1] = 1.0;
        host_particles1.px.data()[0] = 0.0;
        host_particles1.py.data()[0] = 0.0;
        host_particles1.px.data()[1] = 1.0;
        host_particles1.py.data()[1] = 0.0;

        DirectSumForceCuda force1(1.0);
        copyToDeviceAndCompute(host_particles1, force1, device_particles1);
        double ax1 = host_particles1.ax.data()[0];

        EXPECT_NEAR(std::abs(ax0), 1.0, 1e-13);
        EXPECT_NEAR(std::abs(ax1), 1.0 / std::pow(2.0, 1.5), 1e-13);
        EXPECT_GT(std::abs(ax0), std::abs(ax1));
    }

    // Test 8: Acceleration overwrite
    TEST_F(DirectSumForceCUDATest, AccelerationOverwrite)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 2.0;
        host_particles.m.data()[1] = 3.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;
        host_particles.ax.data()[0] = 999.0;
        host_particles.ay.data()[0] = 999.0;
        host_particles.ax.data()[1] = 999.0;
        host_particles.ay.data()[1] = 999.0;

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 3.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ax.data()[1], -2.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[1], 0.0, 1e-13);
    }

    // Test 9: Non-accidental writes
    TEST_F(DirectSumForceCUDATest, NoAccidentalWrites)
    {
        Particles<Space::Host> host_particles(2);
        Particles<Space::Device> device_particles(2);
        host_particles.m.data()[0] = 2.0;
        host_particles.m.data()[1] = 3.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;
        host_particles.vx.data()[0] = 1.0;
        host_particles.vy.data()[0] = 2.0;
        host_particles.vx.data()[1] = 3.0;
        host_particles.vy.data()[1] = 4.0;

        double m0 = host_particles.m.data()[0];
        double m1 = host_particles.m.data()[1];
        double px0 = host_particles.px.data()[0];
        double py0 = host_particles.py.data()[0];
        double px1 = host_particles.px.data()[1];
        double py1 = host_particles.py.data()[1];
        double vx0 = host_particles.vx.data()[0];
        double vy0 = host_particles.vy.data()[0];
        double vx1 = host_particles.vx.data()[1];
        double vy1 = host_particles.vy.data()[1];

        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_EQ(host_particles.m.data()[0], m0);
        EXPECT_EQ(host_particles.m.data()[1], m1);
        EXPECT_EQ(host_particles.px.data()[0], px0);
        EXPECT_EQ(host_particles.py.data()[0], py0);
        EXPECT_EQ(host_particles.px.data()[1], px1);
        EXPECT_EQ(host_particles.py.data()[1], py1);
        EXPECT_EQ(host_particles.vx.data()[0], vx0);
        EXPECT_EQ(host_particles.vy.data()[0], vy0);
        EXPECT_EQ(host_particles.vx.data()[1], vx1);
        EXPECT_EQ(host_particles.vy.data()[1], vy1);
    }

    // Test 10: Empty and zero-length buffers
    TEST_F(DirectSumForceCUDATest, EmptyBuffers)
    {
        Particles<Space::Host> host_particles(0);
        Particles<Space::Device> device_particles(0);
        DirectSumForceCuda force(0.0);
        EXPECT_NO_THROW(copyToDeviceAndCompute(host_particles, force, device_particles));
    }

    TEST_F(DirectSumForceCUDATest, SingleParticleWithEpsilon)
    {
        Particles<Space::Host> host_particles(1);
        Particles<Space::Device> device_particles(1);
        host_particles.m.data()[0] = 5.0;
        host_particles.px.data()[0] = 123.4;
        host_particles.py.data()[0] = -9.9;

        DirectSumForceCuda force(1e-6);
        copyToDeviceAndCompute(host_particles, force, device_particles);

        EXPECT_NEAR(host_particles.ax.data()[0], 0.0, 1e-13);
        EXPECT_NEAR(host_particles.ay.data()[0], 0.0, 1e-13);
    }

    // CUDA-specific: Deterministic repeatability
    TEST_F(DirectSumForceCUDATest, DeterministicRepeatability)
    {
        Particles<Space::Host> host_particles(3);
        Particles<Space::Device> device_particles(3);
        host_particles.m.data()[0] = 1.0;
        host_particles.m.data()[1] = 1.0;
        host_particles.m.data()[2] = 1.0;
        host_particles.px.data()[0] = 0.0;
        host_particles.py.data()[0] = 0.0;
        host_particles.px.data()[1] = 1.0;
        host_particles.py.data()[1] = 0.0;
        host_particles.px.data()[2] = 0.5;
        host_particles.py.data()[2] = std::sqrt(3.0) / 2.0;

        DirectSumForceCuda force(0.0);

        // First run
        copyToDeviceAndCompute(host_particles, force, device_particles);
        double ax0_1 = host_particles.ax.data()[0];
        double ay0_1 = host_particles.ay.data()[0];
        double ax1_1 = host_particles.ax.data()[1];
        double ay1_1 = host_particles.ay.data()[1];
        double ax2_1 = host_particles.ax.data()[2];
        double ay2_1 = host_particles.ay.data()[2];

        // Second run
        copyToDeviceAndCompute(host_particles, force, device_particles);
        double ax0_2 = host_particles.ax.data()[0];
        double ay0_2 = host_particles.ay.data()[0];
        double ax1_2 = host_particles.ax.data()[1];
        double ay1_2 = host_particles.ay.data()[1];
        double ax2_2 = host_particles.ax.data()[2];
        double ay2_2 = host_particles.ay.data()[2];

        EXPECT_NEAR(ax0_1, ax0_2, 1e-13 + 1e-13 * std::abs(ax0_1));
        EXPECT_NEAR(ay0_1, ay0_2, 1e-13 + 1e-13 * std::abs(ay0_1));
        EXPECT_NEAR(ax1_1, ax1_2, 1e-13 + 1e-13 * std::abs(ax1_1));
        EXPECT_NEAR(ay1_1, ay1_2, 1e-13 + 1e-13 * std::abs(ay1_1));
        EXPECT_NEAR(ax2_1, ax2_2, 1e-13 + 1e-13 * std::abs(ax2_1));
        EXPECT_NEAR(ay2_1, ay2_2, 1e-13 + 1e-13 * std::abs(ay2_1));
    }

    // CUDA-specific: Large N no NaN/inf (test with different n)
    TEST_F(DirectSumForceCUDATest, LargeN_NoNaN)
    {
        // Test with n=300 to check for NaN/inf in large systems
        Particles<Space::Host> host_particles300(300);
        Particles<Space::Device> device_particles300(300);
        for (size_t i = 0; i < 300; ++i)
        {
            host_particles300.m.data()[i] = 1.0;
            host_particles300.px.data()[i] = static_cast<double>(i);
            host_particles300.py.data()[i] = 0.0;
        }
        DirectSumForceCuda force(0.0);
        copyToDeviceAndCompute(host_particles300, force, device_particles300);

        // Check no NaN or inf
        for (size_t i = 0; i < 300; ++i)
        {
            EXPECT_TRUE(std::isfinite(host_particles300.ax.data()[i]));
            EXPECT_TRUE(std::isfinite(host_particles300.ay.data()[i]));
        }
    }

} // namespace janus
