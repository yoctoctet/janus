#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <algorithm>
#include "direct_sum_force_cpu.h"
#include "direct_sum_force_cuda.h"
#include "types.h"

namespace janus
{

    class DirectSumForceCrossTest : public ::testing::Test
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

        void computeCPU(Particles<Space::Host> &particles, double epsilon)
        {
            DirectSumForceCPU force(epsilon);
            force.compute(particles);
        }

        void computeCUDA(Particles<Space::Host> &host_particles, double epsilon)
        {
            Particles<Space::Device> device_particles(host_particles.n);
            device_particles.m.copy_from(host_particles.m);
            device_particles.px.copy_from(host_particles.px);
            device_particles.py.copy_from(host_particles.py);
            device_particles.vx.copy_from(host_particles.vx);
            device_particles.vy.copy_from(host_particles.vy);
            device_particles.ax.copy_from(host_particles.ax);
            device_particles.ay.copy_from(host_particles.ay);

            DirectSumForceCuda force(epsilon);
            force.compute(device_particles);

            host_particles.ax.copy_from(device_particles.ax);
            host_particles.ay.copy_from(device_particles.ay);
        }

        bool compareResults(const Particles<Space::Host> &cpu, const Particles<Space::Host> &cuda, double tol = 1e-13)
        {
            for (size_t i = 0; i < cpu.n; ++i)
            {
                double diff_ax = std::abs(cpu.ax.data()[i] - cuda.ax.data()[i]);
                double diff_ay = std::abs(cpu.ay.data()[i] - cuda.ay.data()[i]);
                double max_ax = std::max(std::abs(cpu.ax.data()[i]), std::abs(cuda.ax.data()[i]));
                double max_ay = std::max(std::abs(cpu.ay.data()[i]), std::abs(cuda.ay.data()[i]));
                if (diff_ax > tol + tol * max_ax || diff_ay > tol + tol * max_ay)
                {
                    return false;
                }
            }
            return true;
        }
    };

    // X1: CPU ↔ CUDA numeric agreement (two-body)
    TEST_F(DirectSumForceCrossTest, TwoBodyAgreement)
    {
        Particles<Space::Host> cpu_particles(2);
        cpu_particles.m.data()[0] = 2.0;
        cpu_particles.m.data()[1] = 3.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
    }

    // X2: CPU ↔ CUDA agreement (equilateral triangle)
    TEST_F(DirectSumForceCrossTest, EquilateralTriangleAgreement)
    {
        Particles<Space::Host> cpu_particles(3);
        cpu_particles.m.data()[0] = 1.0;
        cpu_particles.m.data()[1] = 1.0;
        cpu_particles.m.data()[2] = 1.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;
        cpu_particles.px.data()[2] = 0.5;
        cpu_particles.py.data()[2] = std::sqrt(3.0) / 2.0;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
    }

    // X3: CPU ↔ CUDA agreement (random small system + momentum conservation)
    TEST_F(DirectSumForceCrossTest, RandomSystemAgreement)
    {
        const size_t n = 8;
        Particles<Space::Host> cpu_particles(n);
        Particles<Space::Host> cuda_particles(n);

        std::mt19937 gen(42); // fixed seed
        std::uniform_real_distribution<double> mass_dist(0.1, 2.0);
        std::uniform_real_distribution<double> pos_dist(-2.0, 2.0);

        double total_momentum_x = 0.0;
        double total_momentum_y = 0.0;

        for (size_t i = 0; i < n; ++i)
        {
            double m = mass_dist(gen);
            cpu_particles.m.data()[i] = m;
            cuda_particles.m.data()[i] = m;
            cpu_particles.px.data()[i] = pos_dist(gen);
            cuda_particles.px.data()[i] = cpu_particles.px.data()[i];
            cpu_particles.py.data()[i] = pos_dist(gen);
            cuda_particles.py.data()[i] = cpu_particles.py.data()[i];
        }

        computeCPU(cpu_particles, 1e-6);
        computeCUDA(cuda_particles, 1e-6);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));

        // Momentum conservation
        for (size_t i = 0; i < n; ++i)
        {
            total_momentum_x += cpu_particles.m.data()[i] * cpu_particles.ax.data()[i];
            total_momentum_y += cpu_particles.m.data()[i] * cpu_particles.ay.data()[i];
        }
        double total_mass = 0.0;
        for (size_t i = 0; i < n; ++i)
            total_mass += cpu_particles.m.data()[i];
        EXPECT_NEAR(total_momentum_x, 0.0, 1e-12 * total_mass);
        EXPECT_NEAR(total_momentum_y, 0.0, 1e-12 * total_mass);

        // Momentum conservation for CUDA
        double total_momentum_x_cuda = 0.0;
        double total_momentum_y_cuda = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            total_momentum_x_cuda += cuda_particles.m.data()[i] * cuda_particles.ax.data()[i];
            total_momentum_y_cuda += cuda_particles.m.data()[i] * cuda_particles.ay.data()[i];
        }
        double total_mass_cuda = 0.0;
        for (size_t i = 0; i < n; ++i)
            total_mass_cuda += cuda_particles.m.data()[i];
        EXPECT_NEAR(total_momentum_x_cuda, 0.0, 1e-12 * total_mass_cuda);
        EXPECT_NEAR(total_momentum_y_cuda, 0.0, 1e-12 * total_mass_cuda);
    }

    // X4: Translation invariance
    TEST_F(DirectSumForceCrossTest, TranslationInvariance)
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

        Particles<Space::Host> original(particles.n);
        std::copy(particles.m.data(), particles.m.data() + particles.n, original.m.data());
        std::copy(particles.px.data(), particles.px.data() + particles.n, original.px.data());
        std::copy(particles.py.data(), particles.py.data() + particles.n, original.py.data());
        std::copy(particles.vx.data(), particles.vx.data() + particles.n, original.vx.data());
        std::copy(particles.vy.data(), particles.vy.data() + particles.n, original.vy.data());
        computeCPU(original, 0.0);

        // Translate
        double tx = 3.7, ty = -1.2;
        for (size_t i = 0; i < particles.n; ++i)
        {
            particles.px.data()[i] += tx;
            particles.py.data()[i] += ty;
        }

        Particles<Space::Host> translated_cpu(particles.n);
        std::copy(particles.m.data(), particles.m.data() + particles.n, translated_cpu.m.data());
        std::copy(particles.px.data(), particles.px.data() + particles.n, translated_cpu.px.data());
        std::copy(particles.py.data(), particles.py.data() + particles.n, translated_cpu.py.data());
        std::copy(particles.vx.data(), particles.vx.data() + particles.n, translated_cpu.vx.data());
        std::copy(particles.vy.data(), particles.vy.data() + particles.n, translated_cpu.vy.data());

        Particles<Space::Host> translated_cuda(particles.n);
        std::copy(particles.m.data(), particles.m.data() + particles.n, translated_cuda.m.data());
        std::copy(particles.px.data(), particles.px.data() + particles.n, translated_cuda.px.data());
        std::copy(particles.py.data(), particles.py.data() + particles.n, translated_cuda.py.data());
        std::copy(particles.vx.data(), particles.vx.data() + particles.n, translated_cuda.vx.data());
        std::copy(particles.vy.data(), particles.vy.data() + particles.n, translated_cuda.vy.data());

        computeCPU(translated_cpu, 0.0);
        computeCUDA(translated_cuda, 0.0);

        ASSERT_TRUE(compareResults(translated_cpu, translated_cuda));
        ASSERT_TRUE(compareResults(original, translated_cpu));
    }

    // X5: Zero-mass edge consistency
    TEST_F(DirectSumForceCrossTest, ZeroMassConsistency)
    {
        Particles<Space::Host> cpu_particles(2);
        cpu_particles.m.data()[0] = 0.0;
        cpu_particles.m.data()[1] = 5.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
    }

    // X6: Overwrite semantics parity
    TEST_F(DirectSumForceCrossTest, OverwriteSemantics)
    {
        Particles<Space::Host> cpu_particles(2);
        cpu_particles.m.data()[0] = 2.0;
        cpu_particles.m.data()[1] = 3.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;
        cpu_particles.ax.data()[0] = 9e99;
        cpu_particles.ay.data()[0] = 9e99;
        cpu_particles.ax.data()[1] = 9e99;
        cpu_particles.ay.data()[1] = 9e99;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());
        std::copy(cpu_particles.ax.data(), cpu_particles.ax.data() + cpu_particles.n, cuda_particles.ax.data());
        std::copy(cpu_particles.ay.data(), cpu_particles.ay.data() + cpu_particles.n, cuda_particles.ay.data());

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
        EXPECT_NEAR(cpu_particles.ax.data()[0], 3.0, 1e-13);
        EXPECT_NEAR(cuda_particles.ax.data()[0], 3.0, 1e-13);
    }

    // X7: Softening effect parity
    TEST_F(DirectSumForceCrossTest, SofteningEffectParity)
    {
        Particles<Space::Host> cpu_particles(2);
        cpu_particles.m.data()[0] = 1.0;
        cpu_particles.m.data()[1] = 1.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());

        computeCPU(cpu_particles, 1.0);
        computeCUDA(cuda_particles, 1.0);

        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
        EXPECT_NEAR(std::abs(cpu_particles.ax.data()[0]), 1.0 / std::pow(2.0, 1.5), 1e-13);
    }

    // X8: Scaling with target mass
    TEST_F(DirectSumForceCrossTest, ScalingWithTargetMass)
    {
        Particles<Space::Host> cpu_particles(2);
        cpu_particles.m.data()[0] = 2.0;
        cpu_particles.m.data()[1] = 3.0;
        cpu_particles.px.data()[0] = 0.0;
        cpu_particles.py.data()[0] = 0.0;
        cpu_particles.px.data()[1] = 1.0;
        cpu_particles.py.data()[1] = 0.0;

        Particles<Space::Host> cuda_particles(cpu_particles.n);
        std::copy(cpu_particles.m.data(), cpu_particles.m.data() + cpu_particles.n, cuda_particles.m.data());
        std::copy(cpu_particles.px.data(), cpu_particles.px.data() + cpu_particles.n, cuda_particles.px.data());
        std::copy(cpu_particles.py.data(), cpu_particles.py.data() + cpu_particles.n, cuda_particles.py.data());
        std::copy(cpu_particles.vx.data(), cpu_particles.vx.data() + cpu_particles.n, cuda_particles.vx.data());
        std::copy(cpu_particles.vy.data(), cpu_particles.vy.data() + cpu_particles.n, cuda_particles.vy.data());

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        double ax0_cpu = cpu_particles.ax.data()[0];
        double ax1_cpu = cpu_particles.ax.data()[1];
        double ax0_cuda = cuda_particles.ax.data()[0];
        double ax1_cuda = cuda_particles.ax.data()[1];

        // Now scale m[0] by 10x
        cpu_particles.m.data()[0] = 20.0;
        cuda_particles.m.data()[0] = 20.0;

        computeCPU(cpu_particles, 0.0);
        computeCUDA(cuda_particles, 0.0);

        // ax[0] should be unchanged (since it's target mass)
        EXPECT_NEAR(cpu_particles.ax.data()[0], ax0_cpu, 1e-13);
        EXPECT_NEAR(cuda_particles.ax.data()[0], ax0_cuda, 1e-13);
        // ax[1] should change
        EXPECT_NE(cpu_particles.ax.data()[1], ax1_cpu);
        EXPECT_NE(cuda_particles.ax.data()[1], ax1_cuda);
        // But CPU and CUDA still match
        ASSERT_TRUE(compareResults(cpu_particles, cuda_particles));
    }

} // namespace janus
