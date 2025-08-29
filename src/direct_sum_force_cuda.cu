#include "direct_sum_force_cuda.h"

namespace janus
{
    __global__ void zeroAccelerations(double *ax, double *ay, size_t n)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }
    }

    __global__ void computeForces(double *m, double *px, double *py, double *ax, double *ay, size_t n, double epsilon)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            double ax_sum = 0.0;
            double ay_sum = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double r2 = dx * dx + dy * dy + epsilon * epsilon;
                double r = sqrt(r2);
                double inv_r3 = 1.0 / (r2 * r);
                double f = m[i] * m[j] * inv_r3;
                ax_sum += f * dx / m[i];
                ay_sum += f * dy / m[i];
            }
            ax[i] = ax_sum;
            ay[i] = ay_sum;
        }
    }

    void DirectSumForceCuda::compute(Particles<Space::Device> &particles)
    {
        size_t n = particles.n;
        double *m = particles.m.data();
        double *px = particles.px.data();
        double *py = particles.py.data();
        double *ax = particles.ax.data();
        double *ay = particles.ay.data();

        // Zero out accelerations
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        zeroAccelerations<<<blocks, threadsPerBlock>>>(ax, ay, n);
        cudaDeviceSynchronize();

        // Compute forces
        computeForces<<<blocks, threadsPerBlock>>>(m, px, py, ax, ay, n, epsilon_);
        cudaDeviceSynchronize();
    }
}