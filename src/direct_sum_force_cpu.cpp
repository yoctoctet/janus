#include "direct_sum_force_cpu.h"

namespace janus
{
    void DirectSumForceCPU::compute(Particles<Space::Host> &particles)
    {
        size_t n = particles.n;
        double *m = particles.m.data();
        double *px = particles.px.data();
        double *py = particles.py.data();
        double *ax = particles.ax.data();
        double *ay = particles.ay.data();

        // Zero out accelerations
        for (size_t i = 0; i < n; ++i)
        {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }

        // Compute forces
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double r2 = dx * dx + dy * dy + epsilon_ * epsilon_;
                double r = sqrt(r2);
                double inv_r3 = 1.0 / (r2 * r);
                double f = m[j] * inv_r3;
                ax[i] += f * dx;
                ay[i] += f * dy;
            }
        }
    }
}