#pragma once

#include "iforce.h"
#include <cuda_runtime.h>

namespace janus
{
    class DirectSumForceCuda : public IForce<Space::Device>
    {
    public:
        DirectSumForceCuda(double epsilon = 1e-6) : epsilon_(epsilon) {}
        void compute(Particles<Space::Device> &particles) override;

    private:
        double epsilon_;
    };
}