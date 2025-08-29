#pragma once

#include "iforce.h"
#include <cmath>

namespace janus
{
    class DirectSumForceCPU : public IForce<Space::Host>
    {
    public:
        DirectSumForceCPU(double epsilon = 1e-6) : epsilon_(epsilon) {}
        void compute(Particles<Space::Host> &particles) override;

    private:
        double epsilon_;
    };
}