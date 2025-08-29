#pragma once

#include "force.hpp"

namespace janus
{
    class ParticleMeshForceCalculator final : public IForceCalculator
    {
    public:
        ForceMethod method() const noexcept override { return ForceMethod::ParticleMesh; }
    };
}
