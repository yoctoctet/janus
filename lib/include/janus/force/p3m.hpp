#pragma once

#include "force.hpp"

namespace janus
{
    class P3MForceCalculator final : public IForceCalculator
    {
    public:
        ForceMethod method() const noexcept override { return ForceMethod::P3M; }
    };
}
