#pragma once

#include "force.hpp"

namespace janus
{
    class DirectSumForceCalculator final : public IForceCalculator
    {
    public:
        ForceMethod method() const noexcept override { return ForceMethod::DirectSum; }
    };
}
