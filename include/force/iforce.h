#pragma once

#include <cstddef>
#include "types.h"

namespace janus
{
    template <Space S>
    class IForce
    {
    public:
        virtual ~IForce() = default;
        virtual void compute(Particles<S> &particles) = 0;
    };
}