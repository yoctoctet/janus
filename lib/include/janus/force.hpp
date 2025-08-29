#pragma once

#include "janus.hpp"

namespace janus
{

    enum class ForceMethod
    {
        DirectSum,    // Particle-Particle (direct summation)
        ParticleMesh, // Particle-Mesh
        P3M           // Particle-Particle/Particle-Mesh
    };

    class IForceCalculator
    {
    public:
        IForceCalculator(Backend backend = Backend::Auto) : backend_(backend) {}
        virtual ~IForceCalculator() = default;
        virtual void compute(Particles *P) = 0;
        virtual ForceMethod method() const noexcept = 0;

    protected:
        Backend backend_;
    };
}

#include "force/direct_sum.hpp"
#include "force/particle_mesh.hpp"
#include "force/p3m.hpp"