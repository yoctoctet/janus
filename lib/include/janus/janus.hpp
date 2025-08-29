#pragma once

namespace janus
{
    typedef double real;

    enum class Backend
    {
        Auto,
        CPU,
        CUDA
    };

    struct Particles
    {
        size_t n;
        real *m;       // masses
        real *px, *py; // positions
        real *vx, *vy; // velocities
        real *ax, *ay; // accelerations
    };
}