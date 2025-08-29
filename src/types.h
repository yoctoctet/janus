#pragma once

#include "buffer.h"

namespace janus
{
  template <Space S>
  struct Particles
  {
    size_t n;
    Buffer<double, S> m;      // masses
    Buffer<double, S> px, py; // positions
    Buffer<double, S> vx, vy; // velocities
    Buffer<double, S> ax, ay; // accelerations
  };
}