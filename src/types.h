#pragma once

#include "buffer.h"

namespace janus
{
  template <Space S>
  struct Particles
  {
    Particles(size_t num) : n(num), m(num), px(num), py(num), vx(num), vy(num), ax(num), ay(num) {}
    size_t n;
    Buffer<double, S> m;      // masses
    Buffer<double, S> px, py; // positions
    Buffer<double, S> vx, vy; // velocities
    Buffer<double, S> ax, ay; // accelerations
  };
}