#include <iostream>
#include <memory>

#include "types.h"
#include "buffer.h"
#include "force/direct_sum_force_cpu.h"
#include "force/direct_sum_force_cuda.h"

int main(int argc, char *argv[])
{
  try
  {
    // Test host buffer
    janus::Buffer<double, janus::Space::Host> host_buffer(10);
    std::cout << "Created host buffer with " << host_buffer.size() << " elements" << std::endl;

    // Test device buffer
    janus::Buffer<double, janus::Space::Device> device_buffer(10);
    std::cout << "Created device buffer with " << device_buffer.size() << " elements" << std::endl;

    // Test copy from host to device
    host_buffer.fill(3.14);
    device_buffer.copy_from(host_buffer);
    std::cout << "Copied data from host to device" << std::endl;

    // Test copy from device to host
    janus::Buffer<double, janus::Space::Host> host_buffer2(10);
    host_buffer2.copy_from(device_buffer);
    std::cout << "Copied data from device to host" << std::endl;

    std::cout << "All buffer operations completed successfully!" << std::endl;

    // Test direct sum force computation
    std::cout << "\nTesting Direct Sum Force Computation:" << std::endl;

    // Create and initialize host particles
    janus::Particles<janus::Space::Host> host_particles(3);
    host_particles.m.fill(1.0);
    host_particles.vx.fill(0.0);
    host_particles.vy.fill(0.0);

    for (int i = 0; i < 3; i++)
    {
      host_particles.px.data()[i] = (double)rand() / RAND_MAX - 0.5;
      host_particles.py.data()[i] = (double)rand() / RAND_MAX - 0.5;
    }

    // CPU computation
    janus::DirectSumForceCPU cpu_force;
    cpu_force.compute(host_particles);

    std::cout << "CPU Accelerations:" << std::endl;
    for (size_t i = 0; i < host_particles.n; ++i)
    {
      std::cout << "Particle " << i << ": ax=" << host_particles.ax.data()[i] << ", ay=" << host_particles.ay.data()[i] << std::endl;
    }

    // CUDA computation
    janus::Particles<janus::Space::Device> device_particles(3);
    device_particles.m.copy_from(host_particles.m);
    device_particles.px.copy_from(host_particles.px);
    device_particles.py.copy_from(host_particles.py);
    device_particles.vx.copy_from(host_particles.vx);
    device_particles.vy.copy_from(host_particles.vy);

    janus::DirectSumForceCuda cuda_force;
    cuda_force.compute(device_particles);

    // Copy back accelerations
    janus::Buffer<double, janus::Space::Host> host_ax_cuda(3);
    janus::Buffer<double, janus::Space::Host> host_ay_cuda(3);
    host_ax_cuda.copy_from(device_particles.ax);
    host_ay_cuda.copy_from(device_particles.ay);

    std::cout << "CUDA Accelerations:" << std::endl;
    for (size_t i = 0; i < device_particles.n; ++i)
    {
      std::cout << "Particle " << i << ": ax=" << host_ax_cuda.data()[i] << ", ay=" << host_ay_cuda.data()[i] << std::endl;
    }

    std::cout << "Direct sum force test completed!" << std::endl;
  }
  catch (const janus::BufferException &e)
  {
    std::cerr << "Buffer exception: " << e.what() << std::endl;
    return 1;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}