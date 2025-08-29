#include <iostream>
#include <memory>

#include "types.h"
#include "buffer.h"

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