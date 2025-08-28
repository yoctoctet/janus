# Source files
set(SOURCES
    main.cpp
    simulator.cpp
    config.cpp
    test_gpu_verification.cpp
)

set(CUDA_SOURCES
    gpu_kernels.cu
)

# Create executable
add_executable(janus-sim ${SOURCES} ${CUDA_SOURCES})

# Set CUDA separable compilation
set_target_properties(janus-sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(janus-sim
    CUDA::cudart_static
)

# Include directories
target_include_directories(janus-sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
)

# Compiler options
target_compile_options(janus-sim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -Wall>
)

# Conditional compilation
if(JANUS_USE_NVTX)
    target_compile_definitions(janus-sim PRIVATE JANUS_USE_NVTX)
    target_link_libraries(janus-sim NVTX::nvtx)
endif()

if(JANUS_FAST_MATH)
    target_compile_definitions(janus-sim PRIVATE JANUS_FAST_MATH)
endif()

if(JANUS_DETERMINISTIC)
    target_compile_definitions(janus-sim PRIVATE JANUS_DETERMINISTIC)
endif()