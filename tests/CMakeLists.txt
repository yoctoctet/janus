# Test sources
set(TEST_SOURCES
    test_main.cpp
    test_config.cpp
    test_simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/config.cpp
    ${CMAKE_SOURCE_DIR}/src/simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu_kernels.cu
)

# Create test executable
add_executable(janus-tests ${TEST_SOURCES})

# Link with main library (if we had one) or just include source files
target_include_directories(janus-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor
)

# Compiler options
target_compile_options(janus-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Add test
add_test(NAME janus-tests COMMAND janus-tests)