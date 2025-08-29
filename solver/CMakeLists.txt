add_executable(janus-solver
    src/main.cpp)

target_include_directories(janus-solver PRIVATE include)

target_link_libraries(janus-solver PRIVATE janus-lib)