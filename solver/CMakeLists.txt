find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

add_executable(janus-solver
    src/main.cpp)

target_include_directories(janus-solver PRIVATE include)

target_link_libraries(janus-solver PRIVATE janus-lib)
target_link_libraries(janus-solver PRIVATE CLI11::CLI11)
target_link_libraries(janus-solver PRIVATE spdlog::spdlog)