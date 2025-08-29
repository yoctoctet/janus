add_library(janus-lib
    src/lib.cpp)

target_include_directories(janus-lib PUBLIC include)

find_package(GTest REQUIRED)
