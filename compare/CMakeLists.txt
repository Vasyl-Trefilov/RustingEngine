cmake_minimum_required(VERSION 3.14)
project(CompareCubes)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(raylib)

add_executable(CompareCubes compareCubes.cpp)

target_link_libraries(CompareCubes PRIVATE raylib)