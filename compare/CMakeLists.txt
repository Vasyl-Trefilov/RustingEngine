cmake_minimum_required(VERSION 3.10)
project(OpenGLStars)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

find_package(glfw3 CONFIG REQUIRED)

add_executable(OpenGLStars main.cpp)

target_link_libraries(OpenGLStars 
    PRIVATE 
    OpenGL::GL 
    GLEW::GLEW 
    glfw
)