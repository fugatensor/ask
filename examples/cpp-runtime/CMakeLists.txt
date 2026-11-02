cmake_minimum_required(VERSION 3.28)
project(ask_runtime_example LANGUAGES CXX)

find_package(ASK CONFIG REQUIRED COMPONENTS Runtime)
add_executable(ask-runtime-example main.cpp)
target_link_libraries(ask-runtime-example PRIVATE ASK::Runtime)
