cmake_minimum_required(VERSION 3.28)
project(ASKFrontendConsumer LANGUAGES C CXX)

find_package(ASK CONFIG REQUIRED COMPONENTS Core)

add_executable(ask-frontend-example frontend.cpp)
target_link_libraries(ask-frontend-example PRIVATE ASK::Core)
target_compile_features(ask-frontend-example PRIVATE cxx_std_17)
