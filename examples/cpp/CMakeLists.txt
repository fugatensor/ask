cmake_minimum_required(VERSION 3.28)
project(ASKFrontendConsumer LANGUAGES C CXX)

option(ASK_EXAMPLE_CORE "Build the low-level MLIR builder example" ON)
if(ASK_EXAMPLE_CORE)
  find_package(ASK CONFIG REQUIRED COMPONENTS Compiler Core)
else()
  find_package(ASK CONFIG REQUIRED COMPONENTS Compiler)
endif()

if(ASK_EXAMPLE_CORE)
add_executable(ask-frontend-example frontend.cpp)
target_link_libraries(ask-frontend-example PRIVATE ASK::Core)
target_compile_features(ask-frontend-example PRIVATE cxx_std_17)
endif()

add_executable(ask-compiler-example compiler.cpp)
target_link_libraries(ask-compiler-example PRIVATE ASK::Compiler)
