cmake_minimum_required(VERSION 3.28)
include(cmake/AskVersion.cmake)
project(ASK VERSION ${ASK_VERSION} LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(ASK_TARGET_BACKENDS "host" CACHE STRING "none or host")
if(NOT ASK_TARGET_BACKENDS STREQUAL "none" AND NOT ASK_TARGET_BACKENDS STREQUAL "host")
  message(FATAL_ERROR "ASK_TARGET_BACKENDS supports 'none' or 'host'; requested '${ASK_TARGET_BACKENDS}'")
endif()

# LLVM and MLIR come from the pinned SDK; auto-discover .deps/<dir> when unset.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm.lock.json" ASK_LLVM_LOCK)
string(JSON ASK_LLVM_CMAKE_VERSION GET "${ASK_LLVM_LOCK}" cmake_version)
string(JSON ASK_LLVM_PACKAGE_VERSION GET "${ASK_LLVM_LOCK}" version)
string(JSON ASK_LLVM_REVISION GET "${ASK_LLVM_LOCK}" revision)
string(JSON ASK_LLVM_DIRECTORY GET "${ASK_LLVM_LOCK}" directory)
set(ASK_LOCAL_LLVM_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.deps/${ASK_LLVM_DIRECTORY}")
if(NOT LLVM_DIR AND EXISTS "${ASK_LOCAL_LLVM_PREFIX}/lib/cmake/llvm/LLVMConfig.cmake")
  set(LLVM_DIR "${ASK_LOCAL_LLVM_PREFIX}/lib/cmake/llvm")
endif()
if(NOT MLIR_DIR AND EXISTS "${ASK_LOCAL_LLVM_PREFIX}/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "${ASK_LOCAL_LLVM_PREFIX}/lib/cmake/mlir")
endif()
find_package(LLVM ${ASK_LLVM_CMAKE_VERSION} EXACT CONFIG REQUIRED)
find_package(MLIR ${ASK_LLVM_CMAKE_VERSION} EXACT CONFIG REQUIRED HINTS "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir")
if(NOT MLIR_INSTALL_PREFIX STREQUAL LLVM_INSTALL_PREFIX)
  message(FATAL_ERROR "LLVM and MLIR must be from the same SDK installation")
endif()
if(NOT LLVM_PACKAGE_VERSION STREQUAL ASK_LLVM_PACKAGE_VERSION)
  message(FATAL_ERROR "Expected LLVM ${ASK_LLVM_PACKAGE_VERSION} from cmake/llvm.lock.json, found ${LLVM_PACKAGE_VERSION}")
endif()
file(READ "${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h" ASK_LLVM_VCS_HEADER)
string(FIND "${ASK_LLVM_VCS_HEADER}" "${ASK_LLVM_REVISION}" ASK_LLVM_REVISION_MATCH)
if(ASK_LLVM_REVISION_MATCH EQUAL -1)
  message(FATAL_ERROR "LLVM SDK revision does not match cmake/llvm.lock.json; run `python3 tools/fetch_deps.py --only llvm`")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASK_WITH_HOST 0)
if(ASK_TARGET_BACKENDS STREQUAL "host")
  set(ASK_WITH_HOST 1)
endif()
string(SHA256 ASK_BUILD_ID "${CMAKE_CURRENT_BINARY_DIR};${PROJECT_VERSION};${ASK_LLVM_REVISION};${ASK_WITH_HOST}")
configure_file(cmake/BuildConfig.h.in generated/ask/BuildConfig.h @ONLY)

separate_arguments(ASK_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
llvm_map_components_to_libnames(ASK_LLVM_LIBS
  Core Support AsmParser BitReader BitWriter Analysis Target MC CodeGen
  native nativecodegen Passes InstCombine ScalarOpts TransformUtils)

add_library(ask_core STATIC
  lib/IR/ASKDialect.cpp lib/IR/ASKOps.cpp lib/Frontend/Prepare.cpp lib/Frontend/Compiler.cpp lib/Frontend/Partition.cpp
  lib/LLIR/Import.cpp lib/LLIR/Optimize.cpp
  lib/Lowering/Host.cpp lib/Lowering/CUDA.cpp
  lib/Transforms/Optimize.cpp)
add_library(ASK::Core ALIAS ask_core)
target_compile_features(ask_core PUBLIC cxx_std_17)
set_target_properties(ask_core PROPERTIES EXPORT_NAME Core POSITION_INDEPENDENT_CODE ON)
target_include_directories(ask_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
foreach(ASK_DEPENDENCY_INCLUDE IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS)
  target_include_directories(ask_core SYSTEM PUBLIC $<BUILD_INTERFACE:${ASK_DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_options(ask_core PUBLIC ${ASK_LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ask_core PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif()
target_link_libraries(ask_core PUBLIC MLIRIR MLIRParser MLIRFuncDialect MLIRSideEffectInterfaces MLIRSupport ${ASK_LLVM_LIBS})

add_executable(ask-opt tools/ask-opt.cpp)
target_link_libraries(ask-opt PRIVATE ASK::Core)
set_target_properties(ask-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

add_library(ask_compiler SHARED lib/Frontend/CompilerAPI.cpp)
add_library(ASK::Compiler ALIAS ask_compiler)
set_target_properties(ask_compiler PROPERTIES EXPORT_NAME Compiler
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
target_compile_features(ask_compiler PUBLIC cxx_std_17)
target_include_directories(ask_compiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ask_compiler PRIVATE ASK::Core ${CMAKE_DL_LIBS})

add_library(ask_runtime SHARED lib/Runtime/CUDA.cpp)
add_library(ASK::Runtime ALIAS ask_runtime)
set_target_properties(ask_runtime PROPERTIES EXPORT_NAME Runtime POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
target_compile_features(ask_runtime PUBLIC cxx_std_17)
target_include_directories(ask_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ask_runtime PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS ask_core EXPORT ASKCoreTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core)
install(TARGETS ask_compiler EXPORT ASKCompilerTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Compiler)
install(TARGETS ask_runtime EXPORT ASKRuntimeTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime)
install(TARGETS ask-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Core)
install(DIRECTORY include/ask DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Core)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/ask/BuildConfig.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ask COMPONENT Core)
install(EXPORT ASKCoreTargets NAMESPACE ASK:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
install(EXPORT ASKCompilerTargets NAMESPACE ASK:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Compiler)
install(EXPORT ASKRuntimeTargets NAMESPACE ASK:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
configure_package_config_file(cmake/ASKConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
install(FILES cmake/llvm.lock.json DESTINATION ${CMAKE_INSTALL_DATADIR}/ask COMPONENT Core)

option(ASK_TEST_CUDA "Run hardware CUDA runtime tests (requires an NVIDIA GPU)" OFF)
if(BUILD_TESTING)
  add_executable(ask-compiler-test examples/cpp/compiler.cpp)
  target_link_libraries(ask-compiler-test PRIVATE ASK::Compiler ${CMAKE_DL_LIBS})
  add_test(NAME ask-compiler COMMAND ask-compiler-test "${CMAKE_CURRENT_BINARY_DIR}/compiler-test.o")
  add_executable(ask-core-test examples/cpp/frontend.cpp)
  target_link_libraries(ask-core-test PRIVATE ASK::Core)
  add_test(NAME ask-core COMMAND ask-core-test --optimize)
  if(ASK_TEST_CUDA)
    add_executable(ask-runtime-test examples/cpp-runtime/main.cpp)
    target_link_libraries(ask-runtime-test PRIVATE ASK::Runtime)
    add_test(NAME ask-runtime COMMAND ask-runtime-test)
  endif()
endif()

message(STATUS "ASK target backends: ${ASK_TARGET_BACKENDS}; LLVM/MLIR ${LLVM_PACKAGE_VERSION}")
message(STATUS "ASK interfaces: native Compiler, Runtime and Core; no Python build dependency")
