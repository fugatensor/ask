cmake_minimum_required(VERSION 3.28)
project(ASK VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AskCompilerSelection)
include(AskCompilerRegistry)

set(ASK_THIRD_PARTY_COMPILERS "none" CACHE STRING "none, all, or semicolon-separated compiler IDs")
set(ASK_TARGET_BACKENDS "host" CACHE STRING "none or host (initial implementation)")
ask_resolve_compilers("${ASK_THIRD_PARTY_COMPILERS}" "${ASK_SUPPORTED_COMPILERS}" ASK_ENABLED_COMPILERS)
if(NOT ASK_TARGET_BACKENDS STREQUAL "none" AND NOT ASK_TARGET_BACKENDS STREQUAL "host")
  message(FATAL_ERROR "This draft implements ASK_TARGET_BACKENDS=none or host; requested '${ASK_TARGET_BACKENDS}'")
endif()

# Native Python bindings are part of every ASK build, including none.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nanobind.lock.json" ASK_NANOBIND_LOCK)
string(JSON ASK_NANOBIND_VERSION GET "${ASK_NANOBIND_LOCK}" version)
string(JSON ASK_NANOBIND_DIRECTORY GET "${ASK_NANOBIND_LOCK}" directory)
if(NOT nanobind_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE ASK_NANOBIND_QUERY_STATUS OUTPUT_VARIABLE ASK_NANOBIND_CMAKE_DIR
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(ASK_NANOBIND_QUERY_STATUS EQUAL 0)
    set(nanobind_DIR "${ASK_NANOBIND_CMAKE_DIR}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/${ASK_NANOBIND_DIRECTORY}/cmake/nanobind-config.cmake")
    set(nanobind_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/${ASK_NANOBIND_DIRECTORY}/cmake")
  else()
    message(FATAL_ERROR "ASK requires nanobind ${ASK_NANOBIND_VERSION} in every build. Run python3 tools/fetch_nanobind.py or install the locked build dependencies into the selected Python environment.")
  endif()
endif()
find_package(nanobind CONFIG REQUIRED)
file(STRINGS "${nanobind_DIR}/../include/nanobind/nanobind.h" ASK_NANOBIND_HEADER_VERSION REGEX "^#define NB_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "#define NB_VERSION_${part} ([0-9]+)" unused "${ASK_NANOBIND_HEADER_VERSION}")
  set(ASK_NB_${part} "${CMAKE_MATCH_1}")
endforeach()
if(NOT "${ASK_NB_MAJOR}.${ASK_NB_MINOR}.${ASK_NB_PATCH}" VERSION_EQUAL ASK_NANOBIND_VERSION)
  message(FATAL_ERROR "nanobind headers do not match cmake/nanobind.lock.json (${ASK_NANOBIND_VERSION})")
endif()

# LLVM and MLIR must come from the pinned SDK used by the tested producer.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm.lock.json" ASK_LLVM_LOCK)
string(JSON ASK_LLVM_CMAKE_VERSION GET "${ASK_LLVM_LOCK}" cmake_version)
string(JSON ASK_LLVM_PACKAGE_VERSION GET "${ASK_LLVM_LOCK}" version)
string(JSON ASK_LLVM_REVISION GET "${ASK_LLVM_LOCK}" revision)
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
  message(FATAL_ERROR "LLVM SDK revision does not match cmake/llvm.lock.json; run tools/fetch_llvm.py")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASK_WITH_HOST 0)
if(ASK_TARGET_BACKENDS STREQUAL "host")
  set(ASK_WITH_HOST 1)
endif()
string(SHA256 ASK_BUILD_ID "${CMAKE_CURRENT_BINARY_DIR};${PROJECT_VERSION};${ASK_LLVM_REVISION};${ASK_ENABLED_COMPILERS};${ASK_WITH_HOST};${Python_SOABI}")
configure_file(cmake/BuildConfig.h.in generated/ask/BuildConfig.h @ONLY)

add_library(ask_core STATIC
  lib/IR/ASKDialect.cpp lib/IR/ASKOps.cpp lib/Frontend/Prepare.cpp lib/Frontend/Compiler.cpp
  lib/LLIR/Import.cpp lib/LLIR/Optimize.cpp lib/LLIR/Fuse.cpp lib/LLIR/Inline.cpp
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
separate_arguments(ASK_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(ask_core PUBLIC ${ASK_LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ask_core PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif()
llvm_map_components_to_libnames(ASK_LLVM_LIBS Core Support AsmParser BitReader BitWriter Analysis Target MC CodeGen native nativecodegen Passes InstCombine ScalarOpts TransformUtils Linker)
target_link_libraries(ask_core PUBLIC MLIRIR MLIRParser MLIRFuncDialect MLIRSideEffectInterfaces MLIRSupport ${ASK_LLVM_LIBS})

add_executable(ask-opt tools/ask-opt.cpp)
target_link_libraries(ask-opt PRIVATE ASK::Core)
set_target_properties(ask-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

if(SKBUILD)
  set(ASK_PYTHON_INSTALL_DIR "ask")
  set(ASK_ADAPTER_INSTALL_DIR "ask/share/ask/adapters")
  install(TARGETS ask-opt RUNTIME DESTINATION ask/_bin COMPONENT Core)
else()
  set(ASK_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/ask")
  set(ASK_ADAPTER_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/ask/adapters")
endif()
ask_configure_compilers()

# Only this assembly target depends on selected compiler implementations.
# ASK::Core can still be linked by itself without any producer implementation.
add_library(ask_compilers STATIC lib/Adapters/Registry.cpp ${ASK_ADAPTER_OBJECTS})
add_library(ASK::Compilers ALIAS ask_compilers)
set_target_properties(ask_compilers PROPERTIES EXPORT_NAME Compilers POSITION_INDEPENDENT_CODE ON)
target_compile_features(ask_compilers PUBLIC cxx_std_17)
target_include_directories(ask_compilers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
foreach(ASK_DEPENDENCY_INCLUDE IN LISTS LLVM_INCLUDE_DIRS)
  target_include_directories(ask_compilers SYSTEM PUBLIC $<BUILD_INTERFACE:${ASK_DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_options(ask_compilers PUBLIC ${ASK_LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ask_compilers PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif()
target_link_libraries(ask_compilers PUBLIC ${ASK_LLVM_LIBS})
target_link_libraries(ask-opt PRIVATE ASK::Compilers)
install(TARGETS ask_compilers EXPORT ASKCoreTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core)

add_library(ask_runtime STATIC lib/Runtime/CUDA.cpp)
add_library(ASK::Runtime ALIAS ask_runtime)
set_target_properties(ask_runtime PROPERTIES EXPORT_NAME Runtime POSITION_INDEPENDENT_CODE ON)
target_compile_features(ask_runtime PUBLIC cxx_std_17)
target_include_directories(ask_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ask_runtime PRIVATE ${CMAKE_DL_LIBS})

nanobind_add_module(_native NB_STATIC bindings/compiler.cpp bindings/compiler_bridge.cpp bindings/runtime.cpp)
target_link_libraries(_native PRIVATE ASK::Core ASK::Runtime ASK::Compilers)
# LLVM is built without RTTI, but nanobind's own bound runtime classes require it.
# The bridge keeps LLVM headers/types in a separate no-RTTI translation unit.
set_source_files_properties(bindings/compiler.cpp bindings/runtime.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang>:-frtti>")
set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/ask")
# A complete build-tree package, also usable alongside source via extend_path.
file(GLOB ASK_PYTHON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/ask/*.py")
foreach(source IN LISTS ASK_PYTHON_SOURCES)
  get_filename_component(name "${source}" NAME)
  configure_file("${source}" "${CMAKE_CURRENT_BINARY_DIR}/python/ask/${name}" COPYONLY)
endforeach()
install(TARGETS _native LIBRARY DESTINATION "${ASK_PYTHON_INSTALL_DIR}" COMPONENT Core)
install(DIRECTORY python/ask/ DESTINATION "${ASK_PYTHON_INSTALL_DIR}" COMPONENT Core FILES_MATCHING PATTERN "*.py")
install(TARGETS ask_runtime EXPORT ASKCoreTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core)

install(TARGETS ask_core EXPORT ASKCoreTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core)
if(NOT SKBUILD)
  install(TARGETS ask-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Core)
endif()
install(DIRECTORY include/ask DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Core)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/ask/BuildConfig.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ask COMPONENT Core)
install(EXPORT ASKCoreTargets NAMESPACE ASK:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
configure_package_config_file(cmake/ASKConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
install(FILES cmake/llvm.lock.json cmake/nanobind.lock.json DESTINATION ${CMAKE_INSTALL_DATADIR}/ask COMPONENT Core)

if(BUILD_TESTING)
  add_executable(ask-compiler-adapter-test tests/test_compiler_adapter.cpp)
  target_link_libraries(ask-compiler-adapter-test PRIVATE ASK::Core)
  add_test(NAME ask-compiler-adapter COMMAND ask-compiler-adapter-test)
  add_executable(ask-boundary-inline-test tests/test_boundary_inline.cpp)
  target_link_libraries(ask-boundary-inline-test PRIVATE ASK::Core)
  add_test(NAME ask-boundary-inline COMMAND ask-boundary-inline-test)
  add_test(NAME ask-boundary-execution COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_boundary_execution.py"
    "$<TARGET_FILE:ask-boundary-inline-test>" "${LLVM_TOOLS_BINARY_DIR}")
  add_test(NAME ask-python COMMAND "${Python_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -v)
  set_tests_properties(ask-python PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python;ASK_OPT=$<TARGET_FILE:ask-opt>;ASK_NATIVE_REQUIRED=1")
  add_test(NAME ask-native COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native.py" "$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-cuda-lowering COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_lowering.py" "$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-optimization COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_optimization.py" "$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-fusion COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fusion.py" "$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-llir-optimization COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_llir_optimization.py" "$<TARGET_FILE:ask-opt>")
  set_tests_properties(ask-llir-optimization PROPERTIES ENVIRONMENT "ASK_LLVM_TOOLS_DIR=${LLVM_TOOLS_BINARY_DIR}")
  add_test(NAME ask-build-selection COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_build_selection.py")
  add_test(NAME ask-fetch-nanobind COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fetch_nanobind.py")
  add_test(NAME ask-fetch-llvm COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fetch_llvm.py")

endif()
message(STATUS "ASK compiler selection: '${ASK_THIRD_PARTY_COMPILERS}' -> [${ASK_ENABLED_COMPILERS}]")
message(STATUS "ASK target backends: ${ASK_TARGET_BACKENDS}; LLVM/MLIR ${LLVM_PACKAGE_VERSION}")

message(STATUS "ASK native Python: ${Python_EXECUTABLE} (${Python_SOABI}); nanobind ${ASK_NANOBIND_VERSION}")
