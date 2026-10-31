cmake_minimum_required(VERSION 3.28)
project(ASK VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AskCompilerSelection)

set(ASK_THIRD_PARTY_COMPILERS "none" CACHE STRING "none, all, or semicolon-separated compiler IDs")
set(ASK_TARGET_BACKENDS "host" CACHE STRING "none or host (initial implementation)")
set(ASK_TRITON_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python executable in the Triton worker environment")
set(ASK_SUPPORTED_COMPILERS triton)
ask_resolve_compilers("${ASK_THIRD_PARTY_COMPILERS}" "${ASK_SUPPORTED_COMPILERS}" ASK_ENABLED_COMPILERS)
if(NOT ASK_TARGET_BACKENDS STREQUAL "none" AND NOT ASK_TARGET_BACKENDS STREQUAL "host")
  message(FATAL_ERROR "This draft implements ASK_TARGET_BACKENDS=none or host; requested '${ASK_TARGET_BACKENDS}'")
endif()

# The first tested SDK. Do not accidentally combine MLIR with the system LLVM.
find_package(LLVM 18.1.8 EXACT CONFIG REQUIRED)
find_package(MLIR 18.1.8 EXACT CONFIG REQUIRED HINTS "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASK_WITH_TRITON 0)
if("triton" IN_LIST ASK_ENABLED_COMPILERS)
  if(NOT ASK_TRITON_PYTHON_EXECUTABLE OR NOT EXISTS "${ASK_TRITON_PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "Triton selected: set ASK_TRITON_PYTHON_EXECUTABLE to its environment's Python")
  endif()
  execute_process(COMMAND "${ASK_TRITON_PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapters/triton/worker.py" --probe
    RESULT_VARIABLE probe_status OUTPUT_VARIABLE probe_output ERROR_VARIABLE probe_error TIMEOUT 60)
  if(NOT probe_status EQUAL 0)
    message(FATAL_ERROR "Triton environment check failed: ${probe_error}")
  endif()
  set(ASK_WITH_TRITON 1)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/triton-environment.json" "${probe_output}")
  add_custom_target(ask_exporter_triton ALL
    COMMAND "${CMAKE_COMMAND}" -E echo "Triton worker enabled; environment validated at configure time")
  install(FILES adapters/triton/worker.py DESTINATION ${CMAKE_INSTALL_DATADIR}/ask/adapters/triton COMPONENT CompilerTriton)
endif()

set(ASK_WITH_HOST 0)
if(ASK_TARGET_BACKENDS STREQUAL "host")
  set(ASK_WITH_HOST 1)
endif()
configure_file(cmake/BuildConfig.h.in generated/ask/BuildConfig.h @ONLY)

add_library(ask_core STATIC
  lib/IR/ASKDialect.cpp lib/IR/ASKOps.cpp lib/Frontend/Prepare.cpp
  lib/LLIR/Import.cpp lib/Lowering/Host.cpp)
add_library(ASK::Core ALIAS ask_core)
set_target_properties(ask_core PROPERTIES EXPORT_NAME Core POSITION_INDEPENDENT_CODE ON)
target_include_directories(ask_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(ask_core SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(ask_core PUBLIC ${LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ask_core PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif()
llvm_map_components_to_libnames(ASK_LLVM_LIBS Core Support AsmParser BitReader BitWriter Analysis Target MC CodeGen native nativecodegen)
target_link_libraries(ask_core PUBLIC MLIRIR MLIRParser MLIRFuncDialect MLIRSideEffectInterfaces MLIRSupport ${ASK_LLVM_LIBS})

add_executable(ask-opt tools/ask-opt.cpp)
target_link_libraries(ask-opt PRIVATE ASK::Core)
set_target_properties(ask-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

install(TARGETS ask_core EXPORT ASKCoreTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core)
install(TARGETS ask-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Core)
install(DIRECTORY include/ask DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Core)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/ask/BuildConfig.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ask COMPONENT Core)
install(EXPORT ASKCoreTargets NAMESPACE ASK:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)
configure_package_config_file(cmake/ASKConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ASKConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ASKConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ASK COMPONENT Core)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME ask-python COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -v)
  set_tests_properties(ask-python PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;ASK_OPT=$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-native COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native.py" "$<TARGET_FILE:ask-opt>")
  add_test(NAME ask-build-selection COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_build_selection.py")
endif()
message(STATUS "ASK compiler selection: '${ASK_THIRD_PARTY_COMPILERS}' -> [${ASK_ENABLED_COMPILERS}]")
message(STATUS "ASK target backends: ${ASK_TARGET_BACKENDS}; LLVM/MLIR ${LLVM_PACKAGE_VERSION}")
