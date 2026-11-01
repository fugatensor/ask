# Everything needed to adapt Triton lives beside its exporter and lock records.
set(ASK_TRITON_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python in the locked Triton worker environment")
if(NOT ASK_TRITON_PYTHON_EXECUTABLE OR NOT EXISTS "${ASK_TRITON_PYTHON_EXECUTABLE}")
  message(FATAL_ERROR "Triton selected: set ASK_TRITON_PYTHON_EXECUTABLE to its environment's Python")
endif()
execute_process(COMMAND "${ASK_TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/worker.py" --probe
  RESULT_VARIABLE probe_status OUTPUT_VARIABLE probe_output ERROR_VARIABLE probe_error TIMEOUT 60)
if(NOT probe_status EQUAL 0)
  message(FATAL_ERROR "Triton environment check failed: ${probe_error}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/environment.json" "${probe_output}")

add_library(ask_adapter_triton OBJECT lib/TritonBoundary.cpp)
set_target_properties(ask_adapter_triton PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ask_adapter_triton PRIVATE cxx_std_17)
target_include_directories(ask_adapter_triton PRIVATE "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ask_adapter_triton SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_options(ask_adapter_triton PRIVATE ${ASK_LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ask_adapter_triton PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif()
add_custom_target(ask_exporter_triton ALL SOURCES worker.py fusion.py)
install(FILES worker.py fusion.py registration.json dependencies.lock.json requirements.lock
  DESTINATION "${ASK_ADAPTER_INSTALL_DIR}/triton" COMPONENT Core)

if(BUILD_TESTING)
  add_test(NAME ask-triton-worker COMMAND "${Python_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -v)
  set_tests_properties(ask-triton-worker PROPERTIES ENVIRONMENT
    "ASK_TEST_TRITON_PYTHON=${ASK_TRITON_PYTHON_EXECUTABLE};PYTHONPATH=${PROJECT_BINARY_DIR}/python:${PROJECT_SOURCE_DIR}/python")
  add_test(NAME ask-triton-pipeline COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_pipeline.py"
    "$<TARGET_FILE:ask-opt>" "${ASK_TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/worker.py")
  add_test(NAME ask-triton-fusion COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_fusion.py"
    "$<TARGET_FILE:ask-opt>" "${ASK_TRITON_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/worker.py")
endif()
