# Triton adapter: self-contained environment probe, boundary verifier, and tests.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.lock.json"
  "${CMAKE_CURRENT_SOURCE_DIR}/requirements.lock")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.lock.json" _ask_triton_lock)
string(JSON _ask_triton_schema GET "${_ask_triton_lock}" schema_version)
string(JSON _ask_triton_compiler GET "${_ask_triton_lock}" compiler name)
string(JSON ASK_TRITON_COMPILER_VERSION GET "${_ask_triton_lock}" compiler version)
string(JSON ASK_TRITON_ADAPTER_VERSION GET "${_ask_triton_lock}" adapter_version)
string(JSON ASK_TRITON_PROTOCOL_VERSION GET "${_ask_triton_lock}" protocol_version)
string(JSON _ask_triton_llvm_revision GET "${_ask_triton_lock}" producer_llvm revision)
string(JSON _ask_triton_llvm_version GET "${_ask_triton_lock}" producer_llvm upstream_version)
string(JSON _ask_triton_reader_revision GET "${_ask_triton_lock}" ask_reader revision)
string(JSON _ask_triton_reader_version GET "${_ask_triton_lock}" ask_reader version)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/registration.json" _ask_triton_registration)
string(JSON _ask_triton_registered_protocol GET "${_ask_triton_registration}" protocol_version)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/requirements.lock" _ask_triton_requirements
  REGEX "^[^# \t].*")
if(NOT _ask_triton_schema EQUAL 1 OR NOT _ask_triton_compiler STREQUAL "triton" OR
   NOT ASK_TRITON_ADAPTER_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$" OR
   NOT ASK_TRITON_PROTOCOL_VERSION EQUAL 1 OR
   NOT ASK_TRITON_PROTOCOL_VERSION EQUAL _ask_triton_registered_protocol OR
   NOT _ask_triton_llvm_revision STREQUAL ASK_LLVM_REVISION OR
   NOT _ask_triton_llvm_version STREQUAL ASK_LLVM_PACKAGE_VERSION OR
   NOT _ask_triton_reader_revision STREQUAL ASK_LLVM_REVISION OR
   NOT _ask_triton_reader_version STREQUAL ASK_LLVM_PACKAGE_VERSION OR
   NOT _ask_triton_requirements STREQUAL "triton==${ASK_TRITON_COMPILER_VERSION}")
  message(FATAL_ERROR "Triton dependency lock disagrees with registration, requirements or LLVM revision")
endif()
configure_file(TritonVersion.h.in generated/TritonVersion.h @ONLY)

set(ASK_TRITON_PYTHON_EXECUTABLE "" CACHE FILEPATH
  "Python in the locked Triton worker environment (defaults to Python_EXECUTABLE)")
set(_ask_triton_python "${ASK_TRITON_PYTHON_EXECUTABLE}")
if(NOT _ask_triton_python)
  set(_ask_triton_python "${Python_EXECUTABLE}")
endif()
if(NOT _ask_triton_python OR NOT EXISTS "${_ask_triton_python}")
  message(FATAL_ERROR "Triton selected: set ASK_TRITON_PYTHON_EXECUTABLE to the Triton environment's Python")
endif()
execute_process(COMMAND "${_ask_triton_python}" "${CMAKE_CURRENT_SOURCE_DIR}/worker.py" --probe
  RESULT_VARIABLE probe_status OUTPUT_VARIABLE probe_output ERROR_VARIABLE probe_error TIMEOUT 60)
if(NOT probe_status EQUAL 0)
  message(FATAL_ERROR
    "Triton environment check failed using '${_ask_triton_python}' "
    "(pass -DASK_TRITON_PYTHON_EXECUTABLE=/path/to/triton-env/bin/python if Triton lives in another venv): ${probe_error}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/environment.json" "${probe_output}")
set(ASK_RESOLVED_TRITON_PYTHON "${_ask_triton_python}" PARENT_SCOPE)

add_library(ask_adapter_triton OBJECT lib/TritonBoundary.cpp)
set_target_properties(ask_adapter_triton PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ask_adapter_triton PRIVATE "${PROJECT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(ask_adapter_triton PRIVATE ask_llvm_options)

add_custom_target(ask_exporter_triton ALL SOURCES worker.py fusion.py)
install(FILES worker.py fusion.py registration.json dependencies.lock.json requirements.lock
  DESTINATION "${ASK_ADAPTER_INSTALL_DIR}/triton" COMPONENT Core)

if(BUILD_TESTING)
  add_test(NAME ask-triton COMMAND "${Python_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_*.py" -v)
  set_tests_properties(ask-triton PROPERTIES ENVIRONMENT
    "ASK_OPT=$<TARGET_FILE:ask-opt>;ASK_TEST_TRITON_PYTHON=${_ask_triton_python};PYTHONPATH=${PROJECT_BINARY_DIR}/python:${PROJECT_SOURCE_DIR}/python:${PROJECT_SOURCE_DIR}/tests")
endif()
