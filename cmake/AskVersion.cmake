# One native release version, shared by installed libraries and capabilities.
get_filename_component(_ask_version_file "${CMAKE_CURRENT_LIST_DIR}/../VERSION" ABSOLUTE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ask_version_file}")
file(READ "${_ask_version_file}" _ask_version_line)
string(STRIP "${_ask_version_line}" _ask_version_line)
if(NOT _ask_version_line MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)$")
  message(FATAL_ERROR "VERSION must contain a major.minor.patch release version")
endif()
set(ASK_VERSION "${CMAKE_MATCH_1}")
