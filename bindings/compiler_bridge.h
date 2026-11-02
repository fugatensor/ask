#pragma once
#include <string>
#include <utility>

// Keep LLVM's no-RTTI C++ types out of nanobind's RTTI-enabled translation unit.
std::string compilerCapabilitiesJSON();
std::pair<std::string, std::string> prepareJSON(const std::string &source,
    bool optimize, bool foldConstants, bool eliminateDeadCode, bool fuseElementwise);
std::pair<std::string, std::string> partitionJSON(const std::string &source,
    bool optimize, bool foldConstants, bool eliminateDeadCode,
    bool fuseElementwise);
