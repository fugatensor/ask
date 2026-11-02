#pragma once
#include <string>
#include <utility>

// Compiler facade independent of LLVM/MLIR headers and Python.
// Failures throw std::runtime_error; returned reports are JSON protocol text.
namespace ask {
std::string compilerCapabilitiesJSON();
// Actual loaded shared-library path, for content-addressed compilation caches.
std::string compilerLibraryPath();
std::pair<std::string, std::string> prepareJSON(const std::string &source,
    bool optimize = false, bool foldConstants = true, bool eliminateDeadCode = true,
    bool fuseElementwise = true);
std::pair<std::string, std::string> partitionJSON(const std::string &source,
    bool optimize = true, bool foldConstants = true, bool eliminateDeadCode = true,
    bool fuseElementwise = true);
std::pair<std::string, std::string> codegenJSON(const std::string &source,
    const std::string &target, const std::string &output = "", unsigned architecture = 0,
    unsigned blockSize = 128, unsigned ptxVersion = 80, bool optimize = true);

} // namespace ask
