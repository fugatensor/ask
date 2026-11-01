#include "compiler_bridge.h"
#include "ask/Frontend/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include <stdexcept>

std::string compilerCapabilitiesJSON() {
  return llvm::formatv("{0}", llvm::json::Value(ask::compilerCapabilities())).str();
}

std::pair<std::string, std::string> prepareJSON(const std::string &source,
    bool optimize, bool foldConstants, bool eliminateDeadCode, bool fuseElementwise) {
  auto result = ask::prepareSource(source, optimize,
      {foldConstants, eliminateDeadCode, fuseElementwise});
  if (!result) throw std::runtime_error(llvm::toString(result.takeError()));
  return {std::move(result->ir),
          llvm::formatv("{0}", llvm::json::Value(std::move(result->report))).str()};
}
