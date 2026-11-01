#pragma once

#include "ask/Transforms/Optimize.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace ask {
struct PreparedSource {
  std::string ir;
  llvm::json::Object report;
};

// Both the Python extension and CLI report the same build capabilities.
llvm::json::Object compilerCapabilities();
llvm::json::Array externalRequests(mlir::ModuleOp module, bool fusion = false);
// Each call owns its context and diagnostics; no global CLI state or subprocess.
llvm::Expected<PreparedSource> prepareSource(
    llvm::StringRef source, bool optimize = false,
    const OptimizationOptions &options = {});
} // namespace ask
