#pragma once

#include "ask/Transforms/Optimize.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <set>
#include <vector>

namespace ask {
using JSONFloatPaths = std::set<std::vector<std::string>>;
struct PreparedSource {
  std::string ir;
  llvm::json::Object report;
  JSONFloatPaths floatPaths;
};

// Both the Python extension and CLI report the same build capabilities.
llvm::json::Object compilerCapabilities();
llvm::json::Array externalRequests(mlir::ModuleOp module, JSONFloatPaths *floatPaths = nullptr);
// LLVM's JSON writer emits integral doubles as integers, losing constexpr
// type and negative zero. Preserve the MLIR float type at the protocol edge.
std::string reportJSON(const llvm::json::Value &report,
                       const JSONFloatPaths &floatPaths = {}, unsigned indent = 0);
// Each call owns its context and diagnostics; no global CLI state or subprocess.
llvm::Expected<PreparedSource> prepareSource(
    llvm::StringRef source, bool optimize = false,
    const OptimizationOptions &options = {});

enum class CodegenTarget { HostLLVM, HostObject, CUDALLVM };
struct CodegenOptions {
  CodegenTarget target = CodegenTarget::HostLLVM;
  bool optimize = true;
  OptimizationOptions optimization;
  unsigned architecture = 0;
  unsigned blockSize = 128;
  unsigned ptxVersion = 80;
  // Required only for HostObject. Other modes return LLVM assembly in ir.
  std::string outputPath;
};

// Parse, prepare/optimize and lower in one context. Native callers and the CLI
// share this entry; no Python object or companion executable is involved.
llvm::Expected<PreparedSource> codegenSource(
    llvm::StringRef source, const CodegenOptions &options);

struct PartitionOptions {
  bool optimize = true;
  OptimizationOptions optimization;
};

// Partition one static tensor graph into explicit kernel functions. This only
// constructs the execution contract; it does not invoke external compilers.
llvm::Expected<PreparedSource> partitionSource(
    llvm::StringRef source, const PartitionOptions &options = {});
} // namespace ask
