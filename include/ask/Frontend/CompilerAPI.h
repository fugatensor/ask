#pragma once
#include <map>
#include <optional>
#include <string>
#include <utility>

// Compiler facade independent of LLVM/MLIR headers and Python.
// Failures throw std::runtime_error. Calls own independent compiler contexts.
namespace ask {
std::string compilerCapabilitiesJSON();
// Actual loaded shared-library path, for content-addressed compilation caches.
std::string compilerLibraryPath();

namespace compiler {
struct OptimizationOptions {
  bool foldConstants = true;
  bool eliminateDeadCode = true;
  bool fuseElementwise = true;
};

struct PrepareOptions {
  bool optimize = false;
  OptimizationOptions optimization;
};

struct PartitionOptions {
  bool optimize = true;
  OptimizationOptions optimization;
};

enum class CodegenTarget { HostLLVM, HostObject, CUDALLVM };
struct CodegenOptions {
  CodegenTarget target = CodegenTarget::HostLLVM;
  bool optimize = true;
  OptimizationOptions optimization;
  unsigned architecture = 0;
  unsigned blockSize = 128;
  unsigned ptxVersion = 80;
  // Required for HostObject and empty for LLVM assembly targets.
  std::string outputPath;
};

// JSON fields retain their versioned wire contracts. diagnosticsJSON contains
// explanatory metadata only; consumers use the named fields for compilation.
struct PrepareResult {
  std::string ir;
  std::string shapeSnapshotJSON;
  std::string externalRequestsJSON;
  std::string diagnosticsJSON;
};

struct PartitionResult {
  std::string ir;
  std::string shapeSnapshotJSON;
  std::string externalRequestsJSON;
  std::string executionPlanJSON;
  // Keys are execution-plan step IDs; values are standalone ASK source modules.
  std::map<std::string, std::string> stepSources;
  std::string diagnosticsJSON;
};

struct CodegenResult {
  CodegenTarget target;
  // HostObject writes this path and returns empty ir. Other targets return LLVM.
  std::string outputPath;
  std::string ir;
  std::string shapeSnapshotJSON;
  std::string externalRequestsJSON;
  // Host targets return abiJSON; CUDALLVM returns cudaArtifactJSON.
  std::optional<std::string> abiJSON;
  std::optional<std::string> cudaArtifactJSON;
  std::string diagnosticsJSON;
};

PrepareResult prepare(const std::string &source, const PrepareOptions &options = {});
PartitionResult partition(const std::string &source, const PartitionOptions &options = {});
CodegenResult codegen(const std::string &source, const CodegenOptions &options);
} // namespace compiler

// Compatibility wire entrypoints retain their complete JSON reports, as does
// ask-opt. New in-process consumers use the stage-specific results above.
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
