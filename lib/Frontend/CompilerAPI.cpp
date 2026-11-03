#include "ask/Frontend/CompilerAPI.h"
#include "ask/Frontend/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include <stdexcept>
#include <dlfcn.h>
#include <filesystem>
#include <set>

namespace ask {

std::string compilerLibraryPath() {
  Dl_info library{};
  if (!dladdr(reinterpret_cast<void *>(&compilerLibraryPath), &library) || !library.dli_fname)
    throw std::runtime_error("cannot identify the loaded ASK compiler library");
  return std::filesystem::canonical(library.dli_fname).string();
}

std::string compilerCapabilitiesJSON() {
  return llvm::formatv("{0}", llvm::json::Value(ask::compilerCapabilities())).str();
}

namespace {
ask::OptimizationOptions coreOptions(const compiler::OptimizationOptions &options) {
  return {options.foldConstants, options.eliminateDeadCode, options.fuseElementwise};
}

PreparedSource unwrap(llvm::Expected<PreparedSource> result) {
  if (!result) throw std::runtime_error(llvm::toString(result.takeError()));
  return std::move(*result);
}

void requireResult(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("invalid ASK compiler result: " + message);
}

// Removing a report prefix must also relocate its float paths. Otherwise
// integral doubles and negative zero silently change their JSON representation.
std::string takeJSON(PreparedSource &result, llvm::StringRef field,
                     llvm::json::Value::Kind kind) {
  auto *value = result.report.get(field);
  requireResult(value && value->kind() == kind, "missing or invalid " + field.str());
  JSONFloatPaths relativePaths;
  for (const auto &path : result.floatPaths)
    if (!path.empty() && path.front() == field)
      relativePaths.emplace(path.begin() + 1, path.end());
  auto text = reportJSON(*value, relativePaths);
  result.report.erase(field);
  return text;
}

std::string takeDiagnostics(PreparedSource &result) {
  return reportJSON(llvm::json::Value(std::move(result.report)), result.floatPaths);
}
} // namespace

namespace compiler {
PrepareResult prepare(const std::string &source, const PrepareOptions &options) {
  auto result = unwrap(ask::prepareSource(source, options.optimize, coreOptions(options.optimization)));
  requireResult(!result.ir.empty(), "prepare returned empty IR");
  PrepareResult prepared;
  prepared.ir = std::move(result.ir);
  prepared.shapeSnapshotJSON = takeJSON(result, "shape_snapshot", llvm::json::Value::Object);
  prepared.externalRequestsJSON = takeJSON(result, "external_requests", llvm::json::Value::Array);
  prepared.diagnosticsJSON = takeDiagnostics(result);
  return prepared;
}

PartitionResult partition(const std::string &source, const PartitionOptions &options) {
  ask::PartitionOptions core;
  core.optimize = options.optimize;
  core.optimization = coreOptions(options.optimization);
  auto result = unwrap(ask::partitionSource(source, core));
  requireResult(!result.ir.empty(), "partition returned empty IR");
  PartitionResult partitioned;
  partitioned.ir = std::move(result.ir);
  partitioned.shapeSnapshotJSON = takeJSON(result, "shape_snapshot", llvm::json::Value::Object);
  partitioned.externalRequestsJSON = takeJSON(result, "external_requests", llvm::json::Value::Array);
  const auto *sources = result.report.getObject("step_sources");
  requireResult(sources, "missing or invalid step_sources");
  for (const auto &entry : *sources) {
    auto text = entry.second.getAsString();
    requireResult(text && !text->empty(), "empty or invalid step source");
    partitioned.stepSources.emplace(entry.first.str(), text->str());
  }
  const auto *plan = result.report.getObject("execution_plan");
  const auto *steps = plan ? plan->getArray("steps") : nullptr;
  requireResult(steps && steps->size() == partitioned.stepSources.size(),
                "plan steps and sources disagree");
  std::set<std::string> seen;
  for (const auto &step : *steps) {
    const auto *object = step.getAsObject();
    auto id = object ? object->getString("id") : std::nullopt;
    requireResult(id && partitioned.stepSources.count(id->str()) && seen.insert(id->str()).second,
                  "plan step has no unique source");
  }
  result.report.erase("step_sources");
  partitioned.executionPlanJSON = takeJSON(result, "execution_plan", llvm::json::Value::Object);
  partitioned.diagnosticsJSON = takeDiagnostics(result);
  return partitioned;
}

CodegenResult codegen(const std::string &source, const CodegenOptions &options) {
  ask::CodegenOptions core;
  switch (options.target) {
  case CodegenTarget::HostLLVM: core.target = ask::CodegenTarget::HostLLVM; break;
  case CodegenTarget::HostObject: core.target = ask::CodegenTarget::HostObject; break;
  case CodegenTarget::CUDALLVM: core.target = ask::CodegenTarget::CUDALLVM; break;
  default: throw std::runtime_error("unknown ASK code generation target");
  }
  if (options.target != CodegenTarget::HostObject && !options.outputPath.empty())
    throw std::runtime_error("outputPath is only valid for host object generation");
  core.outputPath = options.outputPath;
  core.architecture = options.architecture;
  core.blockSize = options.blockSize;
  core.ptxVersion = options.ptxVersion;
  core.optimize = options.optimize;
  core.optimization = coreOptions(options.optimization);
  auto result = unwrap(ask::codegenSource(source, core));
  requireResult(result.ir.empty() == (options.target == CodegenTarget::HostObject),
                "code generation output disagrees with target");
  CodegenResult generated;
  generated.target = options.target;
  generated.outputPath = options.outputPath;
  generated.ir = std::move(result.ir);
  generated.shapeSnapshotJSON = takeJSON(result, "shape_snapshot", llvm::json::Value::Object);
  generated.externalRequestsJSON = takeJSON(result, "external_requests", llvm::json::Value::Array);
  if (options.target == CodegenTarget::CUDALLVM) {
    requireResult(!result.report.get("abi"), "CUDA result contains Host ABI");
    generated.cudaArtifactJSON = takeJSON(result, "cuda_artifact", llvm::json::Value::Object);
  } else {
    requireResult(!result.report.get("cuda_artifact"), "Host result contains CUDA artifact");
    generated.abiJSON = takeJSON(result, "abi", llvm::json::Value::Object);
  }
  generated.diagnosticsJSON = takeDiagnostics(result);
  return generated;
}
} // namespace compiler

std::pair<std::string, std::string> prepareJSON(const std::string &source,
    bool optimize, bool foldConstants, bool eliminateDeadCode, bool fuseElementwise) {
  auto result = ask::prepareSource(source, optimize,
      {foldConstants, eliminateDeadCode, fuseElementwise});
  if (!result) throw std::runtime_error(llvm::toString(result.takeError()));
  return {std::move(result->ir),
          ask::reportJSON(llvm::json::Value(std::move(result->report)), result->floatPaths)};
}

std::pair<std::string, std::string> partitionJSON(const std::string &source,
    bool optimize, bool foldConstants, bool eliminateDeadCode,
    bool fuseElementwise) {
  ask::PartitionOptions options;
  options.optimize = optimize;
  options.optimization = {foldConstants, eliminateDeadCode, fuseElementwise};
  auto result = ask::partitionSource(source, options);
  if (!result) throw std::runtime_error(llvm::toString(result.takeError()));
  return {std::move(result->ir),
          ask::reportJSON(llvm::json::Value(std::move(result->report)), result->floatPaths)};
}

std::pair<std::string, std::string> codegenJSON(const std::string &source,
    const std::string &target, const std::string &output, unsigned architecture,
    unsigned blockSize, unsigned ptxVersion, bool optimize) {
  ask::CodegenOptions options;
  if (target == "host-llvm") options.target = ask::CodegenTarget::HostLLVM;
  else if (target == "host-object") options.target = ask::CodegenTarget::HostObject;
  else if (target == "cuda-llvm") options.target = ask::CodegenTarget::CUDALLVM;
  else throw std::runtime_error("unknown ASK code generation target: " + target);
  options.outputPath = output;
  options.architecture = architecture;
  options.blockSize = blockSize;
  options.ptxVersion = ptxVersion;
  options.optimize = optimize;
  auto result = ask::codegenSource(source, options);
  if (!result) throw std::runtime_error(llvm::toString(result.takeError()));
  return {std::move(result->ir),
          ask::reportJSON(llvm::json::Value(std::move(result->report)), result->floatPaths)};
}

} // namespace ask
