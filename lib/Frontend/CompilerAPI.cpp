#include "ask/Frontend/CompilerAPI.h"
#include "ask/Frontend/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include <stdexcept>
#include <dlfcn.h>
#include <filesystem>

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
