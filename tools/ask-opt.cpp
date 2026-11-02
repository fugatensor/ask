#include "ask/BuildConfig.h"
#include "ask/Frontend/Prepare.h"
#include "ask/Frontend/Compiler.h"
#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Import.h"
#include "ask/LLIR/Optimize.h"
#include "ask/Lowering/Host.h"
#include "ask/Lowering/CUDA.h"
#include "ask/Transforms/Optimize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;
namespace {
cl::opt<std::string> input(cl::Positional, cl::desc("input ASK MLIR (or LLVM IR with --check-llir)"), cl::init("-"));
cl::opt<std::string> output("o", cl::desc("output path"), cl::init("-"));
cl::opt<bool> prepare("prepare", cl::desc("verify, infer shapes and extract identity markers"));
cl::opt<bool> optimize("optimize", cl::desc("prepare ASK IR, fold constants, eliminate dead code and plan elementwise fusion"));
cl::opt<bool> partitionGraph("partition", cl::desc("partition a CUDA graph into explicit kernel functions and an execution plan"));
cl::opt<bool> partitionNoOptimize("no-optimize", cl::desc("disable graph optimization in --partition"));
cl::opt<bool> noConstantFold("no-constant-fold", cl::desc("disable constant folding in --optimize"));
cl::opt<bool> noDCE("no-dce", cl::desc("disable dead code elimination in --optimize"));
cl::opt<bool> noFusion("no-fusion", cl::desc("disable elementwise fusion in --optimize"));
cl::opt<bool> emitLLVM("emit-llvm", cl::desc("prepare and lower self-owned graph to host LLVM IR"));
cl::opt<bool> emitObject("emit-object", cl::desc("prepare and emit host object file"));
cl::opt<bool> emitCUDALLVM("emit-cuda-llvm", cl::desc("prepare and lower self-owned graph to GPU LLVM IR"));
cl::opt<unsigned> cudaArch("cuda-arch", cl::desc("explicit CUDA target architecture, e.g. 86 for sm86"), cl::init(0));
cl::opt<unsigned> cudaBlockSize("cuda-block-size", cl::desc("threads per block for ASK-owned CUDA kernels"), cl::init(128));
cl::opt<unsigned> cudaPTXVersion("cuda-ptx-version", cl::desc("PTX version for ASK-owned CUDA kernels"), cl::init(80));
cl::opt<bool> checkLLIR("check-llir", cl::desc("parse and verify native LLVM assembly; write verified IR"));
cl::opt<bool> optimizeLLIR("optimize-llir", cl::desc("optimize native LLVM assembly with instcombine, simplifycfg and adce"));
cl::opt<bool> capabilities("capabilities", cl::desc("print this build's capabilities as JSON"));
cl::opt<std::string> reportPath("report-json", cl::desc("write preparation/ABI report"));

int fail(const Twine &message) {
  errs() << "ask-opt: " << message << '\n';
  return 1;
}

Error writeText(StringRef path, StringRef contents) {
  std::error_code ec;
  ToolOutputFile file(path, ec, sys::fs::OF_None);
  if (ec) return errorCodeToError(ec);
  file.os() << contents;
  file.os().flush();
  if (file.os().has_error()) return errorCodeToError(file.os().error());
  file.keep();
  return Error::success();
}

Error writeJSON(StringRef path, const json::Value &value) {
  std::string contents;
  raw_string_ostream stream(contents);
  stream << formatv("{0:2}\n", value);
  return writeText(path, contents);
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "ASK framework frontend and native LLVM driver\n");
  if (capabilities) {
    json::Value result = ask::compilerCapabilities();
    outs() << formatv("{0:2}\n", result);
    return 0;
  }
  if (int(partitionGraph) + int(emitLLVM) + int(emitObject) + int(emitCUDALLVM) + int(checkLLIR) + int(optimizeLLIR) > 1)
    return fail("select only one of --partition, --emit-llvm, --emit-object, --emit-cuda-llvm, --check-llir, --optimize-llir");
  if (!emitCUDALLVM && (cudaArch.getNumOccurrences() || cudaBlockSize.getNumOccurrences() || cudaPTXVersion.getNumOccurrences()))
    return fail("--cuda-arch, --cuda-block-size and --cuda-ptx-version require --emit-cuda-llvm");
  if ((noConstantFold || noDCE || noFusion) && !optimize && !partitionGraph)
    return fail("--no-constant-fold, --no-dce and --no-fusion require --optimize");
  if (partitionNoOptimize && !partitionGraph)
    return fail("--no-optimize requires --partition");
  if (partitionGraph && (prepare || optimize))
    return fail("--partition cannot be combined with --prepare or --optimize; it prepares or optimizes the input itself");
  if ((checkLLIR || optimizeLLIR) && (prepare || optimize))
    return fail("native LLVM input modes cannot be combined with ASK IR preparation or optimization");
  auto buffer = MemoryBuffer::getFileOrSTDIN(input);
  if (!buffer) return fail(buffer.getError().message());
  if (partitionGraph) {
    ask::PartitionOptions options;
    options.optimize = !partitionNoOptimize;
    options.optimization = {!noConstantFold, !noDCE, !noFusion};
    auto result = ask::partitionSource((*buffer)->getBuffer(), options);
    if (!result) return fail(toString(result.takeError()));
    if (auto error = writeText(output, result->ir)) return fail(toString(std::move(error)));
    if (!reportPath.empty())
      if (auto error = writeJSON(reportPath, json::Value(std::move(result->report)))) return fail(toString(std::move(error)));
    return 0;
  }
  if (checkLLIR || optimizeLLIR) {
    LLVMContext context;
    auto module = ask::llir::importLLVMIR((*buffer)->getBuffer(), context, input);
    if (!module) return fail(toString(module.takeError()));
    json::Object report{{"schema_version", 1}, {"phase", "llvm_verified"}};
    if (optimizeLLIR) {
      if (auto error = ask::llir::optimizeLLVMIR(**module, report))
        return fail(toString(std::move(error)));
      report["phase"] = "llvm_optimized";
    }
    std::string text;
    raw_string_ostream stream(text);
    (*module)->print(stream, nullptr);
    if (auto error = writeText(output, text)) return fail(toString(std::move(error)));
    if (!reportPath.empty())
      if (auto error = writeJSON(reportPath, json::Value(std::move(report)))) return fail(toString(std::move(error)));
    return 0;
  }
  mlir::DialectRegistry registry;
  registry.insert<ask::ASKDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  SourceMgr sources;
  sources.AddNewSourceBuffer(std::move(*buffer), SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sources, &context);
  if (!module || mlir::failed(mlir::verify(*module))) return 1;
  json::Object report{{"schema_version", 1}, {"phase", "verified"}};
  if (prepare || optimize || emitLLVM || emitObject || emitCUDALLVM) {
    if (optimize) {
      ask::OptimizationOptions options;
      options.foldConstants = !noConstantFold;
      options.eliminateDeadCode = !noDCE;
      options.fuseElementwise = !noFusion;
      if (mlir::failed(ask::optimizeModule(*module, report, options))) return 1;
    } else if (mlir::failed(ask::prepareModule(*module, report))) {
      return 1;
    }
    report["phase"] = optimize ? "optimized" : "prepared";
    report["external_requests"] = ask::externalRequests(*module);
  }
  std::string result;
  raw_string_ostream stream(result);
  if (emitCUDALLVM) {
    LLVMContext llvmContext;
    json::Object artifact, lowering;
    ask::CUDALoweringOptions options;
    options.architecture = cudaArch;
    options.blockSize = cudaBlockSize;
    options.ptxVersion = cudaPTXVersion;
    auto lowered = ask::lowerToCUDALLVM(*module, llvmContext, options, artifact, lowering);
    if (!lowered) return fail(toString(lowered.takeError()));
    std::string snapshotText;
    raw_string_ostream snapshotStream(snapshotText);
    snapshotStream << formatv("{0}", *report.get("shape_snapshot"));
    (*lowered)->getOrInsertNamedMetadata("ask.shape_snapshot")->addOperand(
        MDNode::get(llvmContext, MDString::get(llvmContext, snapshotText)));
    (*lowered)->print(stream, nullptr);
    artifact["payload"] = result;
    artifact["shape_snapshot"] = *report.get("shape_snapshot");
    report["cuda_artifact"] = std::move(artifact);
    report["cuda_lowering"] = std::move(lowering);
    report["phase"] = "cuda_llvm";
  } else if (emitLLVM || emitObject) {
    if (!ASK_WITH_HOST) return fail("host code generation was disabled at build time");
    LLVMContext llvmContext;
    json::Object abi;
    auto lowered = ask::lowerToHostLLVM(*module, llvmContext, abi);
    if (!lowered) return fail(toString(lowered.takeError()));
    report["abi"] = std::move(abi);
    // Frozen high-level shape information travels with the LLVM artifact.
    // Pointers are never used to reconstruct tensor shapes.
    std::string snapshotText;
    raw_string_ostream snapshotStream(snapshotText);
    snapshotStream << formatv("{0}", *report.get("shape_snapshot"));
    (*lowered)->getOrInsertNamedMetadata("ask.shape_snapshot")->addOperand(
        MDNode::get(llvmContext, MDString::get(llvmContext, snapshotText)));
    report["phase"] = emitObject ? "host_object" : "host_llvm";
    if (emitObject) {
      if (output == "-") return fail("--emit-object requires -o FILE");
      if (auto error = ask::emitHostObject(**lowered, output)) return fail(toString(std::move(error)));
    } else {
      (*lowered)->print(stream, nullptr);
    }
  } else {
    module->print(stream);
    stream << '\n';
  }
  if (!emitObject)
    if (auto error = writeText(output, result)) return fail(toString(std::move(error)));
  if (!reportPath.empty())
    if (auto error = writeJSON(reportPath, json::Value(std::move(report)))) return fail(toString(std::move(error)));
  return 0;
}
