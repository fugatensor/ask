#include "ask/BuildConfig.h"
#include "ask/Adapters/Compiler.h"
#include "ask/Frontend/Prepare.h"
#include "ask/Frontend/Compiler.h"
#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Import.h"
#include "ask/LLIR/Fuse.h"
#include "ask/LLIR/Optimize.h"
#include "ask/Lowering/Host.h"
#include "ask/Lowering/CUDA.h"
#include "ask/Transforms/Optimize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/Metadata.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <map>

using namespace llvm;
namespace {
cl::opt<std::string> input(cl::Positional, cl::desc("input ASK MLIR (or LLVM IR with --check-llir)"), cl::init("-"));
cl::opt<std::string> output("o", cl::desc("output path"), cl::init("-"));
cl::opt<bool> prepare("prepare", cl::desc("verify, infer shapes and extract identity markers"));
cl::opt<bool> optimize("optimize", cl::desc("prepare ASK IR, fold constants, eliminate dead code and plan elementwise fusion"));
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
cl::opt<bool> exportExternal("export-external", cl::desc("prepare and export external DSL to native LLVM IR artifacts"));
cl::opt<bool> fuseExternal("fuse-external", cl::desc("export adapter-attested LLIR and inline ASK pointwise LLVM helpers"));
cl::opt<std::string> reportPath("report-json", cl::desc("write preparation/ABI report"));
cl::opt<std::string> exportDir("export-dir", cl::desc("directory for external export artifacts"));
cl::list<std::string> compilerPythons("compiler-python", cl::desc("worker Python executable: compiler_id=PATH"), cl::ZeroOrMore);
cl::list<std::string> compilerWorkers("compiler-worker", cl::desc("worker path: compiler_id=PATH (default: installed adapter worker)"), cl::ZeroOrMore);

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

Error validateExport(const json::Object &artifact, const json::Object &request,
                     const ask::adapters::CompilerAdapter &adapter) {
  auto nonempty = [](const json::Object &object, StringRef key) {
    auto value = object.getString(key);
    return value && !value->empty();
  };
  auto *producer = artifact.getObject("producer");
  auto *target = artifact.getObject("target");
  auto *invocation = artifact.getObject("invocation");
  auto *specializations = artifact.getObject("specializations");
  if (artifact.get("error") || artifact.getInteger("protocol_version") != adapter.protocolVersion ||
      artifact.getString("format") != "llvm-ir" || !nonempty(artifact, "payload") ||
      !nonempty(artifact, "entry") || !producer ||
      !nonempty(*producer, "name") || !nonempty(*producer, "version") ||
      !target || !invocation || !specializations || !artifact.getObject("validation") || !artifact.getArray("port_map") ||
      !artifact.getArray("dependencies"))
    return createStringError(inconvertibleErrorCode(), "worker response does not match the registered ASK artifact protocol");
  auto matches = [](const json::Object &actual, const json::Object &expected, StringRef key) {
    auto *a = actual.get(key);
    auto *b = expected.get(key);
    return a && b && *a == *b;
  };
  auto *requestedTarget = request.getObject("target");
  if (!requestedTarget) return createStringError(inconvertibleErrorCode(), "missing requested target");
  for (auto &item : *requestedTarget)
    if (!matches(*target, *requestedTarget, item.first))
      return createStringError(inconvertibleErrorCode(), "worker changed requested target");
  if (!matches(*invocation, request, "grid"))
    return createStringError(inconvertibleErrorCode(), "worker changed requested grid");
  for (StringRef key : {"constants", "fixed_options", "source_bindings"})
    if (!matches(*specializations, request, key))
      return createStringError(inconvertibleErrorCode(), "worker changed requested %s", key.str().c_str());
  auto source = request.getString("source");
  if (!source) return createStringError(inconvertibleErrorCode(), "missing requested source");
  SHA256 hash;
  hash.update(*source);
  if (specializations->getString("source_sha256") != toHex(hash.final(), true))
    return createStringError(inconvertibleErrorCode(), "worker source digest does not match request");
  LLVMContext context;
  auto imported = ask::llir::importLLVMIR(*artifact.getString("payload"), context, "external.worker.ll");
  if (!imported) return imported.takeError();
  auto *entry = (*imported)->getFunction(*artifact.getString("entry"));
  auto *arguments = invocation->getArray("arguments");
  if (!entry || entry->isDeclaration() || entry->isVarArg() || !arguments ||
      arguments->size() != entry->arg_size())
    return createStringError(inconvertibleErrorCode(), "worker invocation does not match the LLVM entry ABI");
  if (auto layout = target->getString("data_layout"))
    if (*layout != (*imported)->getDataLayoutStr())
      return createStringError(inconvertibleErrorCode(), "worker data layout differs from its LLVM payload");
  for (auto [index, argument] : enumerate(entry->args())) {
    auto *mapping = (*arguments)[index].getAsObject();
    std::string type;
    raw_string_ostream stream(type);
    argument.getType()->print(stream);
    if (!mapping || mapping->getInteger("argument") != int64_t(index) ||
        mapping->getString("llvm_type") != type || !mapping->getString("binding"))
      return createStringError(inconvertibleErrorCode(), "worker parameter mapping differs from its LLVM entry ABI");
  }
  if (target->getString("backend") == "cuda") {
    auto *codegen = artifact.getObject("codegen");
    auto *required = artifact.getArray("required_element_types");
    if (invocation->getString("kind") != "cuda-kernel" || invocation->getInteger("version") != 2 ||
        !codegen || !required || !entry->getReturnType()->isVoidTy() ||
        entry->getCallingConv() != CallingConv::PTX_Kernel)
      return createStringError(inconvertibleErrorCode(), "CUDA artifact requires normalized invocation version 2 and a kernel entry; re-export legacy artifacts");
    auto policy = codegen->getString("fp_contract");
    auto ptx = codegen->getInteger("ptx_version");
    if (!policy || (*policy != "on" && *policy != "off" && *policy != "fast") ||
        !ptx || *ptx < 10 || *ptx > 999)
      return createStringError(inconvertibleErrorCode(), "CUDA artifact has invalid normalized codegen options");
    for (StringRef field : {"grid", "block"}) {
      auto *dimensions = invocation->getArray(field);
      if (!dimensions || dimensions->size() != 3)
        return createStringError(inconvertibleErrorCode(), "CUDA launch dimensions must have rank 3");
      for (const auto &dimension : *dimensions) {
        auto number = dimension.getAsInteger();
        if (!number || *number <= 0 || uint64_t(*number) > UINT32_MAX)
          return createStringError(inconvertibleErrorCode(), "CUDA launch dimensions must be positive uint32 values");
      }
    }
    auto shared = invocation->getInteger("shared_bytes");
    if (!shared || *shared < 0 || uint64_t(*shared) > UINT32_MAX)
      return createStringError(inconvertibleErrorCode(), "CUDA dynamic shared byte count is invalid");
    auto validElement = [](StringRef type) {
      return type == "bool" || type == "i32" || type == "i64" || type == "f16" || type == "bf16" || type == "f32";
    };
    for (const auto &type : *required)
      if (!type.getAsString() || !validElement(*type.getAsString()))
        return createStringError(inconvertibleErrorCode(), "CUDA required element types contain an unsupported type");
    for (const auto &item : *artifact.getArray("port_map")) {
      auto *port = item.getAsObject();
      auto ordinal = port ? port->getInteger("argument") : std::nullopt;
      auto dtype = port ? port->getString("element_type") : std::nullopt;
      auto binding = port ? port->getString("logical_port") : std::nullopt;
      auto *shape = port ? port->getArray("shape") : nullptr;
      if (!ordinal || *ordinal < 0 || uint64_t(*ordinal) >= arguments->size() || !dtype ||
          !validElement(*dtype) || !binding || !shape ||
          port->getString("layout") != "contiguous_row_major" ||
          (*arguments)[*ordinal].getAsObject()->getString("binding") != *binding)
        return createStringError(inconvertibleErrorCode(), "CUDA logical port lacks a concrete shape or consistent pointer binding");
      for (const auto &dimension : *shape)
        if (!dimension.getAsInteger() || *dimension.getAsInteger() < 0)
          return createStringError(inconvertibleErrorCode(), "CUDA port shapes must contain nonnegative integers");
    }
  }
  if (adapter.validateArtifact)
    return adapter.validateArtifact(artifact, request);
  return Error::success();
}

Expected<std::map<std::string, std::string>> parseRoutes(const cl::list<std::string> &options,
                                                       StringRef optionName) {
  std::map<std::string, std::string> routes;
  for (StringRef option : options) {
    auto [id, path] = option.split('=');
    if (id.empty() || path.empty())
      return createStringError(inconvertibleErrorCode(), "%s requires compiler_id=PATH", optionName.str().c_str());
    if (!ask::adapters::findCompilerAdapter(id))
      return createStringError(inconvertibleErrorCode(), "compiler adapter '%s' is not enabled in this build", id.str().c_str());
    if (!routes.emplace(id.str(), path.str()).second)
      return createStringError(inconvertibleErrorCode(), "duplicate %s route for compiler '%s'", optionName.str().c_str(), id.str().c_str());
  }
  return routes;
}

Error runExports(json::Array &requests, json::Object &report, StringRef executable) {
  // Resolve the complete allowlist and all worker paths before creating any
  // artifacts. Each external request selects its own registered adapter.
  auto pythons = parseRoutes(compilerPythons, "--compiler-python");
  if (!pythons) return pythons.takeError();
  auto workers = parseRoutes(compilerWorkers, "--compiler-worker");
  if (!workers) return workers.takeError();
  for (auto &item : requests) {
    auto *request = item.getAsObject();
    auto id = request ? request->getString("compiler") : std::nullopt;
    auto *adapter = id ? ask::adapters::findCompilerAdapter(*id) : nullptr;
    if (!adapter)
      return createStringError(inconvertibleErrorCode(), "external compiler is not enabled in this build; select it with ASK_THIRD_PARTY_COMPILERS");
    if (fuseExternal && !adapter->prepareFusion)
      return createStringError(inconvertibleErrorCode(), "compiler adapter '%s' does not support value-boundary fusion", id->str().c_str());
    if (request->getInteger("protocol_version") != adapter->protocolVersion)
      return createStringError(inconvertibleErrorCode(), "external request protocol differs from the registered adapter");
    auto python = pythons->find(id->str());
    if (python == pythons->end() && adapter->workerIsPython)
      return createStringError(inconvertibleErrorCode(), "adapter '%s' requires --compiler-python %s=PATH", id->str().c_str(), id->str().c_str());
    if (python != pythons->end()) {
      auto program = sys::findProgramByName(python->second);
      if (!program) return errorCodeToError(program.getError());
      python->second = *program;
    }
    if (!workers->count(id->str())) {
      SmallString<256> path(executable);
      sys::path::remove_filename(path);
      sys::path::append(path, "..", "share", "ask", "adapters");
      sys::path::append(path, *id, adapter->defaultWorker);
      workers->emplace(id->str(), path.str().str());
    }
    if (!sys::fs::exists(workers->at(id->str())))
      return createStringError(inconvertibleErrorCode(), "worker for '%s' not found; set --compiler-worker %s=PATH for a build tree", id->str().c_str(), id->str().c_str());
  }
  if (exportDir.empty())
    return createStringError(inconvertibleErrorCode(), "external export requires --export-dir");
  if (auto ec = sys::fs::create_directories(exportDir)) return errorCodeToError(ec);
  json::Array artifacts;
  for (size_t index = 0; index < requests.size(); ++index) {
    auto &request = *requests[index].getAsObject();
    auto id = *request.getString("compiler");
    auto *adapter = ask::adapters::findCompilerAdapter(id);
    auto python = pythons->find(id.str());
    const auto &worker = workers->at(id.str());
    SmallString<256> requestPath(exportDir), artifactPath(exportDir), irPath(exportDir);
    std::string stem = "external-" + std::to_string(index);
    sys::path::append(requestPath, stem + ".request.json");
    sys::path::append(artifactPath, stem + ".artifact.json");
    sys::path::append(irPath, stem + ".ll");
    // An old artifact must never be mistaken for this worker's response.
    if (sys::fs::exists(artifactPath) || sys::fs::exists(irPath) || sys::fs::exists(requestPath))
      return createStringError(inconvertibleErrorCode(), "export destination already contains %s artifacts; use a fresh directory", stem.c_str());
    json::Object workerRequest;
    for (StringRef key : {"protocol_version", "source", "entry", "signature",
                          "source_bindings", "constants", "fixed_options", "target", "grid", "fusion"})
      if (auto *value = requests[index].getAsObject()->get(key))
        workerRequest[key] = *value;
    if (auto error = writeJSON(requestPath, json::Value(std::move(workerRequest)))) return error;
    SmallVector<StringRef> args;
    if (python != pythons->end()) args.push_back(python->second);
    args.append({worker, "--request", requestPath, "--output", artifactPath});
    StringRef program = args.front();
    std::string executionError;
    bool failed = false;
    int code = sys::ExecuteAndWait(program, args, std::nullopt, {}, 120, 0, &executionError, &failed);
    if (failed || code != 0)
      return createStringError(inconvertibleErrorCode(), "compiler %s export failed (exit %d): %s", id.str().c_str(), code, executionError.c_str());
    auto buffer = MemoryBuffer::getFile(artifactPath);
    if (!buffer) return errorCodeToError(buffer.getError());
    auto artifact = json::parse((*buffer)->getBuffer());
    if (!artifact) return artifact.takeError();
    auto *object = artifact->getAsObject();
    if (!object) return createStringError(inconvertibleErrorCode(), "worker response must be a JSON object");
    if (auto error = validateExport(*object, request, *adapter)) return error;
    if (auto error = writeText(irPath, *object->getString("payload"))) return error;
    artifacts.push_back(json::Object{{"artifact", artifactPath.str().str()},
                                    {"llvm_ir", irPath.str().str()},
                                    {"stage", "exported"}, {"compiler", id.str()}});
  }
  report["external_artifacts"] = std::move(artifacts);
  report["phase"] = "exported";
  return Error::success();
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
  if (int(emitLLVM) + int(emitObject) + int(emitCUDALLVM) + int(checkLLIR) + int(optimizeLLIR) + int(exportExternal) + int(fuseExternal) > 1)
    return fail("select only one of --emit-llvm, --emit-object, --emit-cuda-llvm, --check-llir, --optimize-llir, --export-external, --fuse-external");
  if (!emitCUDALLVM && (cudaArch.getNumOccurrences() || cudaBlockSize.getNumOccurrences() || cudaPTXVersion.getNumOccurrences()))
    return fail("--cuda-arch, --cuda-block-size and --cuda-ptx-version require --emit-cuda-llvm");
  if ((noConstantFold || noDCE || noFusion) && !optimize)
    return fail("--no-constant-fold, --no-dce and --no-fusion require --optimize");
  if ((checkLLIR || optimizeLLIR) && (prepare || optimize))
    return fail("native LLVM input modes cannot be combined with ASK IR preparation or optimization");
  auto buffer = MemoryBuffer::getFileOrSTDIN(input);
  if (!buffer) return fail(buffer.getError().message());
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
  if (prepare || optimize || emitLLVM || emitObject || emitCUDALLVM || exportExternal || fuseExternal) {
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
    report["external_requests"] = ask::externalRequests(*module, fuseExternal);
  }
  if (exportExternal || fuseExternal) {
    std::string executable = sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
    if (auto error = runExports(*report.getArray("external_requests"), report, executable))
      return fail(toString(std::move(error)));
  }
  std::string result;
  raw_string_ostream stream(result);
  if (fuseExternal) {
    auto *artifacts = report.getArray("external_artifacts");
    if (!artifacts || artifacts->size() != 1) return fail("fusion requires exactly one external artifact");
    auto path = *(*artifacts)[0].getAsObject()->getString("artifact");
    auto contents = MemoryBuffer::getFile(path);
    if (!contents) return fail(contents.getError().message());
    auto parsed = json::parse((*contents)->getBuffer());
    if (!parsed) return fail(toString(parsed.takeError()));
    LLVMContext llvmContext;
    std::string helpers;
    auto *requests = report.getArray("external_requests");
    auto *request = requests && requests->size() == 1 ? (*requests)[0].getAsObject() : nullptr;
    auto id = request ? request->getString("compiler") : std::nullopt;
    auto *adapter = id ? ask::adapters::findCompilerAdapter(*id) : nullptr;
    if (!adapter) return fail("fusion requires one enabled compiler adapter");
    auto lowered = ask::llir::fuseExternal(*module, *parsed->getAsObject(), *adapter, llvmContext, report, helpers);
    if (!lowered) return fail(toString(lowered.takeError()));
    std::string snapshot;
    raw_string_ostream snapshotStream(snapshot);
    snapshotStream << formatv("{0}", *report.get("shape_snapshot"));
    (*lowered)->getOrInsertNamedMetadata("ask.shape_snapshot")->addOperand(
        MDNode::get(llvmContext, MDString::get(llvmContext, snapshot)));
    (*lowered)->print(stream, nullptr);
    SmallString<256> helperPath(exportDir), fusedPath(exportDir);
    sys::path::append(helperPath, "ask.helpers.ll");
    sys::path::append(fusedPath, "fused.artifact.json");
    if (auto e = writeText(helperPath, helpers)) return fail(toString(std::move(e)));
    auto fused = *parsed->getAsObject();
    fused["payload"] = result;
    fused["fusion"] = *report.get("fusion");
    json::Value sourceValidation = *fused.get("validation");
    fused["source_validation"] = std::move(sourceValidation);
    fused["validation"] = json::Object{{"llvm_module_verified", true},
        {"device_execution_performed", false}, {"stage", "fused_llvm"}};
    auto *fusionReport = report.getObject("fusion");
    if (!fusionReport || !fusionReport->getObject("codegen") ||
        !fusionReport->getArray("required_element_types"))
      return fail("fusion did not publish normalized execution settings");
    fused["codegen"] = *fusionReport->get("codegen");
    fused["required_element_types"] = *fusionReport->get("required_element_types");
    auto graph = *module->getOps<mlir::func::FuncOp>().begin();
    for (auto &item : *fused.getArray("port_map")) {
      auto *port = item.getAsObject();
      auto binding = port ? port->getString("logical_port") : std::nullopt;
      if (!binding || (*binding != "input:0" && *binding != "output:0"))
        return fail("pointwise fusion requires normalized input:0/output:0 ports");
      auto type = *binding == "input:0" ? graph.getArgument(0).getType() : graph.getResultTypes()[0];
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!tensor || !tensor.hasStaticShape()) return fail("fusion requires concrete port shapes");
      json::Array shape;
      for (int64_t dimension : tensor.getShape()) shape.push_back(dimension);
      (*port)["shape"] = std::move(shape);
      (*port)["layout"] = "contiguous_row_major";
    }
    if (adapter->finalizeArtifact)
      if (auto e = adapter->finalizeArtifact(fused, *fusionReport)) return fail(toString(std::move(e)));
    fused["shape_snapshot"] = *report.get("shape_snapshot");
    if (auto e = writeJSON(fusedPath, json::Value(std::move(fused)))) return fail(toString(std::move(e)));
    report["phase"] = "fused_llvm";
  } else if (emitCUDALLVM) {
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
  if (exportExternal || fuseExternal) {
    SmallString<256> manifest(exportDir);
    sys::path::append(manifest, "manifest.json");
    if (auto error = writeJSON(manifest, json::Value(json::Object(report)))) return fail(toString(std::move(error)));
  }
  if (!reportPath.empty())
    if (auto error = writeJSON(reportPath, json::Value(std::move(report)))) return fail(toString(std::move(error)));
  return 0;
}
