#include "ask/BuildConfig.h"
#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Import.h"
#include "ask/Lowering/Host.h"
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

using namespace llvm;
namespace {
cl::opt<std::string> input(cl::Positional, cl::desc("input ASK MLIR (or LLVM IR with --check-llir)"), cl::init("-"));
cl::opt<std::string> output("o", cl::desc("output path"), cl::init("-"));
cl::opt<bool> prepare("prepare", cl::desc("verify, infer shapes and extract identity markers"));
cl::opt<bool> emitLLVM("emit-llvm", cl::desc("prepare and lower self-owned graph to host LLVM IR"));
cl::opt<bool> emitObject("emit-object", cl::desc("prepare and emit host object file"));
cl::opt<bool> checkLLIR("check-llir", cl::desc("parse and verify native LLVM assembly; write verified IR"));
cl::opt<bool> capabilities("capabilities", cl::desc("print this build's capabilities as JSON"));
cl::opt<bool> exportExternal("export-external", cl::desc("prepare and export external DSL to native LLVM IR artifacts"));
cl::opt<std::string> reportPath("report-json", cl::desc("write preparation/ABI report"));
cl::opt<std::string> exportDir("export-dir", cl::desc("directory for external export artifacts"));
cl::opt<std::string> tritonPython("triton-python", cl::desc("explicit Triton worker Python executable"));
cl::opt<std::string> tritonWorker("triton-worker", cl::desc("explicit worker.py path (default: installed ASK worker)"));

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

json::Value attrJSON(mlir::Attribute attr) {
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    json::Object value;
    for (auto item : dict) value[item.getName().strref()] = attrJSON(item.getValue());
    return value;
  }
  if (auto str = mlir::dyn_cast<mlir::StringAttr>(attr)) return str.getValue();
  if (auto boolean = mlir::dyn_cast<mlir::BoolAttr>(attr)) return boolean.getValue();
  if (auto number = mlir::dyn_cast<mlir::IntegerAttr>(attr)) return number.getInt();
  if (auto array = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    json::Array value;
    for (int64_t item : array.asArrayRef()) value.push_back(item);
    return value;
  }
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    json::Array value;
    for (auto item : array) value.push_back(attrJSON(item));
    return value;
  }
  std::string text;
  raw_string_ostream stream(text);
  attr.print(stream);
  return text;
}

json::Array externalRequests(mlir::ModuleOp module) {
  json::Array requests;
  module.walk([&](ask::ExternalOp op) {
    auto impl = op.resolveImplementation();
    json::Object request{{"protocol_version", 1}};
    for (StringRef key : {"source", "entry", "signature", "source_bindings",
                          "constants", "target", "grid"})
      if (auto attr = impl->getAttr(key)) request[key] = attrJSON(attr);
    request["fixed_options"] = impl->getAttr("options")
        ? attrJSON(impl->getAttr("options")) : json::Value(json::Object{});
    request["compiler"] = attrJSON(impl->getAttr("compiler"));
    request["op_id"] = attrJSON(op->getAttr("ask.id"));
    request["function"] = op->getParentOfType<mlir::func::FuncOp>().getSymName();
    request["implementation"] = attrJSON(impl->getAttr("sym_name"));
    requests.push_back(std::move(request));
  });
  return requests;
}

Error validateExport(const json::Object &artifact, const json::Object &request) {
  auto nonempty = [](const json::Object &object, StringRef key) {
    auto value = object.getString(key);
    return value && !value->empty();
  };
  auto *producer = artifact.getObject("producer");
  auto *target = artifact.getObject("target");
  auto *invocation = artifact.getObject("invocation");
  auto *specializations = artifact.getObject("specializations");
  if (artifact.get("error") || artifact.getInteger("protocol_version") != 1 ||
      artifact.getString("format") != "llvm-ir" || !nonempty(artifact, "payload") ||
      !nonempty(artifact, "entry") || !producer ||
      producer->getString("name") != "triton" || !nonempty(*producer, "version") ||
      !target || !invocation || !specializations || !artifact.getArray("port_map") ||
      !artifact.getArray("dependencies"))
    return createStringError(inconvertibleErrorCode(), "worker response does not match ASK artifact protocol 1");
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
  SHA256 hash;
  hash.update(*request.getString("source"));
  if (specializations->getString("source_sha256") != toHex(hash.final(), true))
    return createStringError(inconvertibleErrorCode(), "worker source digest does not match request");
  return Error::success();
}

Error runExports(json::Array &requests, json::Object &report, StringRef executable) {
  // Check the entire allowlist before creating files or starting any worker.
  for (auto &item : requests) {
    auto compiler = item.getAsObject()->getString("compiler");
    if (!compiler || *compiler != "triton" || !ASK_WITH_TRITON)
      return createStringError(inconvertibleErrorCode(),
          "external compiler is not enabled in this build; rebuild with ASK_THIRD_PARTY_COMPILERS=triton");
  }
  if (exportDir.empty())
    return createStringError(inconvertibleErrorCode(), "--export-external requires --export-dir");
  if (!requests.empty() && tritonPython.empty())
    return createStringError(inconvertibleErrorCode(), "set --triton-python to the configured worker environment's Python");
  std::string worker = tritonWorker;
  if (worker.empty()) {
    SmallString<256> path(executable);
    sys::path::remove_filename(path);
    sys::path::append(path, "..", "share", "ask", "adapters");
    sys::path::append(path, "triton", "worker.py");
    worker = std::string(path);
  }
  if (!requests.empty() && !sys::fs::exists(worker))
    return createStringError(inconvertibleErrorCode(), "worker not found: %s; set --triton-worker for a build tree", worker.c_str());
  if (auto ec = sys::fs::create_directories(exportDir)) return errorCodeToError(ec);
  json::Array artifacts;
  for (size_t index = 0; index < requests.size(); ++index) {
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
                          "source_bindings", "constants", "fixed_options", "target", "grid"})
      if (auto *value = requests[index].getAsObject()->get(key))
        workerRequest[key] = *value;
    if (auto error = writeJSON(requestPath, json::Value(std::move(workerRequest)))) return error;
    SmallVector<StringRef> args{tritonPython, worker, "--request", requestPath, "--output", artifactPath};
    std::string executionError;
    bool failed = false;
    int code = sys::ExecuteAndWait(tritonPython, args, std::nullopt, {}, 120, 0, &executionError, &failed);
    if (failed || code != 0)
      return createStringError(inconvertibleErrorCode(), "Triton export failed (exit %d): %s", code, executionError.c_str());
    auto buffer = MemoryBuffer::getFile(artifactPath);
    if (!buffer) return errorCodeToError(buffer.getError());
    auto artifact = json::parse((*buffer)->getBuffer());
    if (!artifact) return artifact.takeError();
    auto *object = artifact->getAsObject();
    if (!object) return createStringError(inconvertibleErrorCode(), "worker response must be a JSON object");
    if (auto error = validateExport(*object, *requests[index].getAsObject())) return error;
    if (auto error = writeText(irPath, *object->getString("payload"))) return error;
    artifacts.push_back(json::Object{{"artifact", artifactPath.str().str()},
                                    {"llvm_ir", irPath.str().str()},
                                    {"stage", "exported"}});
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
    json::Array compilers;
    if (ASK_WITH_TRITON) compilers.push_back("triton");
    json::Value result = json::Object{
        {"schema_version", 1}, {"ask_version", ASK_VERSION}, {"llvm_version", ASK_LLVM_VERSION},
        {"third_party_compilers", std::move(compilers)}, {"prepare", true},
        {"native_llvm_import", true}, {"host_codegen", bool(ASK_WITH_HOST)},
        {"external_export", bool(ASK_WITH_TRITON)}, {"gpu_execution", false},
        {"supported_element_types", json::Array{"i32"}}};
    outs() << formatv("{0:2}\n", result);
    return 0;
  }
  if (int(emitLLVM) + int(emitObject) + int(checkLLIR) + int(exportExternal) > 1)
    return fail("select only one of --emit-llvm, --emit-object, --check-llir, --export-external");
  auto buffer = MemoryBuffer::getFileOrSTDIN(input);
  if (!buffer) return fail(buffer.getError().message());
  if (checkLLIR) {
    LLVMContext context;
    auto module = ask::llir::importLLVMIR((*buffer)->getBuffer(), context, input);
    if (!module) return fail(toString(module.takeError()));
    std::string text;
    raw_string_ostream stream(text);
    (*module)->print(stream, nullptr);
    if (auto error = writeText(output, text)) return fail(toString(std::move(error)));
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
  if (prepare || emitLLVM || emitObject || exportExternal) {
    if (mlir::failed(ask::prepareModule(*module, report))) return 1;
    report["phase"] = "prepared";
    report["external_requests"] = externalRequests(*module);
  }
  if (exportExternal) {
    std::string executable = sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
    if (auto error = runExports(*report.getArray("external_requests"), report, executable))
      return fail(toString(std::move(error)));
  }
  std::string result;
  raw_string_ostream stream(result);
  if (emitLLVM || emitObject) {
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
  if (exportExternal) {
    SmallString<256> manifest(exportDir);
    sys::path::append(manifest, "manifest.json");
    if (auto error = writeJSON(manifest, json::Value(json::Object(report)))) return fail(toString(std::move(error)));
  }
  if (!reportPath.empty())
    if (auto error = writeJSON(reportPath, json::Value(std::move(report)))) return fail(toString(std::move(error)));
  return 0;
}
