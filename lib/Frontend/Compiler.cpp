#include "ask/Frontend/Compiler.h"
#include "ask/BuildConfig.h"
#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"
#include "ask/Lowering/Host.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace ask {
namespace {
llvm::json::Value attrJSON(mlir::Attribute attr) {
  using namespace llvm;
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    json::Object value;
    for (auto item : dict) {
      // Dictionary keys borrow the context-owned identifier storage. Keep an
      // owned copy so a report can outlive the parsed module.
      value[item.getName().str()] = attrJSON(item.getValue());
    }
    return value;
  }
  // json::Value(StringRef) borrows the bytes. MLIR attributes die with their
  // context, so own the string before the report outlives the parsed module.
  if (auto str = mlir::dyn_cast<mlir::StringAttr>(attr)) return str.getValue().str();
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
  llvm::raw_string_ostream stream(text);
  attr.print(stream);
  return text;
}
} // namespace

llvm::json::Array externalRequests(mlir::ModuleOp module, bool fusion) {
  using namespace llvm;
  json::Array requests;
  module.walk([&](ExternalOp op) {
    auto impl = op.resolveImplementation();
    json::Object request{{"protocol_version", 1}};
    for (StringRef key : {"source", "entry", "signature", "source_bindings", "constants", "target", "grid"})
      if (auto attr = impl->getAttr(key)) request[key] = attrJSON(attr);
    request["fixed_options"] = impl->getAttr("options") ? attrJSON(impl->getAttr("options")) : json::Value(json::Object{});
    if (fusion) request["fusion"] = json::Object{{"enabled", true}};
    request["compiler"] = attrJSON(impl->getAttr("compiler"));
    request["op_id"] = attrJSON(op->getAttr("ask.id"));
    request["function"] = op->getParentOfType<mlir::func::FuncOp>().getSymName();
    request["implementation"] = attrJSON(impl->getAttr("sym_name"));
    json::Object shapes;
    for (auto [index, value] : llvm::enumerate(op->getOperands())) {
      json::Array shape;
      if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType()))
        for (int64_t dimension : tensor.getShape()) shape.push_back(dimension);
      shapes["input:" + std::to_string(index)] = std::move(shape);
    }
    for (auto [index, value] : llvm::enumerate(op->getResults())) {
      json::Array shape;
      if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType()))
        for (int64_t dimension : tensor.getShape()) shape.push_back(dimension);
      shapes["output:" + std::to_string(index)] = std::move(shape);
    }
    request["port_shapes"] = std::move(shapes);
    requests.push_back(std::move(request));
  });
  return requests;
}

llvm::json::Object compilerCapabilities() {
  using namespace llvm;
  json::Array compilers;
  json::Array adapterDescriptions;
  bool supportsFusion = false;
#define ASK_COMPILER_ADAPTER(ID, FACTORY, FUSION) \
  compilers.push_back(ID); \
  adapterDescriptions.push_back(json::Object{{"id", ID}, {"protocol_version", 1}, {"supports_fusion", bool(FUSION)}}); \
  supportsFusion |= bool(FUSION);
#include "ask/Adapters/Manifest.inc"
#undef ASK_COMPILER_ADAPTER
  bool hasCompilers = !compilers.empty();
  return json::Object{
      {"schema_version", 1}, {"build_id", ASK_BUILD_ID}, {"ask_version", ASK_VERSION}, {"llvm_version", ASK_LLVM_VERSION},
      {"llvm_revision", ASK_LLVM_REVISION}, {"third_party_compilers", std::move(compilers)},
      {"compiler_adapters", std::move(adapterDescriptions)},
      {"prepare", true}, {"partition", true}, {"cuda_execution_plan", true},
      {"native_llvm_import", true}, {"host_codegen", bool(ASK_WITH_HOST)},
      {"ir_optimization", true}, {"elementwise_fusion", true}, {"llvm_optimization", true},
      {"external_export", hasCompilers}, {"gpu_execution", true},
      {"cuda_lowering", true}, {"cuda_graph_codegen", true},
      {"external_llir_fusion", supportsFusion}, {"cuda_execution_via_python", true},
      {"supported_element_types", json::Array{"bool", "i32", "i64", "f32", "f16", "bf16"}},
      {"explicit_cast", true}, {"mixed_precision", true},
      {"host_bf16_hardware_supported", bool(ASK_WITH_HOST) && nativeHostSupportsBF16()},
      {"host_supported_element_types", ASK_WITH_HOST ? json::Array{"bool", "i32", "i64", "f32", "f16", "bf16"} : json::Array{}},
      {"gpu_supported_element_types", json::Array{"bool", "i32", "i64", "f32", "f16", "bf16"}},
      {"python_bindings", "nanobind"}, {"python_bindings_enabled", true},
      {"nanobind_version", ASK_NANOBIND_VERSION}, {"python_abi", ASK_PYTHON_SOABI},
      {"native_prepare", true}, {"native_cuda_runtime", true}, {"device_tensors", true},
      {"async_execution", true}, {"executable_cache", true}};
}

llvm::Expected<PreparedSource> prepareSource(llvm::StringRef source, bool optimize,
                                           const OptimizationOptions &options) {
  mlir::DialectRegistry registry;
  registry.insert<ASKDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source, "<ask-python>"), llvm::SMLoc());
  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  mlir::SourceMgrDiagnosticHandler handler(sources, &context, diagnosticStream);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sources, &context);
  PreparedSource result;
  result.report = llvm::json::Object{{"schema_version", 1}};
  if (!module || mlir::failed(mlir::verify(*module)) ||
      mlir::failed(optimize ? optimizeModule(*module, result.report, options)
                            : prepareModule(*module, result.report)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), diagnostics);
  result.report["phase"] = optimize ? "optimized" : "prepared";
  result.report["external_requests"] = externalRequests(*module);
  llvm::raw_string_ostream stream(result.ir);
  module->print(stream);
  stream << '\n';
  return result;
}
} // namespace ask
