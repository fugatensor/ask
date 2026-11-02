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
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"

namespace ask {
namespace {
llvm::json::Value attrJSON(mlir::Attribute attr, JSONFloatPaths *floatPaths = nullptr,
                           std::vector<std::string> path = {}) {
  using namespace llvm;
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    json::Object value;
    for (auto item : dict) {
      // Dictionary keys borrow the context-owned identifier storage. Keep an
      // owned copy so a report can outlive the parsed module.
      path.push_back(item.getName().str());
      value[item.getName().str()] = attrJSON(item.getValue(), floatPaths, path);
      path.pop_back();
    }
    return value;
  }
  // json::Value(StringRef) borrows the bytes. MLIR attributes die with their
  // context, so own the string before the report outlives the parsed module.
  if (auto str = mlir::dyn_cast<mlir::StringAttr>(attr)) return str.getValue().str();
  if (mlir::isa<mlir::UnitAttr>(attr)) return nullptr;
  if (auto boolean = mlir::dyn_cast<mlir::BoolAttr>(attr)) return boolean.getValue();
  if (auto number = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (mlir::cast<mlir::IntegerType>(number.getType()).isUnsigned()) return number.getValue().getZExtValue();
    return number.getInt();
  }
  if (auto number = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    if (floatPaths) floatPaths->insert(std::move(path));
    return number.getValueAsDouble();
  }
  if (auto array = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    json::Array value;
    for (int64_t item : array.asArrayRef()) value.push_back(item);
    return value;
  }
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    json::Array value;
    for (auto [index, item] : llvm::enumerate(array)) {
      path.push_back(std::to_string(index));
      value.push_back(attrJSON(item, floatPaths, path));
      path.pop_back();
    }
    return value;
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  attr.print(stream);
  return text;
}
} // namespace

std::string reportJSON(const llvm::json::Value &report, const JSONFloatPaths &floatPaths,
                       unsigned indent) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::json::OStream json(stream, indent);
  std::vector<std::string> path;
  auto emit = [&](const llvm::json::Value &value, auto &self) -> void {
    if (floatPaths.count(path)) {
      std::string number;
      llvm::raw_string_ostream(number) << llvm::format("%.17e", *value.getAsNumber());
      json.rawValue(number);
    } else if (auto *object = value.getAsObject()) {
      json.object([&] {
        std::vector<std::string> keys;
        for (const auto &item : *object) keys.push_back(item.first.str());
        llvm::sort(keys);
        for (const auto &key : keys) {
          path.push_back(key);
          json.attributeBegin(key);
          self(*object->get(key), self);
          json.attributeEnd();
          path.pop_back();
        }
      });
    } else if (auto *array = value.getAsArray()) {
      json.array([&] {
        for (auto [index, item] : llvm::enumerate(*array)) {
          path.push_back(std::to_string(index));
          self(item, self);
          path.pop_back();
        }
      });
    } else json.value(value);
  };
  emit(report, emit);
  return result;
}

llvm::json::Array externalRequests(mlir::ModuleOp module, JSONFloatPaths *floatPaths) {
  using namespace llvm;
  json::Array requests;
  module.walk([&](ExternalOp op) {
    auto impl = op.resolveImplementation();
    json::Object request{{"protocol_version", 2}};
    for (StringRef key : {"source", "entry", "signature", "source_bindings", "constants", "target", "grid"})
      if (auto attr = impl->getAttr(key))
        request[key] = attrJSON(attr, floatPaths, {"external_requests", std::to_string(requests.size()), key.str()});
    request["fixed_options"] = impl->getAttr("options")
        ? attrJSON(impl->getAttr("options"), floatPaths, {"external_requests", std::to_string(requests.size()), "fixed_options"})
        : json::Value(json::Object{});
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
  return json::Object{
      {"schema_version", 1}, {"build_id", ASK_BUILD_ID}, {"ask_version", ASK_VERSION}, {"llvm_version", ASK_LLVM_VERSION},
      {"llvm_revision", ASK_LLVM_REVISION},
      {"prepare", true}, {"partition", true}, {"cuda_execution_plan", true},
      {"native_llvm_import", true}, {"host_codegen", bool(ASK_WITH_HOST)},
      {"ir_optimization", true}, {"elementwise_fusion", true}, {"llvm_optimization", true},
      {"external_execution", true}, {"external_formats", json::Array{"cubin", "ptx"}},
      {"gpu_execution", true},
      {"cuda_lowering", true}, {"cuda_graph_codegen", true},
      {"cuda_execution_via_python", true},
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
  result.report["external_requests"] = externalRequests(*module, &result.floatPaths);
  llvm::raw_string_ostream stream(result.ir);
  module->print(stream);
  stream << '\n';
  return result;
}
} // namespace ask
