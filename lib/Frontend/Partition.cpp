#include "ask/Frontend/Compiler.h"
#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <functional>

using namespace mlir;

namespace ask {
namespace {
struct Partition {
  llvm::SmallVector<Operation *> operations;
  std::string kind = "owned";
  ExternalOp external;
};
struct Ports {
  llvm::SmallVector<Value> inputs, outputs;
};

std::string identity(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = cast<func::FuncOp>(argument.getOwner()->getParentOp());
    return function.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "ask.value_id").getValue().str();
  }
  auto result = cast<OpResult>(value);
  auto *operation = result.getOwner();
  if (auto name = operation->getAttrOfType<StringAttr>("ask.value_id"))
    return name.getValue().str();
  return (operation->getAttrOfType<StringAttr>("ask.id").getValue() + ".result" +
          llvm::Twine(result.getResultNumber())).str();
}

Ports ports(const Partition &part) {
  Ports result;
  if (part.kind == "external") {
    result.inputs.append(part.external->operand_begin(), part.external->operand_end());
    result.outputs.append(part.external->result_begin(), part.external->result_end());
    return result;
  }
  llvm::DenseSet<Operation *> members(part.operations.begin(), part.operations.end());
  llvm::DenseSet<Value> seenInputs;
  for (auto *operation : part.operations) {
    for (Value operand : operation->getOperands())
      if (isa<RankedTensorType>(operand.getType()) && !members.contains(operand.getDefiningOp()) &&
          seenInputs.insert(operand).second)
        result.inputs.push_back(operand);
    for (Value value : operation->getResults())
      if (isa<RankedTensorType>(value.getType()) && llvm::any_of(value.getUsers(),
          [&](Operation *user) { return !members.contains(user); }))
        result.outputs.push_back(value);
  }
  return result;
}

bool canFuse(const Partition &part) {
  auto ext = part.external;
  if (!ext || ext->getNumOperands() != 1 || ext->getNumResults() != 1 || part.operations.size() < 2)
    return false;
  auto boundary = dyn_cast<RankedTensorType>(ext->getOperand(0).getType());
  if (!boundary || !boundary.hasStaticShape() || boundary.getNumElements() == 0 ||
      ext->getResult(0).getType() != boundary ||
      ext.resolveSemantics()->getAttrOfType<StringAttr>("effects").getValue() != "none")
    return false;
  Ports boundaryPorts = ports(part);
  if (boundaryPorts.inputs.size() != 1 || boundaryPorts.outputs.size() != 1 ||
      boundaryPorts.inputs.front().getType() != boundary || boundaryPorts.outputs.front().getType() != boundary)
    return false;
  for (auto *operation : part.operations)
    for (Value value : operation->getResults())
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
        if (tensor.getShape() != boundary.getShape()) return false;
  std::function<bool(Value, Value)> dependsOnlyOn = [&](Value value, Value leaf) {
    if (value == leaf) return true;
    auto *owner = value.getDefiningOp();
    if (!owner || isa<ExternalOp>(owner)) return false;
    return llvm::all_of(owner->getOperands(), [&](Value operand) { return dependsOnlyOn(operand, leaf); });
  };
  if (!dependsOnlyOn(ext->getOperand(0), boundaryPorts.inputs.front()) ||
      !dependsOnlyOn(boundaryPorts.outputs.front(), ext->getResult(0))) return false;
  return true;
}

std::string printModule(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream << '\n';
  return text;
}

LogicalResult partition(ModuleOp original, llvm::json::Object &report,
                        const PartitionOptions &options, std::string &ir) {
  auto functions = llvm::to_vector(original.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return original.emitError("CUDA partitioning requires exactly one graph function");
  auto function = functions.front();
  auto returned = cast<func::ReturnOp>(function.front().getTerminator());
  for (Type type : llvm::concat<const Type>(function.getArgumentTypes(), function.getResultTypes()))
    if (!isa<RankedTensorType>(type))
      return function.emitError("CUDA partition ports must be static ranked tensors; use tensor<...> for 0-D scalars");

  std::vector<Partition> parts;
  llvm::StringSet<> externalIds;
  for (Operation &operation : function.front()) {
    if (isa<func::ReturnOp>(operation)) continue;
    if (auto external = dyn_cast<ExternalOp>(operation)) {
      for (Type type : llvm::concat<const Type>(external->getOperandTypes(), external->getResultTypes()))
        if (!isa<RankedTensorType>(type))
          return external.emitError("CUDA external ports must be static ranked tensors");
      externalIds.insert(operation.getAttrOfType<StringAttr>("ask.id").getValue());
      parts.push_back({{&operation}, "external", external});
    } else if (llvm::any_of(operation.getResultTypes(), [](Type type) { return isa<RankedTensorType>(type); })) {
      if (parts.empty() || parts.back().kind != "owned") parts.emplace_back();
      parts.back().operations.push_back(&operation);
    }
  }
  for (const auto &id : options.unfusedExternals)
    if (!externalIds.contains(id)) return original.emitError("unknown unfused external ID: ") << id;

  if (options.fuseExternal) {
    for (size_t index = 0; index < parts.size(); ++index) {
      if (parts[index].kind != "external" || options.unfusedExternals.count(
          parts[index].external->getAttrOfType<StringAttr>("ask.id").getValue().str())) continue;
      size_t begin = index, end = index;
      Partition candidate = parts[index];
      candidate.kind = "fused";
      // Try the whole adjacent region first, then each side independently.
      if (index && parts[index - 1].kind == "owned") {
        candidate.operations.insert(candidate.operations.begin(), parts[index - 1].operations.begin(),
                                    parts[index - 1].operations.end());
        begin = index - 1;
      }
      if (index + 1 < parts.size() && parts[index + 1].kind == "owned") {
        candidate.operations.append(parts[index + 1].operations);
        end = index + 1;
      }
      if (!canFuse(candidate)) {
        candidate = parts[index]; candidate.kind = "fused"; begin = end = index;
        if (index && parts[index - 1].kind == "owned") {
          candidate.operations.insert(candidate.operations.begin(), parts[index - 1].operations.begin(),
                                      parts[index - 1].operations.end());
          if (canFuse(candidate)) begin = index - 1;
          else { candidate = parts[index]; candidate.kind = "fused"; }
        }
        if (begin == index && index + 1 < parts.size() && parts[index + 1].kind == "owned") {
          candidate.operations.append(parts[index + 1].operations);
          if (canFuse(candidate)) end = index + 1;
        }
      }
      if (!canFuse(candidate)) continue;
      parts.erase(parts.begin() + begin, parts.begin() + end + 1);
      parts.insert(parts.begin() + begin, std::move(candidate));
      index = begin;
    }
  }

  llvm::json::Array buffers, inputs, outputs, steps;
  llvm::StringSet<> registeredBuffers;
  auto registerBuffer = [&](Value value) {
    std::string id = identity(value);
    if (registeredBuffers.insert(id).second) {
      auto type = cast<RankedTensorType>(value.getType());
      llvm::json::Array shape;
      for (int64_t dimension : type.getShape()) shape.push_back(dimension);
      buffers.push_back(llvm::json::Object{{"id", id}, {"shape", std::move(shape)},
          {"element_type", elementTypeName(type).str()}});
    }
    return id;
  };
  for (Value argument : function.getArguments()) inputs.push_back(registerBuffer(argument));
  OwningOpRef<ModuleOp> result(ModuleOp::create(original.getLoc()));
  (*result)->setAttrs(original->getAttrs());
  if (auto *markers = report.get("markers"))
    (*result)->setAttr("ask.source_markers", StringAttr::get(original.getContext(), llvm::formatv("{0}", *markers).str()));
  for (Operation &operation : original.getBody()->getOperations())
    if (!isa<func::FuncOp>(operation)) result->getBody()->push_back(operation.clone());
  Builder attributes(original.getContext());
  llvm::DenseMap<Value, unsigned> producers;
  llvm::json::Object sources;
  llvm::StringSet<> reservedIds;
  for (Operation &operation : function.front()) {
    if (auto id = operation.getAttrOfType<StringAttr>("ask.id")) reservedIds.insert(id.getValue());
    for (Value value : operation.getResults()) reservedIds.insert(identity(value));
  }
  for (Value argument : function.getArguments()) reservedIds.insert(identity(argument));
  auto fresh = [&](std::string name) {
    while (!reservedIds.insert(name).second) name += "_";
    return name;
  };
  for (const auto &part : parts) {
    Ports boundary = ports(part);
    if (part.kind == "owned" && boundary.outputs.empty()) continue;
    unsigned number = steps.size();
    std::string id = "kernel" + std::to_string(number);
    std::string symbol = "ask_kernel_" + std::to_string(number);
    while (SymbolTable::lookupSymbolIn(original, symbol)) symbol += "_";
    llvm::SmallVector<Type> argumentTypes, resultTypes;
    for (Value value : boundary.inputs) argumentTypes.push_back(value.getType());
    for (Value value : boundary.outputs) resultTypes.push_back(value.getType());
    OpBuilder builder(original.getContext());
    builder.setInsertionPointToEnd(result->getBody());
    auto kernel = func::FuncOp::create(builder, function.getLoc(), symbol,
        FunctionType::get(original.getContext(), argumentTypes, resultTypes));
    for (auto attribute : function->getAttrs())
      if (attribute.getName() != "sym_name" && attribute.getName() != "function_type" &&
          attribute.getName() != "arg_attrs" && attribute.getName() != "res_attrs")
        kernel->setAttr(attribute.getName(), attribute.getValue());
    kernel->setAttr("ask.kernel_id", attributes.getStringAttr(id));
    kernel->setAttr("ask.kernel_kind", attributes.getStringAttr(part.kind));
    kernel->setAttr("ask.source_function", attributes.getStringAttr(function.getSymName()));
    auto *block = kernel.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    IRMapping mapping;
    llvm::json::Array stepInputs, stepOutputs, dependencies;
    std::set<unsigned> predecessors;
    for (auto [index, value] : llvm::enumerate(boundary.inputs)) {
      mapping.map(value, block->getArgument(index));
      kernel.setArgAttr(index, "ask.value_id", attributes.getStringAttr(fresh(id + ".arg" + std::to_string(index))));
      kernel.setArgAttr(index, "ask.source_value_id", attributes.getStringAttr(identity(value)));
      stepInputs.push_back(registerBuffer(value));
      if (auto found = producers.find(value); found != producers.end()) predecessors.insert(found->second);
    }
    unsigned scalarCount = 0;
    std::function<void(Value)> cloneScalar = [&](Value value) {
      if (mapping.contains(value)) return;
      auto *owner = value.getDefiningOp();
      for (Value operand : owner->getOperands()) cloneScalar(operand);
      auto *copy = builder.clone(*owner, mapping);
      std::string scalarId = fresh(id + ".scalar" + std::to_string(scalarCount++));
      copy->setAttr("ask.id", attributes.getStringAttr(scalarId));
      copy->setAttr("ask.source_value_id", attributes.getStringAttr(identity(value)));
      copy->setAttr("ask.value_id", attributes.getStringAttr(fresh(scalarId + ".result0")));
    };
    for (Operation *operation : part.operations) {
      for (Value operand : operation->getOperands())
        if (!mapping.contains(operand)) cloneScalar(operand);
      auto *copy = builder.clone(*operation, mapping);
      if (part.kind == "external") copy->setOperands(block->getArguments());
    }
    llvm::SmallVector<Value> returnValues;
    for (Value value : boundary.outputs) {
      returnValues.push_back(mapping.lookup(value));
      stepOutputs.push_back(registerBuffer(value));
      producers[value] = number;
    }
    func::ReturnOp::create(builder, function.getLoc(), returnValues);
    for (unsigned predecessor : predecessors) dependencies.push_back("kernel" + std::to_string(predecessor));
    llvm::json::Object step{{"id", id}, {"kind", part.kind}, {"function", symbol},
        {"inputs", std::move(stepInputs)}, {"outputs", std::move(stepOutputs)},
        {"depends_on", std::move(dependencies)}};
    if (part.external) step["external_id"] = part.external->getAttrOfType<StringAttr>("ask.id").getValue().str();
    steps.push_back(std::move(step));
    OwningOpRef<ModuleOp> single(ModuleOp::create(original.getLoc()));
    (*single)->setAttrs((*result)->getAttrs());
    for (Operation &declaration : result->getBody()->getOperations())
      if (!isa<func::FuncOp>(declaration)) single->getBody()->push_back(declaration.clone());
    single->getBody()->push_back(kernel->clone());
    sources[id] = printModule(*single);
  }
  for (Value value : returned.getOperands()) outputs.push_back(registerBuffer(value));
  llvm::json::Object plan{{"schema_version", 1}, {"kind", "ask.cuda.execution_plan"},
      {"inputs", std::move(inputs)}, {"outputs", std::move(outputs)}, {"buffers", std::move(buffers)},
      {"steps", std::move(steps)}, {"scheduling", llvm::json::Object{{"policy", "sequential"}, {"stream", "session"}}}};
  (*result)->setAttr("ask.execution_plan", attributes.getStringAttr(llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(plan))).str()));
  if (failed(verify(*result))) return failure();
  report["phase"] = "partitioned";
  report["execution_plan"] = std::move(plan);
  report["step_sources"] = std::move(sources);
  report["external_requests"] = externalRequests(original);
  ir = printModule(*result);
  return success();
}
} // namespace

llvm::Expected<PreparedSource> partitionSource(llvm::StringRef source, const PartitionOptions &options) {
  DialectRegistry registry;
  registry.insert<ASKDialect, func::FuncDialect>();
  MLIRContext context(registry);
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source, "<ask-partition>"), llvm::SMLoc());
  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  SourceMgrDiagnosticHandler handler(sources, &context, diagnosticStream);
  auto module = parseSourceFile<ModuleOp>(sources, &context);
  PreparedSource result;
  result.report["schema_version"] = 1;
  if (!module || failed(verify(*module)) ||
      failed(options.optimize ? optimizeModule(*module, result.report, options.optimization)
                              : prepareModule(*module, result.report)) ||
      failed(partition(*module, result.report, options, result.ir)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), diagnostics);
  return result;
}
} // namespace ask
