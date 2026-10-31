#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace mlir;

namespace ask {
namespace {
std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  return text;
}

std::string printAttribute(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << attribute;
  return text;
}

bool isConcrete(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.hasStaticShape();
  return type.isSignlessInteger(32);
}

LogicalResult checkSize(Operation *owner, Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return success();
  if (!tensor.hasStaticShape())
    return owner->emitError("unresolved tensor dimensions; supply concrete input shapes or a supported shape rule");
  if (llvm::is_contained(tensor.getShape(), int64_t(0)))
    return success();
  uint64_t elements = 1;
  // The prototype uses signed 64-bit byte counts in its external descriptions.
  constexpr uint64_t maxElements = uint64_t(std::numeric_limits<int64_t>::max()) / 4;
  for (int64_t extent : tensor.getShape()) {
    if (extent < 0 || uint64_t(extent) > maxElements / elements)
      return owner->emitError("tensor byte count exceeds the prototype's signed 64-bit size range");
    elements *= uint64_t(extent);
  }
  return success();
}

LogicalResult refine(Operation *op, unsigned index, Type inferred,
                     bool &changed) {
  Type declared = op->getResult(index).getType();
  if (!haveCompatibleShapes(declared, inferred))
    return op->emitOpError() << "shape conflict: declared " << declared
                             << ", inferred " << inferred;
  if (!isConcrete(inferred))
    return op->emitOpError("shape inference did not produce a concrete result");
  if (failed(checkSize(op, inferred)))
    return failure();
  if (declared != inferred) {
    op->getResult(index).setType(inferred);
    changed = true;
  }
  return success();
}

LogicalResult inferOperation(Operation *op, bool &changed) {
  if (isa<ConstantOp>(op))
    return success();
  if (auto splat = dyn_cast<SplatOp>(op)) {
    auto shape = op->getAttrOfType<DenseI64ArrayAttr>("shape");
    return refine(op, 0,
                  RankedTensorType::get(shape.asArrayRef(), splat.getOperand().getType()),
                  changed);
  }
  if (isa<AddOp, MulOp>(op)) {
    Type lhs = op->getOperand(0).getType();
    Type rhs = op->getOperand(1).getType();
    if (lhs != rhs)
      return op->emitOpError() << "requires equal concrete input types; got "
                               << lhs << " and " << rhs;
    return refine(op, 0, lhs, changed);
  }
  if (auto mark = dyn_cast<MarkOp>(op))
    return refine(op, 0, mark.getOperand().getType(), changed);
  if (auto external = dyn_cast<ExternalOp>(op)) {
    auto semantics = external.resolveSemantics();
    auto rule = semantics->getAttrOfType<StringAttr>("shape_rule").getValue();
    auto operand = semantics->getAttrOfType<IntegerAttr>("shape_operand");
    unsigned shapeOperand = operand ? unsigned(operand.getInt()) : 0;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      Type inferred = rule == "same_as" ? op->getOperand(shapeOperand).getType()
                                        : semantics.getFunctionType().getResult(i);
      if (failed(refine(op, i, inferred, changed)))
        return failure();
    }
    return external.verify();
  }
  return op->emitOpError("operation is not supported by the ASK frontend prototype");
}

std::string valueIdFor(Operation *op, unsigned result, StringRef operationId) {
  if (auto explicitId = op->getAttrOfType<StringAttr>("ask.value_id"))
    return explicitId.getValue().str();
  return (operationId + ".result" + llvm::Twine(result)).str();
}

bool matchesPort(StringRef text, StringRef operationId, StringRef direction,
                 unsigned index, unsigned count) {
  std::string base = (operationId + "." + direction).str();
  return text == base + std::to_string(index) ||
         (count == 1 && text == base);
}

llvm::json::Array shapeArray(RankedTensorType type) {
  llvm::json::Array result;
  for (int64_t dimension : type.getShape())
    result.push_back(dimension);
  return result;
}

int64_t readRevision(ModuleOp module, StringRef key) {
  if (auto value = module->getAttrOfType<IntegerAttr>(key))
    return value.getInt();
  return 0;
}

std::string digest(StringRef input) {
  llvm::SHA256 hash;
  hash.update(input);
  auto bytes = hash.final();
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (uint8_t byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}
} // namespace

LogicalResult prepareModule(ModuleOp module, llvm::json::Object &report,
                            bool eraseMarkers) {
  // The caller owns the transaction. Do not publish a partially populated report.
  if (Attribute attribute = module->getAttr("ask.schema_version")) {
    auto version = dyn_cast<IntegerAttr>(attribute);
    if (!version || !version.getType().isSignlessInteger(64) ||
        version.getInt() != 1)
      return module.emitError("ask.schema_version must be the supported i64 value 1");
  }
  for (StringRef name : {"ask.semantic_revision", "ask.ir_revision"}) {
    if (Attribute attribute = module->getAttr(name)) {
      auto revision = dyn_cast<IntegerAttr>(attribute);
      if (!revision || !revision.getType().isSignlessInteger(64))
        return module.emitError() << name << " must be an i64 integer";
    }
  }
  if (failed(mlir::verify(module)))
    return failure();
  llvm::SmallVector<func::FuncOp> functions;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto function = dyn_cast<func::FuncOp>(op))
      functions.push_back(function);
    else if (!isa<SemanticsOp, ImplementationOp>(op))
      return op.emitOpError("module may contain only func.func, ask.semantics, and ask.implementation");
  }
  if (functions.empty())
    return module.emitError("ASK graph module must define at least one graph function");

  int64_t semanticRevision = readRevision(module, "ask.semantic_revision");
  int64_t irRevision = readRevision(module, "ask.ir_revision");
  if (semanticRevision < 0 || irRevision < 0 ||
      semanticRevision == std::numeric_limits<int64_t>::max() ||
      irRevision == std::numeric_limits<int64_t>::max())
    return module.emitError("ASK revisions must be nonnegative and leave room for an update");

  llvm::DenseMap<Value, std::string> logicalIds;
  llvm::DenseMap<Operation *, std::string> operationIds;
  llvm::StringSet<> uniqueValueIds;
  llvm::StringSet<> uniqueOperationIds;
  llvm::StringSet<> uniqueMarkerIds;
  llvm::SmallVector<MarkOp> markers;
  bool typeChanged = false;
  bool irChanged = false;
  Builder builder(module.getContext());

  for (func::FuncOp function : functions) {
    if (function.isExternal() || !llvm::hasSingleElement(function.getBody()))
      return function.emitOpError("prototype graph functions must have exactly one defined basic block");
    if (function.getNumArguments() != function.getFunctionType().getNumInputs())
      return function.emitOpError("argument count does not match the function type");
    unsigned ordinal = 0;
    for (Operation &op : function.front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      if (!isa<ConstantOp, SplatOp, AddOp, MulOp, ExternalOp, MarkOp>(op))
        return op.emitOpError("unsupported graph operation; initial subset is constant, splat, add, mul, external, and mark");
      auto id = op.getAttrOfType<StringAttr>("ask.id");
      if (op.hasAttr("ask.id") && (!id || id.getValue().empty()))
        return op.emitOpError("ask.id must be a nonempty string");
      std::string name = id ? id.getValue().str()
                            : (function.getSymName() + ".op" + llvm::Twine(ordinal)).str();
      ++ordinal;
      if (!uniqueOperationIds.insert(name).second)
        return op.emitOpError() << "duplicate module-wide operation identity '" << name << "'";
      operationIds[&op] = name;
      if (!id) {
        op.setAttr("ask.id", builder.getStringAttr(name));
        irChanged = true;
      }
    }

    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      if (!isSupportedValueType(argument.getType()) || !isConcrete(argument.getType()))
        return function.emitOpError() << "input " << index
             << " requires a concrete ranked i32 tensor or i32 scalar; result constraints cannot supply unknown input dimensions";
      if (failed(checkSize(function, argument.getType())))
        return failure();
      auto declaredId = function.getArgAttrOfType<StringAttr>(index, "ask.value_id");
      if (function.getArgAttr(index, "ask.value_id") &&
          (!declaredId || declaredId.getValue().empty()))
        return function.emitOpError("argument ask.value_id must be a nonempty string");
      std::string id = declaredId ? declaredId.getValue().str()
                                  : (function.getSymName() + ".arg" + llvm::Twine(index)).str();
      if (!uniqueValueIds.insert(id).second)
        return function.emitOpError() << "duplicate logical value identity '" << id << "'";
      logicalIds[argument] = id;
      if (!declaredId) {
        function.setArgAttr(index, "ask.value_id", builder.getStringAttr(id));
        irChanged = true;
      }
    }

    for (Operation &op : function.front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      if (failed(inferOperation(&op, typeChanged)))
        return failure();
      if (auto mark = dyn_cast<MarkOp>(op)) {
        auto found = logicalIds.find(mark.getOperand());
        if (found == logicalIds.end())
          return mark.emitOpError("operand has no logical value identity");
        StringRef requested = op.getAttrOfType<StringAttr>("value_id").getValue();
        if (requested != found->second)
          return mark.emitOpError() << "value_id '" << requested
                                   << "' does not match operand identity '" << found->second << "'";
        if (Attribute identity = op.getAttr("ask.value_id")) {
          auto text = dyn_cast<StringAttr>(identity);
          if (!text || text.getValue() != found->second)
            return mark.emitOpError("mark result must reuse the operand logical value identity");
        }
        if (!uniqueMarkerIds.insert(op.getAttrOfType<StringAttr>("mark_id").getValue()).second)
          return mark.emitOpError("prototype input graphs require unique mark_id values");
        logicalIds[mark.getResult()] = found->second;
        markers.push_back(mark);
      } else {
        auto declaredId = op.getAttrOfType<StringAttr>("ask.value_id");
        if (op.hasAttr("ask.value_id") &&
            (!declaredId || declaredId.getValue().empty() || op.getNumResults() != 1))
          return op.emitOpError("ask.value_id requires a nonempty string on a single-result operation");
        for (auto [index, result] : llvm::enumerate(op.getResults())) {
          std::string id = valueIdFor(&op, index, operationIds[&op]);
          if (!uniqueValueIds.insert(id).second)
            return op.emitOpError() << "duplicate logical value identity '" << id << "'";
          logicalIds[result] = id;
          if (op.getNumResults() == 1 && !declaredId) {
            op.setAttr("ask.value_id", builder.getStringAttr(id));
            irChanged = true;
          }
        }
      }
    }

    auto terminator = dyn_cast<func::ReturnOp>(function.front().getTerminator());
    if (!terminator || terminator.getNumOperands() != function.getFunctionType().getNumResults())
      return function.emitOpError("requires func.return matching the graph result count");
    llvm::SmallVector<Type> returns;
    for (auto [returned, declared] : llvm::zip(terminator.getOperands(), function.getResultTypes())) {
      if (!haveCompatibleShapes(returned.getType(), declared))
        return terminator.emitOpError("returned shape conflicts with graph function result declaration");
      returns.push_back(returned.getType());
    }
    auto refined = FunctionType::get(module.getContext(), function.getArgumentTypes(), returns);
    if (function.getFunctionType() != refined) {
      function.setType(refined);
      typeChanged = true;
    }
  }

  llvm::json::Array markerReport;
  for (MarkOp mark : markers) {
    auto function = mark->getParentOfType<func::FuncOp>();
    StringRef role = mark->getAttrOfType<StringAttr>("role").getValue();
    auto port = mark->getAttrOfType<StringAttr>("port_id");
    if (port) {
      bool matched = false;
      for (Operation &candidate : function.front()) {
        if (isa<MarkOp, func::ReturnOp>(candidate))
          continue;
        StringRef id = operationIds[&candidate];
        if (role != "output")
          for (auto [index, operand] : llvm::enumerate(candidate.getOperands()))
            if (matchesPort(port.getValue(), id, "in", index, candidate.getNumOperands()) &&
                logicalIds.lookup(operand) == logicalIds.lookup(mark.getOperand()))
              matched = true;
        if (role != "input")
          for (auto [index, result] : llvm::enumerate(candidate.getResults()))
            if (matchesPort(port.getValue(), id, "out", index, candidate.getNumResults()) &&
                logicalIds.lookup(result) == logicalIds.lookup(mark.getOperand()))
              matched = true;
      }
      if (!matched)
        return mark.emitOpError("port_id must name an existing data port connected to this logical value (op.in/op.out or op.in0/op.out0)");
    }
    llvm::json::Object entry{
        {"mark_id", mark->getAttrOfType<StringAttr>("mark_id").getValue().str()},
        {"value_id", logicalIds.lookup(mark.getOperand())},
        {"function", function.getSymName().str()},
        {"role", role.str()},
        {"operation_id", operationIds[mark]},
        {"erased", eraseMarkers}};
    if (port)
      entry["port_id"] = port.getValue().str();
    markerReport.push_back(std::move(entry));
  }

  // Type refinement is committed on this caller-owned clone. Verification sees
  // the entire new use-def graph, including refined function return signatures.
  if (failed(mlir::verify(module)))
    return failure();
  if (typeChanged)
    ++semanticRevision;
  if (typeChanged || irChanged || (eraseMarkers && !markers.empty()))
    ++irRevision;

  llvm::json::Array bindings;
  llvm::json::Array specializations;
  llvm::json::Array rules;
  std::string digestInput = "ask.shape.prototype.v1;" + std::to_string(semanticRevision);
  for (SemanticsOp semantics : module.getOps<SemanticsOp>()) {
    std::string descriptor = printAttribute(semantics->getAttrDictionary());
    rules.push_back(llvm::json::Object{{"symbol", semantics.getNameAttr().getValue().str()},
                                     {"definition", descriptor}});
    digestInput += descriptor;
  }
  for (func::FuncOp function : functions) {
    auto appendShape = [&](Value value, bool input, unsigned index) {
      std::string identity = logicalIds.lookup(value);
      std::string type = printType(value.getType());
      digestInput += function.getSymName().str() + ":" + identity + ":" + type + ";";
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) {
        bindings.push_back(llvm::json::Object{
            {"function", function.getSymName().str()}, {"value_id", identity},
            {"rank", tensor.getRank()}, {"shape", shapeArray(tensor)},
            {"element_type", "i32"}, {"type", type}});
      }
      if (input)
        specializations.push_back(llvm::json::Object{
            {"function", function.getSymName().str()}, {"argument", int64_t(index)},
            {"value_id", identity}, {"type", type}});
    };
    for (auto [index, argument] : llvm::enumerate(function.getArguments()))
      appendShape(argument, true, index);
    for (Operation &op : function.front())
      if (!isa<MarkOp, func::ReturnOp>(op))
        for (Value result : op.getResults())
          appendShape(result, false, 0);
  }

  if (eraseMarkers)
    for (MarkOp mark : markers) {
      mark.getResult().replaceAllUsesWith(mark.getOperand());
      mark.erase();
    }
  module->setAttr("ask.semantic_revision", builder.getI64IntegerAttr(semanticRevision));
  module->setAttr("ask.ir_revision", builder.getI64IntegerAttr(irRevision));
  module->setAttr("ask.schema_version", builder.getI64IntegerAttr(1));
  if (failed(mlir::verify(module)))
    return failure();
  llvm::json::Object snapshot{
      {"schema_version", 1}, {"semantic_revision", semanticRevision},
      {"shape_rule_versions", std::move(rules)},
      {"shape_bindings", std::move(bindings)},
      {"input_specialization", std::move(specializations)},
      {"snapshot_digest", digest(digestInput)}};
  report["shape_snapshot"] = std::move(snapshot);
  report["markers"] = std::move(markerReport);
  report["ir_revision"] = irRevision;
  return success();
}
} // namespace ask
