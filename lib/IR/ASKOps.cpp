#include "ask/IR/ASKOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace ask {
bool isSupportedElementType(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

bool isSupportedValueType(Type type) {
  if (isSupportedElementType(type))
    return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isSupportedElementType(tensor.getElementType()) &&
         !tensor.getEncoding();
}

bool haveCompatibleShapes(Type lhs, Type rhs) {
  if (!isSupportedValueType(lhs) || !isSupportedValueType(rhs))
    return false;
  auto a = dyn_cast<RankedTensorType>(lhs);
  auto b = dyn_cast<RankedTensorType>(rhs);
  if (!a || !b)
    return lhs == rhs;
  if (a.getRank() != b.getRank() || a.getElementType() != b.getElementType())
    return false;
  for (auto [x, y] : llvm::zip(a.getShape(), b.getShape()))
    if (!ShapedType::isDynamic(x) && !ShapedType::isDynamic(y) && x != y)
      return false;
  return true;
}

static LogicalResult requireString(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<StringAttr>(name);
  if (!value || value.getValue().empty())
    return op->emitOpError() << "requires a nonempty string '" << name << "'";
  return success();
}

static LogicalResult verifyDataTypes(Operation *op, TypeRange types) {
  for (Type type : types)
    if (!isSupportedValueType(type))
      return op->emitOpError()
             << "supports only i32/f32 scalars and ranked i32/f32 "
                "tensors without encoding; got "
             << type;
  return success();
}

LogicalResult ConstantOp::verify() {
  Type type = getResult().getType();
  if (!isSupportedElementType(type))
    return emitOpError("requires a scalar i32 or f32 result");
  auto value = (*this)->getAttrOfType<TypedAttr>("value");
  if (!value || value.getType() != type ||
      (type.isF32() ? !isa<FloatAttr>(value) : !isa<IntegerAttr>(value)))
    return emitOpError("requires an integer or float 'value' matching the result type");
  return success();
}

LogicalResult SplatOp::verify() {
  if (!isSupportedElementType(getOperand().getType()))
    return emitOpError("requires a scalar i32 or f32 operand");
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!result || !isSupportedValueType(result))
    return emitOpError("requires a ranked i32/f32 tensor result without encoding");
  if (result.getElementType() != getOperand().getType())
    return emitOpError("splat operand and result element types must match");
  auto shape = (*this)->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (!shape || shape.size() != result.getRank())
    return emitOpError("requires a dense i64 'shape' matching the result rank");
  for (auto [extent, declared] : llvm::zip(shape.asArrayRef(), result.getShape())) {
    if (extent < 0)
      return emitOpError("prototype requires nonnegative concrete splat dimensions");
    if (!ShapedType::isDynamic(declared) && declared != extent)
      return emitOpError("shape conflicts with the declared result type");
  }
  return success();
}

static LogicalResult verifyBinary(Operation *op) {
  if (failed(verifyDataTypes(op, op->getOperandTypes())) ||
      failed(verifyDataTypes(op, op->getResultTypes())))
    return failure();
  if (!haveCompatibleShapes(op->getOperand(0).getType(),
                            op->getOperand(1).getType()))
    return op->emitOpError("requires equal input shapes; implicit broadcasting is not supported");
  for (Value operand : op->getOperands())
    if (!haveCompatibleShapes(operand.getType(), op->getResult(0).getType()))
      return op->emitOpError("declared result shape conflicts with an input");
  if (Attribute overflow = op->getAttr("overflow")) {
    Type type = op->getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    if (!type.isSignlessInteger(32))
      return op->emitOpError("overflow is an integer-only attribute; f32 arithmetic uses separate IEEE round-to-nearest operations");
    auto value = dyn_cast<StringAttr>(overflow);
    if (!value || value.getValue() != "wrap")
      return op->emitOpError("prototype supports only overflow = \"wrap\"");
  }
  if (op->hasAttr("fastmath") || op->hasAttr("fastmathFlags"))
    return op->emitOpError("fast-math flags are not supported; floating arithmetic preserves each operation's rounding");
  return success();
}

LogicalResult AddOp::verify() { return verifyBinary(*this); }
LogicalResult MulOp::verify() { return verifyBinary(*this); }

FunctionType SemanticsOp::getFunctionType() {
  auto attribute = (*this)->getAttrOfType<TypeAttr>("function_type");
  return attribute ? dyn_cast<FunctionType>(attribute.getValue()) : FunctionType();
}

LogicalResult SemanticsOp::verify() {
  if (!isa<ModuleOp>((*this)->getParentOp()))
    return emitOpError("must be declared directly in builtin.module");
  auto type = getFunctionType();
  if (!type)
    return emitOpError("requires a function_type TypeAttr");
  if (failed(verifyDataTypes(*this, type.getInputs())) ||
      failed(verifyDataTypes(*this, type.getResults())))
    return failure();
  auto effects = (*this)->getAttrOfType<StringAttr>("effects");
  if (!effects || effects.getValue() != "none")
    return emitOpError("prototype supports effects = \"none\" only; stateful/unknown effects and tokens are not implemented");
  auto rule = (*this)->getAttrOfType<StringAttr>("shape_rule");
  if (!rule || (rule.getValue() != "same_as" && rule.getValue() != "fixed"))
    return emitOpError("requires shape_rule = \"same_as\" or \"fixed\"");
  auto operand = (*this)->getAttrOfType<IntegerAttr>("shape_operand");
  if ((*this)->hasAttr("shape_operand") &&
      (!operand || !operand.getType().isSignlessInteger(64)))
    return emitOpError("shape_operand must be an i64 integer");
  if (rule.getValue() == "same_as") {
    int64_t index = operand ? operand.getInt() : 0;
    if (index < 0 || static_cast<uint64_t>(index) >= type.getNumInputs())
      return emitOpError("shape_operand is outside the semantic input list");
    for (Type output : type.getResults())
      if (!haveCompatibleShapes(type.getInput(index), output))
        return emitOpError("same_as results must be compatible with shape_operand");
  } else {
    for (Type output : type.getResults())
      if (auto tensor = dyn_cast<RankedTensorType>(output))
        if (!tensor.hasStaticShape())
          return emitOpError("fixed shape rule requires concrete semantic result dimensions");
  }
  return success();
}

LogicalResult ImplementationOp::verify() {
  if (!isa<ModuleOp>((*this)->getParentOp()))
    return emitOpError("must be declared directly in builtin.module");
  for (StringRef name : {"compiler", "source", "entry"})
    if (failed(requireString(*this, name)))
      return failure();
  auto signature = (*this)->getAttrOfType<DictionaryAttr>("signature");
  auto bindings = (*this)->getAttrOfType<DictionaryAttr>("source_bindings");
  if (!signature || !bindings || signature.size() != bindings.size())
    return emitOpError("requires signature and source_bindings dictionaries with identical parameter names");
  for (NamedAttribute item : signature) {
    auto type = dyn_cast<StringAttr>(item.getValue());
    auto binding = bindings.getAs<StringAttr>(item.getName());
    if (!type || (type.getValue() != "*i32" && type.getValue() != "i32" &&
                  type.getValue() != "*fp32" && type.getValue() != "fp32"))
      return emitOpError("signature parameters must be i32/fp32 scalars or *i32/*fp32 pointers");
    if (!binding)
      return emitOpError() << "missing string source binding for " << item.getName();
    StringRef text = binding.getValue();
    if (!text.consume_front("input:") && !text.consume_front("output:"))
      return emitOpError("source bindings must use input:<index> or output:<index>");
    unsigned index;
    if (text.empty() || text.getAsInteger(10, index))
      return emitOpError("source binding has an invalid nonnegative port index");
  }
  for (StringRef name : {"constants", "options", "target"})
    if (!(*this)->getAttrOfType<DictionaryAttr>(name))
      return emitOpError() << "requires a dictionary '" << name << "'";
  auto constants = (*this)->getAttrOfType<DictionaryAttr>("constants");
  for (NamedAttribute constant : constants) {
    if (signature.get(constant.getName()))
      return emitOpError("a constexpr constant must not also appear in signature");
    auto value = dyn_cast<IntegerAttr>(constant.getValue());
    if (!value || value.getValue().getBitWidth() > 64)
      return emitOpError("prototype constexpr constants must be integers of at most 64 bits");
  }
  auto target = (*this)->getAttrOfType<DictionaryAttr>("target");
  for (StringRef name : {"options", "target"}) {
    for (NamedAttribute item : (*this)->getAttrOfType<DictionaryAttr>(name)) {
      if (isa<StringAttr>(item.getValue()))
        continue;
      auto value = dyn_cast<IntegerAttr>(item.getValue());
      if (!value || value.getValue().getBitWidth() > 64)
        return emitOpError() << "prototype '" << name
                             << "' values must be strings or integers of at most 64 bits";
    }
  }
  if (!target.getAs<StringAttr>("backend") ||
      target.getAs<StringAttr>("backend").getValue().empty())
    return emitOpError("target requires a nonempty backend string");
  auto grid = (*this)->getAttrOfType<DenseI64ArrayAttr>("grid");
  if (!grid || grid.size() != 3)
    return emitOpError("requires grid as three concrete i64 dimensions");
  for (int64_t dimension : grid.asArrayRef())
    if (dimension <= 0)
      return emitOpError("prototype grid dimensions must be positive");
  return success();
}

SemanticsOp ExternalOp::resolveSemantics() {
  auto reference = (*this)->getAttrOfType<FlatSymbolRefAttr>("semantics");
  return reference ? SymbolTable::lookupNearestSymbolFrom<SemanticsOp>(*this, reference)
                   : SemanticsOp();
}

ImplementationOp ExternalOp::resolveImplementation() {
  auto reference = (*this)->getAttrOfType<FlatSymbolRefAttr>("implementation");
  return reference ? SymbolTable::lookupNearestSymbolFrom<ImplementationOp>(*this, reference)
                   : ImplementationOp();
}

LogicalResult ExternalOp::verify() {
  auto semantics = resolveSemantics();
  auto implementation = resolveImplementation();
  if (!semantics)
    return emitOpError("semantics must resolve to an ask.semantics symbol");
  if (!implementation)
    return emitOpError("implementation must resolve to an ask.implementation symbol");
  if (failed(semantics.verify()) || failed(implementation.verify()))
    return failure();
  FunctionType type = semantics.getFunctionType();
  if (getNumOperands() != type.getNumInputs() ||
      getNumResults() != type.getNumResults())
    return emitOpError("operand/result count does not match semantic function_type");
  for (auto [actual, expected] : llvm::zip(getOperandTypes(), type.getInputs()))
    if (!haveCompatibleShapes(actual, expected))
      return emitOpError("input type/shape conflicts with semantic function_type");
  for (auto [actual, expected] : llvm::zip(getResultTypes(), type.getResults()))
    if (!haveCompatibleShapes(actual, expected))
      return emitOpError("result type/shape conflicts with semantic function_type");

  auto signature = implementation->getAttrOfType<DictionaryAttr>("signature");
  auto bindings = implementation->getAttrOfType<DictionaryAttr>("source_bindings");
  llvm::SmallBitVector inputs(getNumOperands()), outputs(getNumResults());
  for (NamedAttribute item : bindings) {
    StringRef binding = cast<StringAttr>(item.getValue()).getValue();
    bool input = binding.consume_front("input:");
    if (!input)
      binding.consume_front("output:");
    unsigned index = 0;
    if (binding.getAsInteger(10, index))
      return emitOpError("invalid source binding index");
    auto &seen = input ? inputs : outputs;
    if (index >= seen.size() || seen[index])
      return emitOpError("prototype requires each data port to have exactly one valid source binding");
    seen.set(index);
    Type dataType = input ? getOperand(index).getType() : getResult(index).getType();
    auto tensor = dyn_cast<RankedTensorType>(dataType);
    Type elementType = tensor ? tensor.getElementType() : dataType;
    std::string expected = tensor || !input ? "*" : "";
    expected += elementType.isF32() ? "fp32" : "i32";
    if (signature.getAs<StringAttr>(item.getName()).getValue() != expected)
      return emitOpError("source signature is incompatible with the bound data port");
  }
  if (!inputs.all() || !outputs.all())
    return emitOpError("source_bindings must cover every semantic input and output");
  return success();
}

LogicalResult MarkOp::verify() {
  if (getOperand().getType() != getResult().getType())
    return emitOpError("input and result must have exactly the same type");
  if (!isSupportedValueType(getOperand().getType()))
    return emitOpError("supports i32/f32 scalars and ranked tensors only");
  for (StringRef name : {"mark_id", "value_id", "role"})
    if (failed(requireString(*this, name)))
      return failure();
  StringRef role = (*this)->getAttrOfType<StringAttr>("role").getValue();
  if (role != "input" && role != "output" && role != "value")
    return emitOpError("role must be input, output, or value");
  if (role != "value" && failed(requireString(*this, "port_id")))
    return failure();
  if ((*this)->hasAttr("port_id") && failed(requireString(*this, "port_id")))
    return failure();
  return success();
}
} // namespace ask
