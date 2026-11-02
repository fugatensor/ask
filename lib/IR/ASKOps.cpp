#include "ask/IR/ASKOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <cmath>

using namespace mlir;

namespace ask {
llvm::StringRef elementTypeName(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
  if (type.isSignlessInteger(1)) return "bool";
  if (type.isSignlessInteger(32)) return "i32";
  if (type.isSignlessInteger(64)) return "i64";
  if (type.isF32()) return "f32";
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  return "";
}

unsigned elementByteWidth(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
  assert(isSupportedElementType(type) && "requires a supported ASK element type");
  return (type.getIntOrFloatBitWidth() + 7) / 8;
}

bool isSupportedElementType(Type type) {
  return type.isSignlessInteger(1) || type.isSignlessInteger(32) || type.isSignlessInteger(64) || type.isF32() || type.isF16() || type.isBF16();
}

bool isSupportedValueType(Type type) {
  if (isSupportedElementType(type))
    return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isSupportedElementType(tensor.getElementType()) &&
         !tensor.getEncoding();
}

bool haveCompatibleShapeOnly(Type lhs, Type rhs) {
  if (!isSupportedValueType(lhs) || !isSupportedValueType(rhs))
    return false;
  auto a = dyn_cast<RankedTensorType>(lhs);
  auto b = dyn_cast<RankedTensorType>(rhs);
  if (!a || !b)
    return !a && !b;
  if (a.getRank() != b.getRank())
    return false;
  for (auto [x, y] : llvm::zip(a.getShape(), b.getShape()))
    if (!ShapedType::isDynamic(x) && !ShapedType::isDynamic(y) && x != y)
      return false;
  return true;
}

bool haveCompatibleShapes(Type lhs, Type rhs) {
  if (!haveCompatibleShapeOnly(lhs, rhs)) return false;
  if (auto tensor = dyn_cast<RankedTensorType>(lhs))
    return tensor.getElementType() == cast<RankedTensorType>(rhs).getElementType();
  return lhs == rhs;
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
             << "supports only bool/i32/i64/f32/f16/bf16 scalars and ranked bool/i32/i64/f32/f16/bf16 "
                "tensors without encoding; got "
             << type;
  return success();
}

LogicalResult ConstantOp::verify() {
  Type type = getResult().getType();
  if (!isSupportedElementType(type))
    return emitOpError("requires a scalar bool, i32, i64, f32, f16 or bf16 result");
  auto value = (*this)->getAttrOfType<TypedAttr>("value");
  if (!value || value.getType() != type ||
      (isa<FloatType>(type) ? !isa<FloatAttr>(value) : !isa<IntegerAttr>(value)))
    return emitOpError("requires an integer or float 'value' matching the result type");
  return success();
}

LogicalResult SplatOp::verify() {
  if (!isSupportedElementType(getOperand().getType()))
    return emitOpError("requires a scalar bool, i32, i64, f32, f16 or bf16 operand");
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!result || !isSupportedValueType(result))
    return emitOpError("requires a ranked bool/i32/i64/f32/f16/bf16 tensor result without encoding");
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

LogicalResult CastOp::verify() {
  if (failed(verifyDataTypes(*this, (*this)->getOperandTypes())) ||
      failed(verifyDataTypes(*this, (*this)->getResultTypes()))) return failure();
  if (!haveCompatibleShapeOnly(getOperand().getType(), getResult().getType()))
    return emitOpError("cast preserves scalar/tensor kind, rank and shape");
  for (StringRef name : {"rounding", "overflow", "fastmath", "fastmathFlags"})
    if ((*this)->hasAttr(name))
      return emitOpError("cast uses fixed ASK numerical rules; unsupported override: ") << name;
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
  Type type = op->getResult(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
  if (type.isSignlessInteger(1))
    return op->emitOpError("bool add/mul is not defined; insert an explicit numeric cast");
  if (Attribute overflow = op->getAttr("overflow")) {
    if (!type.isSignlessInteger(32) && !type.isSignlessInteger(64))
      return op->emitOpError("overflow is an integer-only attribute; floating arithmetic uses separate IEEE round-to-nearest operations");
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
      if (!haveCompatibleShapeOnly(type.getInput(index), output))
        return emitOpError("same_as results must be compatible with shape_operand");
  } else {
    for (Type output : type.getResults())
      if (auto tensor = dyn_cast<RankedTensorType>(output))
        if (!tensor.hasStaticShape())
          return emitOpError("fixed shape rule requires concrete semantic result dimensions");
  }
  return success();
}

static bool isJSONAttribute(Attribute value) {
  if (isa<UnitAttr, StringAttr, BoolAttr>(value)) return true;
  if (auto number = dyn_cast<IntegerAttr>(value))
    return isa<IntegerType>(number.getType()) && number.getValue().getBitWidth() <= 64;
  if (auto number = dyn_cast<FloatAttr>(value))
    return std::isfinite(number.getValueAsDouble());
  if (auto array = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(array, isJSONAttribute);
  if (auto dict = dyn_cast<DictionaryAttr>(value))
    return llvm::all_of(dict, [](NamedAttribute item) { return isJSONAttribute(item.getValue()); });
  return false;
}

static bool isScalarBinding(StringRef type, Attribute value) {
  if (type == "i1") return isa<BoolAttr>(value);
  if (type == "fp16" || type == "bf16" || type == "fp32" || type == "fp64")
    return !isa<BoolAttr>(value) && isa<IntegerAttr, FloatAttr>(value) && isJSONAttribute(value);
  unsigned bits = llvm::StringSwitch<unsigned>(type)
      .Cases({"i8", "u8"}, 8).Cases({"i16", "u16"}, 16)
      .Cases({"i32", "u32"}, 32).Cases({"i64", "u64"}, 64).Default(0);
  auto integer = dyn_cast<IntegerAttr>(value);
  if (!bits || !integer || isa<BoolAttr>(value) || !isJSONAttribute(value)) return false;
  const auto &number = integer.getValue();
  bool unsignedValue = cast<IntegerType>(integer.getType()).isUnsigned();
  if (type.starts_with("u"))
    return (unsignedValue || !number.isNegative()) && number.getActiveBits() <= bits;
  return unsignedValue ? number.getActiveBits() < bits : number.isSignedIntN(bits);
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
    Attribute value = bindings.get(item.getName());
    if (!type || !value)
      return emitOpError("signature requires string types and a source binding for every parameter");
    if (!type.getValue().starts_with("*")) {
      if (!isScalarBinding(type.getValue(), value))
        return emitOpError() << "invalid scalar source binding for " << item.getName() << " of type " << type;
      continue;
    }
    if (type.getValue() != "*i1" && type.getValue() != "*i32" && type.getValue() != "*i64" &&
        type.getValue() != "*fp32" && type.getValue() != "*fp16" && type.getValue() != "*bf16")
      return emitOpError("pointer parameters require one of the six ASK element types");
    auto binding = dyn_cast<StringAttr>(value);
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
    if (!isJSONAttribute(constant.getValue()))
      return emitOpError("constexpr constants must be finite JSON values with at most 64-bit integers");
  }
  auto target = (*this)->getAttrOfType<DictionaryAttr>("target");
  for (StringRef name : {"options", "target"}) {
    for (NamedAttribute item : (*this)->getAttrOfType<DictionaryAttr>(name)) {
      if (!isJSONAttribute(item.getValue()))
        return emitOpError() << "'" << name << "' values must be finite JSON values with at most 64-bit integers";
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
    if (!signature.getAs<StringAttr>(item.getName()).getValue().starts_with("*")) continue;
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
    std::string expected = "*";
    expected += elementType.isF32() ? "fp32" : elementType.isF16() ? "fp16" :
                elementType.isBF16() ? "bf16" : elementType.isSignlessInteger(1) ? "i1" :
                elementType.isSignlessInteger(64) ? "i64" : "i32";
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
    return emitOpError("supports bool/i32/i64/f32/f16/bf16 scalars and ranked tensors only");
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
