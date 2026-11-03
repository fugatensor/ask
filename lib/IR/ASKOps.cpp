#include "ask/IR/ASKOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <cmath>
#include <limits>

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

LogicalResult ReshapeOp::verify() {
  auto input = dyn_cast<RankedTensorType>(getOperand().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!input || !result || !isSupportedValueType(input) || !isSupportedValueType(result))
    return emitOpError("requires ranked supported tensor operands/results without encoding");
  if (input.getElementType() != result.getElementType())
    return emitOpError("reshape preserves the element type");
  auto shape = (*this)->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (!shape || shape.size() != result.getRank())
    return emitOpError("requires a dense i64 'shape' matching the result rank");
  for (auto [extent, declared] : llvm::zip(shape.asArrayRef(), result.getShape())) {
    if (extent < 0)
      return emitOpError("reshape requires nonnegative concrete dimensions; infer -1 before submitting IR");
    if (!ShapedType::isDynamic(declared) && declared != extent)
      return emitOpError("shape conflicts with the declared result type");
  }
  auto count = [&](llvm::ArrayRef<int64_t> dimensions) -> FailureOr<uint64_t> {
    if (llvm::is_contained(dimensions, int64_t(0))) return uint64_t(0);
    uint64_t elements = 1;
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / elementByteWidth(input);
    for (int64_t extent : dimensions) {
      if (uint64_t(extent) > limit / elements) {
        emitOpError("reshape tensor byte count exceeds the signed 64-bit size range");
        return failure();
      }
      elements *= uint64_t(extent);
    }
    return elements;
  };
  auto outputCount = count(shape.asArrayRef());
  if (failed(outputCount)) return failure();
  if (input.hasStaticShape()) {
    auto inputCount = count(input.getShape());
    if (failed(inputCount)) return failure();
    if (*inputCount != *outputCount)
      return emitOpError("reshape preserves the number of elements");
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

bool isPureComputeOperation(Operation *op) {
  return isa<ConstantOp, SplatOp, CastOp, ReshapeOp, AddOp, MulOp,
             BroadcastInDimOp, TransposeOp, ReduceSumOp, ReduceMaxOp,
             CompareOp, SelectOp, DivOp, ExpOp, LogOp, SqrtOp, MatmulOp>(op);
}

namespace {
Type elementOf(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) return tensor.getElementType();
  return type;
}
SmallVector<int64_t> shapeOf(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return {tensor.getShape().begin(), tensor.getShape().end()};
  return {};
}
bool compatibleExtent(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}
Type withElement(Type type, Type element) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensor.getShape(), element);
  return element;
}
}

FailureOr<Type> inferComputeResultType(Operation *op) {
  auto invalid = [&](StringRef message) -> FailureOr<Type> {
    op->emitOpError(message);
    return failure();
  };
  if (failed(verifyDataTypes(op, op->getOperandTypes()))) return failure();
  const Type input = op->getOperand(0).getType();
  const Type element = elementOf(input);
  auto shape = shapeOf(input);
  if (isa<ReduceSumOp, ReduceMaxOp, CompareOp, DivOp, ExpOp, LogOp, SqrtOp, MatmulOp>(op)) {
    if (op->hasAttr("fastmath") || op->hasAttr("fastmathFlags"))
      return invalid("numerical operations use fixed ASK floating-point rules");
    if (Attribute overflow = op->getAttr("overflow")) {
      auto mode = dyn_cast<StringAttr>(overflow);
      if (!isa<ReduceSumOp>(op) || !isa<IntegerType>(element) || !mode || mode.getValue() != "wrap")
        return invalid("overflow = wrap applies to integer reduce_sum");
    }
  }
  if (isa<BroadcastInDimOp>(op)) {
    auto target = op->getAttrOfType<DenseI64ArrayAttr>("shape");
    auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
    if (!target || !dimensions || dimensions.size() != int64_t(shape.size()))
      return invalid("requires dense i64 shape and dimensions with one dimension per input axis");
    if (llvm::any_of(target.asArrayRef(), [](int64_t dim) { return dim < 0; }))
      return invalid("broadcast shape requires nonnegative concrete dimensions");
    int64_t previous = -1;
    for (auto [index, axis] : llvm::enumerate(dimensions.asArrayRef())) {
      if (axis <= previous || axis >= target.size())
        return invalid("broadcast dimensions must be strictly increasing output axes");
      if (shape[index] != 1 && !compatibleExtent(shape[index], target[axis]))
        return invalid("broadcast input extent must be one or equal its mapped output extent");
      previous = axis;
    }
    return RankedTensorType::get(target.asArrayRef(), element);
  }
  if (isa<TransposeOp>(op)) {
    auto permutation = op->getAttrOfType<DenseI64ArrayAttr>("permutation");
    if (!permutation || permutation.size() != int64_t(shape.size()))
      return invalid("requires a dense i64 permutation containing every input axis");
    llvm::SmallBitVector seen(shape.size());
    SmallVector<int64_t> output;
    for (int64_t axis : permutation.asArrayRef()) {
      if (axis < 0 || uint64_t(axis) >= shape.size() || seen.test(axis))
        return invalid("transpose permutation must contain each input axis exactly once");
      seen.set(axis);
      output.push_back(shape[axis]);
    }
    return isa<RankedTensorType>(input) ? Type(RankedTensorType::get(output, element)) : input;
  }
  if (isa<ReduceSumOp, ReduceMaxOp>(op)) {
    if (element.isInteger(1)) return invalid("reduction requires an explicit numeric dtype");
    auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!axes) return invalid("requires dense i64 sorted unique reduction axes");
    llvm::SmallBitVector reduced(shape.size());
    int64_t previous = -1;
    for (int64_t axis : axes.asArrayRef()) {
      if (axis <= previous || uint64_t(axis) >= shape.size())
        return invalid("reduction axes must be sorted, unique and within the input rank");
      if (isa<ReduceMaxOp>(op) && shape[axis] == 0)
        return invalid("reduce_max requires a nonempty reduction domain");
      reduced.set(axis);
      previous = axis;
    }
    SmallVector<int64_t> output;
    for (auto [index, extent] : llvm::enumerate(shape))
      if (!reduced.test(index)) output.push_back(extent);
    return output.empty() ? element : Type(RankedTensorType::get(output, element));
  }
  if (isa<CompareOp>(op)) {
    if (!haveCompatibleShapes(input, op->getOperand(1).getType()))
      return invalid("compare operands require matching shape and dtype");
    auto predicate = op->getAttrOfType<StringAttr>("predicate");
    if (!predicate || !llvm::is_contained(ArrayRef<StringRef>{"eq", "ne", "lt", "le", "gt", "ge"}, predicate.getValue()))
      return invalid("compare predicate must be eq, ne, lt, le, gt or ge");
    return withElement(input, IntegerType::get(op->getContext(), 1));
  }
  if (isa<SelectOp>(op)) {
    Type yes = op->getOperand(1).getType(), no = op->getOperand(2).getType();
    if (!element.isInteger(1) || !haveCompatibleShapeOnly(input, yes) ||
        !haveCompatibleShapes(yes, no))
      return invalid("select requires a bool condition and matching condition/branch shapes and branch dtypes");
    return yes;
  }
  if (isa<DivOp, ExpOp, LogOp, SqrtOp>(op)) {
    if (!isa<FloatType>(element)) return invalid("requires f32, f16 or bf16 operands");
    if (isa<DivOp>(op) && !haveCompatibleShapes(input, op->getOperand(1).getType()))
      return invalid("div operands require matching shape and dtype");
    return input;
  }
  if (isa<MatmulOp>(op)) {
    auto lhs = dyn_cast<RankedTensorType>(input);
    auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!lhs || !rhs || lhs.getRank() < 2 || lhs.getRank() != rhs.getRank() ||
        !isa<FloatType>(element) || element != rhs.getElementType())
      return invalid("matmul requires same-rank tensors of rank at least two with matching floating dtypes");
    bool transposeA = false, transposeB = false;
    for (auto [name, flag] : {std::pair<StringRef, bool *>{"transpose_a", &transposeA}, {"transpose_b", &transposeB}}) {
      if (auto attr = op->getAttr(name)) {
        auto boolean = dyn_cast<BoolAttr>(attr);
        if (!boolean) return invalid("matmul transpose_a and transpose_b must be bool attributes");
        *flag = boolean.getValue();
      }
    }
    const int64_t rank = lhs.getRank();
    SmallVector<int64_t> output;
    for (int64_t axis = 0; axis < rank - 2; ++axis) {
      if (!compatibleExtent(lhs.getDimSize(axis), rhs.getDimSize(axis)))
        return invalid("matmul batch dimensions must match");
      output.push_back(ShapedType::isDynamic(lhs.getDimSize(axis)) ? rhs.getDimSize(axis) : lhs.getDimSize(axis));
    }
    if (!llvm::is_contained(output, int64_t(0)) &&
        llvm::none_of(output, [](int64_t extent) { return ShapedType::isDynamic(extent); })) {
      uint64_t batchCount = 1;
      for (int64_t extent : output) {
        if (uint64_t(extent) > uint64_t(std::numeric_limits<int64_t>::max()) / batchCount)
          return invalid("matmul batch count exceeds the signed 64-bit size range");
        batchCount *= uint64_t(extent);
      }
    }
    const auto contractingA = lhs.getDimSize(rank - (transposeA ? 2 : 1));
    const auto contractingB = rhs.getDimSize(rank - (transposeB ? 1 : 2));
    if (!compatibleExtent(contractingA, contractingB))
      return invalid("matmul contracting dimensions must match");
    output.push_back(lhs.getDimSize(rank - (transposeA ? 1 : 2)));
    output.push_back(rhs.getDimSize(rank - (transposeB ? 2 : 1)));
    return RankedTensorType::get(output, element);
  }
  return invalid("operation has no compute shape rule");
}

static LogicalResult verifyInferredCompute(Operation *op) {
  if (failed(verifyDataTypes(op, op->getResultTypes()))) return failure();
  auto inferred = inferComputeResultType(op);
  if (failed(inferred)) return failure();
  if (!haveCompatibleShapes(*inferred, op->getResult(0).getType()))
    return op->emitOpError("declared result conflicts with inferred shape or dtype");
  return success();
}

#define ASK_VERIFY_COMPUTE(CLASS) \
LogicalResult CLASS::verify() { return verifyInferredCompute(*this); }
ASK_VERIFY_COMPUTE(BroadcastInDimOp)
ASK_VERIFY_COMPUTE(TransposeOp)
ASK_VERIFY_COMPUTE(ReduceSumOp)
ASK_VERIFY_COMPUTE(ReduceMaxOp)
ASK_VERIFY_COMPUTE(CompareOp)
ASK_VERIFY_COMPUTE(SelectOp)
ASK_VERIFY_COMPUTE(DivOp)
ASK_VERIFY_COMPUTE(ExpOp)
ASK_VERIFY_COMPUTE(LogOp)
ASK_VERIFY_COMPUTE(SqrtOp)
ASK_VERIFY_COMPUTE(MatmulOp)
#undef ASK_VERIFY_COMPUTE

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
