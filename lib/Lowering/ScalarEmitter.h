#pragma once

#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Numeric.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/IRBuilder.h"

#include <functional>

namespace ask::lowering {

// Evaluate a graph value at a row-major element index. Shape changes only map
// indices; reductions build nested loops, so their producers and epilogues stay
// in the caller's output loop without intermediate tensor allocations.
class ScalarEmitter {
  llvm::IRBuilder<> &b;
  llvm::Function *function;
  bool cuda;
  std::function<llvm::Value *(mlir::Value, llvm::Value *)> load;
  llvm::DenseMap<std::pair<mlir::Value, llvm::Value *>, llvm::Value *> values;
  llvm::DenseMap<mlir::Value, llvm::Value *> uniformValues, buffers;

  static llvm::ArrayRef<int64_t> shape(mlir::Type type) {
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) return tensor.getShape();
    return {};
  }
  static uint64_t product(llvm::ArrayRef<int64_t> dimensions) {
    if (llvm::is_contained(dimensions, int64_t(0))) return 0;
    uint64_t count = 1;
    for (int64_t extent : dimensions) count *= uint64_t(extent);
    return count;
  }
  llvm::Value *coordinate(llvm::Value *index, llvm::ArrayRef<int64_t> dimensions, unsigned axis) {
    if (dimensions[axis] == 1) return b.getInt64(0);
    const uint64_t stride = product(dimensions.drop_front(axis + 1));
    return b.CreateURem(b.CreateUDiv(index, b.getInt64(stride)), b.getInt64(dimensions[axis]));
  }
  llvm::Value *atStride(llvm::Value *coordinate, llvm::ArrayRef<int64_t> dimensions, unsigned axis) {
    return b.CreateMul(coordinate, b.getInt64(product(dimensions.drop_front(axis + 1))));
  }
  llvm::Value *binary(mlir::Operation *op, llvm::Value *left, llvm::Value *right) {
    if (left->getType()->isIntegerTy())
      return mlir::isa<AddOp, ReduceSumOp>(op) ? b.CreateAdd(left, right, "add.wrap")
                                            : b.CreateMul(left, right, "mul.wrap");
    const bool bf16 = left->getType()->isBFloatTy();
    if (bf16) {
      left = llir::extendBF16(b, left);
      right = llir::extendBF16(b, right);
    }
    llvm::Value *computed;
    if (mlir::isa<AddOp, ReduceSumOp>(op)) computed = b.CreateFAdd(left, right, "add");
    else if (mlir::isa<DivOp>(op)) computed = b.CreateFDiv(left, right, "div");
    else computed = b.CreateFMul(left, right, "mul");
    return bf16 ? llir::truncateBF16(b, computed) : computed;
  }
  llvm::Value *maximum(llvm::Value *left, llvm::Value *right) {
    if (left->getType()->isIntegerTy())
      return b.CreateSelect(b.CreateICmpSGT(left, right), left, right, "max");
    auto *type = left->getType();
    left = llir::createNumericCast(b, left, b.getFloatTy());
    right = llir::createNumericCast(b, right, b.getFloatTy());
    auto *computed = b.CreateIntrinsic(llvm::Intrinsic::maximum, {b.getFloatTy()}, {left, right});
    return llir::createNumericCast(b, computed, type);
  }
  llvm::Value *compare(mlir::Operation *op, llvm::Value *left, llvm::Value *right) {
    auto predicate = op->getAttrOfType<mlir::StringAttr>("predicate").getValue();
    if (left->getType()->isFloatingPointTy()) {
      left = llir::createNumericCast(b, left, b.getFloatTy());
      right = llir::createNumericCast(b, right, b.getFloatTy());
      auto kind = predicate == "eq" ? llvm::CmpInst::FCMP_OEQ :
                  predicate == "ne" ? llvm::CmpInst::FCMP_UNE :
                  predicate == "lt" ? llvm::CmpInst::FCMP_OLT :
                  predicate == "le" ? llvm::CmpInst::FCMP_OLE :
                  predicate == "gt" ? llvm::CmpInst::FCMP_OGT : llvm::CmpInst::FCMP_OGE;
      return b.CreateFCmp(kind, left, right, "compare");
    }
    const bool boolean = left->getType()->isIntegerTy(1);
    auto kind = predicate == "eq" ? llvm::CmpInst::ICMP_EQ :
                predicate == "ne" ? llvm::CmpInst::ICMP_NE :
                predicate == "lt" ? (boolean ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT) :
                predicate == "le" ? (boolean ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE) :
                predicate == "gt" ? (boolean ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT) :
                                     (boolean ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE);
    return b.CreateICmp(kind, left, right, "compare");
  }
  llvm::Value *unary(mlir::Operation *op, llvm::Value *input) {
    auto *type = input->getType();
    input = llir::createNumericCast(b, input, b.getFloatTy());
    llvm::Value *computed;
    if (mlir::isa<SqrtOp>(op))
      computed = b.CreateIntrinsic(llvm::Intrinsic::sqrt, {b.getFloatTy()}, {input});
    else if (cuda) {
      const char *name = mlir::isa<ExpOp>(op) ? "__nv_expf" : "__nv_logf";
      auto callee = function->getParent()->getOrInsertFunction(name,
          llvm::FunctionType::get(b.getFloatTy(), {b.getFloatTy()}, false));
      computed = b.CreateCall(callee, {input}, "math");
    } else {
      auto intrinsic = mlir::isa<ExpOp>(op) ? llvm::Intrinsic::exp : llvm::Intrinsic::log;
      computed = b.CreateIntrinsic(intrinsic, {b.getFloatTy()}, {input});
    }
    return llir::createNumericCast(b, computed, type);
  }
  template<class Update>
  llvm::Value *accumulate(uint64_t count, llvm::Value *initial, Update update) {
    if (count == 0) return initial;
    if (count == 1) return update(b.getInt64(0), initial);
    ++reductionLoops;
    auto *before = b.GetInsertBlock();
    auto *body = llvm::BasicBlock::Create(b.getContext(), "reduce.loop", function);
    auto *after = llvm::BasicBlock::Create(b.getContext(), "reduce.done", function);
    b.CreateBr(body);
    b.SetInsertPoint(body);
    auto *index = b.CreatePHI(b.getInt64Ty(), 2, "reduce.index");
    auto *accumulator = b.CreatePHI(initial->getType(), 2, "reduce.accumulator");
    index->addIncoming(b.getInt64(0), before);
    accumulator->addIncoming(initial, before);
    auto *result = update(index, accumulator);
    auto *next = b.CreateAdd(index, b.getInt64(1), "reduce.next");
    auto *latch = b.GetInsertBlock();
    index->addIncoming(next, latch);
    accumulator->addIncoming(result, latch);
    b.CreateCondBr(b.CreateICmpULT(next, b.getInt64(count)), body, after);
    b.SetInsertPoint(after);
    return result;
  }
  llvm::Value *reduce(mlir::Operation *op, llvm::Value *index) {
    auto input = op->getOperand(0);
    auto inputShape = shape(input.getType());
    auto axes = op->getAttrOfType<mlir::DenseI64ArrayAttr>("axes").asArrayRef();
    if (axes.empty()) return emit(input, index);
    auto *type = llir::scalarLLVMType(input.getType(), b.getContext());
    llvm::Value *initial;
    if (mlir::isa<ReduceSumOp>(op)) initial = llvm::Constant::getNullValue(type);
    else if (type->isIntegerTy())
      initial = llvm::ConstantInt::get(b.getContext(), llvm::APInt::getSignedMinValue(type->getIntegerBitWidth()));
    else initial = llvm::ConstantFP::getInfinity(type, true);
    llvm::SmallBitVector reduced(inputShape.size());
    llvm::SmallVector<int64_t> reducedShape;
    for (int64_t axis : axes) {
      reduced.set(axis);
      reducedShape.push_back(inputShape[axis]);
    }
    uint64_t count = product(reducedShape);
    if (!count) return initial;
    auto outputShape = shape(op->getResult(0).getType());
    llvm::Value *base = b.getInt64(0);
    unsigned outputAxis = 0;
    for (unsigned axis = 0; axis < inputShape.size(); ++axis)
      if (!reduced.test(axis))
        base = b.CreateAdd(base, atStride(coordinate(index, outputShape, outputAxis++), inputShape, axis));
    return accumulate(count, initial, [&](llvm::Value *reductionIndex, llvm::Value *accumulator) {
      llvm::Value *inputIndex = base;
      for (auto [reducedAxis, axis] : llvm::enumerate(axes))
        inputIndex = b.CreateAdd(inputIndex,
            atStride(coordinate(reductionIndex, reducedShape, reducedAxis), inputShape, axis));
      auto *value = emit(input, inputIndex);
      return mlir::isa<ReduceSumOp>(op) ? binary(op, accumulator, value) : maximum(accumulator, value);
    });
  }
  llvm::Value *matmul(mlir::Operation *op, llvm::Value *index) {
    auto a = op->getOperand(0), c = op->getOperand(1);
    auto aShape = shape(a.getType()), cShape = shape(c.getType());
    auto outputShape = shape(op->getResult(0).getType());
    const unsigned rank = outputShape.size();
    auto transposed = [&](const char *name) {
      auto attribute = op->getAttrOfType<mlir::BoolAttr>(name);
      return attribute && attribute.getValue();
    };
    bool transposeA = transposed("transpose_a"), transposeB = transposed("transpose_b");
    uint64_t rows = outputShape[rank - 2], columns = outputShape[rank - 1];
    uint64_t k = aShape[rank - (transposeA ? 2 : 1)];
    auto *batch = b.CreateUDiv(index, b.getInt64(rows * columns));
    auto *row = coordinate(index, outputShape, rank - 2);
    auto *column = coordinate(index, outputShape, rank - 1);
    auto *aBase = b.CreateMul(batch, b.getInt64(aShape[rank - 2] * aShape[rank - 1]));
    auto *cBase = b.CreateMul(batch, b.getInt64(cShape[rank - 2] * cShape[rank - 1]));
    auto *result = accumulate(k, llvm::ConstantFP::get(b.getFloatTy(), 0.0),
        [&](llvm::Value *inner, llvm::Value *accumulator) {
      auto *aIndex = b.CreateAdd(aBase, b.CreateAdd(
          b.CreateMul(transposeA ? inner : row, b.getInt64(aShape[rank - 1])), transposeA ? row : inner));
      auto *cIndex = b.CreateAdd(cBase, b.CreateAdd(
          b.CreateMul(transposeB ? column : inner, b.getInt64(cShape[rank - 1])), transposeB ? inner : column));
      auto *left = llir::createNumericCast(b, emit(a, aIndex), b.getFloatTy());
      auto *right = llir::createNumericCast(b, emit(c, cIndex), b.getFloatTy());
      return b.CreateFAdd(accumulator, b.CreateFMul(left, right), "matmul.accumulate");
    });
    return llir::createNumericCast(b, result, llir::scalarLLVMType(op->getResult(0).getType(), b.getContext()));
  }

public:
  uint64_t reductionLoops = 0;
  // Reads emitted into this phase, after uniform and per-index memo hits.
  llvm::SmallVector<mlir::Value> bufferReads;
  ScalarEmitter(llvm::IRBuilder<> &builder, llvm::Function *function, bool cuda,
                std::function<llvm::Value *(mlir::Value, llvm::Value *)> loader)
      : b(builder), function(function), cuda(cuda), load(std::move(loader)) {}

  void bindUniform(mlir::Value value, llvm::Value *computed) { uniformValues[value] = computed; }
  void bindBuffer(mlir::Value value, llvm::Value *address) { buffers[value] = address; }

  llvm::Value *emit(mlir::Value value, llvm::Value *index) {
    if (auto uniform = uniformValues.find(value); uniform != uniformValues.end()) return uniform->second;
    if (!mlir::isa<mlir::RankedTensorType>(value.getType())) index = b.getInt64(0);
    auto key = std::make_pair(value, index);
    auto cached = values.find(key);
    if (cached != values.end()) return cached->second;
    llvm::Value *computed;
    if (auto stored = buffers.find(value); stored != buffers.end()) {
      bufferReads.push_back(value);
      auto *type = llir::scalarLLVMType(value.getType(), b.getContext());
      const bool boolean = type->isIntegerTy(1);
      if (boolean) type = b.getInt8Ty();
      auto *address = b.CreateGEP(type, stored->second, index, "cached.address");
      computed = b.CreateAlignedLoad(type, address,
          llvm::Align(elementByteWidth(value.getType())), "cached.value");
      if (boolean) computed = b.CreateICmpNE(computed, b.getInt8(0));
    } else if (mlir::isa<mlir::BlockArgument>(value)) computed = load(value, index);
    else {
      auto *op = value.getDefiningOp();
      if (auto constant = mlir::dyn_cast<ConstantOp>(op)) {
        if (auto integer = constant->getAttrOfType<mlir::IntegerAttr>("value"))
          computed = llvm::ConstantInt::get(b.getContext(), integer.getValue());
        else computed = llvm::ConstantFP::get(b.getContext(), constant->getAttrOfType<mlir::FloatAttr>("value").getValue());
      } else if (mlir::isa<SplatOp, ReshapeOp>(op)) computed = emit(op->getOperand(0), index);
      else if (mlir::isa<CastOp>(op))
        computed = llir::createNumericCast(b, emit(op->getOperand(0), index),
            llir::scalarLLVMType(value.getType(), b.getContext()));
      else if (mlir::isa<BroadcastInDimOp, TransposeOp>(op)) {
        auto input = op->getOperand(0);
        auto inputShape = shape(input.getType()), outputShape = shape(value.getType());
        llvm::Value *mapped = b.getInt64(0);
        if (mlir::isa<BroadcastInDimOp>(op)) {
          auto dimensions = op->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions").asArrayRef();
          for (unsigned axis = 0; axis < inputShape.size(); ++axis)
            if (inputShape[axis] != 1)
              mapped = b.CreateAdd(mapped, atStride(coordinate(index, outputShape, dimensions[axis]), inputShape, axis));
        } else {
          auto permutation = op->getAttrOfType<mlir::DenseI64ArrayAttr>("permutation").asArrayRef();
          for (auto [axis, inputAxis] : llvm::enumerate(permutation))
            mapped = b.CreateAdd(mapped, atStride(coordinate(index, outputShape, axis), inputShape, inputAxis));
        }
        computed = emit(input, mapped);
      } else if (mlir::isa<ReduceSumOp, ReduceMaxOp>(op)) computed = reduce(op, index);
      else if (mlir::isa<MatmulOp>(op)) computed = matmul(op, index);
      else if (mlir::isa<CompareOp>(op)) computed = compare(op, emit(op->getOperand(0), index), emit(op->getOperand(1), index));
      else if (mlir::isa<SelectOp>(op))
        computed = b.CreateSelect(emit(op->getOperand(0), index), emit(op->getOperand(1), index), emit(op->getOperand(2), index), "select");
      else if (mlir::isa<ExpOp, LogOp, SqrtOp>(op)) computed = unary(op, emit(op->getOperand(0), index));
      else computed = binary(op, emit(op->getOperand(0), index), emit(op->getOperand(1), index));
    }
    values[key] = computed;
    return computed;
  }
};
} // namespace ask::lowering
