#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"

namespace ask::llir {
// Element computations use i1 for bool; public buffers are widened separately.
inline llvm::Type *scalarLLVMType(mlir::Type type, llvm::LLVMContext &context) {
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return llvm::IntegerType::get(context, integer.getWidth());
  if (type.isF16()) return llvm::Type::getHalfTy(context);
  if (type.isBF16()) return llvm::Type::getBFloatTy(context);
  assert(type.isF32() && "verified ASK element type required");
  return llvm::Type::getFloatTy(context);
}

// BF16 has the same exponent as f32. Integer rounding preserves subnormals,
// unlike x86 VCVTNEPS2BF16, which flushes them regardless of the FP environment.
inline llvm::Value *extendBF16(llvm::IRBuilder<> &b, llvm::Value *value) {
  auto *bits = b.CreateZExt(b.CreateBitCast(value, b.getInt16Ty()), b.getInt32Ty());
  return b.CreateBitCast(b.CreateShl(bits, b.getInt32(16)), b.getFloatTy());
}
inline llvm::Value *truncateBF16(llvm::IRBuilder<> &b, llvm::Value *value) {
  auto *bits = b.CreateBitCast(value, b.getInt32Ty());
  auto *lsb = b.CreateAnd(b.CreateLShr(bits, b.getInt32(16)), b.getInt32(1));
  auto *rounded = b.CreateAdd(bits, b.CreateAdd(b.getInt32(0x7fff), lsb));
  auto *ordinary = b.CreateLShr(rounded, b.getInt32(16));
  auto *nan = b.CreateICmpUGT(b.CreateAnd(bits, b.getInt32(0x7fffffff)), b.getInt32(0x7f800000));
  auto *quiet = b.CreateOr(b.CreateLShr(bits, b.getInt32(16)), b.getInt32(0x40));
  return b.CreateBitCast(b.CreateTrunc(b.CreateSelect(nan, quiet, ordinary), b.getInt16Ty()),
                         b.getBFloatTy());
}

// Round integer magnitude directly to the destination precision. Going through
// f32 can round twice (e.g. 2^40 + 2^32 + 1 via f32 differs from direct bf16).
inline llvm::Value *integerToLowFloat(llvm::IRBuilder<> &b, llvm::Value *value,
                                      llvm::Type *destination) {
  auto *integer = b.CreateSExtOrTrunc(value, b.getInt64Ty());
  auto *negative = b.CreateICmpSLT(integer, b.getInt64(0));
  auto *magnitude = b.CreateSelect(negative, b.CreateNeg(integer), integer);
  auto *leading = b.CreateIntrinsic(llvm::Intrinsic::ctlz, {b.getInt64Ty()},
                                    {magnitude, b.getFalse()});
  auto *exponent = b.CreateSub(b.getInt64(63), leading);
  unsigned fractionBits = destination->isHalfTy() ? 10 : 7;
  unsigned bias = destination->isHalfTy() ? 15 : 127;
  // For zero, exponent wraps. Clamp every shift before it becomes an operand;
  // the final select alone would not protect against poison from overshifting.
  auto *nonzero = b.CreateICmpNE(magnitude, b.getInt64(0));
  exponent = b.CreateSelect(nonzero, exponent, b.getInt64(0));
  auto *right = b.CreateSelect(b.CreateICmpUGT(exponent, b.getInt64(fractionBits)),
                               b.CreateSub(exponent, b.getInt64(fractionBits)), b.getInt64(0));
  auto *left = b.CreateSelect(b.CreateICmpULT(exponent, b.getInt64(fractionBits)),
                              b.CreateSub(b.getInt64(fractionBits), exponent), b.getInt64(0));
  auto *retained = b.CreateLShr(magnitude, right);
  auto *unit = b.CreateShl(b.getInt64(1), right);
  auto *remainder = b.CreateAnd(magnitude, b.CreateSub(unit, b.getInt64(1)));
  auto *half = b.CreateLShr(unit, b.getInt64(1));
  auto *odd = b.CreateICmpNE(b.CreateAnd(retained, b.getInt64(1)), b.getInt64(0));
  auto *roundUp = b.CreateAnd(b.CreateICmpNE(right, b.getInt64(0)),
      b.CreateOr(b.CreateICmpUGT(remainder, half), b.CreateAnd(b.CreateICmpEQ(remainder, half), odd)));
  auto *rounded = b.CreateAdd(retained, b.CreateZExt(roundUp, b.getInt64Ty()));
  rounded = b.CreateShl(rounded, left);
  auto *carry = b.CreateICmpEQ(rounded, b.getInt64(uint64_t(1) << (fractionBits + 1)));
  auto *significand = b.CreateSelect(carry, b.CreateLShr(rounded, b.getInt64(1)), rounded);
  auto *biasedExponent = b.CreateAdd(b.CreateAdd(exponent, b.getInt64(bias)),
                                      b.CreateZExt(carry, b.getInt64Ty()));
  auto *payload = b.CreateOr(b.CreateShl(biasedExponent, b.getInt64(fractionBits)),
      b.CreateAnd(significand, b.getInt64((uint64_t(1) << fractionBits) - 1)));
  if (destination->isHalfTy())
    payload = b.CreateSelect(b.CreateICmpUGE(biasedExponent, b.getInt64(31)), b.getInt64(0x7c00), payload);
  auto *sign = b.CreateSelect(negative, b.getInt64(0x8000), b.getInt64(0));
  auto *bits = b.CreateSelect(nonzero, b.CreateOr(sign, payload), b.getInt64(0));
  return b.CreateBitCast(b.CreateTrunc(bits, b.getInt16Ty()), destination);
}

inline llvm::Value *createNumericCast(llvm::IRBuilder<> &b, llvm::Value *value,
                                      llvm::Type *destination) {
  auto *source = value->getType();
  if (source == destination) return value;
  if (destination->isIntegerTy(1))
    return source->isFloatingPointTy()
        ? b.CreateFCmpUNE(value, llvm::ConstantFP::get(source, 0.0), "cast.bool")
        : b.CreateICmpNE(value, llvm::ConstantInt::get(source, 0), "cast.bool");
  if (source->isIntegerTy(1))
    return destination->isIntegerTy() ? b.CreateZExt(value, destination, "cast.bool.integer")
                                      : b.CreateUIToFP(value, destination, "cast.bool.float");
  if (source->isIntegerTy() && destination->isIntegerTy())
    return b.CreateSExtOrTrunc(value, destination, "cast.integer");
  if (source->isIntegerTy()) {
    if (destination->isHalfTy() || destination->isBFloatTy())
      return integerToLowFloat(b, value, destination);
    return b.CreateSIToFP(value, destination, "cast.float");
  }
  if (source->isBFloatTy()) {
    value = extendBF16(b, value);
    source = value->getType();
  }
  if (destination->isIntegerTy())
    return b.CreateIntrinsic(llvm::Intrinsic::fptosi_sat, {destination, source}, {value},
                              nullptr, "cast.integer.saturating");
  if (destination->isBFloatTy()) {
    if (source->isHalfTy()) value = b.CreateFPExt(value, b.getFloatTy());
    return truncateBF16(b, value);
  }
  if (source == destination) return value;
  return source->getScalarSizeInBits() < destination->getScalarSizeInBits()
      ? b.CreateFPExt(value, destination, "cast.extend")
      : b.CreateFPTrunc(value, destination, "cast.round");
}
} // namespace ask::llir
