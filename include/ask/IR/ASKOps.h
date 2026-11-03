#ifndef ASK_IR_ASKOPS_H
#define ASK_IR_ASKOPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace ask {

// The first implementation uses MLIR's generic operation syntax. These are
// registered operations, not unregistered names accepted by a permissive parser.
#define ASK_OP_COMMON(NAME)                                                    \
  using Op::Op;                                                               \
  static llvm::StringRef getOperationName() { return NAME; }                    \
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }      \
  mlir::LogicalResult verify()

#define ASK_NO_MEMORY_EFFECTS                                                  \
  void getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &) {}

#define ASK_SINGLE_RESULT                                                      \
  mlir::Value getResult() { return getOperation()->getResult(0); }

class ConstantOp final
    : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.constant");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

class SplatOp final
    : public mlir::Op<SplatOp, mlir::OpTrait::OneOperand,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.splat");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

class CastOp final
    : public mlir::Op<CastOp, mlir::OpTrait::OneOperand,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.cast");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

class ReshapeOp final
    : public mlir::Op<ReshapeOp, mlir::OpTrait::OneOperand,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.reshape");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

class AddOp final
    : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.add");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

class MulOp final
    : public mlir::Op<MulOp, mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.mul");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

#define ASK_COMPUTE_OP(CLASS, NAME, OPERANDS)                                      \
class CLASS final                                                             \
    : public mlir::Op<CLASS, OPERANDS,              \
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,     \
                      mlir::OpTrait::ZeroSuccessors,                           \
                      mlir::MemoryEffectOpInterface::Trait> {                 \
public:                                                                       \
  ASK_OP_COMMON(NAME);                                                        \
  ASK_SINGLE_RESULT                                                           \
  ASK_NO_MEMORY_EFFECTS                                                       \
};

ASK_COMPUTE_OP(BroadcastInDimOp, "ask.broadcast_in_dim", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(TransposeOp, "ask.transpose", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(ReduceSumOp, "ask.reduce_sum", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(ReduceMaxOp, "ask.reduce_max", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(CompareOp, "ask.compare", mlir::OpTrait::NOperands<2>::Impl)
ASK_COMPUTE_OP(SelectOp, "ask.select", mlir::OpTrait::NOperands<3>::Impl)
ASK_COMPUTE_OP(DivOp, "ask.div", mlir::OpTrait::NOperands<2>::Impl)
ASK_COMPUTE_OP(ExpOp, "ask.exp", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(LogOp, "ask.log", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(SqrtOp, "ask.sqrt", mlir::OpTrait::OneOperand)
ASK_COMPUTE_OP(MatmulOp, "ask.matmul", mlir::OpTrait::NOperands<2>::Impl)
#undef ASK_COMPUTE_OP

class SemanticsOp final
    : public mlir::Op<SemanticsOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::ZeroResults, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::SymbolOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.semantics");
  mlir::FunctionType getFunctionType();
};

class ImplementationOp final
    : public mlir::Op<ImplementationOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::ZeroResults, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::SymbolOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.implementation");
};

class ExternalOp final
    : public mlir::Op<ExternalOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::VariadicResults, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors> {
public:
  ASK_OP_COMMON("ask.external");
  SemanticsOp resolveSemantics();
  ImplementationOp resolveImplementation();
};

class MarkOp final
    : public mlir::Op<MarkOp, mlir::OpTrait::OneOperand,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::OpTrait::ZeroSuccessors,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  ASK_OP_COMMON("ask.mark");
  ASK_SINGLE_RESULT
  ASK_NO_MEMORY_EFFECTS
};

// Accept an element type or ranked tensor type. Names use Python/API spelling;
// bool is represented by MLIR i1 but occupies one byte in buffers.
llvm::StringRef elementTypeName(mlir::Type type);
unsigned elementByteWidth(mlir::Type type);
bool isSupportedElementType(mlir::Type type);
bool isSupportedValueType(mlir::Type type);
bool haveCompatibleShapes(mlir::Type lhs, mlir::Type rhs);
bool haveCompatibleShapeOnly(mlir::Type lhs, mlir::Type rhs);
bool isPureComputeOperation(mlir::Operation *op);
mlir::FailureOr<mlir::Type> inferComputeResultType(mlir::Operation *op);

#undef ASK_OP_COMMON
#undef ASK_NO_MEMORY_EFFECTS
#undef ASK_SINGLE_RESULT
} // namespace ask

#endif
