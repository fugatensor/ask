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

bool isSupportedValueType(mlir::Type type);
bool haveCompatibleShapes(mlir::Type lhs, mlir::Type rhs);

#undef ASK_OP_COMMON
#undef ASK_NO_MEMORY_EFFECTS
#undef ASK_SINGLE_RESULT
} // namespace ask

#endif
