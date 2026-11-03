#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"

namespace ask {
ASKDialect::ASKDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, mlir::TypeID::get<ASKDialect>()) {
  addOperations<ConstantOp, SplatOp, CastOp, ReshapeOp, AddOp, MulOp, SemanticsOp,
                ImplementationOp, ExternalOp, MarkOp, BroadcastInDimOp, TransposeOp,
                ReduceSumOp, ReduceMaxOp, CompareOp, SelectOp, DivOp,
                ExpOp, LogOp, SqrtOp, MatmulOp>();
}
} // namespace ask
