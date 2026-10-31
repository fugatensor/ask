#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"

namespace ask {
ASKDialect::ASKDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, mlir::TypeID::get<ASKDialect>()) {
  addOperations<ConstantOp, SplatOp, AddOp, MulOp, SemanticsOp,
                ImplementationOp, ExternalOp, MarkOp>();
}
} // namespace ask
