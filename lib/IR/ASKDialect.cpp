#include "ask/IR/ASKDialect.h"
#include "ask/IR/ASKOps.h"

namespace ask {
ASKDialect::ASKDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, mlir::TypeID::get<ASKDialect>()) {
  addOperations<ConstantOp, SplatOp, CastOp, ReshapeOp, AddOp, MulOp, SemanticsOp,
                ImplementationOp, ExternalOp, MarkOp, BroadcastInDimOp, TransposeOp,
                ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp, ReduceAnyOp,
                ReduceAllOp, ArgmaxOp, ArgminOp, CompareOp, SelectOp, DivOp,
                FloordivOp, ModOp, PowOp, MaximumOp, MinimumOp, ExpOp, LogOp,
                SqrtOp, SinOp, CosOp, TanhOp, ErfOp, AbsOp, FloorOp, CeilOp,
                RoundOp, BitwiseNotOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp,
                ShiftLeftOp, ShiftRightOp, IotaOp, SliceOp, PadOp, ConcatenateOp,
                GatherOp, ScatterAddOp, MatmulOp>();
}
} // namespace ask
