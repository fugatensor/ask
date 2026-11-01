#ifndef ASK_TRANSFORMS_OPTIMIZE_H
#define ASK_TRANSFORMS_OPTIMIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"

namespace ask {

struct OptimizationOptions {
  bool foldConstants = true;
  bool eliminateDeadCode = true;
  bool fuseElementwise = true;
};

// Prepare and optimize the supported ASK graph subset in place. The caller must
// clone the module if it needs rollback on failure; report is published only on
// success. External operations and their declarations are never folded/deleted.
// Fusion records a plan in ask.fusion_group; a backend must explicitly consume
// and validate that plan. Disabling fusion removes any existing group attributes.
mlir::LogicalResult optimizeModule(
    mlir::ModuleOp module, llvm::json::Object &report,
    const OptimizationOptions &options = {});

} // namespace ask

#endif
