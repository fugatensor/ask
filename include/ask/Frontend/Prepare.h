#ifndef ASK_FRONTEND_PREPARE_H
#define ASK_FRONTEND_PREPARE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"

namespace ask {
/// Validate the prototype's graph subset, infer concrete shapes, assign/check
/// logical identities, and collect/optionally erase identity markers.
/// This mutates module. A caller needing transactions must pass a clone and
/// commit it only on success. No DSL compilation or exporter loading occurs.
/// On success report contains "shape_snapshot" and "markers".
mlir::LogicalResult prepareModule(mlir::ModuleOp module,
                                  llvm::json::Object &report,
                                  bool eraseMarkers = true);
} // namespace ask

#endif
