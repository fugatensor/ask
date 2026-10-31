#ifndef ASK_IR_ASKDIALECT_H
#define ASK_IR_ASKDIALECT_H

#include "mlir/IR/Dialect.h"

namespace ask {
class ASKDialect final : public mlir::Dialect {
public:
  explicit ASKDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "ask"; }
};
} // namespace ask

#endif
