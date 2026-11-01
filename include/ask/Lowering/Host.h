#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace ask {
// Pure feature policy for target inspection and tests; execution uses detected
// native features, never caller-supplied overrides.
bool hostSupportsBF16(llvm::StringRef architecture, const llvm::StringMap<bool> &features);
bool nativeHostSupportsBF16();

// Contiguous, disjoint tensors. Bool buffers/scalars use one byte (nonzero input
// is true, outputs are 0/1); LLVM computations use i1. Call prepareModule first.
// BF16 requires a supported native CPU instruction feature.
llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp module, llvm::LLVMContext &context,
                llvm::json::Object &abiReport);
llvm::Error emitHostObject(llvm::Module &module, llvm::StringRef path);
}
