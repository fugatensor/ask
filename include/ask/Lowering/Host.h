#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace ask {
// Check the target dtype boundary before graph optimization or LLVM lowering.
llvm::Error validateHostElementTypes(mlir::ModuleOp module);

// Contiguous, disjoint tensors. Bool buffers/scalars use one byte (nonzero input
// is true, outputs are 0/1); LLVM computations use i1. Call prepareModule first.
// Host values use bool, i32, i64, f32 or f16.
llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp module, llvm::LLVMContext &context,
                llvm::json::Object &abiReport);
llvm::Error emitHostObject(llvm::Module &module, llvm::StringRef path);
}
