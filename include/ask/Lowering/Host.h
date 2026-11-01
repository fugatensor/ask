#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace ask {
// Contiguous, disjoint i32/f32 tensor ABI. Call prepareModule before lowering.
llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp module, llvm::LLVMContext &context,
                llvm::json::Object &abiReport);
llvm::Error emitHostObject(llvm::Module &module, llvm::StringRef path);
}
