#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace ask {
struct CUDALoweringOptions {
  unsigned architecture = 0; // Explicit target; never inferred from a producer.
  unsigned blockSize = 128;
  unsigned ptxVersion = 80;
};

// Lower one prepared static indexed graph, including reductions.
// All public arguments are pointers:
// inputs first, then disjoint output buffers. Scalars use shape [] and one
// element; bool storage uses one byte. No external compiler is involved.
// The returned artifact contains the normalized CUDA invocation (version 2).
// Callers attach the final LLVM payload after adding any module metadata.
llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToCUDALLVM(mlir::ModuleOp source, llvm::LLVMContext &context,
                const CUDALoweringOptions &options,
                llvm::json::Object &artifact, llvm::json::Object &report);
} // namespace ask
