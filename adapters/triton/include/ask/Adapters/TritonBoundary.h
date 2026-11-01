#pragma once

#include "ask/Adapters/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace ask::adapters {
struct TritonBoundaries {
  llvm::SmallVector<llvm::CallInst *> inputs;
  llvm::SmallVector<llvm::CallInst *> outputs;
  unsigned valueArgument = 0;
  unsigned addressArgument = 1;
  unsigned indexArgument = 2;
  unsigned maskArgument = 3;
};

// Producer-specific evidence is checked here. The native LLVM inliner knows
// nothing about Triton versions, PTX identity strings, grid or warp sizes.
// module must be the verified, unmodified import of artifact.payload. Returned
// pointers remain valid only while that module and those instructions survive.
llvm::Expected<TritonBoundaries>
bindTritonBoundaries(llvm::Module &module, const llvm::json::Object &artifact);

// After binding succeeds, represent the producer's permitted multiply-add
// contraction on its own instructions. Call before inserting ASK helpers:
// their strict arithmetic must not inherit Triton's global codegen option.
llvm::Expected<llvm::json::Object>
normalizeTritonNumerics(llvm::Module &module, const llvm::json::Object &artifact);

// Referenced only by the generated enabled-adapter assembly registry.
const CompilerAdapter &tritonCompilerAdapter();
}
