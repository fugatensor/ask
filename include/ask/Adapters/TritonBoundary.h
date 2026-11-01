#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace ask::adapters {
struct TritonBoundaries {
  llvm::SmallVector<llvm::CallInst *> inputs;
  llvm::SmallVector<llvm::CallInst *> outputs;
};

// Producer-specific evidence is checked here. The native LLVM inliner knows
// nothing about Triton versions, PTX identity strings, grid or warp sizes.
// module must be the verified, unmodified import of artifact.payload. Returned
// pointers remain valid only while that module and those instructions survive.
llvm::Expected<TritonBoundaries>
bindTritonBoundaries(llvm::Module &module, const llvm::json::Object &artifact);
}
