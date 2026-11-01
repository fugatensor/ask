#pragma once

#include "ask/Adapters/Compiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace ask::llir {
// ASK graph -> typed helpers -> producer-independent native LLVM inlining.
// The assembly layer selects and injects an enabled compiler adapter. Core has
// no dependency on the production registry or any concrete adapter library.
// The adapter's boundary contract is a trusted compiler artifact, not a proof
// inferred from unknown inline assembly. Unsupported graphs fail closed.
// graph must have passed prepareModule (including marker extraction).
llvm::Expected<std::unique_ptr<llvm::Module>>
fuseExternal(mlir::ModuleOp graph, const llvm::json::Object &artifact,
             const adapters::CompilerAdapter &adapter,
             llvm::LLVMContext &context, llvm::json::Object &report,
             std::string &helperIR);
}
