#ifndef ASK_LLIR_OPTIMIZE_H
#define ASK_LLIR_OPTIMIZE_H

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace ask::llir {

/// Verify, then run InstCombine, SimplifyCFG and ADCE on function definitions.
/// This is a target-independent function pipeline, not an optimization level;
/// it neither inlines functions nor changes the module's target or ABI.
/// Name-based libc inference is disabled: third-party functions do not acquire
/// strlen/memcpy/etc. semantics solely from their symbol names. Explicit LLVM
/// intrinsics and attributes still supply their normal LLVM semantics.
/// Functions carrying optnone are skipped. Existing LLVM effects and numerical
/// flags govern transformations; verification alone does not prove equivalence.
///
/// Module-level ask.abi and ask.shape_snapshot remain semantic records. All
/// ask.* instruction attachments and known module/function/global
/// trace/BoundaryMap records are removed because their SSA locations may become
/// stale. Other ask.* named metadata is retained but reported as unvalidated,
/// never as an exact map.
/// LLVM/target metadata is left to LLVM's own preservation rules.
///
/// This mutates module. A caller needing transactions must pass a clone and
/// commit it only on success; discard it on failure. The report is updated only
/// on success, under "llir_optimization". Caller-owned trace/BoundaryMap data
/// outside the module must also be invalidated after a successful call.
/// Pass status is "applied" only when that pass changes IR; metadata cleanup
/// is reported separately by "metadata_changed".
llvm::Error optimizeLLVMIR(llvm::Module &module, llvm::json::Object &report);

} // namespace ask::llir

#endif
