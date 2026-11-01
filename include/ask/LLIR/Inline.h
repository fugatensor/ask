#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace llvm {
class CallInst;
class Function;
class Value;
}

namespace ask::llir {

/// A value boundary already located and justified by the caller's analysis.
/// marker is an identity call in consumer; helper is defined in helpers.
/// arguments, predicate and passthrough refer to values in consumer. Constants
/// are also accepted. Their types must exactly match the helper/call result.
///
/// The adapter/analysis must establish that replacing this identity value is a
/// legal graph transformation, including logical element correspondence, memory
/// effects, aliases, ordering and launch policy. This API does not decode asm or
/// infer a kernel's semantics from the marker's name. Identifying a CallInst is
/// not itself such a proof.
struct BoundaryInlineSite {
  llvm::CallInst *marker = nullptr;
  llvm::Function *helper = nullptr;
  llvm::SmallVector<llvm::Value *> arguments;
  llvm::Value *predicate = nullptr;
  llvm::Value *passthrough = nullptr;
};

/// Link pure LLVM value helpers and inline them at explicit identity boundaries.
/// Consumer may contain arbitrary target-specific instructions and control flow;
/// helper types support scalar integers/floats and fixed-width vectors. Helpers
/// are inspected rather than trusted on attributes: no memory, pointer values,
/// side effects, unknown calls, integer division, cyclic control flow or recursion.
/// A scalar predicate guards helper evaluation and keeps passthrough when false.
/// A vector predicate selects per lane; it requires straight-line helpers whose
/// speculative evaluation cannot introduce a trap or poison-dependent branch.
///
/// Both modules must share a context. Nonempty helper target/layout descriptions
/// must match consumer. A dedicated helper module must contain only pure helper
/// functions and permitted intrinsic declarations, not globals/aliases/ifuncs or
/// module flags. Function-level target/numerical string attributes must be
/// normalized by the caller, as must helper/callsite parameter and return
/// attributes (such as noundef/range). Instruction-level numerical flags are
/// preserved. Recognized intrinsic declarations retain their defined contracts.
/// Floating helpers cannot inherit consumer strictfp, trapping, flush-to-zero,
/// aggressive contraction or relaxed NaN/Inf/zero assumptions. This includes
/// integer-signature helpers with floating intermediates or nested callees.
/// Neutral attributes and unrelated target attributes remain intact. Backend
/// options are outside the module; the caller must also preserve per-operation
/// rounding when invoking code generation after inlining.
///
/// Returns a new verified module. Inputs and report are unchanged on failure;
/// inputs are also unchanged on success. All site pointers describe the inputs,
/// not the returned clone. Caller-owned low-level analyses must be invalidated
/// after success; this API does not claim to preserve instruction mappings.
llvm::Expected<std::unique_ptr<llvm::Module>>
inlineValueBoundaries(const llvm::Module &consumer, const llvm::Module &helpers,
                      llvm::ArrayRef<BoundaryInlineSite> sites,
                      llvm::json::Object &report);

} // namespace ask::llir
