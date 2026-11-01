#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>

namespace ask::adapters {

// A producer has proved that this identity call is a replaceable element value.
// The generic inliner validates types, dominance and helper purity. It does not
// decode the call's protocol or guess which argument carries a mask/value.
struct ValueBoundary {
  llvm::CallInst *marker = nullptr;
  llvm::Value *value = nullptr;
  llvm::Value *predicate = nullptr;
  llvm::Value *passthrough = nullptr;
};

struct FusionRequest {
  const llvm::json::Object &exportRequest;
  llvm::StringRef elementType;
  int64_t elementCount;
};

struct PreparedFusion {
  llvm::SmallVector<ValueBoundary> inputs;
  llvm::SmallVector<ValueBoundary> outputs;
  std::string protocol;
  llvm::json::Object producerNumerics;
  // Standard backend options after the producer's numerical policy has been
  // made local to its own instructions, before inserting strict ASK helpers.
  llvm::json::Object codegen;
};

struct CompilerAdapter {
  llvm::StringRef id;
  unsigned protocolVersion = 1;
  llvm::Error (*validateArtifact)(const llvm::json::Object &artifact,
                                 const llvm::json::Object &request) = nullptr;
  // Called on a fresh imported module. Any mutation belongs to that disposable
  // import. Pointers in PreparedFusion refer only to this module/revision.
  llvm::Expected<PreparedFusion> (*prepareFusion)(
      llvm::Module &module, const llvm::json::Object &artifact,
      const FusionRequest &request) = nullptr;
  // Move/invalidate private evidence after the generic writer has replaced the
  // payload and standard execution fields. Never attest the rewritten payload
  // with provenance that describes only the producer's original module.
  llvm::Error (*finalizeArtifact)(llvm::json::Object &artifact,
                                 const llvm::json::Object &fusionReport) = nullptr;
  // A selected adapter owns its installed worker name and interpreter policy.
  // Native worker binaries need no Python interpreter.
  llvm::StringRef defaultWorker = "worker";
  bool workerIsPython = false;
};

// These belong to the assembly library, not ASK::Core. Its generated manifest
// contains only the adapters selected at build time. Tests inject a descriptor
// directly into fuseExternal; test descriptors never enter this production set.
llvm::ArrayRef<const CompilerAdapter *> registeredCompilerAdapters();
const CompilerAdapter *findCompilerAdapter(llvm::StringRef id);

} // namespace ask::adapters
