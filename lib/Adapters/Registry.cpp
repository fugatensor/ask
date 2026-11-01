#include "ask/Adapters/Compiler.h"

namespace ask::adapters {
#define ASK_COMPILER_ADAPTER(ID, FACTORY, FUSION) const CompilerAdapter &FACTORY();
#include "ask/Adapters/Manifest.inc"
#undef ASK_COMPILER_ADAPTER

llvm::ArrayRef<const CompilerAdapter *> registeredCompilerAdapters() {
  static const CompilerAdapter *const descriptors[] = {
      nullptr,
#define ASK_COMPILER_ADAPTER(ID, FACTORY, FUSION) &FACTORY(),
#include "ask/Adapters/Manifest.inc"
#undef ASK_COMPILER_ADAPTER
  };
  return llvm::ArrayRef<const CompilerAdapter *>(descriptors).drop_front();
}

const CompilerAdapter *findCompilerAdapter(llvm::StringRef id) {
  for (auto *adapter : registeredCompilerAdapters())
    if (adapter->id == id) return adapter;
  return nullptr;
}
} // namespace ask::adapters
