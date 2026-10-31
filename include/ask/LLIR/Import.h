#ifndef ASK_LLIR_IMPORT_H
#define ASK_LLIR_IMPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
class LLVMContext;
}

namespace ask::llir {

/// Parse and verify native LLVM assembly without repairing or lowering it.
/// The caller must keep context alive as long as the returned module.
llvm::Expected<std::unique_ptr<llvm::Module>>
importLLVMIR(llvm::StringRef text, llvm::LLVMContext &context,
             llvm::StringRef sourceName = "<ask-llir>");

} // namespace ask::llir

#endif
