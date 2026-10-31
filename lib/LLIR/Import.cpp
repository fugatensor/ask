#include "ask/LLIR/Import.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace ask::llir {

llvm::Expected<std::unique_ptr<llvm::Module>>
importLLVMIR(llvm::StringRef text, llvm::LLVMContext &context,
             llvm::StringRef sourceName) {
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(text, diagnostic, context);
  if (!module) {
    std::string details;
    llvm::raw_string_ostream stream(details);
    diagnostic.print(sourceName.str().c_str(), stream);
    stream.flush();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "LLVM IR parse failed with reader " LLVM_VERSION_STRING
        "; incompatible producer IR is one possible cause. No IR was "
        "rewritten.\n%s",
        details.c_str());
  }

  std::string details;
  llvm::raw_string_ostream stream(details);
  if (llvm::verifyModule(*module, &stream)) {
    stream.flush();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "LLVM IR verification failed with reader " LLVM_VERSION_STRING
        " for %s:\n%s",
        sourceName.str().c_str(), details.c_str());
  }
  module->setModuleIdentifier(sourceName);
  return std::move(module);
}

} // namespace ask::llir
