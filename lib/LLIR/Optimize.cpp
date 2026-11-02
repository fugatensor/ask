#include "ask/LLIR/Optimize.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

namespace ask::llir {
namespace {
llvm::Error verify(llvm::Module &module, llvm::StringRef stage) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  if (!llvm::verifyModule(module, &stream))
    return llvm::Error::success();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "LLVM IR verification failed %s optimization:\n%s",
                                stage.str().c_str(), stream.str().c_str());
}

llvm::json::Object counts(const llvm::Module &module) {
  int64_t functions = 0, blocks = 0, instructions = 0;
  for (const llvm::Function &function : module) {
    if (function.isDeclaration())
      continue;
    ++functions;
    blocks += function.size();
    instructions += function.getInstructionCount();
  }
  return llvm::json::Object{{"functions", functions},
                            {"basic_blocks", blocks},
                            {"instructions", instructions}};
}

bool isTraceMetadata(llvm::StringRef name) {
  return name.starts_with("ask.trace") ||
         name.starts_with("ask.instruction_map") ||
         name.starts_with("ask.ssa_map") || name.starts_with("ask.value_map");
}

// These records describe module interfaces and logical shapes, not instructions.
bool isSemanticMetadata(llvm::StringRef name) {
  return name == "ask.abi" || name == "ask.shape_snapshot";
}

struct FunctionABI {
  std::string name;
  llvm::FunctionType *type;
  llvm::CallingConv::ID callingConvention;
  llvm::GlobalValue::LinkageTypes linkage;
  llvm::GlobalValue::VisibilityTypes visibility;
  llvm::GlobalValue::DLLStorageClassTypes storage;
  unsigned addressSpace;
  llvm::AttributeList attributes;

  bool matches(const llvm::Module &module) const {
    const llvm::Function *function = module.getFunction(name);
    return function && function->getFunctionType() == type &&
           function->getCallingConv() == callingConvention &&
           function->getLinkage() == linkage &&
           function->getVisibility() == visibility &&
           function->getDLLStorageClass() == storage &&
           function->getAddressSpace() == addressSpace &&
           function->getAttributes() == attributes;
  }
};
} // namespace

llvm::Error optimizeLLVMIR(llvm::Module &module, llvm::json::Object &report) {
  if (auto error = verify(module, "before"))
    return error;

  auto before = counts(module);
  const auto triple = module.getTargetTriple();
  const std::string layout = module.getDataLayoutStr();
  llvm::SmallVector<FunctionABI> interfaces;
  for (const llvm::Function &function : module)
    if (!function.hasLocalLinkage() && !function.isIntrinsic())
      interfaces.push_back({function.getName().str(), function.getFunctionType(),
                            function.getCallingConv(), function.getLinkage(),
                            function.getVisibility(), function.getDLLStorageClass(),
                            function.getAddressSpace(), function.getAttributes()});

  llvm::json::Array preservedMetadata, removedMetadata, unvalidatedMetadata;
  llvm::SmallVector<llvm::NamedMDNode *> staleRecords;
  for (llvm::NamedMDNode &record : module.named_metadata()) {
    llvm::StringRef name = record.getName();
    if (isSemanticMetadata(name))
      preservedMetadata.push_back(name.str());
    else if (isTraceMetadata(name)) {
      removedMetadata.push_back(name.str());
      staleRecords.push_back(&record);
    } else if (name.starts_with("ask."))
      unvalidatedMetadata.push_back(name.str());
  }
  for (llvm::NamedMDNode *record : staleRecords)
    module.eraseNamedMetadata(record);

  llvm::SmallVector<llvm::StringRef> metadataKinds;
  module.getMDKindNames(metadataKinds);
  int64_t removedObjectAttachments = 0;
  for (llvm::GlobalObject &object : module.global_objects()) {
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> attachments;
    object.getAllMetadata(attachments);
    for (auto [kind, node] : attachments)
      if (isTraceMetadata(metadataKinds[kind])) {
        object.setMetadata(kind, nullptr);
        ++removedObjectAttachments;
      }
  }
  int64_t removedAttachments = 0;
  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> attachments;
      instruction.getAllMetadata(attachments);
      for (auto [kind, node] : attachments)
        if (metadataKinds[kind].starts_with("ask.")) {
          instruction.setMetadata(kind, nullptr);
          ++removedAttachments;
        }
    }

  // Register analysis dependencies without selecting any default O1/O2/O3
  // pipeline, target machine, vectorizer, loop transform, or inliner.
  llvm::LoopAnalysisManager loopAnalyses;
  llvm::FunctionAnalysisManager functionAnalyses;
  llvm::CGSCCAnalysisManager cgsccAnalyses;
  llvm::ModuleAnalysisManager moduleAnalyses;
  // A third-party symbol named strlen/memcpy/etc. does not itself establish a
  // libc contract. Disable name-based library assumptions for this generic
  // entry point; explicit LLVM intrinsics and IR attributes keep their meaning.
  llvm::TargetLibraryInfoImpl libraryInfo(module.getTargetTriple());
  libraryInfo.disableAllFunctions();
  functionAnalyses.registerPass([libraryInfo] {
    return llvm::TargetLibraryAnalysis(libraryInfo);
  });
  llvm::PassBuilder builder;
  builder.registerLoopAnalyses(loopAnalyses);
  builder.registerFunctionAnalyses(functionAnalyses);
  builder.registerCGSCCAnalyses(cgsccAnalyses);
  builder.registerModuleAnalyses(moduleAnalyses);
  builder.crossRegisterProxies(loopAnalyses, functionAnalyses, cgsccAnalyses,
                               moduleAnalyses);
  llvm::FunctionPassManager combine, simplify, eliminate;
  combine.addPass(llvm::InstCombinePass());
  simplify.addPass(llvm::SimplifyCFGPass());
  eliminate.addPass(llvm::ADCEPass());
  llvm::FunctionPassManager *pipelines[] = {&combine, &simplify, &eliminate};
  int64_t changedByPass[] = {0, 0, 0};
  int64_t processedFunctions = 0, changedFunctions = 0, skippedFunctions = 0;
  for (llvm::Function &function : module) {
    if (function.isDeclaration())
      continue;
    if (function.hasFnAttribute(llvm::Attribute::OptimizeNone)) {
      ++skippedFunctions;
      continue;
    }
    bool changed = false;
    for (unsigned i = 0; i < 3; ++i) {
      // Each selected LLVM pass preserves all analyses when it makes no change.
      // A one-pass manager also performs the corresponding FAM invalidation.
      if (!pipelines[i]->run(function, functionAnalyses).areAllPreserved()) {
        ++changedByPass[i];
        changed = true;
      }
    }
    ++processedFunctions;
    changedFunctions += changed;
  }

  if (auto error = verify(module, "after"))
    return error;
  if (module.getTargetTriple() != triple || module.getDataLayoutStr() != layout)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "LLVM optimization unexpectedly changed target information");
  for (const FunctionABI &interface : interfaces)
    if (!interface.matches(module))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "LLVM optimization unexpectedly changed the ABI of %s",
                                    interface.name.c_str());

  const char *status = changedFunctions ? "applied" : "skipped";
  llvm::json::Array passes;
  const char *names[] = {"instcombine", "simplifycfg", "adce"};
  for (unsigned i = 0; i < 3; ++i)
    passes.push_back(llvm::json::Object{
        {"name", names[i]},
        {"status", changedByPass[i] ? "applied" : "skipped"},
        {"functions_changed", changedByPass[i]}});
  report["llir_optimization"] = llvm::json::Object{
      {"status", status}, {"passes", std::move(passes)},
      {"library_function_inference", false},
      {"before", std::move(before)}, {"after", counts(module)},
      {"functions_processed", processedFunctions},
      {"functions_changed", changedFunctions},
      {"functions_skipped_optnone", skippedFunctions},
      {"metadata_changed", !staleRecords.empty() || removedAttachments != 0 ||
                               removedObjectAttachments != 0},
      {"preserved_semantic_metadata", std::move(preservedMetadata)},
      {"removed_trace_metadata", std::move(removedMetadata)},
      {"removed_instruction_attachments", removedAttachments},
      {"removed_object_attachments", removedObjectAttachments},
      {"unvalidated_named_metadata", std::move(unvalidatedMetadata)},
      {"invalidated_analyses", llvm::json::Array{"llvm_ssa_locations", "trace"}}};
  return llvm::Error::success();
}

} // namespace ask::llir
