#include "ask/LLIR/Inline.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace ask::llir {
namespace {
llvm::Error error(llvm::Twine text) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), text.str());
}

bool valueType(llvm::Type *type) {
  if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type))
    type = vector->getElementType();
  return type->isIntegerTy() || type->isFloatingPointTy();
}

bool containsPointer(const llvm::Value *value) {
  if (value->getType()->getScalarType()->isPointerTy()) return true;
  if (auto *constant = llvm::dyn_cast<llvm::Constant>(value))
    for (const auto &operand : constant->operands())
      if (containsPointer(operand.get())) return true;
  return false;
}

bool safeIntrinsic(llvm::Intrinsic::ID id) {
  // The effect contract comes from LLVM's intrinsic definition, not an arbitrary
  // declaration carrying user-supplied memory(none)/speculatable attributes.
  switch (id) {
  case llvm::Intrinsic::abs:
  case llvm::Intrinsic::bswap:
  case llvm::Intrinsic::bitreverse:
  case llvm::Intrinsic::ctpop:
  case llvm::Intrinsic::ctlz:
  case llvm::Intrinsic::cttz:
  case llvm::Intrinsic::smax:
  case llvm::Intrinsic::smin:
  case llvm::Intrinsic::umax:
  case llvm::Intrinsic::umin:
  case llvm::Intrinsic::fshl:
  case llvm::Intrinsic::fshr:
  case llvm::Intrinsic::fabs:
  case llvm::Intrinsic::sqrt:
  case llvm::Intrinsic::floor:
  case llvm::Intrinsic::ceil:
  case llvm::Intrinsic::trunc:
  case llvm::Intrinsic::round:
  case llvm::Intrinsic::roundeven:
  case llvm::Intrinsic::copysign:
  case llvm::Intrinsic::minimum:
  case llvm::Intrinsic::maximum:
  case llvm::Intrinsic::minnum:
  case llvm::Intrinsic::maxnum:
  case llvm::Intrinsic::fma:
  case llvm::Intrinsic::fptosi_sat:
    return true;
  default:
    return false;
  }
}

bool acyclic(const llvm::BasicBlock *block,
             llvm::DenseMap<const llvm::BasicBlock *, unsigned> &colors) {
  unsigned &color = colors[block];
  if (color == 1) return false;
  if (color == 2) return true;
  color = 1;
  for (auto *successor : llvm::successors(block))
    if (!acyclic(successor, colors)) return false;
  colors[block] = 2;
  return true;
}

llvm::Error checkHelper(const llvm::Function &function, const llvm::Module &module,
                        llvm::DenseMap<const llvm::Function *, unsigned> &colors) {
  unsigned color = colors.lookup(&function);
  if (color == 1) return error("recursive value helpers cannot be speculated");
  if (color == 2) return llvm::Error::success();
  if (function.isDeclaration()) {
    if (!safeIntrinsic(function.getIntrinsicID()))
      return error("unknown external helper call: " + function.getName());
    colors[&function] = 2;
    return llvm::Error::success();
  }
  if (function.isVarArg() || !valueType(function.getReturnType()) ||
      function.getCallingConv() != llvm::CallingConv::C || function.hasPersonalityFn() ||
      function.hasFnAttribute(llvm::Attribute::NoInline) ||
      function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
      function.hasFnAttribute(llvm::Attribute::StrictFP) ||
      function.hasFnAttribute(llvm::Attribute::Naked) ||
      function.hasFnAttribute(llvm::Attribute::ReturnsTwice) ||
      function.hasFnAttribute(llvm::Attribute::Convergent))
    return error("helper must be an inlineable ordinary numeric value function: " + function.getName());
  // String attributes can select a different target or floating-point model.
  // Dropping them during inlining or merging them into the consumer would both
  // change semantics. This layer accepts instruction-local numerical flags;
  // callers must normalize function-level assumptions before handing it helpers.
  for (auto attribute : function.getAttributes().getFnAttrs())
    if (attribute.isStringAttribute())
      return error("helper function-level target/numerical attributes require normalization");
  if (function.getAttributes().getRetAttrs().hasAttributes())
    return error("helper return attributes require normalization");
  for (unsigned index = 0; index < function.arg_size(); ++index)
    if (function.getAttributes().getParamAttrs(index).hasAttributes())
      return error("helper parameter attributes require normalization");
  for (const auto &argument : function.args())
    if (!valueType(argument.getType())) return error("helper arguments must be numeric values");
  llvm::DenseMap<const llvm::BasicBlock *, unsigned> blocks;
  for (const auto &block : function)
    if (!acyclic(&block, blocks)) return error("cyclic helper control flow cannot be speculated");
  colors[&function] = 1;
  for (const auto &block : function) {
    for (const auto &instruction : block) {
      if (instruction.getType()->getScalarType()->isPointerTy()) return error("pointer computation in value helper");
      auto *asCall = llvm::dyn_cast<llvm::CallInst>(&instruction);
      for (const auto &operand : instruction.operands())
        if ((!asCall || operand.get() != asCall->getCalledOperand()) && containsPointer(operand.get()))
          return error("pointer operand in value helper");
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
        auto *callee = call->getCalledFunction();
        if (!callee || callee->getParent() != &module || call->hasOperandBundles() ||
            call->isConvergent() || call->isMustTailCall())
          return error("unknown, bundled or convergent helper call");
        if (call->getAttributes().getRetAttrs().hasAttributes())
          return error("helper call return attributes require normalization");
        for (unsigned index = 0; index < call->arg_size(); ++index)
          if (call->getAttributes().getParamAttrs(index).hasAttributes())
            return error("helper call parameter attributes require normalization");
        if (auto e = checkHelper(*callee, module, colors)) return e;
        continue;
      }
      switch (instruction.getOpcode()) {
      case llvm::Instruction::Ret:
      case llvm::Instruction::Br:
      case llvm::Instruction::Switch:
      case llvm::Instruction::PHI:
      case llvm::Instruction::Select:
      case llvm::Instruction::Freeze:
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::And:
      case llvm::Instruction::Or:
      case llvm::Instruction::Xor:
      case llvm::Instruction::Shl:
      case llvm::Instruction::LShr:
      case llvm::Instruction::AShr:
      case llvm::Instruction::FNeg:
      case llvm::Instruction::FAdd:
      case llvm::Instruction::FSub:
      case llvm::Instruction::FMul:
      case llvm::Instruction::FDiv:
      case llvm::Instruction::FRem:
      case llvm::Instruction::ICmp:
      case llvm::Instruction::FCmp:
      case llvm::Instruction::Trunc:
      case llvm::Instruction::ZExt:
      case llvm::Instruction::SExt:
      case llvm::Instruction::FPToUI:
      case llvm::Instruction::FPToSI:
      case llvm::Instruction::UIToFP:
      case llvm::Instruction::SIToFP:
      case llvm::Instruction::FPTrunc:
      case llvm::Instruction::FPExt:
      case llvm::Instruction::BitCast:
      case llvm::Instruction::ExtractElement:
      case llvm::Instruction::InsertElement:
      case llvm::Instruction::ShuffleVector:
        break;
      default:
        return error("unsupported or potentially trapping value helper instruction: " +
                     llvm::Twine(instruction.getOpcodeName()));
      }
    }
  }
  colors[&function] = 2;
  return llvm::Error::success();
}

bool available(llvm::Value *value, llvm::CallInst *marker, llvm::DominatorTree &dominators) {
  if (!value || &value->getContext() != &marker->getContext()) return false;
  if (llvm::isa<llvm::Constant>(value)) return valueType(value->getType());
  if (auto *argument = llvm::dyn_cast<llvm::Argument>(value))
    return argument->getParent() == marker->getFunction();
  auto *instruction = llvm::dyn_cast<llvm::Instruction>(value);
  return instruction && instruction != marker &&
         instruction->getFunction() == marker->getFunction() && dominators.dominates(instruction, marker);
}

bool straightLine(const llvm::Function &function) {
  if (function.isDeclaration()) return true; // already checked intrinsic
  if (function.size() != 1) return false;
  for (const auto &instruction : function.front())
    if (const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
      if (!straightLine(*call->getCalledFunction())) return false;
  return true;
}

bool usesFloatingPoint(const llvm::Function &function,
                       llvm::SmallPtrSetImpl<const llvm::Function *> &visited) {
  if (!visited.insert(&function).second) return false;
  auto floating = [](llvm::Type *type) { return type->getScalarType()->isFloatingPointTy(); };
  if (floating(function.getReturnType())) return true;
  for (const auto &argument : function.args())
    if (floating(argument.getType())) return true;
  for (const auto &block : function)
    for (const auto &instruction : block) {
      if (floating(instruction.getType())) return true;
      for (const auto &operand : instruction.operands())
        if (floating(operand->getType())) return true;
      if (const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
        if (usesFloatingPoint(*call->getCalledFunction(), visited)) return true;
    }
  return false;
}

llvm::Error checkConsumerNumerics(const llvm::Function &consumer,
                                  const llvm::Function &helper) {
  // The helper may return an integer after doing floating-point work. Checking
  // only the marker's result type would miss conversions and nested helpers.
  llvm::SmallPtrSet<const llvm::Function *, 16> visited;
  if (!usesFloatingPoint(helper, visited)) return llvm::Error::success();
  if (consumer.hasFnAttribute(llvm::Attribute::StrictFP))
    return error("floating helper cannot inherit consumer strictfp semantics");
  for (llvm::StringRef name : {"unsafe-fp-math", "no-nans-fp-math", "no-infs-fp-math",
                               "no-signed-zeros-fp-math", "approx-func-fp-math",
                               "less-precise-fpmad"}) {
    auto attribute = consumer.getFnAttribute(name);
    if (attribute.isValid() && attribute.getValueAsString() != "false")
      return error("floating helper cannot inherit consumer numerical attribute: " + name);
  }
  for (llvm::StringRef name : {"denormal-fp-math", "denormal-fp-math-f32"}) {
    auto attribute = consumer.getFnAttribute(name);
    if (attribute.isValid() && attribute.getValueAsString() != "ieee" &&
        attribute.getValueAsString() != "ieee,ieee")
      return error("floating helper cannot inherit consumer numerical attribute: " + name);
  }
  auto trapping = consumer.getFnAttribute("no-trapping-math");
  if (trapping.isValid() && trapping.getValueAsString() != "true")
    return error("floating helper cannot inherit consumer numerical attribute: no-trapping-math");
  auto contraction = consumer.getFnAttribute("fp-contract");
  if (contraction.isValid() && contraction.getValueAsString() != "off" &&
      contraction.getValueAsString() != "on")
    return error("floating helper cannot inherit consumer numerical attribute: fp-contract");
  return llvm::Error::success();
}
} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
inlineValueBoundaries(const llvm::Module &consumer, const llvm::Module &helpers,
                      llvm::ArrayRef<BoundaryInlineSite> sites, llvm::json::Object &report) {
  if (&consumer.getContext() != &helpers.getContext()) return error("LLVM contexts must match");
  if (llvm::verifyModule(consumer, &llvm::errs()) || llvm::verifyModule(helpers, &llvm::errs()))
    return error("invalid input LLVM module");
  if ((!helpers.getTargetTriple().empty() && helpers.getTargetTriple().str() != consumer.getTargetTriple().str()) ||
      (!helpers.getDataLayoutStr().empty() && helpers.getDataLayoutStr() != consumer.getDataLayoutStr()))
    return error("value helper target/data layout differs from consumer");
  if (!helpers.global_empty() || !helpers.alias_empty() || !helpers.ifunc_empty() ||
      !helpers.getModuleInlineAsm().empty() || helpers.getModuleFlagsMetadata())
    return error("value helper module must not introduce globals, aliases, ifuncs, module asm or flags");
  llvm::DenseMap<const llvm::Function *, unsigned> helperColors;
  for (const auto &function : helpers)
    if (auto e = checkHelper(function, helpers, helperColors)) return std::move(e);
  llvm::SmallPtrSet<llvm::CallInst *, 16> seen;
  for (const auto &site : sites) {
    if (!site.marker || !site.helper || site.marker->getModule() != &consumer ||
        site.helper->getParent() != &helpers || site.helper->isDeclaration() ||
        !seen.insert(site.marker).second || site.marker->hasOperandBundles() ||
        site.marker->isMustTailCall() || site.marker->isConvergent() || !valueType(site.marker->getType()))
      return error("boundary must identify a unique numeric identity call and a defined helper");
    if (site.helper->getReturnType() != site.marker->getType() ||
        site.helper->arg_size() != site.arguments.size()) return error("boundary/helper signature mismatch");
    if (auto e = checkConsumerNumerics(*site.marker->getFunction(), *site.helper))
      return std::move(e);
    llvm::DominatorTree dominators(*site.marker->getFunction());
    for (size_t i = 0; i < site.arguments.size(); ++i)
      if (!available(site.arguments[i], site.marker, dominators) ||
          site.arguments[i]->getType() != site.helper->getArg(i)->getType())
        return error("helper argument type or dominance mismatch");
    if (site.predicate) {
      if (!available(site.predicate, site.marker, dominators) ||
          !available(site.passthrough, site.marker, dominators) ||
          site.passthrough->getType() != site.marker->getType())
        return error("invalid predicated boundary passthrough or dominance");
      auto *predicateType = site.predicate->getType();
      if (!predicateType->isIntegerTy(1)) {
        auto *mask = llvm::dyn_cast<llvm::FixedVectorType>(predicateType);
        auto *value = llvm::dyn_cast<llvm::FixedVectorType>(site.marker->getType());
        if (!mask || !value || !mask->getElementType()->isIntegerTy(1) ||
            mask->getNumElements() != value->getNumElements())
          return error("boundary predicate must be i1 or matching vector<i1>");
        if (!straightLine(*site.helper))
          return error("vector-predicated helpers must be straight-line computations");
      }
    } else if (site.passthrough) {
      return error("passthrough requires a predicate");
    }
  }
  llvm::ValueToValueMapTy consumerMap, helperMap;
  auto result = llvm::CloneModule(consumer, consumerMap);
  auto copiedHelpers = llvm::CloneModule(helpers, helperMap);
  copiedHelpers->setTargetTriple(consumer.getTargetTriple());
  copiedHelpers->setDataLayout(consumer.getDataLayout());
  llvm::SmallVector<std::string> helperNames;
  unsigned ordinal = 0;
  for (auto &function : *copiedHelpers) {
    if (function.isIntrinsic()) continue;
    std::string name;
    do { name = "__ask_value_helper_" + std::to_string(ordinal++); }
    while (result->getNamedValue(name) || copiedHelpers->getNamedValue(name));
    function.setName(name);
    function.setLinkage(llvm::GlobalValue::ExternalLinkage);
    function.setComdat(nullptr);
    helperNames.push_back(name);
  }
  llvm::SmallVector<std::string> siteHelperNames;
  for (const auto &site : sites)
    siteHelperNames.push_back(llvm::cast<llvm::Function>(helperMap.lookup(site.helper))->getName().str());
  if (llvm::Linker::linkModules(*result, std::move(copiedHelpers))) return error("LLVM helper link failed");
  for (size_t index = 0; index < sites.size(); ++index) {
    const auto &site = sites[index];
    auto *marker = llvm::cast<llvm::CallInst>(consumerMap.lookup(site.marker));
    llvm::SmallVector<llvm::Value *> arguments;
    auto mapped = [&](llvm::Value *value) -> llvm::Value * {
      return llvm::MapValue(value, consumerMap);
    };
    for (auto *argument : site.arguments) arguments.push_back(mapped(argument));
    llvm::IRBuilder<> builder(marker);
    auto *function = result->getFunction(siteHelperNames[index]);
    llvm::CallInst *call;
    llvm::Value *replacement;
    if (site.predicate && site.predicate->getType()->isIntegerTy(1)) {
      llvm::Instruction *thenTerminator, *elseTerminator;
      llvm::SplitBlockAndInsertIfThenElse(mapped(site.predicate), marker,
                                         &thenTerminator, &elseTerminator);
      builder.SetInsertPoint(thenTerminator);
      call = builder.CreateCall(function, arguments);
      builder.SetInsertPoint(marker);
      auto *merged = builder.CreatePHI(marker->getType(), 2);
      merged->addIncoming(call, thenTerminator->getParent());
      merged->addIncoming(mapped(site.passthrough), elseTerminator->getParent());
      replacement = merged;
    } else {
      call = builder.CreateCall(function, arguments);
      replacement = call;
    }
    if (site.predicate && !site.predicate->getType()->isIntegerTy(1))
      replacement = builder.CreateSelect(mapped(site.predicate), call, mapped(site.passthrough));
    marker->replaceAllUsesWith(replacement);
    marker->eraseFromParent();
    llvm::InlineFunctionInfo information;
    if (!llvm::InlineFunction(*call, information).isSuccess()) return error("LLVM boundary helper inlining failed");
  }
  // Helpers may call other pure helpers. Keep remaining definitions internal so
  // no new public ABI appears; unused roots are removed without touching consumer
  // functions that happened to share their original names.
  for (const auto &name : helperNames) {
    auto *function = result->getFunction(name);
    function->setLinkage(llvm::GlobalValue::InternalLinkage);
  }
  bool removed;
  do {
    removed = false;
    for (auto iterator = result->begin(); iterator != result->end();) {
      auto &function = *iterator++;
      if (llvm::is_contained(helperNames, function.getName().str()) && function.use_empty()) {
        function.eraseFromParent();
        removed = true;
      }
    }
  } while (removed);
  if (llvm::verifyModule(*result, &llvm::errs())) return error("invalid LLVM after boundary inlining");
  report["boundary_inlining"] = llvm::json::Object{{"schema_version", 1},
      {"sites", int64_t(sites.size())}, {"helper_definitions", int64_t(helperNames.size())},
      {"input_modules_unchanged", true}, {"llvm_verified", true},
      {"caller_instruction_mappings", "invalidated"}};
  return std::move(result);
}
} // namespace ask::llir
