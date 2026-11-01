#include "ask/Adapters/TritonBoundary.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace ask::adapters {
namespace {
llvm::Error error(llvm::Twine text) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), text.str());
}
std::string digest(llvm::StringRef text) {
  llvm::SHA256 hash;
  hash.update(text);
  return llvm::toHex(hash.final(), true);
}

llvm::Error proveMasks(llvm::Module &module, llvm::Function &entry,
                      const TritonBoundaries &boundaries, int64_t n) {
  // Ask LLVM to normalize actual == expected on a disposable copy. Producer
  // optimizations may turn zext(index < N) into a sign-bit extraction or fold
  // it to a constant. No instruction spelling is itself the legality rule.
  llvm::ValueToValueMapTy mapping;
  auto proof = llvm::CloneModule(module, mapping);
  auto *proofEntry = llvm::cast<llvm::Function>(mapping.lookup(&entry));
  auto &context = module.getContext();
  auto *sink = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::Type::getInt1Ty(context)}, false),
      llvm::GlobalValue::ExternalLinkage, "__ask_mask_proof", *proof);
  llvm::SmallVector<llvm::WeakTrackingVH> checks;
  for (const auto *group : {&boundaries.inputs, &boundaries.outputs}) {
    for (auto *site : *group) {
      auto *copy = llvm::cast<llvm::CallInst>(mapping.lookup(site));
      llvm::IRBuilder<> builder(copy);
      auto *expected = builder.CreateZExt(builder.CreateICmpSLT(copy->getArgOperand(boundaries.indexArgument),
                                       builder.getInt32(n)), builder.getInt32Ty());
      auto *equal = builder.CreateICmpEQ(copy->getArgOperand(boundaries.maskArgument), expected);
      checks.emplace_back(builder.CreateCall(sink, {equal}));
    }
  }
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::ModuleAnalysisManager modules;
  llvm::TargetLibraryInfoImpl libraryInfo(module.getTargetTriple());
  libraryInfo.disableAllFunctions();
  functions.registerPass([libraryInfo] { return llvm::TargetLibraryAnalysis(libraryInfo); });
  llvm::PassBuilder builder;
  builder.registerLoopAnalyses(loops);
  builder.registerFunctionAnalyses(functions);
  builder.registerCGSCCAnalyses(cgscc);
  builder.registerModuleAnalyses(modules);
  builder.crossRegisterProxies(loops, functions, cgscc, modules);
  llvm::FunctionPassManager normalize;
  normalize.addPass(llvm::InstCombinePass());
  normalize.addPass(llvm::EarlyCSEPass());
  normalize.addPass(llvm::InstCombinePass());
  normalize.run(*proofEntry, functions);
  for (auto &handle : checks) {
    auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(handle);
    auto *answer = call ? llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0)) : nullptr;
    if (!answer || !answer->isOne())
      return error("cannot prove boundary mask equivalent to signed index < N predicate");
  }
  return llvm::Error::success();
}
} // namespace

llvm::Expected<TritonBoundaries>
bindTritonBoundaries(llvm::Module &m, const llvm::json::Object &artifact) {
  auto *contract = artifact.getObject("fusion_contract");
  auto payload = artifact.getString("payload");
  auto entryName = artifact.getString("entry");
  auto *producer = artifact.getObject("producer");
  auto *target = artifact.getObject("target");
  auto *specializations = artifact.getObject("specializations");
  if (!contract || !payload || !entryName || !producer || !target || !specializations ||
      contract->getString("protocol") != "ask.triton.boundary.v1" ||
      producer->getString("name") != "triton" || producer->getString("version") != "3.7.1" ||
      contract->getString("llir_sha256") != digest(*payload) ||
      contract->getString("entry") != entryName)
    return error("missing or inconsistent Triton boundary contract / LLVM digest");
  auto dtype = contract->getString("dtype");
  llvm::Type *valueType;
  llvm::StringRef constraints;
  if (dtype == "i32") {
    valueType = llvm::Type::getInt32Ty(m.getContext());
    constraints = "=r,r,l,r,r";
  } else if (dtype == "f32") {
    valueType = llvm::Type::getFloatTy(m.getContext());
    constraints = "=f,r,f,l,r";
  } else if (dtype == "f16" || dtype == "bf16") {
    valueType = dtype == "f16" ? llvm::Type::getHalfTy(m.getContext())
                                : llvm::Type::getBFloatTy(m.getContext());
    constraints = "=h,r,h,l,r";
  } else {
    return error("unsupported Triton boundary dtype");
  }
  uint64_t elementBytes = m.getDataLayout().getTypeStoreSize(valueType).getFixedValue();
  auto *proof = contract->getObject("proof");
  if (!proof || proof->getInteger("input_loads") != 1 || proof->getInteger("output_stores") != 1)
    return error("boundary contract lacks complete input/output coverage");
  for (llvm::StringRef field : {"same_index", "same_mask", "no_pointer_escape",
                                "only_output_store_has_effect", "requires_disjoint_input_output"})
    if (proof->getBoolean(field) != true) return error("incomplete boundary proof: " + field);
  auto *layout = proof->getObject("layout");
  if (!layout || layout->getBoolean("matched_uninstrumented_baseline") != true)
    return error("missing evidence that tracing preserved the producer layout");
  auto *constants = specializations->getObject("constants");
  auto *options = specializations->getObject("fixed_options");
  auto *invocation = artifact.getObject("invocation");
  if (!constants || !options || !invocation) return error("missing Triton launch/specialization contract");
  auto n = constants->getInteger("N"), block = constants->getInteger("BLOCK");
  auto warps = options->getInteger("num_warps"), warpSize = target->getInteger("warp_size");
  if (!n || !block || !warps || !warpSize || *n < 1 || *n > 2147483520 ||
      *block < 1 || *block > 1024 || (*block & (*block - 1)) ||
      (*warps != 4 && *warps != 8) || *warpSize != 32 ||
      target->getString("backend") != "cuda" || target->getInteger("arch") != 80)
    return error("unsupported Triton launch contract");
  auto *shape = contract->getArray("shape");
  if (!shape || shape->size() != 1 || (*shape)[0].getAsInteger() != n ||
      contract->getInteger("block") != block || contract->getInteger("num_warps") != warps ||
      !contract->get("grid") || !invocation->get("grid") ||
      *contract->get("grid") != *invocation->get("grid"))
    return error("Triton boundary shape/launch evidence does not match specialization");
  // Keep the requested policy; do not choose another layout to enable fusion.
  int64_t threads = *warps * *warpSize;
  llvm::json::Array grid{(*n + *block - 1) / *block, 1, 1};
  if (!invocation->get("grid") || *invocation->get("grid") != llvm::json::Value(std::move(grid)) ||
      invocation->getInteger("num_warps") != warps || invocation->getInteger("num_ctas") != 1 ||
      invocation->getInteger("shared_bytes") != 0 || !invocation->get("block") ||
      *invocation->get("block") != llvm::json::Value(llvm::json::Array{threads, 1, 1}))
    return error("boundary launch grid does not cover the logical tensor under the requested policy");
  auto *entry = m.getFunction(*entryName);
  if (!entry || entry->arg_size() != 4 ||
      entry->getCallingConv() != llvm::CallingConv::PTX_Kernel || !entry->getReturnType()->isVoidTy() ||
      entry->isVarArg() || !entry->hasFnAttribute("nvvm.reqntid") ||
      entry->getFnAttribute("nvvm.reqntid").getValueAsString() != std::to_string(threads))
    return error("unsupported external LLVM entry ABI/control flow");
  for (auto &arg : entry->args())
    if (!arg.getType()->isPointerTy() || arg.getType()->getPointerAddressSpace() != 1 ||
        (arg.getArgNo() >= 2 && !arg.use_empty()))
      return error("unsupported data or scratch pointer ABI");
  if (target->getString("data_layout") != m.getDataLayoutStr())
    return error("LLVM data layout differs from artifact target");
  if (!m.getTargetTriple().empty() && m.getTargetTriple().str() != "nvptx64-nvidia-cuda")
    return error("external LLVM target is not nvptx64-nvidia-cuda");
  m.setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));

  TritonBoundaries result;
  if (dtype != "i32") {
    result.valueArgument = 1;
    result.addressArgument = 2;
    result.indexArgument = 0;
  }
  llvm::SmallVector<llvm::CallInst *> *sites[] = {&result.inputs, &result.outputs};
  std::string asmText[] = {"// ASK_BOUNDARY_INPUT_V1\nmov.b",
                           "// ASK_BOUNDARY_OUTPUT_V1\nmov.b"};
  for (auto &text : asmText)
    text += std::to_string(elementBytes * 8) + " $0, $" +
            std::to_string(result.valueArgument + 1) + ";";
  for (auto &bb : *entry) for (auto &instruction : bb) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
    if (!call || !call->isInlineAsm()) continue;
    auto *assembly = llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
    for (unsigned i = 0; i < 2; ++i) {
      if (assembly->getAsmString() != asmText[i]) continue;
      if (assembly->getConstraintString() != constraints || !assembly->hasSideEffects() ||
          call->arg_size() != 4 || call->getType() != valueType ||
          call->getArgOperand(result.valueArgument)->getType() != valueType ||
          !call->getArgOperand(result.addressArgument)->getType()->isPointerTy() ||
          !call->getArgOperand(result.indexArgument)->getType()->isIntegerTy(32) ||
          !call->getArgOperand(result.maskArgument)->getType()->isIntegerTy(32))
        return error("malformed ASK boundary marker");
      sites[i]->push_back(call);
    }
  }
  if (result.inputs.empty() || result.outputs.empty())
    return error("producer did not retain both ASK boundary markers");
  auto *ports = artifact.getArray("port_map");
  if (!ports || ports->size() != 2) return error("missing port map");
  llvm::TargetLibraryInfoImpl tliImpl(m.getTargetTriple());
  tliImpl.disableAllFunctions();
  llvm::TargetLibraryInfo tli(tliImpl);
  llvm::AssumptionCache assumptions(*entry);
  llvm::DominatorTree dominance(*entry);
  llvm::LoopInfo loops(dominance);
  llvm::ScalarEvolution evolution(*entry, tli, assumptions, dominance, loops);
  auto *indexType = llvm::Type::getInt64Ty(m.getContext());
  for (unsigned i = 0; i < 2; ++i) {
    llvm::StringRef binding = i == 0 ? "input:0" : "output:0";
    auto *boundary = contract->getObject(i == 0 ? "input" : "output");
    if (!boundary || boundary->getString("marker_asm") != asmText[i] ||
        boundary->getString("constraints") != constraints ||
        boundary->getString("source_binding") != binding ||
        boundary->getInteger("marker_count") != static_cast<int64_t>(sites[i]->size()))
      return error("boundary marker definition/count differs from the exported evidence");
    auto *arguments = boundary->getObject("arguments");
    if (!arguments || arguments->getInteger("value") != result.valueArgument ||
        arguments->getInteger("address") != result.addressArgument ||
        arguments->getInteger("index") != result.indexArgument ||
        arguments->getInteger("mask") != result.maskArgument)
      return error("boundary marker argument mapping differs from its dtype");
    llvm::Argument *arg = nullptr;
    for (auto &port : *ports) {
      auto *p = port.getAsObject();
      if (p && p->getString("logical_port") == binding) {
        if (p->getString("element_type") != dtype)
          return error("boundary dtype does not match data port element type");
        auto index = p->getInteger("argument");
        if (!index || *index < 0 || *index > 1) return error("invalid data argument mapping");
        arg = entry->getArg(*index);
      }
    }
    if (!arg) return error("missing data argument mapping");
    for (auto *site : *sites[i]) {
      auto *index = site->getArgOperand(result.indexArgument);
      auto *address = site->getArgOperand(result.addressArgument);
      if (llvm::getUnderlyingObject(address) != arg)
        return error("boundary address is not rooted at the declared data port");
      // Compare LLVM address expressions, not one textual GEP/sext pattern.
      auto *offset = evolution.getSignExtendExpr(evolution.getSCEV(index), indexType);
      auto *bytes = evolution.getMulExpr(offset, evolution.getConstant(indexType, elementBytes));
      auto *expected = evolution.getAddExpr(evolution.getSCEV(arg), bytes);
      if (evolution.getSCEV(address) != expected)
        return error("boundary address does not match its signed i32 element index");
    }
  }
  if (auto e = proveMasks(m, *entry, result, *n)) return std::move(e);
  return result;
}

llvm::Expected<llvm::json::Object>
normalizeTritonNumerics(llvm::Module &module, const llvm::json::Object &artifact) {
  auto *contract = artifact.getObject("fusion_contract");
  auto *specializations = artifact.getObject("specializations");
  auto *invocation = artifact.getObject("invocation");
  auto *numerics = contract ? contract->getObject("numerics") : nullptr;
  auto *options = specializations ? specializations->getObject("fixed_options") : nullptr;
  auto *metadata = invocation ? invocation->getObject("compiler_metadata") : nullptr;
  auto entryName = artifact.getString("entry");
  auto *entry = entryName ? module.getFunction(*entryName) : nullptr;
  if (!numerics || !options || !metadata || !entry)
    return error("missing Triton floating numerical policy");
  auto enabled = numerics->getBoolean("enable_fp_fusion");
  auto reflectFTZ = numerics->getBoolean("enable_reflect_ftz");
  if (!enabled || !reflectFTZ || metadata->getBoolean("enable_fp_fusion") != enabled ||
      metadata->getBoolean("enable_reflect_ftz") != reflectFTZ ||
      *enabled != options->getBoolean("enable_fp_fusion").value_or(true))
    return error("Triton floating numerical policy does not match compiler metadata/options");
  // Public Triton export returns unflagged fmul/fadd and applies its global
  // enable_fp_fusion option in LLVM-to-PTX codegen. Encode that permission on
  // original F instructions before strict ASK instructions enter this module.
  llvm::SmallVector<llvm::BinaryOperator *> candidates;
  for (auto &block : *entry) for (auto &instruction : block) {
    auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
    if (!binary || (binary->getOpcode() != llvm::Instruction::FMul &&
                    binary->getOpcode() != llvm::Instruction::FAdd &&
                    binary->getOpcode() != llvm::Instruction::FSub)) continue;
    if (!*enabled && binary->hasAllowContract())
      return error("Triton LLVM permits contraction despite enable_fp_fusion=false");
    candidates.push_back(binary);
  }
  if (*enabled)
    for (auto *binary : candidates) binary->setHasAllowContract(true);
  return llvm::json::Object{
      {"producer_enable_fp_fusion", *enabled}, {"producer_enable_reflect_ftz", *reflectFTZ},
      {"representation", "original-F-instruction-contract-flags"},
      {"tagged_instructions", static_cast<int64_t>(*enabled ? candidates.size() : 0)}};
}
} // namespace ask::adapters
