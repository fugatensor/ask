#include "ask/LLIR/Fuse.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Import.h"
#include "ask/LLIR/Optimize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace ask::llir {
namespace {
llvm::Error error(llvm::Twine text) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), text.str());
}
std::string digest(llvm::StringRef text) {
  llvm::SHA256 hash;
  hash.update(text);
  return llvm::toHex(hash.final(), true);
}

llvm::json::Value attributeJSON(mlir::Attribute attr) {
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
    llvm::json::Object result;
    for (auto item : dict) result[item.getName().strref()] = attributeJSON(item.getValue());
    return result;
  }
  if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr)) return s.getValue();
  if (auto n = mlir::dyn_cast<mlir::IntegerAttr>(attr)) return n.getInt();
  if (auto array = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    llvm::json::Array result;
    for (auto n : array.asArrayRef()) result.push_back(n);
    return result;
  }
  return nullptr;
}

// Both helpers are scalar, pure i32 functions. Tensor indices, masks, memory
// operations and the external kernel's launch strategy stay in the producer IR.
llvm::Expected<llvm::Value *> emit(mlir::Value value, mlir::Value leaf,
                                  llvm::Value *argument, llvm::IRBuilder<> &b,
                                  llvm::DenseMap<mlir::Value, llvm::Value *> &memo) {
  if (value == leaf) return argument;
  if (auto it = memo.find(value); it != memo.end()) return it->second;
  auto *op = value.getDefiningOp();
  if (!op) return error("fusion expression depends on a different graph input");
  llvm::Value *result;
  if (auto constant = mlir::dyn_cast<ConstantOp>(op)) {
    auto attr = constant->getAttrOfType<mlir::IntegerAttr>("value");
    result = b.getInt(attr.getValue());
  } else if (mlir::isa<SplatOp>(op)) {
    auto scalar = emit(op->getOperand(0), leaf, argument, b, memo);
    if (!scalar) return scalar.takeError();
    result = *scalar;
  } else if (mlir::isa<AddOp, MulOp>(op)) {
    auto lhs = emit(op->getOperand(0), leaf, argument, b, memo);
    if (!lhs) return lhs.takeError();
    auto rhs = emit(op->getOperand(1), leaf, argument, b, memo);
    if (!rhs) return rhs.takeError();
    result = mlir::isa<AddOp>(op) ? b.CreateAdd(*lhs, *rhs) : b.CreateMul(*lhs, *rhs);
  } else {
    return error("fusion supports only constant/splat/add/mul around one external operation");
  }
  memo[value] = result;
  return result;
}

llvm::Expected<llvm::Function *> helper(llvm::Module &m, llvm::StringRef name,
                                       mlir::Value root, mlir::Value leaf) {
  llvm::IRBuilder<> b(m.getContext());
  auto *f = llvm::Function::Create(llvm::FunctionType::get(b.getInt32Ty(), {b.getInt32Ty()}, false),
                                  llvm::GlobalValue::ExternalLinkage, name, m);
  f->setDoesNotAccessMemory();
  f->setDoesNotThrow();
  b.SetInsertPoint(llvm::BasicBlock::Create(m.getContext(), "entry", f));
  llvm::DenseMap<mlir::Value, llvm::Value *> memo;
  auto value = emit(root, leaf, f->getArg(0), b, memo);
  if (!value) return value.takeError();
  b.CreateRet(*value);
  return f;
}
} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
fuseExternal(mlir::ModuleOp graph, const llvm::json::Object &artifact,
             llvm::LLVMContext &context, llvm::json::Object &report, std::string &helperIR) {
  llvm::SmallVector<mlir::func::FuncOp> functions(graph.getOps<mlir::func::FuncOp>());
  if (functions.size() != 1) return error("fusion requires exactly one graph function");
  auto func = functions[0];
  if (func.getNumArguments() != 1 || func.getNumResults() != 1 || !func.getBody().hasOneBlock())
    return error("fusion requires one static tensor input and one output in one block");
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() || !type.getElementType().isSignlessInteger(32))
    return error("fusion requires a static one-dimensional i32 tensor");
  int64_t n = type.getDimSize(0);
  if (func.getResultTypes()[0] != type || n < 1 || n > 2147483520)
    return error("fusion requires matching shapes and indices within signed i32 range");
  ExternalOp external;
  for (auto &op : func.front()) {
    if (!mlir::isa<ConstantOp, SplatOp, AddOp, MulOp, ExternalOp, mlir::func::ReturnOp>(op))
      return error("unsupported operation in fusion graph");
    for (auto result : op.getResults())
      if (mlir::isa<mlir::TensorType>(result.getType()) && result.getType() != type)
        return error("fusion requires matching pointwise tensor shapes");
    if (auto f = mlir::dyn_cast<ExternalOp>(op)) {
      if (external) return error("fusion requires exactly one external operation");
      external = f;
    }
  }
  if (!external || external->getNumOperands() != 1 || external->getNumResults() != 1)
    return error("fusion requires one external operation with one input and one output");
  if (external.resolveSemantics()->getAttrOfType<mlir::StringAttr>("effects").getValue() != "none")
    return error("fusion requires a pure external semantic contract");
  auto *contract = artifact.getObject("fusion_contract");
  auto payload = artifact.getString("payload");
  auto entryName = artifact.getString("entry");
  auto *producer = artifact.getObject("producer");
  auto *specializations = artifact.getObject("specializations");
  if (!contract || !payload || !entryName || !producer || !specializations ||
      contract->getString("protocol") != "ask.triton.boundary.v1" ||
      producer->getString("name") != "triton" || producer->getString("version") != "3.7.1" ||
      contract->getString("llir_sha256") != digest(*payload) ||
      contract->getString("entry") != entryName || contract->getString("dtype") != "i32")
    return error("missing or inconsistent Triton boundary contract / LLVM digest");
  auto impl = external.resolveImplementation();
  auto source = impl->getAttrOfType<mlir::StringAttr>("source").getValue();
  if (impl->getAttrOfType<mlir::StringAttr>("compiler").getValue() != "triton" ||
      impl->getAttrOfType<mlir::StringAttr>("entry").getValue() != entryName)
    return error("artifact entry/compiler does not match this implementation");
  for (auto pair : {std::make_pair("constants", "constants"),
                    std::make_pair("options", "fixed_options"),
                    std::make_pair("source_bindings", "source_bindings")}) {
    auto attr = impl->getAttr(pair.first);
    if (!attr || !specializations->get(pair.second) ||
        attributeJSON(attr) != *specializations->get(pair.second))
      return error("artifact specialization does not match implementation: " + llvm::Twine(pair.first));
  }
  auto *target = artifact.getObject("target");
  auto declaredTarget = impl->getAttrOfType<mlir::DictionaryAttr>("target");
  if (!target || !declaredTarget) return error("missing target contract");
  for (auto field : declaredTarget)
    if (!target->get(field.getName().strref()) ||
        *target->get(field.getName().strref()) != attributeJSON(field.getValue()))
      return error("artifact target does not match implementation");
  if (contract->getString("original_source_sha256") != digest(source) ||
      specializations->getString("source_sha256") != digest(source))
    return error("boundary contract does not describe this graph's DSL source");
  auto instrumented = artifact.getString("instrumented_source");
  if (!instrumented || contract->getString("instrumented_source_sha256") != digest(*instrumented))
    return error("instrumented source digest does not match boundary contract");
  auto *shape = contract->getArray("shape");
  auto *proof = contract->getObject("proof");
  if (!proof || proof->getInteger("input_loads") != 1 || proof->getInteger("output_stores") != 1)
    return error("boundary contract lacks complete input/output coverage");
  for (llvm::StringRef field : {"same_index", "same_mask", "no_pointer_escape",
                                "only_output_store_has_effect", "requires_disjoint_input_output"})
    if (proof->getBoolean(field) != true) return error("incomplete boundary proof: " + field);
  auto *constants = specializations->getObject("constants");
  auto *invocation = artifact.getObject("invocation");
  if (!shape || shape->size() != 1 || (*shape)[0].getAsInteger() != n || !constants ||
      constants->getInteger("N") != n || constants->getInteger("BLOCK") != 128 || !invocation)
    return error("boundary shape/constexpr N does not match the prepared graph");
  llvm::json::Array grid{std::max<int64_t>(1, (n + 127) / 128), 1, 1};
  if (!invocation->get("grid") || *invocation->get("grid") != llvm::json::Value(std::move(grid)) ||
      !impl->getAttr("grid") || *invocation->get("grid") != attributeJSON(impl->getAttr("grid")) ||
      invocation->getInteger("num_warps") != 4 || invocation->getInteger("num_ctas") != 1 ||
      invocation->getInteger("shared_bytes") != 0 || !invocation->get("block") ||
      *invocation->get("block") != llvm::json::Value(llvm::json::Array{128, 1, 1}))
    return error("boundary launch grid does not cover exactly the logical tensor");
  auto imported = importLLVMIR(*payload, context, "triton.boundary.ll");
  if (!imported) return imported.takeError();
  auto &m = **imported;
  auto *entry = m.getFunction(*entryName);
  if (!entry || entry->arg_size() != 4 || entry->size() != 1 ||
      entry->getCallingConv() != llvm::CallingConv::PTX_Kernel || !entry->getReturnType()->isVoidTy() ||
      entry->isVarArg() || !entry->hasFnAttribute("nvvm.reqntid") ||
      entry->getFnAttribute("nvvm.reqntid").getValueAsString() != "128")
    return error("unsupported external LLVM entry ABI/control flow");
  for (auto &arg : entry->args())
    if (!arg.getType()->isPointerTy() || arg.getType()->getPointerAddressSpace() != 1 ||
        (arg.getArgNo() >= 2 && !arg.use_empty()))
      return error("unsupported data or scratch pointer ABI");
  for (auto &f : m)
    if (&f != entry && !f.isDeclaration()) return error("unexpected auxiliary external function body");
  if (target->getString("data_layout") != m.getDataLayoutStr())
    return error("LLVM data layout differs from artifact target");
  if (!m.getTargetTriple().empty() && m.getTargetTriple().str() != "nvptx64-nvidia-cuda")
    return error("external LLVM target is not nvptx64-nvidia-cuda");
  m.setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
  llvm::CallInst *boundaries[2] = {nullptr, nullptr};
  const char *asmText[] = {"// ASK_BOUNDARY_INPUT_V1\nmov.b32 $0, $1;",
                           "// ASK_BOUNDARY_OUTPUT_V1\nmov.b32 $0, $1;"};
  for (auto &instruction : entry->front()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
    if (!call || !call->isInlineAsm()) continue;
    auto *assembly = llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
    for (unsigned i = 0; i < 2; ++i) {
      if (assembly->getAsmString() != asmText[i]) continue;
      if (boundaries[i] || assembly->getConstraintString() != "=r,r,l,r,r" ||
          !assembly->hasSideEffects() || call->arg_size() != 4 || !call->getType()->isIntegerTy(32) ||
          !call->getArgOperand(0)->getType()->isIntegerTy(32) ||
          !call->getArgOperand(1)->getType()->isPointerTy() ||
          !call->getArgOperand(2)->getType()->isIntegerTy(32) ||
          !call->getArgOperand(3)->getType()->isIntegerTy(32))
        return error("malformed or repeated ASK boundary marker");
      boundaries[i] = call;
    }
  }
  if (!boundaries[0] || !boundaries[1]) return error("producer did not retain both ASK boundary markers");
  if (boundaries[0]->getArgOperand(2) != boundaries[1]->getArgOperand(2) ||
      boundaries[0]->getArgOperand(3) != boundaries[1]->getArgOperand(3))
    return error("input/output boundary index or mask differs");
  auto *mask = llvm::dyn_cast<llvm::ZExtInst>(boundaries[0]->getArgOperand(3));
  auto *predicate = mask ? llvm::dyn_cast<llvm::ICmpInst>(mask->getOperand(0)) : nullptr;
  auto *limit = predicate ? llvm::dyn_cast<llvm::ConstantInt>(predicate->getOperand(1)) : nullptr;
  if (!predicate || predicate->getPredicate() != llvm::ICmpInst::ICMP_SLT || !limit ||
      predicate->getOperand(0) != boundaries[0]->getArgOperand(2) || limit->getSExtValue() != n)
    return error("boundary mask must be the declared signed index < N predicate");
  auto *ports = artifact.getArray("port_map");
  if (!ports || ports->size() != 2) return error("missing port map");
  for (unsigned i = 0; i < 2; ++i) {
    llvm::StringRef binding = i == 0 ? "input:0" : "output:0";
    auto *boundary = contract->getObject(i == 0 ? "input" : "output");
    if (!boundary || boundary->getString("marker_asm") != asmText[i] ||
        boundary->getString("constraints") != "=r,r,l,r,r" ||
        boundary->getString("source_binding") != binding)
      return error("boundary marker definition differs from the supported protocol");
    llvm::Argument *arg = nullptr;
    for (auto &port : *ports) {
      auto *p = port.getAsObject();
      if (p && p->getString("logical_port") == binding) {
        auto index = p->getInteger("argument");
        if (!index || *index < 0 || *index > 1) return error("invalid data argument mapping");
        arg = entry->getArg(*index);
      }
    }
    if (!arg || llvm::getUnderlyingObject(boundaries[i]->getArgOperand(1)) != arg)
      return error("boundary address is not rooted at the declared data port");
    auto *address = llvm::dyn_cast<llvm::GetElementPtrInst>(boundaries[i]->getArgOperand(1));
    auto *offset = address && address->getNumIndices() == 1
        ? llvm::dyn_cast<llvm::SExtInst>(address->getOperand(1)) : nullptr;
    if (!address || !address->getSourceElementType()->isIntegerTy(32) ||
        address->getPointerOperand() != arg || !offset || !offset->getType()->isIntegerTy(64) ||
        offset->getOperand(0) != boundaries[i]->getArgOperand(2))
      return error("boundary address must be data pointer + signed i32 element index");
  }
  auto helpers = std::make_unique<llvm::Module>("ask.scalar.helpers", context);
  helpers->setTargetTriple(m.getTargetTriple());
  helpers->setDataLayout(m.getDataLayout());
  auto pre = helper(*helpers, "__ask_pre", external->getOperand(0), func.getArgument(0));
  if (!pre) return pre.takeError();
  auto ret = mlir::cast<mlir::func::ReturnOp>(func.front().getTerminator());
  auto post = helper(*helpers, "__ask_post", ret.getOperand(0), external->getResult(0));
  if (!post) return post.takeError();
  if (llvm::verifyModule(*helpers, &llvm::errs())) return error("invalid ASK helper LLVM IR");
  llvm::raw_string_ostream helperStream(helperIR);
  helpers->print(helperStream, nullptr);
  if (m.getFunction("__ask_pre") || m.getFunction("__ask_post")) return error("reserved helper symbol collision");
  if (llvm::Linker::linkModules(m, std::move(helpers))) return error("LLVM helper link failed");
  for (unsigned i = 0; i < 2; ++i) {
    auto *marker = boundaries[i];
    llvm::IRBuilder<> b(marker);
    auto *f = m.getFunction(i == 0 ? "__ask_pre" : "__ask_post");
    auto *call = b.CreateCall(f, {marker->getArgOperand(0)});
    llvm::Value *value = call;
    if (i == 0) {
      auto *valid = b.CreateICmpNE(marker->getArgOperand(3), b.getInt32(0));
      value = b.CreateSelect(valid, call, marker->getArgOperand(0));
    }
    marker->replaceAllUsesWith(value);
    marker->eraseFromParent();
    llvm::InlineFunctionInfo info;
    if (!llvm::InlineFunction(*call, info).isSuccess()) return error("ASK helper inlining failed");
    if (!f->use_empty()) return error("ASK helper unexpectedly retained users");
    f->eraseFromParent();
  }
  if (auto e = optimizeLLVMIR(m, report)) return std::move(e);
  if (llvm::verifyModule(m, &llvm::errs())) return error("fused LLVM IR failed verification");
  report["fusion"] = llvm::json::Object{{"protocol", "ask.triton.boundary.v1"},
      {"external_op", external->getAttrOfType<mlir::StringAttr>("ask.id").getValue()},
      {"entry", *entryName}, {"inlined_helpers", 2}, {"removed_boundary_markers", 2},
      {"original_llir_sha256", digest(*payload)}, {"ask_helpers_sha256", digest(helperIR)},
      {"launch_policy", "preserved"}, {"input_mask_other", "preserved"},
      {"llvm_verified", true}, {"device_execution_performed", false}};
  return std::move(*imported);
}
} // namespace ask::llir
