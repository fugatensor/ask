#include "ask/LLIR/Fuse.h"
#include "ask/IR/ASKOps.h"
#include "ask/Adapters/TritonBoundary.h"
#include "ask/LLIR/Inline.h"
#include "ask/LLIR/Import.h"
#include "ask/LLIR/Optimize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"

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
  if (auto boolean = mlir::dyn_cast<mlir::BoolAttr>(attr)) return boolean.getValue();
  if (auto n = mlir::dyn_cast<mlir::IntegerAttr>(attr)) return n.getInt();
  if (auto array = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    llvm::json::Array result;
    for (auto n : array.asArrayRef()) result.push_back(n);
    return result;
  }
  return nullptr;
}

// Both helpers are scalar, pure functions of the ASK element type. Indices, masks, memory
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
    auto attr = constant->getAttr("value");
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      result = b.getInt(integer.getValue());
    else if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attr))
      result = llvm::ConstantFP::get(b.getContext(), floating.getValue());
    else
      return error("unsupported ASK constant element type");
  } else if (mlir::isa<SplatOp>(op)) {
    auto scalar = emit(op->getOperand(0), leaf, argument, b, memo);
    if (!scalar) return scalar.takeError();
    result = *scalar;
  } else if (mlir::isa<AddOp, MulOp>(op)) {
    auto lhs = emit(op->getOperand(0), leaf, argument, b, memo);
    if (!lhs) return lhs.takeError();
    auto rhs = emit(op->getOperand(1), leaf, argument, b, memo);
    if (!rhs) return rhs.takeError();
    if ((*lhs)->getType()->isFloatingPointTy())
      result = mlir::isa<AddOp>(op) ? b.CreateFAdd(*lhs, *rhs) : b.CreateFMul(*lhs, *rhs);
    else
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
  auto elementType = leaf.getType();
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(elementType))
    elementType = tensor.getElementType();
  llvm::Type *scalarType;
  if (elementType.isSignlessInteger(32)) scalarType = b.getInt32Ty();
  else if (elementType.isF32()) scalarType = b.getFloatTy();
  else if (elementType.isF16()) scalarType = b.getHalfTy();
  else if (elementType.isBF16()) scalarType = b.getBFloatTy();
  else return error("unsupported ASK helper element type");
  auto *f = llvm::Function::Create(llvm::FunctionType::get(scalarType, {scalarType}, false),
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
  if (!type || !type.hasStaticShape() || !isSupportedElementType(type.getElementType()))
    return error("ASK graph lowering requires a static i32, f32, f16 or bf16 tensor");
  auto elementType = type.getElementType();
  llvm::StringRef dtype = elementType.isF32() ? "f32" : elementType.isF16() ? "f16" :
                          elementType.isBF16() ? "bf16" : "i32";
  int64_t n = 1;
  for (int64_t dim : type.getShape()) {
    if (dim == 0 || dim > INT64_MAX / n)
      return error("fusion requires a nonempty tensor with a representable element count");
    n *= dim;
  }
  if (func.getResultTypes()[0] != type)
    return error("fusion requires matching pointwise input/output shapes");
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
  if (!contract || !payload || !entryName || !producer || !specializations)
    return error("missing external fusion artifact / boundary contract fields");
  if (contract->getString("dtype") != dtype)
    return error("external boundary dtype does not match the ASK graph element type");
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
  if (!shape || shape->size() != 1 || (*shape)[0].getAsInteger() != n)
    return error("boundary shape/constexpr element count does not match the prepared graph");
  auto *invocation = artifact.getObject("invocation");
  if (!invocation || !invocation->get("grid") || !impl->getAttr("grid") ||
      *invocation->get("grid") != attributeJSON(impl->getAttr("grid")))
    return error("artifact launch grid does not match implementation");
  auto imported = importLLVMIR(*payload, context, "external.boundary.ll");
  if (!imported) return imported.takeError();
  auto &m = **imported;
  // Dispatch is producer-specific; native LLVM transformation is independent.
  auto boundaries = adapters::bindTritonBoundaries(m, artifact);
  if (!boundaries) return boundaries.takeError();
  // The producer's backend settings are not always encoded in its LLIR. Make
  // F's contraction permission local before adding ASK's separately rounded ops.
  auto numerics = adapters::normalizeTritonNumerics(m, artifact);
  if (!numerics) return numerics.takeError();
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
  llvm::SmallVector<BoundaryInlineSite> sites;
  for (auto *marker : boundaries->inputs) {
    llvm::IRBuilder<> b(marker);
    auto *valid = b.CreateICmpNE(marker->getArgOperand(boundaries->maskArgument), b.getInt32(0));
    auto *value = marker->getArgOperand(boundaries->valueArgument);
    sites.push_back({marker, *pre, {value}, valid, value});
  }
  for (auto *marker : boundaries->outputs)
    sites.push_back({marker, *post, {marker->getArgOperand(boundaries->valueArgument)}, nullptr, nullptr});
  auto fused = inlineValueBoundaries(m, *helpers, sites, report);
  if (!fused) return fused.takeError();
  if (auto e = optimizeLLVMIR(**fused, report)) return std::move(e);
  if (llvm::verifyModule(**fused, &llvm::errs())) return error("fused LLVM IR failed verification");
  report["fusion"] = llvm::json::Object{{"protocol", "ask.triton.boundary.v1"},
      {"external_op", external->getAttrOfType<mlir::StringAttr>("ask.id").getValue().str()},
      {"entry", entryName->str()}, {"inlined_helpers", static_cast<int64_t>(sites.size())},
      {"dtype", dtype},
      {"ask_numerics", dtype == "i32" ? std::string("i32-wrap") :
                       (dtype + "-rne-no-fast-math-no-contract").str()},
      {"producer_numerics", std::move(*numerics)},
      {"removed_boundary_markers", static_cast<int64_t>(sites.size())},
      {"helper_functions", 2}, {"boundary_inliner", "ask.llvm.value-boundary.v1"},
      {"original_llir_sha256", digest(*payload)}, {"ask_helpers_sha256", digest(helperIR)},
      {"launch_policy", "preserved"}, {"input_mask_other", "preserved"},
      {"llvm_verified", true}, {"device_execution_performed", false}};
  return std::move(*fused);
}
} // namespace ask::llir
