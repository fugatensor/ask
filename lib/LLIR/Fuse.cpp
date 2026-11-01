#include "ask/LLIR/Fuse.h"
#include "ask/IR/ASKOps.h"
#include "ask/Frontend/Compiler.h"
#include "ask/LLIR/Inline.h"
#include "ask/LLIR/Import.h"
#include "ask/LLIR/Optimize.h"
#include "ask/LLIR/Numeric.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include <set>

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
  } else if (mlir::isa<CastOp>(op)) {
    auto operand = emit(op->getOperand(0), leaf, argument, b, memo);
    if (!operand) return operand.takeError();
    result = createNumericCast(b, *operand, scalarLLVMType(value.getType(), b.getContext()));
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
    return error("fusion supports only constant/splat/add/mul/cast around one external operation");
  }
  memo[value] = result;
  return result;
}

llvm::Expected<llvm::Function *> helper(llvm::Module &m, llvm::StringRef name,
                                       mlir::Value root, mlir::Value leaf) {
  llvm::IRBuilder<> b(m.getContext());
  auto *inputType = scalarLLVMType(leaf.getType(), b.getContext());
  auto *outputType = scalarLLVMType(root.getType(), b.getContext());
  auto *f = llvm::Function::Create(llvm::FunctionType::get(outputType, {inputType}, false),
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
             const adapters::CompilerAdapter &adapter, llvm::LLVMContext &context, llvm::json::Object &report, std::string &helperIR) {
  llvm::SmallVector<mlir::func::FuncOp> functions(graph.getOps<mlir::func::FuncOp>());
  if (functions.size() != 1) return error("fusion requires exactly one graph function");
  auto func = functions[0];
  if (func.getNumArguments() != 1 || func.getNumResults() != 1 || !func.getBody().hasOneBlock())
    return error("fusion requires one static tensor input and one output in one block");
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
  if (!type || !type.hasStaticShape() || !isSupportedElementType(type.getElementType()))
    return error("ASK graph lowering requires a static supported tensor");
  llvm::StringRef dtype = elementTypeName(type);
  std::set<std::string> requiredTypes{dtype.str()};
  int64_t n = 1;
  for (int64_t dim : type.getShape()) {
    if (dim == 0 || dim > INT64_MAX / n)
      return error("fusion requires a nonempty tensor with a representable element count");
    n *= dim;
  }
  if (func.getResultTypes()[0] != type)
    return error("fusion requires matching pointwise input/output shapes and storage types");
  ExternalOp external;
  for (auto &op : func.front()) {
    if (!mlir::isa<ConstantOp, SplatOp, AddOp, MulOp, CastOp, ExternalOp, mlir::func::ReturnOp>(op))
      return error("unsupported operation in fusion graph");
    for (auto result : op.getResults()) {
      requiredTypes.insert(elementTypeName(result.getType()).str());
      if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(result.getType()))
        if (tensor.getShape() != type.getShape())
          return error("fusion requires matching pointwise tensor shapes");
    }
    if (auto f = mlir::dyn_cast<ExternalOp>(op)) {
      if (external) return error("fusion requires exactly one external operation");
      external = f;
    }
  }
  if (!external || external->getNumOperands() != 1 || external->getNumResults() != 1)
    return error("fusion requires one external operation with one input and one output");
  if (external->getOperand(0).getType() != type || external->getResult(0).getType() != type)
    return error("fusion external boundaries must preserve graph storage types and shapes");
  if (external.resolveSemantics()->getAttrOfType<mlir::StringAttr>("effects").getValue() != "none")
    return error("fusion requires a pure external semantic contract");
  auto impl = external.resolveImplementation();
  auto compiler = impl->getAttrOfType<mlir::StringAttr>("compiler").getValue();
  if (compiler != adapter.id)
    return error("selected adapter does not match the graph implementation compiler");
  if (!adapter.prepareFusion)
    return error("selected compiler adapter does not support value-boundary fusion");
  auto payload = artifact.getString("payload");
  auto entryName = artifact.getString("entry");
  if (!payload || payload->empty() || !entryName || entryName->empty() ||
      artifact.getString("format") != "llvm-ir" ||
      artifact.getInteger("protocol_version") != adapter.protocolVersion)
    return error("missing or incompatible external LLVM artifact fields");
  auto requests = ask::externalRequests(graph, true);
  if (requests.size() != 1 || !requests[0].getAsObject())
    return error("fusion requires one prepared external request");
  const auto &request = *requests[0].getAsObject();
  if (adapter.validateArtifact)
    if (auto e = adapter.validateArtifact(artifact, request)) return std::move(e);
  if (auto *types = artifact.getArray("required_element_types"))
    for (const auto &element : *types) {
      auto name = element.getAsString();
      if (!name || name->empty()) return error("invalid external element type requirement");
      requiredTypes.insert(name->str());
    }
  auto imported = importLLVMIR(*payload, context, "external.boundary.ll");
  if (!imported) return imported.takeError();
  auto &m = **imported;
  auto *entry = m.getFunction(*entryName);
  if (!entry || entry->isDeclaration() || entry->isVarArg())
    return error("external LLVM entry must be a defined non-variadic function");
  // Producer-local BF16 casts also require hardware support, even when graph
  // storage and all ASK expressions are f32. Include operand types for void
  // instructions such as stores as well as computed result types.
  for (auto &function : m)
    for (auto &block : function)
      for (auto &instruction : block) {
        if (instruction.getType()->getScalarType()->isBFloatTy()) requiredTypes.insert("bf16");
        for (auto &operand : instruction.operands())
          if (operand->getType()->getScalarType()->isBFloatTy()) requiredTypes.insert("bf16");
      }
  auto boundaries = adapter.prepareFusion(m, artifact, {request, dtype, n});
  if (!boundaries) return boundaries.takeError();
  if (boundaries->protocol.empty() || boundaries->inputs.empty() || boundaries->outputs.empty())
    return error("compiler adapter did not provide complete input/output value boundaries");
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
  for (const auto &boundary : boundaries->inputs)
    sites.push_back({boundary.marker, *pre, {boundary.value}, boundary.predicate, boundary.passthrough});
  for (const auto &boundary : boundaries->outputs)
    sites.push_back({boundary.marker, *post, {boundary.value}, boundary.predicate, boundary.passthrough});
  auto fused = inlineValueBoundaries(m, *helpers, sites, report);
  if (!fused) return fused.takeError();
  if (auto e = optimizeLLVMIR(**fused, report)) return std::move(e);
  if (llvm::verifyModule(**fused, &llvm::errs())) return error("fused LLVM IR failed verification");
  llvm::json::Array computeTypes, requiredElementTypes;
  for (const auto &element : requiredTypes) {
    computeTypes.push_back(element);
    requiredElementTypes.push_back(element);
  }
  report["fusion"] = llvm::json::Object{{"protocol", boundaries->protocol}, {"compiler", adapter.id},
      {"external_op", external->getAttrOfType<mlir::StringAttr>("ask.id").getValue().str()},
      {"entry", entryName->str()}, {"inlined_helpers", static_cast<int64_t>(sites.size())},
      {"dtype", dtype},
      {"ask_numerics", dtype == "bool" ? std::string("bool-exact") :
                       (dtype == "i32" || dtype == "i64") ? (dtype + "-wrap").str() :
                       (dtype + "-rne-no-fast-math-no-contract").str()},
      {"compute_types", std::move(computeTypes)},
      {"required_element_types", std::move(requiredElementTypes)},
      {"cast_semantics", "rne-saturating-fptosi-nan-zero-v1"},
      {"producer_numerics", std::move(boundaries->producerNumerics)},
      {"codegen", std::move(boundaries->codegen)},
      {"removed_boundary_markers", static_cast<int64_t>(sites.size())},
      {"helper_functions", 2}, {"boundary_inliner", "ask.llvm.value-boundary.v1"},
      {"original_llir_sha256", digest(*payload)}, {"ask_helpers_sha256", digest(helperIR)},
      {"launch_policy", "preserved"}, {"input_predicate_passthrough", "preserved"},
      {"llvm_verified", true}, {"device_execution_performed", false}};
  return std::move(*fused);
}
} // namespace ask::llir
