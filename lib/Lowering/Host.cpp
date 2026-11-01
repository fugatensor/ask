#include "ask/Lowering/Host.h"
#include "ask/BuildConfig.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Numeric.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>

namespace ask {
bool hostSupportsBF16(llvm::StringRef architecture, const llvm::StringMap<bool> &features) {
  auto enabled = [&](llvm::StringRef feature) {
    auto it = features.find(feature);
    return it != features.end() && it->second;
  };
  // AMX-BF16 alone supplies matrix dot products, not the scalar/vector format
  // conversions used by this backend. Do not infer support from its name.
  if (architecture == "x86_64" || architecture == "i386")
    return enabled("avx512bf16") || enabled("avxneconvert");
  if (architecture == "aarch64" || architecture == "aarch64_be")
    return enabled("bf16");
  return false;
}
bool nativeHostSupportsBF16() {
  llvm::Triple triple(llvm::sys::getProcessTriple());
  return hostSupportsBF16(triple.getArchName(), llvm::sys::getHostCPUFeatures());
}
namespace {
llvm::Error error(llvm::Twine message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message.str());
}

llvm::Expected<std::unique_ptr<llvm::TargetMachine>> makeHostMachine() {
  static std::once_flag initialized;
  static bool unavailable = false;
  std::call_once(initialized, [] {
    unavailable = llvm::InitializeNativeTarget() ||
                  llvm::InitializeNativeTargetAsmPrinter();
  });
  if (unavailable)
    return error("LLVM was built without the native target or assembly printer");
  llvm::Triple triple(llvm::sys::getProcessTriple());
  std::string reason;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, reason);
  if (!target)
    return error("cannot create host target: " + reason);
  llvm::TargetOptions options;
  llvm::SubtargetFeatures features;
  for (const auto &feature : llvm::sys::getHostCPUFeatures())
    features.AddFeature(feature.getKey(), feature.getValue());
  auto cpu = llvm::sys::getHostCPUName();
  if (cpu.empty()) cpu = "generic";
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, cpu, features.getString(), options, llvm::Reloc::PIC_));
  if (!machine)
    return error("LLVM native target cannot create a target machine");
  return std::move(machine);
}

llvm::Expected<uint64_t> elementCount(mlir::Type type) {
  if (isSupportedElementType(type))
    return uint64_t(1);
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || !isSupportedValueType(type))
    return error("host lowering requires concrete ranked supported tensors or scalars");
  if (llvm::is_contained(tensor.getShape(), int64_t(0)))
    return uint64_t(0);
  uint64_t count = 1;
  const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) /
      elementByteWidth(tensor);
  for (int64_t dimension : tensor.getShape()) {
    if (dimension < 0 || uint64_t(dimension) > limit / count)
      return error("host tensor byte count overflows the supported size range");
    count *= uint64_t(dimension);
  }
  return count;
}

mlir::Type elementType(mlir::Type type) {
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

llvm::Type *llvmElementType(mlir::Type type, llvm::LLVMContext &context) {
  return llir::scalarLLVMType(type, context);
}
llvm::Type *llvmStorageType(mlir::Type type, llvm::LLVMContext &context) {
  auto *element = llvmElementType(type, context);
  return element->isIntegerTy(1) ? llvm::Type::getInt8Ty(context) : element;
}
bool sameShape(mlir::Type lhs, mlir::Type rhs) {
  auto a = mlir::dyn_cast<mlir::RankedTensorType>(lhs);
  auto b = mlir::dyn_cast<mlir::RankedTensorType>(rhs);
  return a && b && a.getShape() == b.getShape();
}

std::string valueIdentity(mlir::Value value, mlir::func::FuncOp graph) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (auto id = graph.getArgAttrOfType<mlir::StringAttr>(argument.getArgNumber(), "ask.value_id"))
      return id.getValue().str();
    return (graph.getSymName() + ".arg" + llvm::Twine(argument.getArgNumber())).str();
  }
  auto result = mlir::cast<mlir::OpResult>(value);
  mlir::Operation *producer = result.getOwner();
  if (auto id = producer->getAttrOfType<mlir::StringAttr>("ask.value_id"))
    return id.getValue().str();
  auto id = producer->getAttrOfType<mlir::StringAttr>("ask.id");
  return (id.getValue() + ".result" + llvm::Twine(result.getResultNumber())).str();
}

llvm::json::Object describeArgument(mlir::Value value, mlir::func::FuncOp graph,
                                   unsigned argument) {
  mlir::Type type = value.getType();
  llvm::json::Array shape;
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (tensor)
    for (int64_t dimension : tensor.getShape())
      shape.push_back(dimension);
  return llvm::json::Object{{"argument", int64_t(argument)},
                            {"value_id", valueIdentity(value, graph)},
                            {"kind", tensor ? "tensor" : "scalar"},
                            {"dtype", elementTypeName(type)},
                            {"element_bytes", int64_t(elementByteWidth(type))},
                            {"minimum_alignment", int64_t(elementByteWidth(type))},
                            {"shape", std::move(shape)}};
}

llvm::Error validateFusionGroups(mlir::func::FuncOp function) {
  struct Group {
    mlir::Type type;
    unsigned operations = 0;
  };
  llvm::StringMap<Group> groups;
  llvm::StringSet<> closedGroups;
  std::string active;
  for (mlir::Operation &op : function.front()) {
    mlir::Attribute attribute = op.getAttr("ask.fusion_group");
    if (!attribute) {
      // Pure scalar arithmetic can occur between tensor operations in a group.
      // Any other tensor computation ends the current contiguous tensor group.
      if (llvm::any_of(op.getResultTypes(), [](mlir::Type type) {
            return mlir::isa<mlir::RankedTensorType>(type);
          })) {
        if (!active.empty()) closedGroups.insert(active);
        active.clear();
      }
      continue;
    }
    auto name = mlir::dyn_cast<mlir::StringAttr>(attribute);
    auto tensor = op.getNumResults() == 1
                      ? mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType())
                      : mlir::RankedTensorType();
    if (!name || name.getValue().empty() || !tensor || !tensor.hasStaticShape() ||
        !mlir::isa<SplatOp, AddOp, MulOp, CastOp>(op))
      return error("ask.fusion_group requires a nonempty string on a static tensor splat/add/mul/cast");
    llvm::StringRef id = name.getValue();
    if (active != id) {
      if (!active.empty()) closedGroups.insert(active);
      if (closedGroups.contains(id))
        return error("ask.fusion_group cannot resume across another tensor group or ungrouped tensor operation");
      active = id.str();
    }
    Group &group = groups[id];
    if (group.operations && !sameShape(group.type, tensor))
      return error("ask.fusion_group members must have exactly the same tensor shape");
    group.type = tensor;
    ++group.operations;
  }
  for (const auto &entry : groups)
    if (entry.second.operations < 2)
      return error("ask.fusion_group must contain at least two tensor operations");
  return llvm::Error::success();
}

llvm::Error validateFunction(mlir::func::FuncOp function, bool supportsBF16) {
  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()))
    return error("host lowering requires a defined single-block graph function");
  for (mlir::Type type : function.getArgumentTypes()) {
    if (elementType(type).isBF16() && !supportsBF16)
      return error("bf16 requires hardware support; this native CPU does not report a supported BF16 instruction feature");
    auto count = elementCount(type);
    if (!count)
      return count.takeError();
  }
  for (mlir::Operation &op : function.front()) {
    if (mlir::isa<ExternalOp>(op))
      return error("host lowering cannot compile ask.external; export its DSL through the selected compiler separately");
    if (!mlir::isa<ConstantOp, SplatOp, AddOp, MulOp, CastOp, mlir::func::ReturnOp>(op))
      return error("host lowering requires prepared constant/splat/add/mul/cast graphs; unsupported operation " + op.getName().getStringRef());
    for (mlir::Type type : op.getResultTypes()) {
      if (elementType(type).isBF16() && !supportsBF16)
        return error("bf16 requires hardware support; this native CPU does not report a supported BF16 instruction feature");
      auto count = elementCount(type);
      if (!count)
        return count.takeError();
    }
    if (op.getNumResults() && !op.getAttrOfType<mlir::StringAttr>("ask.value_id") &&
        !op.getAttrOfType<mlir::StringAttr>("ask.id"))
      return error("host lowering requires prepared result identities");
    if (mlir::isa<AddOp, MulOp>(op) &&
        (op.getOperand(0).getType() != op.getOperand(1).getType() ||
         op.getOperand(0).getType() != op.getResult(0).getType()))
      return error("host arithmetic requires exactly equal prepared operand/result types");
  }
  if (!mlir::isa<mlir::func::ReturnOp>(function.front().getTerminator()))
    return error("host graph requires func.return");
  return validateFusionGroups(function);
}
} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp source, llvm::LLVMContext &context,
                llvm::json::Object &abiReport) {
  if (!ASK_WITH_HOST)
    return error("host code generation was disabled at build time");
  if (mlir::failed(mlir::verify(source)))
    return error("cannot lower invalid ASK IR");
  auto target = makeHostMachine();
  if (!target)
    return target.takeError();
  const bool supportsBF16 = nativeHostSupportsBF16();
  auto result = std::make_unique<llvm::Module>("ask.host", context);
  result->setTargetTriple((*target)->getTargetTriple());
  result->setDataLayout((*target)->createDataLayout());
  auto *pointer = llvm::PointerType::getUnqual(context);
  llvm::json::Array functions;
  uint64_t totalLoops = 0, totalUnfusedLoops = 0, totalFusedOutputGroups = 0;
  uint64_t totalInputLoads = 0;

  for (mlir::Operation &op : source.getBody()->getOperations()) {
    if (mlir::isa<SemanticsOp, ImplementationOp>(op))
      continue;
    auto graph = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!graph)
      return error("unsupported module-level operation in host lowering");
    if (llvm::Error failure = validateFunction(graph, supportsBF16))
      return std::move(failure);

    llvm::SmallVector<llvm::Type *> parameters;
    for (mlir::Type type : graph.getArgumentTypes())
      parameters.push_back(mlir::isa<mlir::RankedTensorType>(type)
                               ? static_cast<llvm::Type *>(pointer)
                               : llvmStorageType(type, context));
    for (unsigned i = 0; i < graph.getNumResults(); ++i)
      parameters.push_back(pointer);
    auto *functionType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), parameters, false);
    auto *function = llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage,
                                             graph.getSymName(), result.get());
    function->setCallingConv(llvm::CallingConv::C);
    function->addFnAttr("target-cpu", (*target)->getTargetCPU());
    function->addFnAttr("target-features", (*target)->getTargetFeatureString());
    llvm::DenseMap<mlir::Value, llvm::Value *> arguments;
    llvm::json::Array inputs;
    llvm::json::Array outputs;
    for (auto [index, argument] : llvm::enumerate(graph.getArguments())) {
      llvm::Argument *physical = function->getArg(index);
      physical->setName("arg" + std::to_string(index));
      arguments[argument] = physical;
      if (physical->getType()->isPointerTy()) {
        physical->addAttr(llvm::Attribute::ReadOnly);
        physical->addAttr(llvm::Attribute::getWithCaptureInfo(context, llvm::CaptureInfo::none()));
      }
      inputs.push_back(describeArgument(argument, graph, index));
    }
    for (auto [index, type] : llvm::enumerate(graph.getResultTypes())) {
      unsigned physicalIndex = graph.getNumArguments() + index;
      auto *physical = function->getArg(physicalIndex);
      physical->setName("out" + std::to_string(index));
      physical->addAttr(llvm::Attribute::NoAlias);
      physical->addAttr(llvm::Attribute::getWithCaptureInfo(context, llvm::CaptureInfo::none()));
      physical->addAttr(llvm::Attribute::WriteOnly);
      auto returned = mlir::cast<mlir::func::ReturnOp>(graph.front().getTerminator());
      outputs.push_back(describeArgument(returned.getOperand(index), graph, physicalIndex));
    }

    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
    auto returned = mlir::cast<mlir::func::ReturnOp>(graph.front().getTerminator());
    struct OutputBatch {
      llvm::SmallVector<unsigned> outputs;
      mlir::Type type;
      std::string group;
      uint64_t count;
    };
    llvm::SmallVector<OutputBatch> batches;
    llvm::StringMap<unsigned> batchForGroup;
    uint64_t loopCount = 0, unfusedLoops = 0, fusedOutputGroups = 0;
    uint64_t inputLoads = 0, emptyOutputs = 0;
    for (auto [outputIndex, value] : llvm::enumerate(returned.getOperands())) {
      auto count = elementCount(value.getType());
      if (!count)
        return count.takeError();
      if (*count != 0) ++unfusedLoops;
      std::string group;
      if (mlir::isa<mlir::RankedTensorType>(value.getType()))
        if (auto *producer = value.getDefiningOp())
          if (auto name = producer->getAttrOfType<mlir::StringAttr>("ask.fusion_group"))
            group = name.getValue().str();
      if (!group.empty()) {
        auto previous = batchForGroup.find(group);
        if (previous != batchForGroup.end()) {
          OutputBatch &batch = batches[previous->second];
          if (!sameShape(batch.type, value.getType()) || batch.count != *count)
            return error("fused output roots require identical tensor shape");
          batch.outputs.push_back(outputIndex);
          continue;
        }
        batchForGroup[group] = batches.size();
      }
      batches.push_back(OutputBatch{{unsigned(outputIndex)}, value.getType(), group, *count});
    }

    llvm::json::Array fusedGroups;
    for (const OutputBatch &batch : batches) {
      if (batch.count == 0) {
        emptyOutputs += batch.outputs.size();
        continue; // No input or output pointer is dereferenced for empty results.
      }
      ++loopCount;
      bool shared = batch.outputs.size() > 1;
      if (shared) {
        ++fusedOutputGroups;
        llvm::json::Array outputIndices;
        for (unsigned outputIndex : batch.outputs) outputIndices.push_back(int64_t(outputIndex));
        fusedGroups.push_back(llvm::json::Object{
            {"group", batch.group}, {"outputs", std::move(outputIndices)},
            {"element_count", int64_t(batch.count)}});
      }

      llvm::BasicBlock *before = builder.GetInsertBlock();
      llvm::BasicBlock *body = llvm::BasicBlock::Create(
          context, shared ? "fused.output.loop" : "output.loop", function);
      llvm::BasicBlock *after = llvm::BasicBlock::Create(context, "output.done", function);
      builder.CreateBr(body);
      builder.SetInsertPoint(body);
      auto *index = builder.CreatePHI(builder.getInt64Ty(), 2, "i");
      index->addIncoming(builder.getInt64(0), before);
      llvm::DenseMap<mlir::Value, llvm::Value *> emitted;
      std::function<llvm::Value *(mlir::Value)> emit = [&](mlir::Value current) -> llvm::Value * {
        auto cached = emitted.find(current);
        if (cached != emitted.end())
          return cached->second;
        llvm::Value *computed = nullptr;
        auto argument = arguments.find(current);
        if (argument != arguments.end()) {
          if (mlir::isa<mlir::RankedTensorType>(current.getType())) {
            auto *element = llvmStorageType(current.getType(), context);
            auto *address = builder.CreateGEP(element, argument->second, index, "input.address");
            computed = builder.CreateAlignedLoad(element, address, llvm::Align(elementByteWidth(current.getType())), "input");
            ++inputLoads;
          } else {
            computed = argument->second;
          }
          if (elementType(current.getType()).isInteger(1))
            computed = builder.CreateICmpNE(computed, builder.getInt8(0), "input.bool");
        } else {
          mlir::Operation *producer = current.getDefiningOp();
          if (auto constant = mlir::dyn_cast<ConstantOp>(producer)) {
            if (auto value = constant->getAttrOfType<mlir::IntegerAttr>("value"))
              computed = llvm::ConstantInt::get(context, value.getValue());
            else
              computed = llvm::ConstantFP::get(
                  context, constant->getAttrOfType<mlir::FloatAttr>("value").getValue());
          } else if (mlir::isa<SplatOp>(producer)) {
            computed = emit(producer->getOperand(0));
          } else if (mlir::isa<CastOp>(producer)) {
            computed = llir::createNumericCast(builder, emit(producer->getOperand(0)),
                                               llvmElementType(current.getType(), context));
          } else {
            llvm::Value *lhs = emit(producer->getOperand(0));
            llvm::Value *rhs = emit(producer->getOperand(1));
            if (lhs->getType()->isFloatingPointTy()) {
              // No reassociation or contraction: each ASK operation rounds
              // independently at its declared floating-point width.
              bool bf16 = lhs->getType()->isBFloatTy();
              if (bf16) {
                lhs = llir::extendBF16(builder, lhs);
                rhs = llir::extendBF16(builder, rhs);
              }
              computed = mlir::isa<AddOp>(producer) ? builder.CreateFAdd(lhs, rhs, "add")
                                                   : builder.CreateFMul(lhs, rhs, "mul");
              if (bf16) computed = llir::truncateBF16(builder, computed);
            } else {
              computed = mlir::isa<AddOp>(producer) ? builder.CreateAdd(lhs, rhs, "add.wrap")
                                                   : builder.CreateMul(lhs, rhs, "mul.wrap");
            }
          }
        }
        emitted[current] = computed;
        return computed;
      };
      // The memo is shared only within this iteration of the same-shape batch.
      // Every public output still gets its own store under the existing noalias
      // contract, including repeated returns of the same logical value.
      for (unsigned outputIndex : batch.outputs) {
        llvm::Value *computed = emit(returned.getOperand(outputIndex));
        if (computed->getType()->isIntegerTy(1))
          computed = builder.CreateZExt(computed, builder.getInt8Ty(), "output.bool");
        llvm::Value *output = function->getArg(graph.getNumArguments() + outputIndex);
        auto *address = builder.CreateGEP(computed->getType(), output, index, "output.address");
        builder.CreateAlignedStore(computed, address, llvm::Align(computed->getType()->getScalarSizeInBits() / 8));
      }
      llvm::Value *next = builder.CreateAdd(index, builder.getInt64(1), "next");
      index->addIncoming(next, body);
      builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(batch.count)), body, after);
      builder.SetInsertPoint(after);
    }
    builder.CreateRetVoid();
    unsigned bufferAlignment = 1;
    for (auto type : graph.getArgumentTypes())
      bufferAlignment = std::max(bufferAlignment, elementByteWidth(type));
    for (auto type : graph.getResultTypes())
      bufferAlignment = std::max(bufferAlignment, elementByteWidth(type));
    functions.push_back(llvm::json::Object{
        {"name", graph.getSymName().str()}, {"calling_convention", "c"},
        {"inputs", std::move(inputs)}, {"outputs", std::move(outputs)},
        {"layout", "contiguous_row_major"}, {"minimum_alignment", int64_t(bufferAlignment)},
        {"requires_disjoint_outputs", true}, {"outputs_must_not_alias_inputs", true},
        {"lowering", llvm::json::Object{
            {"statistics_stage", "host_lowering"},
            {"loop_count", int64_t(loopCount)},
            {"unfused_output_loop_count", int64_t(unfusedLoops)},
            {"fused_output_groups", int64_t(fusedOutputGroups)},
            {"input_load_count", int64_t(inputLoads)},
            {"zero_element_outputs", int64_t(emptyOutputs)},
            {"groups", std::move(fusedGroups)}}}});
    totalLoops += loopCount;
    totalUnfusedLoops += unfusedLoops;
    totalFusedOutputGroups += fusedOutputGroups;
    totalInputLoads += inputLoads;
  }
  if (functions.empty())
    return error("host lowering requires at least one graph function");
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  if (llvm::verifyModule(*result, &stream))
    return error("host lowering produced invalid LLVM IR: " + stream.str());
  abiReport = llvm::json::Object{{"schema_version", 1},
                                {"target_triple", result->getTargetTriple().str()},
                                {"data_layout", result->getDataLayoutStr()},
                                {"target_cpu", (*target)->getTargetCPU().str()},
                                {"target_features", (*target)->getTargetFeatureString().str()},
                                {"bf16_hardware_supported", supportsBF16},
                                {"bool_storage", "one-byte-zero-or-one"},
                                {"cast_semantics", "rne-saturating-fptosi-nan-zero-v1"},
                                {"functions", std::move(functions)},
                                {"lowering", llvm::json::Object{
                                    {"statistics_stage", "host_lowering"},
                                    {"loop_count", int64_t(totalLoops)},
                                    {"unfused_output_loop_count", int64_t(totalUnfusedLoops)},
                                    {"fused_output_groups", int64_t(totalFusedOutputGroups)},
                                    {"input_load_count", int64_t(totalInputLoads)}}}};
  std::string abiText;
  llvm::raw_string_ostream abiStream(abiText);
  abiStream << llvm::json::Value(llvm::json::Object(abiReport));
  result->getOrInsertNamedMetadata("ask.abi")->addOperand(
      llvm::MDNode::get(context, llvm::MDString::get(context, abiText)));
  return std::move(result);
}

llvm::Error emitHostObject(llvm::Module &module, llvm::StringRef path) {
  if (!ASK_WITH_HOST)
    return error("host code generation was disabled at build time");
  auto target = makeHostMachine();
  if (!target)
    return target.takeError();
  if (module.getTargetTriple() != (*target)->getTargetTriple() ||
      module.getDataLayout() != (*target)->createDataLayout())
    return error("host object emission requires the native triple and matching DataLayout; retargeting imported IR is not supported");
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  if (llvm::verifyModule(module, &stream))
    return error("cannot emit an object for invalid LLVM IR: " + stream.str());
  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return llvm::errorCodeToError(ec);
  llvm::legacy::PassManager passes;
  if ((*target)->addPassesToEmitFile(passes, output, nullptr, llvm::CodeGenFileType::ObjectFile))
    return error("LLVM native target does not support object emission");
  passes.run(module);
  output.flush();
  if (output.has_error())
    return error("failed while writing the host object file");
  return llvm::Error::success();
}
} // namespace ask
