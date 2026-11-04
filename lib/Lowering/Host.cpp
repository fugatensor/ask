#include "ask/Lowering/Host.h"
#include "ask/BuildConfig.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Numeric.h"
#include "ScalarEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
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
  options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
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
        (!isPureComputeOperation(&op) || mlir::isa<MatmulOp>(op)))
      return error("ask.fusion_group requires a nonempty string on a static tensor computation");
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

struct WorkspacePlan {
  llvm::DenseSet<mlir::Value> live, materialized;
};

// Inputs and results have already passed the static byte-size checks.
uint64_t concreteCount(mlir::Type type) {
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (llvm::is_contained(tensor.getShape(), int64_t(0))) return 0;
    return uint64_t(tensor.getNumElements());
  }
  return 1;
}

WorkspacePlan planWorkspace(mlir::func::FuncOp graph) {
  WorkspacePlan plan;
  std::function<void(mlir::Value)> visit = [&](mlir::Value value) {
    if (!plan.live.insert(value).second) return;
    if (concreteCount(value.getType()) == 0) return;
    if (auto *owner = value.getDefiningOp())
      for (auto operand : owner->getOperands()) visit(operand);
  };
  for (auto value : mlir::cast<mlir::func::ReturnOp>(graph.front().getTerminator()).getOperands()) visit(value);
  llvm::DenseMap<mlir::Value, bool> expensive;
  std::function<bool(mlir::Value)> hasReduction = [&](mlir::Value value) {
    if (!mlir::isa<mlir::RankedTensorType>(value.getType()) ||
        concreteCount(value.getType()) == 0 || plan.materialized.contains(value)) return false;
    auto found = expensive.find(value);
    if (found != expensive.end()) return found->second;
    auto *owner = value.getDefiningOp();
    bool result = owner && ((mlir::isa<ReduceSumOp, ReduceMaxOp>(owner) &&
        concreteCount(owner->getOperand(0).getType()) > concreteCount(value.getType())) ||
        llvm::any_of(owner->getOperands(), hasReduction));
    expensive[value] = result;
    return result;
  };
  for (auto &op : graph.front()) {
    if (!op.getNumResults() || !plan.live.contains(op.getResult(0)) ||
        concreteCount(op.getResult(0).getType()) == 0) continue;
    if (mlir::isa<MatmulOp>(op) && llvm::any_of(op.getResult(0).getUsers(),
        [&](mlir::Operation *user) {
          return user->getNumResults() && plan.live.contains(user->getResult(0));
        })) {
      plan.materialized.insert(op.getResult(0));
      expensive.clear();
    }
    for (auto operand : op.getOperands())
      if ((mlir::isa<MatmulOp>(op) || concreteCount(op.getResult(0).getType()) > concreteCount(operand.getType())) &&
          hasReduction(operand)) {
        plan.materialized.insert(operand);
        expensive.clear();
      }
  }
  return plan;
}

template<class Emit>
void emitHostLoop(llvm::IRBuilder<> &builder, llvm::Function *function,
                  uint64_t count, uint64_t step, Emit emit) {
  if (!count) return;
  if (count == step) { emit(builder.getInt64(0));return; }
  auto *before = builder.GetInsertBlock();
  auto *body = llvm::BasicBlock::Create(builder.getContext(), "tile.loop", function);
  auto *after = llvm::BasicBlock::Create(builder.getContext(), "tile.done", function);
  builder.CreateBr(body);
  builder.SetInsertPoint(body);
  auto *index = builder.CreatePHI(builder.getInt64Ty(), 2, "tile.index");
  index->addIncoming(builder.getInt64(0), before);
  emit(index);
  auto *next = builder.CreateAdd(index, builder.getInt64(step));
  index->addIncoming(next, builder.GetInsertBlock());
  builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(count)), body, after);
  builder.SetInsertPoint(after);
}

struct WorkspaceBuffer {
  mlir::Value value;
  llvm::GetElementPtrInst *address;
  uint64_t bytes;
  unsigned alignment;
  std::size_t first, last, slot = 0;
};
struct WorkspaceSlot {
  uint64_t bytes, offset = 0;
  unsigned alignment;
  std::size_t last;
};

llvm::Error validateWorkspace(llvm::ArrayRef<WorkspaceBuffer> buffers,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> phases,
    llvm::ArrayRef<WorkspaceSlot> slots, uint64_t bytes) {
  llvm::DenseMap<mlir::Value, const WorkspaceBuffer *> definitions;
  for (const auto &buffer : buffers) {
    if (!definitions.try_emplace(buffer.value, &buffer).second ||
        buffer.slot >= slots.size() || buffer.first > buffer.last || buffer.last >= phases.size())
      return error("invalid host workspace definition or lifetime");
    const auto &slot = slots[buffer.slot];
    if (buffer.bytes > slot.bytes || slot.offset % buffer.alignment ||
        slot.offset > bytes || slot.bytes > bytes - slot.offset)
      return error("host workspace slot violates capacity or alignment");
  }
  for (auto [phase, reads] : llvm::enumerate(phases))
    for (auto value : reads) {
      auto found = definitions.find(value);
      if (found == definitions.end() || found->second->first >= phase || found->second->last < phase)
        return error("host workspace read is outside its produced lifetime");
    }
  for (auto [index, slot] : llvm::enumerate(slots))
    if (index && slots[index - 1].offset + slots[index - 1].bytes > slot.offset)
      return error("host workspace slots overlap");
  llvm::SmallVector<const WorkspaceBuffer *> previous(slots.size(), nullptr);
  for (const auto &buffer : buffers) {
    auto *prior = previous[buffer.slot];
    if (prior && prior->last >= buffer.first)
      return error("host workspace aliases live materializations");
    previous[buffer.slot] = &buffer;
  }
  return llvm::Error::success();
}

llvm::Expected<uint64_t> assignWorkspace(llvm::MutableArrayRef<WorkspaceBuffer> buffers,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> phases, uint64_t limit) {
  llvm::DenseMap<mlir::Value, std::size_t> indices;
  for (auto [index, buffer] : llvm::enumerate(buffers)) indices[buffer.value] = index;
  for (auto [phase, reads] : llvm::enumerate(phases))
    for (auto value : reads) {
      auto found = indices.find(value);
      if (found == indices.end()) return error("host workspace read lacks a materialization");
      buffers[found->second].last = phase;
    }

  llvm::SmallVector<WorkspaceSlot> slots;
  for (auto &buffer : buffers) {
    std::size_t selected = slots.size();
    uint64_t growth = buffer.bytes, capacity = std::numeric_limits<uint64_t>::max();
    for (auto [index, slot] : llvm::enumerate(slots)) {
      if (slot.last >= buffer.first) continue;
      uint64_t added = buffer.bytes > slot.bytes ? buffer.bytes - slot.bytes : 0;
      if (added < growth || (added == growth && slot.bytes < capacity)) {
        selected = index;
        growth = added;
        capacity = slot.bytes;
      }
    }
    buffer.slot = selected;
    if (selected == slots.size()) slots.push_back({buffer.bytes, 0, buffer.alignment, buffer.last});
    else {
      auto &slot = slots[selected];
      slot.bytes = std::max(slot.bytes, buffer.bytes);
      slot.alignment = std::max(slot.alignment, buffer.alignment);
      slot.last = buffer.last;
    }
  }
  uint64_t bytes = 0;
  for (auto &slot : slots) {
    const uint64_t padding = slot.alignment - 1;
    if (bytes > limit - padding) return error("host reduction workspace size overflows the address range");
    slot.offset = (bytes + padding) & ~padding;
    if (slot.bytes > limit - slot.offset) return error("host reduction workspace size overflows the address range");
    bytes = slot.offset + slot.bytes;
  }
  if (auto failure = validateWorkspace(buffers, phases, slots, bytes)) return std::move(failure);
  for (const auto &buffer : buffers)
    buffer.address->setOperand(1, llvm::ConstantInt::get(
        buffer.address->getOperand(1)->getType(), slots[buffer.slot].offset));
  return bytes;
}

llvm::Error validateFunction(mlir::func::FuncOp function) {
  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()))
    return error("host lowering requires a defined single-block graph function");
  for (mlir::Type type : function.getArgumentTypes()) {
    auto count = elementCount(type);
    if (!count)
      return count.takeError();
  }
  for (mlir::Operation &op : function.front()) {
    if (mlir::isa<ExternalOp>(op))
      return error("host lowering cannot compile ask.external; export its DSL through the selected compiler separately");
    if (!isPureComputeOperation(&op) && !mlir::isa<mlir::func::ReturnOp>(op))
      return error("unsupported host graph operation " + op.getName().getStringRef());
    for (mlir::Type type : op.getResultTypes()) {
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

llvm::Error validateHostElementTypes(mlir::ModuleOp module) {
  auto supported = [](mlir::Type type) {
    type = elementType(type);
    return type.isSignlessInteger(1) || type.isSignlessInteger(32) ||
           type.isSignlessInteger(64) || type.isF32() || type.isF16();
  };
  auto result = module.walk([&](mlir::Operation *op) {
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op))
      if (!llvm::all_of(function.getArgumentTypes(), supported) ||
          !llvm::all_of(function.getResultTypes(), supported))
        return mlir::WalkResult::interrupt();
    return llvm::all_of(op->getResultTypes(), supported)
        ? mlir::WalkResult::advance() : mlir::WalkResult::interrupt();
  });
  return result.wasInterrupted() ? error("Host backend supports only bool/i32/i64/f32/f16")
                                 : llvm::Error::success();
}

llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp source, llvm::LLVMContext &context,
                llvm::json::Object &abiReport) {
  if (!ASK_WITH_HOST)
    return error("host code generation was disabled at build time");
  if (mlir::failed(mlir::verify(source)))
    return error("cannot lower invalid ASK IR");
  if (auto failure = validateHostElementTypes(source)) return std::move(failure);
  auto target = makeHostMachine();
  if (!target)
    return target.takeError();
  auto result = std::make_unique<llvm::Module>("ask.host", context);
  result->setTargetTriple((*target)->getTargetTriple());
  result->setDataLayout((*target)->createDataLayout());
  auto *pointer = llvm::PointerType::getUnqual(context);
  llvm::json::Array functions;
  uint64_t totalLoops = 0, totalUnfusedLoops = 0, totalFusedOutputGroups = 0;
  uint64_t totalInputLoads = 0, totalReductionLoops = 0;

  for (mlir::Operation &op : source.getBody()->getOperations()) {
    if (mlir::isa<SemanticsOp, ImplementationOp>(op))
      continue;
    auto graph = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!graph)
      return error("unsupported module-level operation in host lowering");
    if (llvm::Error failure = validateFunction(graph))
      return std::move(failure);
    const unsigned pointerBits = result->getDataLayout().getPointerSizeInBits();
    const uint64_t workspaceLimit = pointerBits < 64 ? (uint64_t(1) << pointerBits) - 1
        : uint64_t(std::numeric_limits<int64_t>::max());
    const auto workspacePlan = planWorkspace(graph);

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
    uint64_t inputLoads = 0, emptyOutputs = 0, reductionLoops = 0;
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
    llvm::CallInst *workspace = nullptr;
    llvm::SmallVector<WorkspaceBuffer> workspaceBuffers;
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> workspacePhases;
    if (llvm::any_of(workspacePlan.materialized, [](mlir::Value value) {
          return concreteCount(value.getType()) > 1;
        })) {
      auto *sizeType = builder.getIntPtrTy(result->getDataLayout());
      // Layout constants are filled after collecting the emitted phase reads.
      workspace = builder.CreateCall(result->getOrInsertFunction("malloc",
          llvm::FunctionType::get(pointer, {sizeType}, false)),
          {llvm::ConstantInt::get(sizeType, 0)}, "workspace");
      auto *ready = llvm::BasicBlock::Create(context, "workspace.ready", function);
      auto *failed = llvm::BasicBlock::Create(context, "workspace.failed", function);
      builder.CreateCondBr(builder.CreateICmpNE(workspace, llvm::ConstantPointerNull::get(pointer)), ready, failed);
      builder.SetInsertPoint(failed);
      auto *abort = builder.CreateCall(result->getOrInsertFunction("abort",
          llvm::FunctionType::get(builder.getVoidTy(), {}, false)));
      abort->setDoesNotReturn();
      builder.CreateUnreachable();
      builder.SetInsertPoint(ready);
    }
    llvm::DenseMap<mlir::Value, llvm::Value *> uniformValues, cachedBuffers;
    auto load = [&](mlir::Value value, llvm::Value *offset) -> llvm::Value * {
      llvm::Value *computed = arguments.lookup(value);
      if (mlir::isa<mlir::RankedTensorType>(value.getType())) {
        auto *type = llvmStorageType(value.getType(), context);
        auto *address = builder.CreateGEP(type, computed, offset, "input.address");
        computed = builder.CreateAlignedLoad(type, address,
            llvm::Align(elementByteWidth(value.getType())), "input");
        ++inputLoads;
      }
      if (elementType(value.getType()).isInteger(1))
        computed = builder.CreateICmpNE(computed, builder.getInt8(0), "input.bool");
      return computed;
    };
    auto bind = [&](lowering::ScalarEmitter &emitter) {
      for (const auto &entry : uniformValues) emitter.bindUniform(entry.first, entry.second);
      for (const auto &entry : cachedBuffers) emitter.bindBuffer(entry.first, entry.second);
    };
    auto emitMatmul = [&](mlir::Value value, llvm::Value *address) {
      auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
      const uint64_t columns = type.getShape().back();
      const uint64_t rows = type.getShape()[type.getRank() - 2];
      constexpr unsigned width = 16, height = 4;
      const uint64_t fullColumns = columns / width * width;
      const uint64_t fullRows = rows / height * height;
      llvm::SmallVector<mlir::Value> reads;
      emitHostLoop(builder, function, concreteCount(type) / (rows * columns), 1, [&](llvm::Value *batch) {
        auto *batchBase = builder.CreateMul(batch, builder.getInt64(rows * columns));
        auto rowTile = [&](llvm::Value *row, unsigned tileRows) {
          auto *base = builder.CreateAdd(batchBase, builder.CreateMul(row, builder.getInt64(columns)));
          auto tile = [&](llvm::Value *column, unsigned lanes) {
            auto *index = builder.CreateAdd(base, column);
            lowering::ScalarEmitter emitter(builder, function, false, load);
            bind(emitter);
            auto *computed = emitter.emitMatmulTile(value, index, lanes, tileRows);
            for (unsigned offset = 0; offset < tileRows; ++offset) {
              llvm::Value *rowValue = computed;
              if (tileRows > 1) {
                if (lanes == 1) rowValue = builder.CreateExtractElement(computed, offset);
                else {
                  llvm::SmallVector<int> mask;
                  for (unsigned lane = 0; lane < lanes; ++lane) mask.push_back(offset * lanes + lane);
                  rowValue = builder.CreateShuffleVector(computed, mask);
                }
              }
              auto *outputIndex = builder.CreateAdd(index, builder.getInt64(offset * columns));
              auto *destination = builder.CreateGEP(llvmStorageType(type, context), address, outputIndex);
              builder.CreateAlignedStore(rowValue, destination, llvm::Align(elementByteWidth(type)));
            }
            reductionLoops += emitter.reductionLoops;
            reads.append(emitter.bufferReads);
          };
          emitHostLoop(builder, function, fullColumns, width,
              [&](llvm::Value *column) { tile(column, width); });
          if (columns != fullColumns) tile(builder.getInt64(fullColumns), columns - fullColumns);
        };
        emitHostLoop(builder, function, fullRows, height,
            [&](llvm::Value *row) { rowTile(row, height); });
        if (rows != fullRows) rowTile(builder.getInt64(fullRows), rows - fullRows);
      });
      return reads;
    };
    for (auto &op : graph.front()) {
      if (!op.getNumResults()) continue;
      auto value = op.getResult(0);
      if (!workspacePlan.live.contains(value)) continue;
      const bool scalar = !mlir::isa<mlir::RankedTensorType>(value.getType());
      if (!scalar && !workspacePlan.materialized.contains(value)) continue;
      if (scalar || concreteCount(value.getType()) == 1) {
        lowering::ScalarEmitter emitter(builder, function, false, load);
        bind(emitter);
        uniformValues[value] = emitter.emit(value, builder.getInt64(0));
        reductionLoops += emitter.reductionLoops;
        workspacePhases.push_back(std::move(emitter.bufferReads));
        continue;
      }
      auto *address = llvm::GetElementPtrInst::Create(builder.getInt8Ty(), workspace,
          {builder.getInt64(0)}, "cached.buffer", builder.GetInsertBlock());
      if (mlir::isa<MatmulOp>(op)) {
        auto reads = emitMatmul(value, address);
        cachedBuffers[value] = address;
        const auto phase = workspacePhases.size();
        workspaceBuffers.push_back({value, address,
            concreteCount(value.getType()) * elementByteWidth(value.getType()),
            elementByteWidth(value.getType()), phase, phase});
        workspacePhases.push_back(std::move(reads));
        continue;
      }
      auto *before = builder.GetInsertBlock();
      auto *body = llvm::BasicBlock::Create(context, "materialize.loop", function);
      auto *after = llvm::BasicBlock::Create(context, "materialize.done", function);
      builder.CreateBr(body);
      builder.SetInsertPoint(body);
      auto *index = builder.CreatePHI(builder.getInt64Ty(), 2, "materialize.index");
      index->addIncoming(builder.getInt64(0), before);
      lowering::ScalarEmitter emitter(builder, function, false, load);
      bind(emitter);
      auto *computed = emitter.emit(value, index);
      if (computed->getType()->isIntegerTy(1)) computed = builder.CreateZExt(computed, builder.getInt8Ty());
      builder.CreateAlignedStore(computed, builder.CreateGEP(computed->getType(), address, index),
          llvm::Align(elementByteWidth(value.getType())));
      reductionLoops += emitter.reductionLoops;
      auto *next = builder.CreateAdd(index, builder.getInt64(1));
      index->addIncoming(next, builder.GetInsertBlock());
      builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(concreteCount(value.getType()))), body, after);
      builder.SetInsertPoint(after);
      cachedBuffers[value] = address;
      const auto phase = workspacePhases.size();
      workspaceBuffers.push_back({value, address,
          concreteCount(value.getType()) * elementByteWidth(value.getType()),
          elementByteWidth(value.getType()), phase, phase});
      workspacePhases.push_back(std::move(emitter.bufferReads));
    }
    for (const OutputBatch &batch : batches) {
      if (batch.count == 0) {
        emptyOutputs += batch.outputs.size();
        continue; // No input or output pointer is dereferenced for empty results.
      }
      ++loopCount;
      auto value = returned.getOperand(batch.outputs.front());
      if (value.getDefiningOp<MatmulOp>() && !cachedBuffers.contains(value) &&
          !uniformValues.contains(value)) {
        workspacePhases.push_back(emitMatmul(value,
            function->getArg(graph.getNumArguments() + batch.outputs.front())));
        continue;
      }
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
      lowering::ScalarEmitter emitter(builder, function, false, load);
      bind(emitter);
      // The memo is shared only within this iteration of the same-shape batch.
      // Every public output still gets its own store under the existing noalias
      // contract, including repeated returns of the same logical value.
      for (unsigned outputIndex : batch.outputs) {
        llvm::Value *computed = emitter.emit(returned.getOperand(outputIndex), index);
        if (computed->getType()->isIntegerTy(1))
          computed = builder.CreateZExt(computed, builder.getInt8Ty(), "output.bool");
        llvm::Value *output = function->getArg(graph.getNumArguments() + outputIndex);
        auto *address = builder.CreateGEP(computed->getType(), output, index, "output.address");
        builder.CreateAlignedStore(computed, address, llvm::Align(computed->getType()->getScalarSizeInBits() / 8));
      }
      reductionLoops += emitter.reductionLoops;
      llvm::Value *next = builder.CreateAdd(index, builder.getInt64(1), "next");
      index->addIncoming(next, builder.GetInsertBlock());
      builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(batch.count)), body, after);
      builder.SetInsertPoint(after);
      workspacePhases.push_back(std::move(emitter.bufferReads));
    }
    auto workspaceBytes = assignWorkspace(workspaceBuffers, workspacePhases, workspaceLimit);
    if (!workspaceBytes) return workspaceBytes.takeError();
    if (workspace)
      workspace->setArgOperand(0, llvm::ConstantInt::get(workspace->getArgOperand(0)->getType(), *workspaceBytes));
    if (workspace)
      builder.CreateCall(result->getOrInsertFunction("free",
          llvm::FunctionType::get(builder.getVoidTy(), {pointer}, false)), {workspace});
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
            {"reduction_loop_count", int64_t(reductionLoops)},
            {"temporary_workspace_bytes", int64_t(*workspaceBytes)},
            {"materialized_tensor_count", int64_t(cachedBuffers.size())},
            {"zero_element_outputs", int64_t(emptyOutputs)},
            {"groups", std::move(fusedGroups)}}}});
    totalLoops += loopCount;
    totalUnfusedLoops += unfusedLoops;
    totalFusedOutputGroups += fusedOutputGroups;
    totalInputLoads += inputLoads;
    totalReductionLoops += reductionLoops;
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
                                {"bool_storage", "one-byte-zero-or-one"},
                                {"cast_semantics", "rne-saturating-fptosi-nan-zero-v1"},
                                {"functions", std::move(functions)},
                                {"lowering", llvm::json::Object{
                                    {"statistics_stage", "host_lowering"},
                                    {"loop_count", int64_t(totalLoops)},
                                    {"unfused_output_loop_count", int64_t(totalUnfusedLoops)},
                                    {"fused_output_groups", int64_t(totalFusedOutputGroups)},
                                    {"input_load_count", int64_t(totalInputLoads)},
                                    {"reduction_loop_count", int64_t(totalReductionLoops)}}}};
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
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager calls;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder optimizer(target->get());
  optimizer.registerLoopAnalyses(loops);
  optimizer.registerFunctionAnalyses(functions);
  optimizer.registerCGSCCAnalyses(calls);
  optimizer.registerModuleAnalyses(modules);
  optimizer.crossRegisterProxies(loops, functions, calls, modules);
  auto pipeline = optimizer.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  pipeline.run(module, modules);
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
