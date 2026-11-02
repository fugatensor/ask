#include "ask/Lowering/CUDA.h"
#include "ask/BuildConfig.h"
#include "ask/IR/ASKOps.h"
#include "ask/LLIR/Numeric.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace ask {
namespace {
llvm::Error error(llvm::Twine message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message.str());
}

llvm::Expected<uint64_t> elementCount(mlir::Type type) {
  if (isSupportedElementType(type)) return uint64_t(1);
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || !isSupportedValueType(type))
    return error("CUDA lowering requires concrete ranked supported tensors or scalars");
  if (llvm::is_contained(tensor.getShape(), int64_t(0))) return uint64_t(0);
  uint64_t count = 1;
  const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / elementByteWidth(type);
  for (int64_t dimension : tensor.getShape()) {
    if (dimension < 0 || uint64_t(dimension) > limit / count)
      return error("CUDA tensor byte count overflows the supported size range");
    count *= uint64_t(dimension);
  }
  return count;
}

llvm::Type *storageType(mlir::Type type, llvm::LLVMContext &context) {
  auto *element = llir::scalarLLVMType(type, context);
  return element->isIntegerTy(1) ? llvm::Type::getInt8Ty(context) : element;
}

llvm::SmallVector<int64_t> shapeOf(mlir::Type type) {
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return {tensor.getShape().begin(), tensor.getShape().end()};
  return {};
}

llvm::json::Object describePort(mlir::Type type, unsigned argument, llvm::StringRef binding) {
  llvm::json::Array shape;
  for (int64_t dimension : shapeOf(type)) shape.push_back(dimension);
  return llvm::json::Object{{"argument", int64_t(argument)}, {"logical_port", binding.str()},
      {"element_type", elementTypeName(type)}, {"shape", std::move(shape)},
      {"layout", "contiguous_row_major"}, {"storage_bytes", int64_t(elementByteWidth(type))}};
}

llvm::Error validateFunction(mlir::func::FuncOp graph,
                              const CUDALoweringOptions &options,
                              llvm::StringSet<> &requiredTypes) {
  if (graph.isExternal() || !llvm::hasSingleElement(graph.getBody()))
    return error("CUDA lowering requires one defined single-block graph");
  if (graph.getNumResults() == 0)
    return error("CUDA lowering requires at least one graph result");
  if (graph.getNumArguments() + graph.getNumResults() > 64)
    return error("CUDA pointer ABI supports at most 64 input and output buffers");
  // The runtime deliberately accepts portable C identifier entry names.
  auto name = graph.getSymName();
  auto letter = [](char ch) { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'; };
  if (name.empty() || !letter(name.front()) ||
      llvm::any_of(name.drop_front(), [&](char ch) { return !letter(ch) && !(ch >= '0' && ch <= '9'); }))
    return error("CUDA graph entry must be an ASCII identifier");
  auto checkType = [&](mlir::Type type) -> llvm::Error {
    auto count = elementCount(type);
    if (!count) return count.takeError();
    auto element = elementTypeName(type);
    requiredTypes.insert(element);
    if (element == "bf16" && options.architecture < 80)
      return error("bf16 compilation requires target sm80 or newer; no unsupported-device fallback is enabled");
    return llvm::Error::success();
  };
  for (auto type : graph.getArgumentTypes())
    if (auto failure = checkType(type)) return failure;
  for (auto type : graph.getResultTypes())
    if (auto failure = checkType(type)) return failure;
  for (mlir::Operation &op : graph.front()) {
    if (mlir::isa<ExternalOp>(op))
      return error("ASK-owned CUDA lowering cannot compile ask.external; provide an external executable through an execution plan");
    if (!mlir::isa<ConstantOp, SplatOp, AddOp, MulOp, CastOp, ReshapeOp, mlir::func::ReturnOp>(op))
      return error("CUDA lowering requires prepared constant/splat/add/mul/cast/reshape graphs; unsupported operation " + op.getName().getStringRef());
    for (auto type : op.getResultTypes())
      if (auto failure = checkType(type)) return failure;
    if (mlir::isa<AddOp, MulOp>(op) &&
        (op.getOperand(0).getType() != op.getOperand(1).getType() ||
         op.getOperand(0).getType() != op.getResult(0).getType()))
      return error("CUDA arithmetic requires exactly equal prepared operand/result types");
    if (mlir::isa<CastOp>(op) && shapeOf(op.getOperand(0).getType()) != shapeOf(op.getResult(0).getType()))
      return error("CUDA cast requires exactly equal prepared shapes");
  }
  if (!mlir::isa<mlir::func::ReturnOp>(graph.front().getTerminator()))
    return error("CUDA graph requires func.return");
  return llvm::Error::success();
}
} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToCUDALLVM(mlir::ModuleOp source, llvm::LLVMContext &context,
                const CUDALoweringOptions &options,
                llvm::json::Object &artifact, llvm::json::Object &report) {
  if (options.architecture < 10 || options.architecture > 999)
    return error("CUDA lowering requires an explicit --cuda-arch between 10 and 999");
  if (options.blockSize == 0 || options.blockSize > 1024)
    return error("CUDA block size must be between 1 and 1024");
  if (options.ptxVersion < 10 || options.ptxVersion > 999)
    return error("CUDA PTX version must be between 10 and 999");
  if (mlir::failed(mlir::verify(source))) return error("cannot lower invalid ASK IR");
  mlir::func::FuncOp graph;
  for (mlir::Operation &op : source.getBody()->getOperations()) {
    if (mlir::isa<SemanticsOp, ImplementationOp>(op)) continue;
    auto candidate = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!candidate || graph) return error("CUDA lowering requires exactly one graph function");
    graph = candidate;
  }
  if (!graph) return error("CUDA lowering requires exactly one graph function");
  llvm::StringSet<> requiredTypes;
  if (auto failure = validateFunction(graph, options, requiredTypes)) return std::move(failure);

  auto result = std::make_unique<llvm::Module>("ask.cuda", context);
  result->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
  result->setDataLayout("e-p:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  auto *pointer = llvm::PointerType::get(context, 1);
  llvm::SmallVector<llvm::Type *> parameters(graph.getNumArguments() + graph.getNumResults(), pointer);
  auto *functionType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), parameters, false);
  auto *function = llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage,
                                          graph.getSymName(), result.get());
  function->setCallingConv(llvm::CallingConv::PTX_Kernel);
  llvm::json::Array ports, invocationArguments;
  for (auto [index, type] : llvm::enumerate(graph.getArgumentTypes())) {
    auto *argument = function->getArg(index);
    argument->setName("input" + std::to_string(index));
    argument->addAttr(llvm::Attribute::ReadOnly);
    argument->addAttr(llvm::Attribute::getWithCaptureInfo(context, llvm::CaptureInfo::none()));
    std::string binding = "input:" + std::to_string(index);
    ports.push_back(describePort(type, index, binding));
    invocationArguments.push_back(llvm::json::Object{{"argument", int64_t(index)},
        {"binding", binding}, {"llvm_type", "ptr addrspace(1)"}});
  }
  for (auto [index, type] : llvm::enumerate(graph.getResultTypes())) {
    unsigned argumentIndex = graph.getNumArguments() + index;
    auto *argument = function->getArg(argumentIndex);
    argument->setName("output" + std::to_string(index));
    argument->addAttr(llvm::Attribute::NoAlias);
    argument->addAttr(llvm::Attribute::WriteOnly);
    argument->addAttr(llvm::Attribute::getWithCaptureInfo(context, llvm::CaptureInfo::none()));
    std::string binding = "output:" + std::to_string(index);
    ports.push_back(describePort(type, argumentIndex, binding));
    invocationArguments.push_back(llvm::json::Object{{"argument", int64_t(argumentIndex)},
        {"binding", binding}, {"llvm_type", "ptr addrspace(1)"}});
  }

  struct Batch {
    llvm::SmallVector<int64_t> shape;
    uint64_t count;
    llvm::SmallVector<unsigned> outputs;
  };
  llvm::SmallVector<Batch> batches;
  uint64_t maximumCount = 0, emptyOutputs = 0;
  auto returned = mlir::cast<mlir::func::ReturnOp>(graph.front().getTerminator());
  for (auto [index, value] : llvm::enumerate(returned.getOperands())) {
    auto count = elementCount(value.getType());
    if (!count) return count.takeError();
    maximumCount = std::max(maximumCount, *count);
    if (*count == 0) { ++emptyOutputs; continue; }
    auto shape = shapeOf(value.getType());
    auto batch = llvm::find_if(batches, [&](const Batch &item) { return item.shape == shape; });
    if (batch == batches.end()) batches.push_back({std::move(shape), *count, {unsigned(index)}});
    else batch->outputs.push_back(index);
  }
  // A grid-stride loop bounds the launch size without bounding tensor lengths.
  // For empty outputs the entry returns without touching any pointer.
  uint64_t grid = std::max<uint64_t>(1, std::min<uint64_t>(65535,
      maximumCount / options.blockSize + (maximumCount % options.blockSize != 0)));
  uint64_t stride = grid * options.blockSize;
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
  auto registerValue = [&](llvm::StringRef name) -> llvm::Value * {
    auto callee = result->getOrInsertFunction(name,
        llvm::FunctionType::get(builder.getInt32Ty(), {}, false));
    return builder.CreateZExt(builder.CreateCall(callee), builder.getInt64Ty());
  };
  llvm::Value *start = nullptr;
  if (!batches.empty()) {
    auto *block = registerValue("llvm.nvvm.read.ptx.sreg.ctaid.x");
    auto *thread = registerValue("llvm.nvvm.read.ptx.sreg.tid.x");
    start = builder.CreateAdd(builder.CreateMul(block, builder.getInt64(options.blockSize)), thread, "start");
  }
  uint64_t inputLoads = 0;
  llvm::json::Array groups;
  for (const Batch &batch : batches) {
    auto *before = builder.GetInsertBlock();
    auto *body = llvm::BasicBlock::Create(context, "element.loop", function);
    auto *after = llvm::BasicBlock::Create(context, "output.done", function);
    builder.CreateCondBr(builder.CreateICmpULT(start, builder.getInt64(batch.count)), body, after);
    builder.SetInsertPoint(body);
    auto *index = builder.CreatePHI(builder.getInt64Ty(), 2, "index");
    index->addIncoming(start, before);
    llvm::DenseMap<mlir::Value, llvm::Value *> emitted;
    std::function<llvm::Value *(mlir::Value)> emit = [&](mlir::Value value) -> llvm::Value * {
      auto cached = emitted.find(value);
      if (cached != emitted.end()) return cached->second;
      llvm::Value *computed;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        auto *type = storageType(value.getType(), context);
        llvm::Value *offset = mlir::isa<mlir::RankedTensorType>(value.getType())
                                ? static_cast<llvm::Value *>(index) : builder.getInt64(0);
        auto *address = builder.CreateGEP(type, function->getArg(argument.getArgNumber()), offset, "input.address");
        computed = builder.CreateAlignedLoad(type, address, llvm::Align(elementByteWidth(value.getType())), "input");
        ++inputLoads;
        if (llir::scalarLLVMType(value.getType(), context)->isIntegerTy(1))
          computed = builder.CreateICmpNE(computed, builder.getInt8(0), "input.bool");
      } else {
        auto *producer = value.getDefiningOp();
        if (auto constant = mlir::dyn_cast<ConstantOp>(producer)) {
          if (auto integer = constant->getAttrOfType<mlir::IntegerAttr>("value"))
            computed = llvm::ConstantInt::get(context, integer.getValue());
          else computed = llvm::ConstantFP::get(context, constant->getAttrOfType<mlir::FloatAttr>("value").getValue());
        } else if (mlir::isa<SplatOp, ReshapeOp>(producer)) {
          computed = emit(producer->getOperand(0));
        } else if (mlir::isa<CastOp>(producer)) {
          computed = llir::createNumericCast(builder, emit(producer->getOperand(0)),
                                              llir::scalarLLVMType(value.getType(), context));
        } else {
          auto *lhs = emit(producer->getOperand(0));
          auto *rhs = emit(producer->getOperand(1));
          if (lhs->getType()->isFloatingPointTy()) {
            const bool bf16 = lhs->getType()->isBFloatTy();
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
      emitted[value] = computed;
      return computed;
    };
    llvm::json::Array outputIndices;
    for (unsigned outputIndex : batch.outputs) {
      auto *computed = emit(returned.getOperand(outputIndex));
      if (computed->getType()->isIntegerTy(1)) computed = builder.CreateZExt(computed, builder.getInt8Ty(), "output.bool");
      auto *pointer = function->getArg(graph.getNumArguments() + outputIndex);
      auto *address = builder.CreateGEP(computed->getType(), pointer, index, "output.address");
      builder.CreateAlignedStore(computed, address,
          llvm::Align(elementByteWidth(returned.getOperand(outputIndex).getType())));
      outputIndices.push_back(int64_t(outputIndex));
    }
    auto *next = builder.CreateAdd(index, builder.getInt64(stride), "next");
    index->addIncoming(next, body);
    builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(batch.count)), body, after);
    builder.SetInsertPoint(after);
    groups.push_back(llvm::json::Object{{"outputs", std::move(outputIndices)}, {"element_count", int64_t(batch.count)}});
  }
  builder.CreateRetVoid();
  std::string diagnostics;
  llvm::raw_string_ostream diagnosticStream(diagnostics);
  if (llvm::verifyModule(*result, &diagnosticStream))
    return error("CUDA lowering produced invalid LLVM IR: " + diagnosticStream.str());

  llvm::json::Array required;
  // Stable artifact bytes independent of StringSet iteration order.
  for (llvm::StringRef type : {"bool", "i32", "i64", "f32", "f16", "bf16"})
    if (requiredTypes.contains(type)) required.push_back(type);
  artifact = llvm::json::Object{{"format", "llvm-ir"}, {"entry", graph.getSymName().str()},
      {"producer", llvm::json::Object{{"name", "ASK"}, {"version", ASK_VERSION}}},
      {"target", llvm::json::Object{{"backend", "cuda"}, {"arch", int64_t(options.architecture)},
          {"triple", result->getTargetTriple().str()}, {"data_layout", result->getDataLayoutStr()}}},
      {"codegen", llvm::json::Object{{"fp_contract", "off"}, {"ptx_version", int64_t(options.ptxVersion)}}},
      {"required_element_types", std::move(required)}, {"port_map", std::move(ports)},
      {"invocation", llvm::json::Object{{"version", 2}, {"kind", "cuda-kernel"},
          {"grid", llvm::json::Array{int64_t(grid), 1, 1}},
          {"block", llvm::json::Array{int64_t(options.blockSize), 1, 1}}, {"shared_bytes", 0},
          {"arguments", std::move(invocationArguments)}}},
      {"requires_disjoint_outputs", true}, {"outputs_must_not_alias_inputs", true},
      {"validation", llvm::json::Object{{"llvm_module_verified", true}, {"device_execution_performed", false}, {"stage", "ask_cuda_llvm"}}}};
  report = llvm::json::Object{{"statistics_stage", "cuda_lowering"}, {"kernel_count", 1},
      {"indexing", "linear_grid_stride"}, {"output_loop_count", int64_t(batches.size())},
      {"input_load_count", int64_t(inputLoads)}, {"zero_element_outputs", int64_t(emptyOutputs)},
      {"temporary_buffer_count", 0}, {"groups", std::move(groups)},
      {"bool_storage", "one-byte-zero-or-one"}, {"cast_semantics", "rne-saturating-fptosi-nan-zero-v1"}};
  return std::move(result);
}
} // namespace ask
