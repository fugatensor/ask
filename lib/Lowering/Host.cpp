#include "ask/Lowering/Host.h"
#include "ask/IR/ASKOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

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
  std::string triple = llvm::sys::getDefaultTargetTriple();
  std::string reason;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, reason);
  if (!target)
    return error("cannot create host target: " + reason);
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, "generic", "", options, llvm::Reloc::PIC_));
  if (!machine)
    return error("LLVM native target cannot create a target machine");
  return std::move(machine);
}

llvm::Expected<uint64_t> elementCount(mlir::Type type) {
  if (type.isSignlessInteger(32))
    return uint64_t(1);
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || !isSupportedValueType(type))
    return error("host lowering requires concrete ranked i32 tensors or scalar i32");
  if (llvm::is_contained(tensor.getShape(), int64_t(0)))
    return uint64_t(0);
  uint64_t count = 1;
  constexpr uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / 4;
  for (int64_t dimension : tensor.getShape()) {
    if (dimension < 0 || uint64_t(dimension) > limit / count)
      return error("host tensor byte count overflows the supported size range");
    count *= uint64_t(dimension);
  }
  return count;
}

llvm::json::Object describeArgument(mlir::Type type, unsigned argument) {
  llvm::json::Array shape;
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (tensor)
    for (int64_t dimension : tensor.getShape())
      shape.push_back(dimension);
  return llvm::json::Object{{"argument", int64_t(argument)},
                            {"kind", tensor ? "tensor" : "scalar"},
                            {"dtype", "i32"}, {"shape", std::move(shape)}};
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
    if (!mlir::isa<ConstantOp, SplatOp, AddOp, MulOp, mlir::func::ReturnOp>(op))
      return error("host lowering requires prepared constant/splat/add/mul graphs; unsupported operation " + op.getName().getStringRef());
    for (mlir::Type type : op.getResultTypes()) {
      auto count = elementCount(type);
      if (!count)
        return count.takeError();
    }
    if (mlir::isa<AddOp, MulOp>(op) &&
        (op.getOperand(0).getType() != op.getOperand(1).getType() ||
         op.getOperand(0).getType() != op.getResult(0).getType()))
      return error("host arithmetic requires exactly equal prepared operand/result types");
  }
  if (!mlir::isa<mlir::func::ReturnOp>(function.front().getTerminator()))
    return error("host graph requires func.return");
  return llvm::Error::success();
}
} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
lowerToHostLLVM(mlir::ModuleOp source, llvm::LLVMContext &context,
                llvm::json::Object &abiReport) {
  if (mlir::failed(mlir::verify(source)))
    return error("cannot lower invalid ASK IR");
  auto target = makeHostMachine();
  if (!target)
    return target.takeError();
  auto result = std::make_unique<llvm::Module>("ask.host", context);
  result->setTargetTriple((*target)->getTargetTriple().str());
  result->setDataLayout((*target)->createDataLayout());
  auto *i32 = llvm::Type::getInt32Ty(context);
  auto *pointer = llvm::PointerType::getUnqual(context);
  llvm::json::Array functions;

  for (mlir::Operation &op : source.getBody()->getOperations()) {
    if (mlir::isa<SemanticsOp, ImplementationOp>(op))
      continue;
    auto graph = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!graph)
      return error("unsupported module-level operation in host lowering");
    if (llvm::Error failure = validateFunction(graph))
      return std::move(failure);

    llvm::SmallVector<llvm::Type *> parameters;
    for (mlir::Type type : graph.getArgumentTypes())
      parameters.push_back(mlir::isa<mlir::RankedTensorType>(type)
                               ? static_cast<llvm::Type *>(pointer)
                               : static_cast<llvm::Type *>(i32));
    for (unsigned i = 0; i < graph.getNumResults(); ++i)
      parameters.push_back(pointer);
    auto *functionType = llvm::FunctionType::get(llvm::Type::getVoidTy(context), parameters, false);
    auto *function = llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage,
                                             graph.getSymName(), result.get());
    function->setCallingConv(llvm::CallingConv::C);
    llvm::DenseMap<mlir::Value, llvm::Value *> arguments;
    llvm::json::Array inputs;
    llvm::json::Array outputs;
    for (auto [index, argument] : llvm::enumerate(graph.getArguments())) {
      llvm::Argument *physical = function->getArg(index);
      physical->setName("arg" + std::to_string(index));
      arguments[argument] = physical;
      if (physical->getType()->isPointerTy()) {
        physical->addAttr(llvm::Attribute::ReadOnly);
        physical->addAttr(llvm::Attribute::NoCapture);
      }
      inputs.push_back(describeArgument(argument.getType(), index));
    }
    for (auto [index, type] : llvm::enumerate(graph.getResultTypes())) {
      unsigned physicalIndex = graph.getNumArguments() + index;
      auto *physical = function->getArg(physicalIndex);
      physical->setName("out" + std::to_string(index));
      physical->addAttr(llvm::Attribute::NoAlias);
      physical->addAttr(llvm::Attribute::NoCapture);
      physical->addAttr(llvm::Attribute::WriteOnly);
      outputs.push_back(describeArgument(type, physicalIndex));
    }

    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
    auto returned = mlir::cast<mlir::func::ReturnOp>(graph.front().getTerminator());
    for (auto [outputIndex, value] : llvm::enumerate(returned.getOperands())) {
      auto count = elementCount(value.getType());
      if (!count)
        return count.takeError();
      if (*count == 0)
        continue; // No input or output pointer is dereferenced for empty results.

      llvm::BasicBlock *before = builder.GetInsertBlock();
      llvm::BasicBlock *body = llvm::BasicBlock::Create(context, "output.loop", function);
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
            auto *address = builder.CreateGEP(i32, argument->second, index, "input.address");
            computed = builder.CreateAlignedLoad(i32, address, llvm::Align(4), "input");
          } else {
            computed = argument->second;
          }
        } else {
          mlir::Operation *producer = current.getDefiningOp();
          if (auto constant = mlir::dyn_cast<ConstantOp>(producer)) {
            auto bits = constant->getAttrOfType<mlir::IntegerAttr>("value").getValue();
            computed = llvm::ConstantInt::get(context, bits);
          } else if (mlir::isa<SplatOp>(producer)) {
            computed = emit(producer->getOperand(0));
          } else {
            llvm::Value *lhs = emit(producer->getOperand(0));
            llvm::Value *rhs = emit(producer->getOperand(1));
            computed = mlir::isa<AddOp>(producer) ? builder.CreateAdd(lhs, rhs, "add.wrap")
                                                   : builder.CreateMul(lhs, rhs, "mul.wrap");
          }
        }
        emitted[current] = computed;
        return computed;
      };
      llvm::Value *computed = emit(value);
      llvm::Value *output = function->getArg(graph.getNumArguments() + outputIndex);
      auto *address = builder.CreateGEP(i32, output, index, "output.address");
      builder.CreateAlignedStore(computed, address, llvm::Align(4));
      llvm::Value *next = builder.CreateAdd(index, builder.getInt64(1), "next");
      index->addIncoming(next, body);
      builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt64(*count)), body, after);
      builder.SetInsertPoint(after);
    }
    builder.CreateRetVoid();
    functions.push_back(llvm::json::Object{
        {"name", graph.getSymName().str()}, {"calling_convention", "c"},
        {"inputs", std::move(inputs)}, {"outputs", std::move(outputs)},
        {"layout", "contiguous_row_major"}, {"minimum_alignment", 4},
        {"requires_disjoint_outputs", true}});
  }
  if (functions.empty())
    return error("host lowering requires at least one graph function");
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  if (llvm::verifyModule(*result, &stream))
    return error("host lowering produced invalid LLVM IR: " + stream.str());
  abiReport = llvm::json::Object{{"schema_version", 1},
                                {"target_triple", result->getTargetTriple()},
                                {"data_layout", result->getDataLayoutStr()},
                                {"functions", std::move(functions)}};
  return std::move(result);
}

llvm::Error emitHostObject(llvm::Module &module, llvm::StringRef path) {
  auto target = makeHostMachine();
  if (!target)
    return target.takeError();
  if (module.getTargetTriple() != (*target)->getTargetTriple().str() ||
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
