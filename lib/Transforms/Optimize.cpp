#include "ask/Transforms/Optimize.h"
#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <limits>

using namespace mlir;

namespace ask {
namespace {
StringRef operationId(Operation *op) {
  return op->getAttrOfType<StringAttr>("ask.id").getValue();
}

std::string valueId(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = cast<func::FuncOp>(argument.getOwner()->getParentOp());
    return function.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                                "ask.value_id").getValue().str();
  }
  auto result = cast<OpResult>(value);
  Operation *owner = result.getOwner();
  if (auto id = owner->getAttrOfType<StringAttr>("ask.value_id"))
    return id.getValue().str();
  return (operationId(owner) + ".result" + Twine(result.getResultNumber())).str();
}

bool isKnownPure(Operation *op) {
  return isa<ConstantOp, SplatOp, AddOp, MulOp>(op);
}

bool namesPort(StringRef port, Operation *op) {
  StringRef id = operationId(op);
  for (auto [direction, count] :
       {std::pair<StringRef, unsigned>{"in", op->getNumOperands()},
        std::pair<StringRef, unsigned>{"out", op->getNumResults()}}) {
    std::string prefix = (id + "." + direction).str();
    if (count == 1 && port == prefix)
      return true;
    for (unsigned index = 0; index < count; ++index)
      if (port == prefix + std::to_string(index))
        return true;
  }
  return false;
}

// Preserve the owner of an input marker's port before folds change operation
// kind/arity or DCE erases it. The marker operation itself is already erased.
llvm::StringMap<std::string> findMarkerPortOwners(ModuleOp module,
                                                const llvm::json::Array &markers) {
  llvm::StringMap<std::string> owners;
  for (const llvm::json::Value &value : markers) {
    const auto &marker = *value.getAsObject();
    auto port = marker.getString("port_id");
    if (!port)
      continue;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.getSymName() != *marker.getString("function"))
        continue;
      for (Operation &op : function.front())
        if (!isa<func::ReturnOp>(op) && namesPort(*port, &op))
          owners[*marker.getString("mark_id")] = operationId(&op).str();
    }
  }
  return owners;
}

class IdAllocator {
public:
  explicit IdAllocator(ModuleOp module) {
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      for (Value argument : function.getArguments())
        values.insert(valueId(argument));
      for (Operation &op : function.front()) {
        if (isa<func::ReturnOp>(op))
          continue;
        operations.insert(operationId(&op));
        for (Value result : op.getResults())
          values.insert(valueId(result));
      }
    }
  }

  std::string scalarFor(StringRef foldedId) {
    std::string base = (foldedId + ".folded_scalar").str();
    std::string candidate = base;
    for (uint64_t suffix = 1;
         operations.contains(candidate) || values.contains(candidate + ".result0");
         ++suffix)
      candidate = base + "." + std::to_string(suffix);
    operations.insert(candidate);
    values.insert(candidate + ".result0");
    return candidate;
  }

private:
  llvm::StringSet<> operations;
  llvm::StringSet<> values;
};

Operation *createConstant(OpBuilder &builder, Location location, Type type,
                          const llvm::APInt &bits, NamedAttrList attributes) {
  attributes.erase("overflow");
  attributes.erase("shape");
  attributes.set("value", builder.getIntegerAttr(type, bits));
  OperationState state(location, ConstantOp::getOperationName());
  state.addTypes(type);
  state.addAttributes(attributes);
  return builder.create(state);
}

int64_t foldConstants(ModuleOp module, llvm::StringSet<> &rewrittenOperations,
                      llvm::StringSet<> &rewrittenValues) {
  IdAllocator ids(module);
  int64_t folds = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    // A value in this map is either a scalar constant or a uniform tensor.
    // One APInt per SSA value avoids materializing tensors, including huge ones.
    llvm::DenseMap<Value, llvm::APInt> constants;
    for (Operation &op : llvm::make_early_inc_range(function.front())) {
      if (auto constant = dyn_cast<ConstantOp>(op)) {
        constants.try_emplace(constant.getResult(),
                              op.getAttrOfType<IntegerAttr>("value").getValue());
        continue;
      }
      if (auto splat = dyn_cast<SplatOp>(op)) {
        auto known = constants.find(splat.getOperand());
        if (known != constants.end())
          constants.try_emplace(splat.getResult(), known->second);
        continue;
      }
      if (!isa<AddOp, MulOp>(op))
        continue;
      auto lhs = constants.find(op.getOperand(0));
      auto rhs = constants.find(op.getOperand(1));
      if (lhs == constants.end() || rhs == constants.end())
        continue;

      // Both operands are verified i32. APInt arithmetic wraps at exactly 32 bits.
      llvm::APInt folded = isa<AddOp>(op) ? lhs->second + rhs->second
                                        : lhs->second * rhs->second;
      rewrittenOperations.insert(operationId(&op));
      rewrittenValues.insert(valueId(op.getResult(0)));
      OpBuilder builder(&op);
      Operation *replacement = nullptr;
      Type type = op.getResult(0).getType();
      NamedAttrList attributes(op.getAttrs());
      if (type.isSignlessInteger(32)) {
        replacement = createConstant(builder, op.getLoc(), type, folded, attributes);
      } else {
        std::string scalarId = ids.scalarFor(operationId(&op));
        NamedAttrList scalarAttributes;
        scalarAttributes.set("ask.id", builder.getStringAttr(scalarId));
        scalarAttributes.set("ask.value_id", builder.getStringAttr(scalarId + ".result0"));
        Operation *scalar = createConstant(builder, op.getLoc(), builder.getI32Type(),
                                           folded, scalarAttributes);
        constants.try_emplace(scalar->getResult(0), folded);
        attributes.erase("overflow");
        attributes.set("shape", builder.getDenseI64ArrayAttr(
                                    cast<RankedTensorType>(type).getShape()));
        OperationState state(op.getLoc(), SplatOp::getOperationName());
        state.addOperands(scalar->getResult(0));
        state.addTypes(type);
        state.addAttributes(attributes);
        replacement = builder.create(state);
      }
      Value result = replacement->getResult(0);
      constants.try_emplace(result, folded);
      op.getResult(0).replaceAllUsesWith(result);
      op.erase();
      ++folds;
    }
  }
  return folds;
}

int64_t eliminateDeadCode(ModuleOp module) {
  int64_t removed = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    // In a verified one-block SSA graph, every user follows its definition.
    // Walking backwards therefore also removes an entire newly dead chain.
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(function.front())))
      if (isKnownPure(&op) && op.use_empty()) {
        op.erase();
        ++removed;
      }
  }
  return removed;
}

llvm::json::Array planFusion(ModuleOp module, bool enabled, bool &changed) {
  llvm::DenseMap<Operation *, StringAttr> planned;
  llvm::json::Array groups;
  Builder builder(module.getContext());
  if (enabled) {
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      llvm::SmallVector<Operation *> pending;
      RankedTensorType groupType;
      auto finish = [&] {
        if (pending.size() >= 2) {
          // Operation IDs are module-wide, so this is unique and stable across
          // repeated runs even when unrelated functions are added or removed.
          std::string id = ("fusion." + operationId(pending.front())).str();
          StringAttr attribute = builder.getStringAttr(id);
          llvm::json::Array operations;
          llvm::json::Array shape;
          for (Operation *op : pending) {
            planned[op] = attribute;
            operations.push_back(operationId(op).str());
          }
          for (int64_t extent : groupType.getShape())
            shape.push_back(extent);
          groups.push_back(llvm::json::Object{
              {"id", id}, {"function", function.getSymName().str()},
              {"shape", std::move(shape)}, {"operations", std::move(operations)}});
        }
        pending.clear();
        groupType = {};
      };
      for (Operation &op : function.front()) {
        if (!isKnownPure(&op)) {
          finish(); // In particular, never carry a group across ask.external.
          continue;
        }
        auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!type)
          continue; // Scalar arithmetic may occur between tensor operations.
        if (!type.hasStaticShape() || !isa<SplatOp, AddOp, MulOp>(op)) {
          finish();
          continue;
        }
        if (groupType && type != groupType)
          finish();
        if (!groupType)
          groupType = type;
        pending.push_back(&op);
      }
      finish();
    }
  }
  module.walk([&](Operation *op) {
    Attribute previous = op->getAttr("ask.fusion_group");
    StringAttr next = planned.lookup(op);
    if (previous == next)
      return;
    changed = true;
    if (next)
      op->setAttr("ask.fusion_group", next);
    else
      op->removeAttr("ask.fusion_group");
  });
  return groups;
}

llvm::json::Array updateMarkers(
    ModuleOp module, const llvm::json::Array &input,
    const llvm::StringMap<std::string> &portOwners,
    const llvm::StringSet<> &rewrittenOperations,
    const llvm::StringSet<> &rewrittenValues) {
  llvm::StringSet<> liveValues;
  llvm::StringSet<> liveOperations;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Value argument : function.getArguments())
      liveValues.insert(valueId(argument));
    for (Operation &op : function.front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      liveOperations.insert(operationId(&op));
      for (Value value : op.getResults())
        liveValues.insert(valueId(value));
    }
  }
  llvm::json::Array output;
  for (const llvm::json::Value &value : input) {
    llvm::json::Object marker(*value.getAsObject());
    StringRef identity = *marker.getString("value_id");
    auto owner = portOwners.find(*marker.getString("mark_id"));
    if (!liveValues.contains(identity)) {
      marker["status"] = "eliminated";
      marker["reason"] = "logical value was removed by dead-code elimination";
    } else if (owner != portOwners.end() && !liveOperations.contains(owner->second)) {
      marker["status"] = "eliminated";
      marker["reason"] = "referenced port was removed by dead-code elimination";
    } else if (rewrittenValues.contains(identity) ||
               (owner != portOwners.end() && rewrittenOperations.contains(owner->second))) {
      marker["status"] = "rewritten";
      marker["reason"] = "constant folding replaced the defining operation or referenced port";
    } else {
      marker["status"] = "exact";
    }
    output.push_back(std::move(marker));
  }
  return output;
}
} // namespace

LogicalResult optimizeModule(ModuleOp module, llvm::json::Object &report,
                              const OptimizationOptions &options) {
  llvm::json::Object input;
  if (failed(prepareModule(module, input)))
    return failure();
  llvm::json::Array inputMarkers(*input.getArray("markers"));
  auto portOwners = findMarkerPortOwners(module, inputMarkers);
  llvm::StringSet<> rewrittenOperations;
  llvm::StringSet<> rewrittenValues;
  int64_t folds = options.foldConstants
                      ? foldConstants(module, rewrittenOperations, rewrittenValues) : 0;
  int64_t dead = options.eliminateDeadCode ? eliminateDeadCode(module) : 0;
  bool fusionChanged = false;
  llvm::json::Array groups = planFusion(module, options.fuseElementwise, fusionChanged);
  bool semanticChanged = folds != 0 || dead != 0;
  if (semanticChanged || fusionChanged) {
    int64_t semantic = module->getAttrOfType<IntegerAttr>("ask.semantic_revision").getInt();
    int64_t ir = module->getAttrOfType<IntegerAttr>("ask.ir_revision").getInt();
    // prepare requires one spare revision value for a future update.
    constexpr int64_t limit = std::numeric_limits<int64_t>::max() - 1;
    if (ir >= limit || (semanticChanged && semantic >= limit))
      return module.emitError("ASK revision counter exhausted during optimization");
    Builder builder(module.getContext());
    if (semanticChanged)
      module->setAttr("ask.semantic_revision", builder.getI64IntegerAttr(semantic + 1));
    module->setAttr("ask.ir_revision", builder.getI64IntegerAttr(ir + 1));
  }
  llvm::json::Object current;
  if (failed(prepareModule(module, current)))
    return failure();

  // Copy the caller's unrelated report fields only after every pass verifies.
  // No partial snapshot or marker status escapes a failed optimization call.
  llvm::json::Object complete(report);
  complete["input_shape_snapshot"] = *input.get("shape_snapshot");
  complete["input_markers"] = llvm::json::Array(inputMarkers);
  complete["shape_snapshot"] = *current.get("shape_snapshot");
  complete["ir_revision"] = *current.get("ir_revision");
  complete["markers"] = updateMarkers(module, inputMarkers, portOwners,
                                       rewrittenOperations, rewrittenValues);
  complete["optimization"] = llvm::json::Object{
      {"constant_folds", folds}, {"dead_ops", dead},
      {"fusion_groups", std::move(groups)}};
  report = std::move(complete);
  return success();
}
} // namespace ask
