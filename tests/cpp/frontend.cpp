#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKDialect.h"
#include "ask/Transforms/Optimize.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  bool optimize = argc == 2 && llvm::StringRef(argv[1]) == "--optimize";
  if (argc > 1 && !optimize) {
    llvm::errs() << "usage: ask-core-test [--optimize]\n";
    return 1;
  }
  mlir::DialectRegistry registry;
  registry.insert<ask::ASKDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto i32 = builder.getI32Type();

  // An upper framework constructs main(x: i32) -> i32, computing x + 1.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  (*module)->setAttr("ask.schema_version", builder.getI64IntegerAttr(1));
  auto function = mlir::func::FuncOp::create(
      location, "main", builder.getFunctionType({i32}, {i32}));
  module->push_back(function);
  auto *entry = function.addEntryBlock();
  function.setArgAttr(0, "ask.value_id", builder.getStringAttr("main.arg0"));
  builder.setInsertionPointToStart(entry);

  mlir::OperationState constant(location, "ask.constant");
  constant.addTypes(i32);
  constant.addAttribute("value", builder.getI32IntegerAttr(1));
  constant.addAttribute("ask.id", builder.getStringAttr("one"));
  auto *one = builder.create(constant);

  mlir::OperationState add(location, "ask.add");
  add.addOperands({entry->getArgument(0), one->getResult(0)});
  add.addTypes(i32);
  add.addAttribute("overflow", builder.getStringAttr("wrap"));
  add.addAttribute("ask.id", builder.getStringAttr("P"));
  add.addAttribute("ask.value_id", builder.getStringAttr("v_y"));
  auto *sum = builder.create(add);

  mlir::OperationState returned(location, "func.return");
  returned.addOperands(sum->getResult(0));
  builder.create(returned);
  if (mlir::failed(mlir::verify(*module)))
    return 1;

  // prepareModule mutates its argument. Keep the framework's original intact
  // until preparation succeeds, and publish only the successful candidate.
  mlir::OwningOpRef<mlir::ModuleOp> prepared(
      mlir::cast<mlir::ModuleOp>(module->getOperation()->clone()));
  llvm::json::Object report;
  if (mlir::failed(optimize ? ask::optimizeModule(*prepared, report)
                            : ask::prepareModule(*prepared, report)))
    return 1;

  if (!report.getObject("shape_snapshot") ||
      (optimize && !report.getObject("optimization"))) {
    llvm::errs() << "missing preparation or optimization report\n";
    return 1;
  }
  if ((*module)->hasAttr("ask.ir_revision") ||
      !(*prepared)->hasAttr("ask.ir_revision")) {
    llvm::errs() << "preparing the clone modified the original graph\n";
    return 1;
  }
  auto result = prepared->lookupSymbol<mlir::func::FuncOp>("main");
  if (!result || result.getFunctionType() != function.getFunctionType() ||
      mlir::failed(mlir::verify(*prepared))) {
    llvm::errs() << "preparation changed the public function signature\n";
    return 1;
  }
  llvm::outs() << "native graph preparation and report contracts passed\n";
  return 0;
}
