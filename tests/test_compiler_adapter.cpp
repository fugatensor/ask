#include "ask/Adapters/Compiler.h"
#include "ask/Frontend/Compiler.h"
#include "ask/Frontend/Prepare.h"
#include "ask/IR/ASKDialect.h"
#include "ask/LLIR/Fuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;
using namespace ask::adapters;
namespace {
void require(bool condition, StringRef message) {
  if (!condition) {
    errs() << "compiler-adapter test failed: " << message << '\n';
    std::exit(1);
  }
}
Error error(StringRef message) {
  return createStringError(inconvertibleErrorCode(), message);
}

const char *graphIR = R"(
module {
  "ask.semantics"() {sym_name = "scale", function_type = (tensor<1xi32>) -> tensor<1xi32>,
    shape_rule = "same_as", shape_operand = 0 : i64, effects = "none"} : () -> ()
  "ask.implementation"() {sym_name = "fixture_impl", compiler = "fixture", entry = "source_entry",
    source = "fixture scale two", signature = {in = "*i32", out = "*i32"},
    source_bindings = {in = "input:0", out = "output:0"}, constants = {}, options = {},
    target = {backend = "fixture-host"}, grid = array<i64: 1, 1, 1>} : () -> ()
  func.func @pipeline(%x: tensor<1xi32>) -> tensor<1xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 1>} : (i32) -> tensor<1xi32>
    %p = "ask.add"(%x, %ones) {overflow = "wrap"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %f = "ask.external"(%p) {ask.id = "F", semantics = @scale, implementation = @fixture_impl} : (tensor<1xi32>) -> tensor<1xi32>
    %three = "ask.constant"() {value = 3 : i32} : () -> i32
    %threes = "ask.splat"(%three) {shape = array<i64: 1>} : (i32) -> tensor<1xi32>
    %q = "ask.add"(%f, %threes) {overflow = "wrap"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %q : tensor<1xi32>
  }
})";

// The deliberately different protocol has i1 predicates and value argument 2,
// ordinary LLVM calls, a renamed lowered entry, and no GPU or producer metadata.
const char *producerIR = R"(
declare i32 @fixture_input(i1, i64, i32)
declare i32 @fixture_output(i32, i64)
define i32 @lowered_entry() {
  %x = call i32 @fixture_input(i1 true, i64 99, i32 3)
  %f = mul i32 %x, 2
  %y = call i32 @fixture_output(i32 %f, i64 88)
  ret i32 %y
}
define i32 @inactive_lane() {
  %x = call i32 @fixture_input(i1 false, i64 99, i32 3)
  %f = mul i32 %x, 2
  %y = call i32 @fixture_output(i32 %f, i64 88)
  ret i32 %y
}
)";

Error validateFixture(const json::Object &artifact, const json::Object &request) {
  if (request.getString("compiler") != "fixture" ||
      request.getString("source") != "fixture scale two" ||
      request.getString("entry") != "source_entry" ||
      artifact.getString("entry") != "lowered_entry")
    return error("fixture source/entry mismatch");
  return Error::success();
}
Expected<PreparedFusion> prepareFixture(Module &module, const json::Object &,
                                        const FusionRequest &request) {
  if (request.elementType != "i32" || request.elementCount != 1)
    return error("fixture shape/type mismatch");
  PreparedFusion result;
  result.protocol = "fixture.value.v9";
  result.producerNumerics = json::Object{{"integer_semantics", "wrap"}};
  result.codegen = json::Object{{"fp_contract", "off"}};
  for (auto &function : module)
    for (auto &block : function)
      for (auto &instruction : block) {
        auto *call = dyn_cast<CallInst>(&instruction);
        if (!call || !call->getCalledFunction()) continue;
        if (call->getCalledFunction()->getName() == "fixture_input") {
          auto *value = call->getArgOperand(2);
          result.inputs.push_back({call, value, call->getArgOperand(0), value});
        } else if (call->getCalledFunction()->getName() == "fixture_output") {
          result.outputs.push_back({call, call->getArgOperand(0), nullptr, nullptr});
        }
      }
  return result;
}
Expected<PreparedFusion> invalidBoundary(Module &module, const json::Object &artifact,
                                         const FusionRequest &request) {
  auto result = prepareFixture(module, artifact, request);
  if (!result) return result.takeError();
  result->inputs.front().value = nullptr;
  return result;
}
const CompilerAdapter fixture{"fixture", 1, &validateFixture, &prepareFixture};

void expectFailure(mlir::ModuleOp graph, const json::Object &artifact,
                   const CompilerAdapter &adapter, StringRef expected) {
  LLVMContext context;
  json::Object report;
  std::string helper;
  auto fused = ask::llir::fuseExternal(graph, artifact, adapter, context, report, helper);
  require(!fused, "invalid adapter input was accepted");
  auto message = toString(fused.takeError());
  require(StringRef(message).contains(expected), message);
  require(!report.get("fusion"), "failed fusion must not report success");
}
} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<ask::ASKDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext mlirContext(registry);
  auto graph = mlir::parseSourceString<mlir::ModuleOp>(graphIR, &mlirContext);
  require(bool(graph), "graph fixture must parse");
  json::Object preparation;
  require(mlir::succeeded(ask::prepareModule(*graph, preparation)), "graph must prepare");
  json::Object artifact{{"protocol_version", 1}, {"format", "llvm-ir"},
                        {"entry", "lowered_entry"}, {"payload", producerIR}};
  LLVMContext context;
  json::Object report;
  std::string helper;
  auto fused = ask::llir::fuseExternal(*graph, artifact, fixture, context, report, helper);
  if (!fused) {
    errs() << toString(fused.takeError()) << '\n';
    return 1;
  }
  for (auto pair : {std::make_pair("lowered_entry", 11), std::make_pair("inactive_lane", 9)}) {
    auto *function = (*fused)->getFunction(pair.first);
    require(function && function->size() == 1, "fused fixture entry missing");
    auto *ret = dyn_cast<ReturnInst>(function->front().getTerminator());
    auto *value = ret ? dyn_cast<ConstantInt>(ret->getReturnValue()) : nullptr;
    require(value && value->getSExtValue() == pair.second,
            "fused result or predicate passthrough changed");
    for (auto &instruction : function->front())
      require(!isa<CallInst>(instruction), "marker/helper calls survived fusion");
  }
  auto *fusion = report.getObject("fusion");
  require(fusion && fusion->getString("compiler") == "fixture" &&
          fusion->getString("protocol") == "fixture.value.v9" &&
          fusion->getInteger("inlined_helpers") == 4,
          "fusion report must use adapter-provided identity/protocol");
  require(fusion->getObject("codegen") &&
          fusion->getObject("codegen")->getString("fp_contract") == "off",
          "adapter backend policy was lost");
  require(artifact.getString("payload") == producerIR, "source artifact was mutated");
  auto capabilities = ask::compilerCapabilities();
  for (const auto &id : *capabilities.getArray("third_party_compilers"))
    require(id.getAsString() != "fixture", "test adapter leaked into production manifest");
  auto wrong = fixture;
  wrong.id = "other";
  expectFailure(*graph, artifact, wrong, "selected adapter does not match");
  auto unavailable = fixture;
  unavailable.prepareFusion = nullptr;
  expectFailure(*graph, artifact, unavailable, "does not support value-boundary fusion");
  auto invalid = fixture;
  invalid.prepareFusion = &invalidBoundary;
  expectFailure(*graph, artifact, invalid, "argument type or dominance mismatch");
  artifact["entry"] = "wrong";
  expectFailure(*graph, artifact, fixture, "fixture source/entry mismatch");

  // The public fusion API verifies the actual lowered entry even when an
  // adapter needs no producer-specific validator. Finding valid value markers
  // elsewhere in the module does not make a missing/declaration/vararg entry
  // executable. These checks must not depend on the CLI envelope validator.
  auto noValidator = fixture;
  noValidator.validateArtifact = nullptr;
  expectFailure(*graph, artifact, noValidator, "defined non-variadic function");
  artifact["entry"] = "declared_entry";
  artifact["payload"] = std::string(producerIR) + "\ndeclare i32 @declared_entry()\n";
  expectFailure(*graph, artifact, noValidator, "defined non-variadic function");
  artifact["entry"] = "variadic_entry";
  artifact["payload"] = std::string(producerIR) +
      "\ndefine i32 @variadic_entry(...) { ret i32 0 }\n";
  expectFailure(*graph, artifact, noValidator, "defined non-variadic function");
  outs() << "generic adapter fusion: 2 numerical/predicate cases and 7 rejection cases passed\n";
  return 0;
}
