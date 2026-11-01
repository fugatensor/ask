#include "ask/LLIR/Inline.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;
using ask::llir::BoundaryInlineSite;

namespace {
void require(bool condition, StringRef message) {
  if (!condition) {
    errs() << "boundary-inline test failed: " << message << '\n';
    std::exit(1);
  }
}

std::unique_ptr<Module> parse(LLVMContext &context, StringRef text) {
  SMDiagnostic diagnostic;
  auto module = parseAssemblyString(text, diagnostic, context);
  if (!module) diagnostic.print("test_boundary_inline", errs());
  require(bool(module), "LLVM fixture must parse");
  return module;
}

std::string print(const Module &module) {
  std::string text;
  raw_string_ostream stream(text);
  module.print(stream, nullptr);
  return text;
}

CallInst *marker(Module &module, StringRef function, unsigned ordinal = 0) {
  for (auto &block : *module.getFunction(function))
    for (auto &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (ordinal-- == 0) return call;
  require(false, "missing fixture marker");
  return nullptr;
}

const char *consumerIR = R"(
declare i64 @identity_i64(i64)
declare i32 @identity_i32(i32)
declare float @identity_f32(float)
declare double @identity_f64(double)
declare <4 x float> @identity_vector(<4 x float>)

define i64 @twice_i64(i64 %x) {
  %a = call i64 @identity_i64(i64 %x)
  %b = call i64 @identity_i64(i64 %a)
  ret i64 %b
}
define i32 @multiple_arguments(i32 %x, i32 %y) {
  %v = call i32 @identity_i32(i32 %x)
  ret i32 %v
}
define float @scalar_f32(float %x) {
  %v = call float @identity_f32(float %x)
  ret float %v
}
define double @scalar_f64(double %x) {
  %v = call double @identity_f64(double %x)
  ret double %v
}
define float @intrinsic_abs(float %x) {
  %v = call float @identity_f32(float %x)
  ret float %v
}
define i64 @guarded_i64(i64 %x, i32 %enabled) {
  %valid = icmp ne i32 %enabled, 0
  %v = call i64 @identity_i64(i64 %x)
  ret i64 %v
}
define void @masked_vector(ptr %input, ptr %output) {
  %x = load <4 x float>, ptr %input, align 4
  %v = call <4 x float> @identity_vector(<4 x float> %x)
  store <4 x float> %v, ptr %output, align 4
  ret void
}
define i32 @__ask_value_helper_0(i32 %x) {
  ret i32 %x
}
)";

const char *helperIR = R"(
declare float @llvm.fabs.f32(float)
define i64 @increment(i64 %x) {
  %v = add i64 %x, 7
  ret i64 %v
}
define i32 @mix(i32 %x, i32 %y) {
  %twice = mul i32 %y, 2
  %v = add i32 %x, %twice
  ret i32 %v
}
define float @scale_f32(float %x) {
  %v = fmul float %x, 2.0
  %r = fadd float %v, 3.0
  ret float %r
}
define double @scale_f64(double %x) {
  %v = fmul double %x, 2.0
  %r = fadd double %v, 3.0
  ret double %r
}
define float @absolute(float %x) {
  %v = call float @llvm.fabs.f32(float %x)
  ret float %v
}
define i64 @branching(i64 %x) {
  %v = add nsw i64 %x, 1
  %positive = icmp sgt i64 %v, 0
  br i1 %positive, label %yes, label %no
yes:
  ret i64 %v
no:
  %negative = sub i64 0, %v
  ret i64 %negative
}
define <4 x float> @vector_add(<4 x float> %x) {
  %r = fadd <4 x float> %x, <float 5.0, float 5.0, float 5.0, float 5.0>
  ret <4 x float> %r
}
)";

std::unique_ptr<Module> success(LLVMContext &context) {
  auto consumer = parse(context, consumerIR);
  auto helpers = parse(context, helperIR);
  auto originalConsumer = print(*consumer), originalHelpers = print(*helpers);
  SmallVector<BoundaryInlineSite> sites;
  for (unsigned i = 0; i < 2; ++i) {
    auto *call = marker(*consumer, "twice_i64", i);
    sites.push_back({call, helpers->getFunction("increment"), {call->getArgOperand(0)}});
  }
  auto *mixed = marker(*consumer, "multiple_arguments");
  sites.push_back({mixed, helpers->getFunction("mix"),
                  {mixed->getFunction()->getArg(0), mixed->getFunction()->getArg(1)}});
  for (auto names : {std::make_pair("scalar_f32", "scale_f32"),
                     std::make_pair("scalar_f64", "scale_f64")}) {
    auto *call = marker(*consumer, names.first);
    sites.push_back({call, helpers->getFunction(names.second), {call->getArgOperand(0)}});
  }
  auto *guarded = marker(*consumer, "guarded_i64");
  auto *intrinsic = marker(*consumer, "intrinsic_abs");
  sites.push_back({intrinsic, helpers->getFunction("absolute"), {intrinsic->getArgOperand(0)}});
  sites.push_back({guarded, helpers->getFunction("branching"), {guarded->getArgOperand(0)},
                  &guarded->getFunction()->front().front(), guarded->getArgOperand(0)});
  auto *vector = marker(*consumer, "masked_vector");
  auto *mask = ConstantVector::get({ConstantInt::getTrue(context), ConstantInt::getFalse(context),
                                     ConstantInt::getTrue(context), ConstantInt::getFalse(context)});
  sites.push_back({vector, helpers->getFunction("vector_add"), {vector->getArgOperand(0)},
                  mask, vector->getArgOperand(0)});
  json::Object report;
  auto output = ask::llir::inlineValueBoundaries(*consumer, *helpers, sites, report);
  if (!output) {
    errs() << toString(output.takeError()) << '\n';
    require(false, "generic fusion must succeed");
  }
  require(print(*consumer) == originalConsumer && print(*helpers) == originalHelpers,
          "successful transaction must not mutate inputs");
  require(report.getObject("boundary_inlining")->getInteger("sites") == 8, "all sites must be reported");
  require((*output)->getFunction("__ask_value_helper_0") != nullptr, "consumer helper-like names must survive");
  for (auto &function : **output)
    if (!function.isDeclaration())
      for (auto &block : function)
        for (auto &instruction : block)
          if (auto *call = dyn_cast<CallInst>(&instruction))
            require(call->getCalledFunction() && call->getCalledFunction()->isIntrinsic(),
                    "all leaf helpers and identity calls must be inlined");
  return std::move(*output);
}

void rejection(LLVMContext &context, StringRef helper, StringRef expected) {
  auto consumer = parse(context, "declare i64 @identity(i64)\n"
                                "define i64 @f(i64 %x) { %r = call i64 @identity(i64 %x) ret i64 %r }");
  auto helpers = parse(context, helper);
  auto *call = marker(*consumer, "f");
  BoundaryInlineSite site{call, helpers->getFunction("h"), {call->getArgOperand(0)}};
  auto before = print(*consumer), helperBefore = print(*helpers);
  json::Object report{{"sentinel", 42}};
  auto result = ask::llir::inlineValueBoundaries(*consumer, *helpers, {site}, report);
  if (result) {
    errs() << "Unexpectedly accepted helper; expected diagnostic: " << expected << '\n';
    errs() << helper << '\n';
    require(false, "invalid helper must be rejected");
  }
  auto diagnostic = toString(result.takeError());
  require(diagnostic.find(expected.str()) != std::string::npos, diagnostic);
  require(print(*consumer) == before && print(*helpers) == helperBefore,
          "failed transaction must leave both input modules unchanged");
  require(report.size() == 1 && report.getInteger("sentinel") == 42,
          "failed transaction must not publish a success report");
}

void consumerNumerics(LLVMContext &context, StringRef attributes,
                      bool floating, bool accept) {
  // Both signatures are integer-only; the floating helper hides its operations
  // in a nested function. Numerical checks must inspect the whole call graph.
  auto consumer = parse(context, "declare i64 @identity(i64)\n"
      "define i64 @f(i64 %x) " + attributes.str() +
      " { %r=call i64 @identity(i64 %x) ret i64 %r }");
  auto helpers = parse(context, floating ? R"(
      define i64 @inner(i64 %x) {
        %f = sitofp i64 %x to double
        %a = fadd double %f, 1.0
        %r = fptosi double %a to i64
        ret i64 %r
      }
      define i64 @h(i64 %x) { %r=call i64 @inner(i64 %x) ret i64 %r }
      )" : "define i64 @h(i64 %x) { %r=add i64 %x, 1 ret i64 %r }");
  auto *call = marker(*consumer, "f");
  BoundaryInlineSite site{call, helpers->getFunction("h"), {call->getArgOperand(0)}};
  auto before = print(*consumer), helperBefore = print(*helpers);
  json::Object report{{"sentinel", 42}};
  auto result = ask::llir::inlineValueBoundaries(*consumer, *helpers, {site}, report);
  require(print(*consumer) == before && print(*helpers) == helperBefore,
          "numerical policy validation must preserve input modules");
  if (accept) {
    if (!result) errs() << toString(result.takeError()) << '\n';
    require(bool(result), "compatible consumer numerical attributes must be accepted");
    for (auto attribute : consumer->getFunction("f")->getAttributes().getFnAttrs())
      if (attribute.isStringAttribute())
        require((*result)->getFunction("f")->getFnAttribute(attribute.getKindAsString()) == attribute,
                "consumer target and numerical attributes must not be erased or rewritten");
  } else {
    require(!result, "incompatible consumer numerical attributes must be rejected");
    auto diagnostic = toString(result.takeError());
    require(diagnostic.find("floating helper cannot inherit consumer") != std::string::npos, diagnostic);
    require(report.size() == 1 && report.getInteger("sentinel") == 42,
            "numerical rejection must not modify the report");
  }
}
} // namespace

int main(int argc, char **argv) {
  LLVMContext context;
  auto result = success(context);
  rejection(context, R"(define i64 @h(i64 %x) memory(none) { store i64 %x, ptr null ret i64 %x })", "pointer operand");
  rejection(context, R"(define i64 @h(i64 %x) {
                        %v=add i64 %x, ptrtoint (ptr @h to i64) ret i64 %v })", "pointer operand");
  rejection(context, R"(declare i64 @foreign(i64) memory(none) speculatable
                        define i64 @h(i64 %x) { %v=call i64 @foreign(i64 %x) ret i64 %v })", "unknown external");
  rejection(context, R"(define i64 @h(i64 %x) { %v=udiv i64 10, %x ret i64 %v })", "potentially trapping");
  rejection(context, R"(define i64 @h(i64 %x) { br label %loop
                        loop: br label %loop })", "cyclic");
  rejection(context, R"(define i64 @h(i64 %x) { %v=call i64 @h(i64 %x) ret i64 %v })", "recursive");
  rejection(context, R"(define i64 @h(i64 %x) { %v=call i64 asm sideeffect "", "=r,r"(i64 %x) ret i64 %v })", "unknown");
  rejection(context, R"(target triple = "different-target"
                        define i64 @h(i64 %x) { ret i64 %x })", "target/data layout");
  rejection(context, R"(define i64 @h(i64 %x) { ret i64 %x }
                        !llvm.module.flags = !{!0}
                        !0 = !{i32 1, !"arbitrary-semantic-flag", i32 1})", "module asm or flags");
  rejection(context, R"(define i64 @h(i64 %x) "target-cpu"="different" { ret i64 %x })", "normalization");
  rejection(context, R"(define noundef i64 @h(i64 %x) { ret i64 %x })", "return attributes");
  rejection(context, R"(define i64 @h(i64 noundef %x) { ret i64 %x })", "parameter attributes");
  rejection(context, R"(define i64 @inner(i64 %x) { ret i64 %x }
                        define i64 @h(i64 %x) { %v=call i64 @inner(i64 noundef %x) ret i64 %v })", "call parameter");
  rejection(context, R"(define float @h(i64 %x) { ret float 0.0 })", "signature mismatch");

  for (StringRef attributes : {R"("unsafe-fp-math"="true")", R"("no-nans-fp-math"="true")",
       R"("no-infs-fp-math"="true")", R"("no-signed-zeros-fp-math"="true")",
       R"("approx-func-fp-math"="true")", R"("less-precise-fpmad"="true")",
       R"("no-trapping-math"="false")", R"("denormal-fp-math"="preserve-sign,preserve-sign")",
       R"("denormal-fp-math-f32"="positive-zero,positive-zero")", R"("fp-contract"="fast")",
       "strictfp"}) {
    consumerNumerics(context, attributes, true, false);
    consumerNumerics(context, attributes, false, true);
  }
  consumerNumerics(context, R"("unsafe-fp-math"="false" "no-nans-fp-math"="false"
      "no-infs-fp-math"="false" "no-signed-zeros-fp-math"="false"
      "approx-func-fp-math"="false" "less-precise-fpmad"="false"
      "no-trapping-math"="true" "denormal-fp-math"="ieee,ieee"
      "denormal-fp-math-f32"="ieee" "fp-contract"="off"
      "nvvm.reqntid"="128" "target-cpu"="generic")", true, true);
  consumerNumerics(context, R"("fp-contract"="on")", true, true);

  // A value produced after the marker is not available at the rewrite site.
  auto consumer = parse(context, "declare i64 @identity(i64)\n"
      "define i64 @f(i64 %x) { %r=call i64 @identity(i64 %x) %late=add i64 %x, 1 ret i64 %late }");
  auto helpers = parse(context, "define i64 @h(i64 %x) { ret i64 %x }");
  auto *call = marker(*consumer, "f");
  BoundaryInlineSite site{call, helpers->getFunction("h"), {call->getNextNode()}};
  json::Object report;
  auto invalid = ask::llir::inlineValueBoundaries(*consumer, *helpers, {site}, report);
  require(!invalid, "non-dominating helper argument must fail");
  consumeError(invalid.takeError());
  site.arguments[0] = call->getArgOperand(0);
  auto duplicate = ask::llir::inlineValueBoundaries(*consumer, *helpers, {site, site}, report);
  require(!duplicate, "duplicate marker must fail");
  consumeError(duplicate.takeError());
  LLVMContext differentContext;
  auto different = parse(differentContext, "define i64 @h(i64 %x) { ret i64 %x }");
  auto mismatch = ask::llir::inlineValueBoundaries(*consumer, *different, {}, report);
  require(!mismatch, "different LLVM contexts must fail");
  consumeError(mismatch.takeError());
  if (argc == 2) {
    std::error_code error;
    raw_fd_ostream file(argv[1], error);
    require(!error, "cannot create generated LLVM fixture");
    result->print(file, nullptr);
  }
  outs() << "generic LLVM boundary inlining: scalar/vector/multi-site/mask tests passed; rejection transactions preserved inputs\n";
  return 0;
}
