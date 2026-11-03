#include <ask/Frontend/CompilerAPI.h>
#include "support/primitive_graphs.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
std::size_t occurrences(const std::string &text, const std::string &part) {
  std::size_t result = 0, pos = 0;
  while ((pos = text.find(part, pos)) != std::string::npos) { ++result;pos += part.size(); }
  return result;
}
void compile(const std::string &source, bool host) {
  ask::compiler::CodegenOptions options;
  if (host) {
    const bool unsupportedBF16 = source.find("bf16") != std::string::npos &&
        ask::compilerCapabilitiesJSON().find("\"host_bf16_hardware_supported\":true") == std::string::npos;
    if (unsupportedBF16) {
      bool rejected = false;
      try { ask::compiler::codegen(source, options); }
      catch (const std::exception &error) {
        rejected = std::string(error.what()).find("bf16 requires hardware support") != std::string::npos;
      }
      require(rejected, "unsupported Host bf16 must be rejected");
    } else require(!ask::compiler::codegen(source, options).ir.empty(), "missing Host LLVM");
  }
  options.target = ask::compiler::CodegenTarget::CUDALLVM;options.architecture = 86;
  require(!ask::compiler::codegen(source, options).ir.empty(), "missing CUDA LLVM");
}
}
int main() {
  try {
    const bool host = ask::compilerCapabilitiesJSON().find("\"host_codegen\":true") != std::string::npos;
    for (const std::string operation : {"reduce_sum", "reduce_max"})
      for (int64_t extent : {0, 1, 3})
        for (bool reduceAxis : {false, true}) {
          if (operation == "reduce_max" && reduceAxis && extent == 0) continue;
          auto plan = ask::compiler::partition(ask::test::reductionExpansion(extent, reduceAxis, operation));
          const std::size_t expectedSteps = reduceAxis && extent > 1 ? 2 : 1;
          require(plan.stepSources.size() == expectedSteps, "materialize only a repeated multi-element reduction");
          for (const auto &step : plan.stepSources) compile(step.second, host);
        }
    for (const std::string dtype : {"f32", "f16", "bf16"})
      for (const std::vector<int64_t> shape : {std::vector<int64_t>{}, {7}, {2, 3}, {0, 3}}) {
        auto source = ask::test::pointwise(shape, dtype);
        auto prepared = ask::compiler::prepare(source, {true, {}});
        require(prepared.ir.find("ask.fusion_group") != std::string::npos, "new pointwise operations lost fusion");
        auto plan = ask::compiler::partition(source);
        require(plan.stepSources.size() == 1, "pointwise chain requires one owned step");
        require(occurrences(plan.executionPlanJSON, "\"element_type\":") == 3,
                "pointwise intermediates escaped into plan buffers");
        compile(source, host);
      }
    for (const std::string dtype : {"i1", "i32", "i64", "f32", "f16", "bf16"}) {
      auto source = ask::test::spatial(dtype);
      auto plan = ask::compiler::partition(source);
      require(plan.stepSources.size() == 1, "index transformations split an owned region");
      require(occurrences(plan.executionPlanJSON, "\"element_type\":") == 2,
              "index transformation created a plan intermediate");
      compile(source, host);
    }
    for (const std::string operation : {"reduce_sum", "reduce_max"})
      for (bool full : {false, true}) {
        auto plan = ask::compiler::partition(ask::test::reductionBroadcast(full, operation));
        require(plan.stepSources.size() == 2, "broadcast must reuse a materialized reduction");
        if (full) require(plan.executionPlanJSON.find("\"shape\":[]") != std::string::npos,
                          "full reduction lost its scalar buffer");
        for (const auto &step : plan.stepSources) compile(step.second, host);
      }
    for (const std::string dtype : {"f32", "f16", "bf16"})
      for (int64_t batch : {0, 2})
        for (bool transposeA : {false, true})
          for (bool transposeB : {false, true}) {
            auto source = ask::test::matmul(dtype, batch, transposeA, transposeB);
            auto plan = ask::compiler::partition(source);
            require(plan.stepSources.size() == 3, "matmul lost the owned/library/owned boundary");
            require(occurrences(plan.executionPlanJSON, "\"library\":\"cublaslt\"") == 1,
                    "matmul did not select cuBLASLt");
            for (const auto &field : {"\"m\":5", "\"n\":7", "\"k\":3"})
              require(plan.executionPlanJSON.find(field) != std::string::npos, "wrong logical matmul dimensions");
            for (const auto &step : plan.stepSources)
              if (step.second.find("\"ask.matmul\"") == std::string::npos) compile(step.second, host);
          }
    std::cout << "primitive shape, fusion, reduction and library partition matrices passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';return 1;
  }
}
