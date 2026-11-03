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
    if (source.find("bf16") != std::string::npos) {
      bool rejected = false;
      try { ask::compiler::codegen(source, options); }
      catch (const std::exception &error) {
        rejected = std::string(error.what()).find("Host backend supports only bool/i32/i64/f32/f16") != std::string::npos;
      }
      require(rejected, "unsupported Host bf16 must be rejected");
    } else require(!ask::compiler::codegen(source, options).ir.empty(), "missing Host LLVM");
  }
  options.target = ask::compiler::CodegenTarget::CUDALLVM;options.architecture = 86;
  require(!ask::compiler::codegen(source, options).ir.empty(), "missing CUDA LLVM");
}
void hostWorkspace(const std::string &source, std::size_t bytes, std::size_t buffers) {
  auto result = ask::compiler::codegen(source, {});
  const auto &abi = result.abiJSON.value();
  require(abi.find("\"temporary_workspace_bytes\":" + std::to_string(bytes)) != std::string::npos,
          "wrong Host reduction workspace capacity");
  require(abi.find("\"materialized_tensor_count\":" + std::to_string(buffers)) != std::string::npos,
          "wrong Host reduction materialization count");
  require(occurrences(result.ir, "call ptr @malloc(") == (bytes ? 1 : 0) &&
          occurrences(result.ir, "call void @free(") == (bytes ? 1 : 0),
          "Host workspace must have one allocation lifetime");
}
}
int main() {
  try {
    const bool host = ask::compilerCapabilitiesJSON().find("\"host_codegen\":true") != std::string::npos;
    for (const std::string dtype : {"f32", "f16", "bf16"})
      for (bool constant : {false, true}) {
        auto plan = ask::compiler::partition(ask::test::publicScalar(dtype, constant));
        require(plan.stepSources.size() == 1, "public scalar requires one producing owned step");
        require(occurrences(plan.executionPlanJSON, "\"shape\":[]") == (constant ? 1 : 2),
                "public scalars must use one-element device buffers");
        compile(plan.stepSources.begin()->second, host);
      }
    if (host)
      for (const std::string operation : {"reduce_sum", "reduce_max"})
        for (int64_t inputSize : {0, 1, 8})
          for (int64_t outputSize : {0, 8}) {
            if (operation == "reduce_max" && inputSize == 0) continue;
            auto source = ask::test::scalarReductionExpansion(inputSize, outputSize, operation);
            hostWorkspace(source, 0, 0);
            auto lowered = ask::compiler::codegen(source, {});
            if (outputSize == 0 || inputSize <= 1)
              require(lowered.ir.find("reduce.loop:") == std::string::npos,
                      "empty output and trivial reduction require no reduction loop");
            else require(lowered.ir.find("reduce.loop:") < lowered.ir.find("output.loop:"),
                         "scalar reduction must run once before splat");
          }
    for (const std::string operation : {"reduce_sum", "reduce_max"})
      for (int64_t extent : {0, 1, 3})
        for (bool reduceAxis : {false, true}) {
          if (operation == "reduce_max" && reduceAxis && extent == 0) continue;
          auto plan = ask::compiler::partition(ask::test::reductionExpansion(extent, reduceAxis, operation));
          const std::size_t expectedSteps = reduceAxis && extent > 1 ? 2 : 1;
          require(plan.stepSources.size() == expectedSteps, "materialize only a repeated multi-element reduction");
          for (const auto &step : plan.stepSources) compile(step.second, host);
          if (host) hostWorkspace(ask::test::reductionExpansion(extent, reduceAxis, operation),
                                  expectedSteps == 2 ? 8 : 0, expectedSteps - 1);
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
        auto source = ask::test::reductionBroadcast(full, operation);
        auto plan = ask::compiler::partition(source);
        require(plan.stepSources.size() == 2, "broadcast must reuse a materialized reduction");
        if (full) require(plan.executionPlanJSON.find("\"shape\":[]") != std::string::npos,
                          "full reduction lost its scalar buffer");
        for (const auto &step : plan.stepSources) compile(step.second, host);
        if (host) {
          hostWorkspace(source, full ? 0 : 8, full ? 0 : 1);
          if (full) {
            auto lowered = ask::compiler::codegen(source, {});
            require(lowered.ir.find("reduce.loop:") < lowered.ir.find("output.loop:"),
                    "scalar reduction must precede the expanding output loop");
          }
        }
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
