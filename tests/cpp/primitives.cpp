#include <ask/Frontend/CompilerAPI.h>
#include "support/primitive_graphs.h"

#include <algorithm>
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
bool integerField(const std::string &json, const std::string &name, std::size_t value) {
  auto token = "\"" + name + "\":" + std::to_string(value);
  auto found = json.find(token);
  if (found == std::string::npos) return false;
  auto end = json.find_first_not_of(" \t\n\r", found + token.size());
  return end != std::string::npos && (json[end] == ',' || json[end] == '}');
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
ask::compiler::CodegenResult hostWorkspace(const std::string &source, std::size_t bytes,
    std::size_t buffers, const ask::compiler::CodegenOptions &options = {}) {
  auto result = ask::compiler::codegen(source, options);
  const auto &abi = result.abiJSON.value();
  require(integerField(abi, "temporary_workspace_bytes", bytes),
          "wrong Host reduction workspace capacity");
  require(integerField(abi, "materialized_tensor_count", buffers),
          "wrong Host reduction materialization count");
  require(occurrences(result.ir, "call ptr @malloc(") == (bytes ? 1 : 0) &&
          occurrences(result.ir, "call void @free(") == (bytes ? 1 : 0),
          "Host workspace must have one allocation lifetime");
  return result;
}
void workspaceMatrix() {
  const std::vector<std::pair<std::string, std::size_t>> types{
      {"i1", 1}, {"i32", 4}, {"i64", 8}, {"f32", 4}, {"f16", 2}};
  std::size_t cases = 0;
  for (const auto &[dtype, elementBytes] : types)
    for (int64_t rows : {0, 1, 5})
      for (int64_t width : {0, 1, 3})
        for (const std::string operation : {"reduce_sum", "reduce_max"}) {
          if (width == 0 && operation == "reduce_max") continue;
          for (bool transposed : {false, true})
            for (bool retainFirst : {false, true})
              for (bool fused : {false, true}) {
                ask::compiler::CodegenOptions options;
                options.optimization.fuseElementwise = fused;
                const bool stored = rows > 1 && width > 1;
                const auto buffers = stored ? 3 : 0;
                const auto slots = stored ? (retainFirst ? 3 : 2) : 0;
                auto result = hostWorkspace(ask::test::materializationChain(
                    {dtype, dtype, dtype}, rows, width, operation, transposed, retainFirst),
                    slots * rows * elementBytes, buffers, options);
                const auto outputLoops = rows && width ? (fused ? 1 : 2) + retainFirst : 0;
                const auto reductionLoops = rows && width > 1 ? 3 : 0;
                const auto branches = outputLoops + reductionLoops + buffers + (stored ? 1 : 0);
                require(occurrences(result.ir, "br i1 ") == std::size_t(branches),
                        "Host workspace validation added execution branches");
                ++cases;
              }
        }
  for (const auto &[first, firstBytes] : types)
    for (const auto &[next, nextBytes] : types)
      for (bool captureScalar : {false, true}) {
        // The first slot is reused by a differently sized and aligned dtype.
        const std::size_t rows = 5;
        const auto alignment = std::max(firstBytes, nextBytes);
        const auto firstCapacity = rows * alignment;
        const auto secondOffset = (firstCapacity + nextBytes - 1) & ~(nextBytes - 1);
        hostWorkspace(ask::test::materializationChain({first, next, next}, rows, 3,
                      "reduce_sum", false, false, captureScalar),
                      secondOffset + rows * nextBytes, 3);
        ++cases;
      }
  std::cout << "Host workspace matrix: " << cases << " cases\n";
}
void hostMatmulMatrix() {
  std::size_t cases = 0;
  for (const std::string dtype : {"f32", "f16"})
    for (int64_t batch : {0, 2})
     for (int64_t rows : {3, 4, 5, 8})
      for (int64_t columns : {1, 15, 16, 17, 32, 33})
        for (bool transposeA : {false, true})
          for (bool transposeB : {false, true})
            for (bool epilogue : {false, true}) {
              const auto elements = (batch ? batch : 1) * rows * columns;
              const auto bytes = elements * (dtype == "f32" ? 4 : 2);
              auto result = hostWorkspace(ask::test::hostMatmul(
                  dtype, batch, rows, columns, transposeA, transposeB, epilogue),
                  epilogue ? bytes : 0, epilogue ? 1 : 0);
              if (dtype == "f32") {
                require(result.ir.find("call void @ask_cblas_sgemm") != std::string::npos &&
                        result.abiJSON->find("\"link_libraries\":[\"openblas\"]") != std::string::npos,
                        "Host f32 matmul must declare and invoke its OpenBLAS dependency");
              } else {
                require(result.ir.find("fmul <") != std::string::npos,
                        "Host f16 matmul must accumulate independent columns in f32");
                require(result.ir.find("fast ") == std::string::npos &&
                        result.ir.find("contract ") == std::string::npos &&
                        result.ir.find("@llvm.fma") == std::string::npos &&
                        result.ir.find("ask_cblas_sgemm") == std::string::npos,
                        "Host f16 matmul changed separate multiply/add rounding");
              }
              require(result.ir.find("@llvm.masked") == std::string::npos,
                      "static matmul tails must not require runtime lane masks");
              ++cases;
            }
  std::cout << "Host matmul layout matrix: " << cases << " cases\n";
  for (unsigned axis = 0; axis < 3; ++axis) for (int64_t dimension : {int64_t(2147483647), int64_t(2147483648)}) {
    int64_t m = axis == 0 ? dimension : 1, n = axis == 1 ? dimension : 1, k = axis == 2 ? dimension : 1;
    auto a = ask::test::tensor({m, k}, "f32"), b = ask::test::tensor({k, n}, "f32"), c = ask::test::tensor({m, n}, "f32");
    const std::string source = "module { func.func @boundary(%a: " + a + ", %b: " + b + ") -> " + c +
        " { %y = \"ask.matmul\"(%a, %b) : (" + a + ", " + b + ") -> " + c + " return %y : " + c + " } }";
    bool rejected = false;
    try { ask::compiler::codegen(source, {}); }
    catch (const std::runtime_error &error) { rejected = std::string(error.what()).find("OpenBLAS LP64 range") != std::string::npos; }
    require(rejected == (dimension > 2147483647), "OpenBLAS matrix dimensions were not checked before code generation");
  }
}
void cudaReductionMatrix() {
  ask::compiler::CodegenOptions options;
  options.target = ask::compiler::CodegenTarget::CUDALLVM;
  options.architecture = 86;
  std::size_t cases = 0;
  for (const std::string dtype : {"i32", "i64", "f32", "f16", "bf16"})
    for (int64_t rows : {0, 1, 9})
      for (int64_t width : {1, 31, 32, 33, 64, 65})
        for (bool transposed : {false, true})
          for (bool epilogue : {false, true})
            for (bool retainInput : {false, true}) {
              auto result = ask::compiler::codegen(ask::test::cudaReduction(
                  dtype, rows, width, transposed, epilogue, retainInput), options);
              const bool cooperative = rows && width >= 32;
              const auto call = "call " + std::string(dtype == "i32" || dtype == "i64" ? "i32" : "float") +
                  " @llvm.nvvm.shfl.sync.bfly.";
              require(occurrences(result.ir, call) == (cooperative ? (dtype == "i64" ? 10 : 5) : 0),
                      "CUDA max selected an invalid warp reduction");
              require((result.ir.find("reduce.tail:") != std::string::npos) == (cooperative && width % 32),
                      "CUDA max must emit a tail load only for a partial warp");
              const int64_t workers = cooperative ? 4 : 128;
              auto blocks = rows / workers + (rows % workers != 0);
              if (retainInput) blocks = std::max(blocks, rows * width / 128 + (rows * width % 128 != 0));
              blocks = std::max<int64_t>(blocks, 1);
              require(result.cudaArtifactJSON->find("\"grid\":[" + std::to_string(blocks) + ",1,1]") != std::string::npos,
                      "CUDA mixed output groups have an incorrect launch grid");
              require(integerField(*result.cudaArtifactJSON, "shared_bytes", 0),
                      "warp max must preserve the zero shared-memory launch contract");
              ++cases;
            }
  for (unsigned block : {31, 32, 33, 128})
    for (unsigned architecture : {60, 70, 86})
      for (const std::string operation : {"reduce_max", "reduce_sum"}) {
        options.blockSize = block;
        options.architecture = architecture;
        auto result = ask::compiler::codegen(
            ask::test::cudaReduction("f32", 9, 65, false, true, false, operation), options);
        const bool cooperative = (operation == "reduce_max" || operation == "reduce_sum") && architecture >= 70 && block % 32 == 0;
        require((result.ir.find("@llvm.nvvm.shfl.sync") != std::string::npos) == cooperative,
                "CUDA warp choice changed the architecture/block/sum contract");
        ++cases;
      }
  options.blockSize = 128;
  options.architecture = 86;
  for (const std::string dtype : {"i32", "i64", "f32", "f16", "bf16"})
    for (int64_t rows : {1, 9})
      for (int64_t width : {32, 65}) {
        auto result = ask::compiler::codegen(ask::test::nestedCudaMaximum(dtype, rows, width), options);
        require(result.ir.find("@llvm.nvvm.shfl.sync") == std::string::npos,
                "nested reductions must retain the indexed scalar schedule");
        ++cases;
      }
  std::cout << "CUDA reduction schedule matrix: " << cases << " cases\n";
}
}
static void stage2Tests(bool host) {
  using namespace ask::compiler;
  const auto checkIR = [&](const std::string &mlir) {
    (void)prepare(mlir, {true, {}});
    compile(mlir, host);
  };
  CodegenOptions cudaOpts;
  cudaOpts.target = CodegenTarget::CUDALLVM;
  cudaOpts.architecture = 86;
  cudaOpts.blockSize = 128;

  // Unary/binary float & int math + bitwise + shifts + iota
  checkIR(R"(
    module {
      func.func @main(%x: tensor<2x4xf32>, %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
        %0 = "ask.sin"(%x) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %1 = "ask.cos"(%y) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %2 = "ask.tanh"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %3 = "ask.erf"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %4 = "ask.abs"(%2) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %5 = "ask.floor"(%3) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %6 = "ask.ceil"(%4) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %7 = "ask.round"(%5) : (tensor<2x4xf32>) -> tensor<2x4xf32>
        %8 = "ask.pow"(%6, %7) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %9 = "ask.maximum"(%8, %x) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %10 = "ask.minimum"(%9, %y) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %11 = "ask.floordiv"(%10, %y) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %12 = "ask.mod"(%11, %y) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %13 = "ask.iota"() {dimension = 1 : i64} : () -> tensor<2x4xf32>
        %14 = "ask.add"(%12, %13) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        return %14 : tensor<2x4xf32>
      }
    }
  )");
  checkIR(R"(
    module {
      func.func @main(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
        %0 = "ask.abs"(%a) : (tensor<4xi32>) -> tensor<4xi32>
        %1 = "ask.floordiv"(%0, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %2 = "ask.mod"(%1, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %3 = "ask.pow"(%2, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %4 = "ask.bitwise_and"(%3, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %5 = "ask.bitwise_or"(%4, %a) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %6 = "ask.bitwise_xor"(%5, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %7 = "ask.bitwise_not"(%6) : (tensor<4xi32>) -> tensor<4xi32>
        %8 = "ask.shift_left"(%7, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %9 = "ask.shift_right"(%8, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        return %9 : tensor<4xi32>
      }
    }
  )");
  // Slice (negative step), Pad (interior), Concatenate, Gather, Argmax/Argmin, Boolean/Prod/Min reductions
  checkIR(R"(
    module {
      func.func @main(%x: tensor<4x5xf32>, %p: f32, %idx: tensor<3xi64>, %flag: tensor<2x3xi1>)
          -> (tensor<2x3xf32>, tensor<1x3xf32>, tensor<5xi64>, tensor<2xi1>) {
        %0 = "ask.slice"(%x) {starts = array<i64: 3, 1>, stops = array<i64: -1, 4>, steps = array<i64: -2, 1>}
            : (tensor<4x5xf32>) -> tensor<2x3xf32>
        %1 = "ask.pad"(%x, %p) {pad_low = array<i64: 1, 0>, pad_high = array<i64: 0, 1>, interior = array<i64: 1, 0>}
            : (tensor<4x5xf32>, f32) -> tensor<8x6xf32>
        %2 = "ask.reduce_min"(%1) {axes = array<i64: 0>} : (tensor<8x6xf32>) -> tensor<6xf32>
        %3 = "ask.reduce_prod"(%0) {axes = array<i64: 0>} : (tensor<2x3xf32>) -> tensor<3xf32>
        %4 = "ask.reshape"(%3) {shape = array<i64: 1, 3>} : (tensor<3xf32>) -> tensor<1x3xf32>
        %5 = "ask.gather"(%x, %idx) {axis = 1 : i64} : (tensor<4x5xf32>, tensor<3xi64>) -> tensor<4x3xf32>
        %6 = "ask.concatenate"(%0, %5) {axis = 0 : i64} : (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>
        %7 = "ask.slice"(%6) {starts = array<i64: 0, 0>, stops = array<i64: 1, 3>, steps = array<i64: 1, 1>}
            : (tensor<6x3xf32>) -> tensor<1x3xf32>
        %8 = "ask.add"(%4, %7) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
        %9 = "ask.argmax"(%x) {axis = 0 : i64} : (tensor<4x5xf32>) -> tensor<5xi64>
        %10 = "ask.reduce_any"(%flag) {axes = array<i64: 1>} : (tensor<2x3xi1>) -> tensor<2xi1>
        return %0, %8, %9, %10 : tensor<2x3xf32>, tensor<1x3xf32>, tensor<5xi64>, tensor<2xi1>
      }
    }
  )");
  // Argmax must not attempt warp shuffle on CUDA; ReduceMin warp reduction uses shuffle.
  {
    auto cudaArg = codegen(R"(
      module {
        func.func @main(%x: tensor<4x64xf32>) -> tensor<4xi64> {
          %0 = "ask.argmax"(%x) {axis = 1 : i64} : (tensor<4x64xf32>) -> tensor<4xi64>
          return %0 : tensor<4xi64>
        }
      }
    )", cudaOpts);
    require(cudaArg.ir.find("@llvm.nvvm.shfl.sync") == std::string::npos,
            "argmax must use sequential loop without warp shuffle");

    auto cudaMin = codegen(R"(
      module {
        func.func @main(%x: tensor<4x64xf32>) -> tensor<4xf32> {
          %0 = "ask.reduce_min"(%x) {axes = array<i64: 1>} : (tensor<4x64xf32>) -> tensor<4xf32>
          return %0 : tensor<4xf32>
        }
      }
    )", cudaOpts);
    require(cudaMin.ir.find("@llvm.nvvm.shfl.sync") != std::string::npos,
            "reduce_min should use warp shuffle when domain >= 32");
  }
  // ScatterAdd partition isolation & gather reduction materialization
  {
    const std::string scatterSrc = R"(
      module {
        func.func @main(%b: tensor<4x3xf32>, %idx: tensor<2xi64>, %u: tensor<2x3xf32>) -> tensor<4x3xf32> {
          %0 = "ask.scatter_add"(%b, %idx, %u) {axis = 0 : i64}
              : (tensor<4x3xf32>, tensor<2xi64>, tensor<2x3xf32>) -> tensor<4x3xf32>
          %1 = "ask.abs"(%0) : (tensor<4x3xf32>) -> tensor<4x3xf32>
          return %1 : tensor<4x3xf32>
        }
      }
    )";
    auto p = partition(scatterSrc);
    require(p.stepSources.size() == 2, "scatter_add must stay isolated in its own partition");
    bool foundAtomic = false;
    for (const auto &entry : p.stepSources)
      if (entry.second.find("ask.scatter_add") != std::string::npos)
        foundAtomic |= codegen(entry.second, cudaOpts).ir.find("atomicrmw fadd") != std::string::npos;
    require(foundAtomic, "scatter_add CUDA LLVM must emit atomicrmw fadd");
    if (host) {
      auto h = codegen(scatterSrc, {});
      require(h.ir.find("scatter.host.apply") != std::string::npos,
              "scatter_add Host LLVM lowering failed");
    }
    // Gather reading a multi-element reduction must materialize the reduction first.
    auto gp = partition(R"(
      module {
        func.func @main(%x: tensor<4x3xf32>, %idx: tensor<2xi64>) -> tensor<2xf32> {
          %0 = "ask.reduce_sum"(%x) {axes = array<i64: 1>} : (tensor<4x3xf32>) -> tensor<4xf32>
          %1 = "ask.gather"(%0, %idx) {axis = 0 : i64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<2xf32>
          return %1 : tensor<2xf32>
        }
      }
    )");
    require(gp.stepSources.size() == 2, "gather over reduction must materialize reduction");
  }
  // Negative validation cases
  const auto rejectIR = [](const std::string &mlir) {
    bool rejected = false;
    try { (void)prepare(mlir, {}); } catch (const std::exception &) { rejected = true; }
    require(rejected, "invalid Stage 2 IR unexpectedly passed prepare/verify");
  };
  rejectIR(R"(module { func.func @main(%x: tensor<4xi32>) -> tensor<4xi32> { %0 = "ask.sin"(%x) : (tensor<4xi32>) -> tensor<4xi32> return %0 : tensor<4xi32> } })");
  rejectIR(R"(module { func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> { %0 = "ask.bitwise_not"(%x) : (tensor<4xf32>) -> tensor<4xf32> return %0 : tensor<4xf32> } })");
  rejectIR(R"(module { func.func @main(%x: tensor<0x4xf32>) -> tensor<4xi64> { %0 = "ask.argmax"(%x) {axis = 0 : i64} : (tensor<0x4xf32>) -> tensor<4xi64> return %0 : tensor<4xi64> } })");
  rejectIR(R"(module { func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> { %0 = "ask.slice"(%x) {starts = array<i64: 0>, stops = array<i64: 5>, steps = array<i64: 1>} : (tensor<4xf32>) -> tensor<4xf32> return %0 : tensor<4xf32> } })");
}

int main() {
  try {
    const bool host = ask::compilerCapabilitiesJSON().find("\"host_codegen\":true") != std::string::npos;
    if (host) { workspaceMatrix();hostMatmulMatrix(); }
    cudaReductionMatrix();
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
    stage2Tests(host);
    std::cout << "primitive shape, fusion, reduction and library partition matrices passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';return 1;
  }
}
