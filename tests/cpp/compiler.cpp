#include "ask/Frontend/CompilerAPI.h"

#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
const std::string source = R"mlir(module attributes {ask.schema_version = 1 : i64} {
  func.func @literals(%x: tensor<1xi64>) -> (tensor<1xi64>, tensor<1xf32>) {
    %limit = "ask.constant"() {value = 9223372036854775807 : i64} : () -> i64
    %wide = "ask.splat"(%limit) {shape = array<i64: 1>} : (i64) -> tensor<1xi64>
    %sum = "ask.add"(%x, %wide) {overflow = "wrap"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %zero = "ask.constant"() {value = -0.0 : f32} : () -> f32
    %zeros = "ask.splat"(%zero) {shape = array<i64: 1>} : (f32) -> tensor<1xf32>
    return %sum, %zeros : tensor<1xi64>, tensor<1xf32>
  }
})mlir";

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void checkDiagnostics(const std::string &diagnostics) {
  for (const auto *field : {"shape_snapshot", "external_requests", "execution_plan",
                            "step_sources", "abi", "cuda_artifact"})
    require(diagnostics.find(std::string("\"") + field + "\":") == std::string::npos,
            "diagnostics contains a machine contract");
  require(diagnostics.find("\"phase\":") != std::string::npos, "diagnostics lost its phase");
}

void checkTypedCompiler(bool host, const std::string &objectPath) {
  using namespace ask::compiler;
  static_assert(!std::is_same_v<PrepareResult, PartitionResult>);
  static_assert(!std::is_same_v<PrepareResult, CodegenResult>);
  PrepareOptions preparation;
  preparation.optimize = true;
  preparation.optimization = {false, false, false};
  auto prepared = prepare(source, preparation);
  auto wire = ask::prepareJSON(source, true, false, false, false);
  require(prepared.ir == wire.first, "typed prepare options disagree with wire entrypoint");
  require(wire.second.find("\"shape_snapshot\":" + prepared.shapeSnapshotJSON) != std::string::npos,
          "typed shape snapshot differs from wire contract");
  require(prepared.externalRequestsJSON == "[]", "owned graph has external requests");
  checkDiagnostics(prepared.diagnosticsJSON);

  PartitionOptions partitioning;
  partitioning.optimization = preparation.optimization;
  auto partitioned = partition(source, partitioning);
  auto planWire = ask::partitionJSON(source, true, false, false, false);
  require(partitioned.ir == planWire.first, "typed partition options disagree with wire entrypoint");
  require(planWire.second.find("\"execution_plan\":" + partitioned.executionPlanJSON) != std::string::npos,
          "typed execution plan differs from wire contract");
  require(partitioned.stepSources.size() == 1, "owned graph should have one step source");
  require(!prepare(partitioned.stepSources.begin()->second).ir.empty(), "step source is not standalone");
  checkDiagnostics(partitioned.diagnosticsJSON);

  CodegenOptions cudaOptions;
  cudaOptions.target = CodegenTarget::CUDALLVM;
  cudaOptions.architecture = 86;
  auto cuda = codegen(source, cudaOptions);
  auto cudaWire = ask::codegenJSON(source, "cuda-llvm", "", 86);
  require(cuda.target == CodegenTarget::CUDALLVM && cuda.outputPath.empty() && !cuda.ir.empty(),
          "CUDA result lost its target or assembly");
  require(cuda.ir == cudaWire.first && cuda.cudaArtifactJSON && !cuda.abiJSON,
          "CUDA result has wrong payload kind");
  require(cudaWire.second.find("\"cuda_artifact\":" + *cuda.cudaArtifactJSON) != std::string::npos,
          "typed CUDA artifact differs from wire contract");
  require(cuda.ir.find("9223372036854775807") != std::string::npos &&
              cuda.ir.find("-0.000000e+00") != std::string::npos,
          "typed codegen changed literal precision");
  checkDiagnostics(cuda.diagnosticsJSON);

  for (auto bad : {CodegenOptions{}, cudaOptions}) {
    bad.outputPath = objectPath;
    bool rejected = false;
    try { codegen(source, bad); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "LLVM target accepted an object output path");
  }
  CodegenOptions invalid;
  invalid.target = static_cast<CodegenTarget>(-1);
  bool rejected = false;
  try { codegen(source, invalid); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "invalid codegen enum was accepted");
  invalid.target = CodegenTarget::HostObject;
  rejected = false;
  try { codegen(source, invalid); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "object mode accepted a missing output path");

  if (host) {
    CodegenOptions hostOptions;
    auto hostLLVM = codegen(source, hostOptions);
    require(hostLLVM.abiJSON && !hostLLVM.cudaArtifactJSON && !hostLLVM.ir.empty() && hostLLVM.outputPath.empty(),
            "Host LLVM result has wrong payload kind");
    checkDiagnostics(hostLLVM.diagnosticsJSON);
    hostOptions.target = CodegenTarget::HostObject;
    hostOptions.outputPath = objectPath;
    auto object = codegen(source, hostOptions);
    require(object.target == CodegenTarget::HostObject && object.outputPath == objectPath && object.ir.empty() &&
                object.abiJSON && !object.cudaArtifactJSON, "Host object result has wrong output contract");
    checkDiagnostics(object.diagnosticsJSON);
    std::ifstream file(object.outputPath, std::ios::binary);
    char magic[4]{};
    file.read(magic, sizeof(magic));
    require(std::string(magic, 4) == "\x7f" "ELF", "typed codegen did not write a host object");
  }
}

void checkReshape(bool host) {
  const std::string reshape = R"mlir(module {
    func.func @reshape(%x: tensor<4xf32>) -> (tensor<?x2xf32>, tensor<4xf32>) {
      %matrix = "ask.reshape"(%x) {shape = array<i64: 2, 2>, ask.id = "matrix"} : (tensor<4xf32>) -> tensor<?x2xf32>
      %square = "ask.mul"(%matrix, %matrix) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
      %flat = "ask.reshape"(%matrix) {shape = array<i64: 4>} : (tensor<?x2xf32>) -> tensor<4xf32>
      return %square, %flat : tensor<?x2xf32>, tensor<4xf32>
    }
  })mlir";
  auto prepared = ask::prepareJSON(reshape, true);
  require(prepared.first.find('?') == std::string::npos, "reshape did not refine its declared shape");
  require(prepared.first.find("ask.fusion_group") != std::string::npos, "reshape prevents owned fusion");
  require(ask::prepareJSON(prepared.first, true).first == prepared.first, "reshape preparation is not stable");
  auto cuda = ask::codegenJSON(reshape, "cuda-llvm", "", 86);
  require(cuda.first.find("fmul") != std::string::npos, "reshape lost its producer arithmetic");
  if (host) require(!ask::codegenJSON(reshape, "host-llvm").first.empty(), "reshape host lowering failed");

  auto graph = [](const std::string &input, const std::string &output, const std::string &shape) {
    return "module { func.func @view(%x: " + input + ") -> " + output +
        " { %r = \"ask.reshape\"(%x) {shape = array<i64" + shape + ">} : (" + input +
        ") -> " + output + " return %r : " + output + " } }";
  };
  for (const auto &valid : {graph("tensor<1xi1>", "tensor<i1>", ""),
                             graph("tensor<0x4xf32>", "tensor<3x0xf32>", ": 3, 0")}) {
    ask::prepareJSON(valid);
    ask::codegenJSON(valid, "cuda-llvm", "", 86);
    if (host) ask::codegenJSON(valid, "host-llvm");
  }
  for (const auto &invalid : {
           graph("tensor<4xf32>", "tensor<3xf32>", ": 3"),
           graph("tensor<4xf32>", "tensor<2x2xi32>", ": 2, 2"),
           graph("tensor<4xf32>", "tensor<2x2xf32>", ": 4"),
           graph("tensor<4xf32>", "tensor<?xf32>", ": -1"),
           graph("f32", "tensor<1xf32>", ": 1"),
           graph("tensor<1xf32>", "tensor<9223372036854775807xf32>", ": 9223372036854775807")}) {
    bool rejected = false;
    try { ask::prepareJSON(invalid); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid reshape was accepted");
  }
  const std::string external = R"mlir(module {
    "ask.semantics"() {sym_name = "sem", function_type = (tensor<2x2xf32>) -> tensor<2x2xf32>, shape_rule = "fixed", effects = "none"} : () -> ()
    "ask.implementation"() {sym_name = "impl", compiler = "test", source = "source", entry = "kernel", signature = {X = "*fp32", Y = "*fp32"}, source_bindings = {X = "input:0", Y = "output:0"}, constants = {whole = 1.0 : f64, nested = {values = [-0.0 : f64, 9223372036854775807 : i64]}}, options = {}, target = {backend = "cuda", arch = 86 : i64}, grid = array<i64: 1, 1, 1>} : () -> ()
    func.func @external_shape(%x: tensor<4xf32>) -> tensor<2x2xf32> {
      %matrix = "ask.reshape"(%x) {shape = array<i64: 2, 2>, ask.id = "matrix", ask.value_id = "reshaped"} : (tensor<4xf32>) -> tensor<2x2xf32>
      %marked = "ask.mark"(%matrix) {mark_id = "matrix_shape", value_id = "reshaped", role = "value"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
      %result = "ask.external"(%marked) {semantics = @sem, implementation = @impl} : (tensor<2x2xf32>) -> tensor<2x2xf32>
      return %result : tensor<2x2xf32>
    }
  })mlir";
  auto plan = ask::partitionJSON(external);
  require(plan.second.find("\"port_shapes\":{\"input:0\":[2,2],\"output:0\":[2,2]}") != std::string::npos,
          "reshape flattened external kernel shape metadata");
  require(plan.second.find("matrix_shape") != std::string::npos, "reshape lost its source marker");
  require(plan.second.find("\"kind\":\"owned\"") != std::string::npos &&
              plan.second.find("\"kind\":\"external\"") != std::string::npos,
          "reshape was not partitioned as owned computation");
  auto typedPlan = ask::compiler::partition(external);
  auto typedPreparation = ask::compiler::prepare(external);
  for (const auto &requests : {typedPlan.externalRequestsJSON, typedPreparation.externalRequestsJSON}) {
    require(requests.find("\"whole\":1.00000000000000000e+00") != std::string::npos,
            "typed external requests changed an integral float to an integer");
    require(requests.find("[-0.00000000000000000e+00,9223372036854775807]") != std::string::npos,
            "typed external requests lost nested negative zero or i64 precision");
    require(plan.second.find("\"external_requests\":" + requests) != std::string::npos,
            "typed external requests differ from the wire contract");
  }
  require(typedPlan.stepSources.size() == 2, "mixed graph lost partition source modules");
  checkDiagnostics(typedPlan.diagnosticsJSON);
  checkDiagnostics(typedPreparation.diagnosticsJSON);
}

void checkHostTypes(const std::string &capabilities, const std::string &objectPath) {
  using namespace ask::compiler;
  require(capabilities.find("\"host_supported_element_types\":[\"bool\",\"i32\",\"i64\",\"f32\",\"f16\"]") != std::string::npos,
          "Host type capabilities disagree with its dtype boundary");
  for (const std::string type : {"bf16", "tensor<bf16>", "tensor<0xbf16>", "tensor<2xbf16>"}) {
    const std::string graph = "module { func.func @identity(%x: " + type + ") -> " + type +
        " { return %x : " + type + " } }";
    for (const auto target : {CodegenTarget::HostLLVM, CodegenTarget::HostObject})
      for (bool optimize : {false, true}) {
        CodegenOptions options;
        options.target = target;
        options.optimize = optimize;
        if (target == CodegenTarget::HostObject) options.outputPath = objectPath;
        bool rejected = false;
        try { codegen(graph, options); }
        catch (const std::runtime_error &error) {
          rejected = std::string(error.what()).find("Host backend supports only bool/i32/i64/f32/f16") != std::string::npos;
        }
        require(rejected, "Host codegen accepted a bf16 graph");
      }
    CodegenOptions cuda;
    cuda.target = CodegenTarget::CUDALLVM;
    cuda.architecture = 86;
    require(!codegen(graph, cuda).ir.empty(), "CUDA bf16 support changed with the Host boundary");
  }
  // A constant fold or DCE must not erase an unsupported input graph's dtype.
  for (bool returned : {false, true}) {
    const std::string graph = "module { func.func @constant(%x: f32) -> f32 { "
        "%b = \"ask.constant\"() {value = 1.0 : bf16} : () -> bf16 "
        "%f = \"ask.cast\"(%b) : (bf16) -> f32 return " +
        std::string(returned ? "%f" : "%x") + " : f32 } }";
    bool rejected = false;
    try { codegen(graph, {}); }
    catch (const std::runtime_error &error) {
      rejected = std::string(error.what()).find("Host backend supports only bool/i32/i64/f32/f16") != std::string::npos;
    }
    require(rejected, "Host optimization bypassed its bf16 dtype boundary");
    require(!prepare(graph).ir.empty(), "target-independent preparation rejected bf16");
  }
}
} // namespace

// Compiler-only consumer: the public facade exposes no LLVM/MLIR or Python type.
int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: ask-compiler-test OUTPUT.o");
    auto prepared = ask::prepareJSON(source, true);
    require(prepared.second.find("shape_snapshot") != std::string::npos, "missing shape report");
    auto partitioned = ask::partitionJSON(source);
    require(partitioned.second.find("execution_plan") != std::string::npos, "missing execution plan");
    auto cuda = ask::codegenJSON(source, "cuda-llvm", "", 86);
    require(cuda.first.find("ptx_kernel") != std::string::npos, "missing CUDA entry");
    require(cuda.first.find("9223372036854775807") != std::string::npos, "i64 literal lost precision");
    require(cuda.first.find("-0.000000e+00") != std::string::npos, "negative zero lost its sign");
    bool rejected = false;
    try { ask::prepareJSON("module { unknown }"); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid source was accepted");
    const auto capabilities = ask::compilerCapabilitiesJSON();
    checkTypedCompiler(capabilities.find("\"host_codegen\":true") != std::string::npos, argv[1]);
    checkReshape(capabilities.find("\"host_codegen\":true") != std::string::npos);
    require(std::ifstream(ask::compilerLibraryPath()).good(), "loaded compiler library is not identifiable");
    if (capabilities.find("\"host_codegen\":true") != std::string::npos) {
      checkHostTypes(capabilities, argv[1]);
      auto host = ask::codegenJSON(source, "host-llvm");
      require(host.second.find("\"abi\"") != std::string::npos, "missing Host ABI");
      auto object = ask::codegenJSON(source, "host-object", argv[1]);
      require(object.first.empty(), "object mode unexpectedly returned assembly");
      std::ifstream file(argv[1], std::ios::binary);
      char magic[4]{};
      file.read(magic, sizeof(magic));
      require(std::string(magic, 4) == "\x7f" "ELF", "invalid host object");
    }
    std::vector<std::future<std::pair<std::string, std::string>>> concurrent;
    for (int i = 0; i < 4; ++i)
      concurrent.push_back(std::async(std::launch::async, [] { return ask::prepareJSON(source, true); }));
    for (auto &result : concurrent) require(result.get() == prepared, "concurrent contexts disagree");
    std::cout << "native prepare, partition, codegen and independent contexts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
