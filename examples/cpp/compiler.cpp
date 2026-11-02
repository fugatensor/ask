#include "ask/Frontend/CompilerAPI.h"

#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
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
    require(std::ifstream(ask::compilerLibraryPath()).good(), "loaded compiler library is not identifiable");
    if (capabilities.find("\"host_codegen\":true") != std::string::npos) {
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
