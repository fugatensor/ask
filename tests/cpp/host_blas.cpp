#include "ask/Frontend/CompilerAPI.h"
#include "support/primitive_graphs.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

extern char **environ;

namespace {
struct Case { int batches, rows, columns, inner;bool transposeA, transposeB, fused; };

std::string graph(const Case &c, const std::string &name) {
  auto a = ask::test::tensor({c.batches, c.transposeA ? c.inner : c.rows,
                             c.transposeA ? c.rows : c.inner}, "f32");
  auto b = ask::test::tensor({c.batches, c.transposeB ? c.columns : c.inner,
                             c.transposeB ? c.inner : c.columns}, "f32");
  auto out = ask::test::tensor({c.batches, c.rows, c.columns}, "f32");
  std::string text = "func.func @" + name + "(%a: " + a + ", %b: " + b + ") -> " + out + " {\n";
  if (c.fused) text += "%aa = \"ask.add\"(%a, %a) : (" + a + ", " + a + ") -> " + a + "\n";
  text += "%m = \"ask.matmul\"(" + std::string(c.fused ? "%aa" : "%a") + ", %b) {transpose_a = " +
      (c.transposeA ? "true" : "false") + ", transpose_b = " + (c.transposeB ? "true" : "false") +
      "} : (" + a + ", " + b + ") -> " + out + "\n";
  if (c.fused) text += "%y = \"ask.add\"(%m, %m) : (" + out + ", " + out + ") -> " + out + "\n";
  return text + "return " + (c.fused ? "%y" : "%m") + " : " + out + "\n}\n";
}

void link(const std::vector<std::string> &arguments) {
  std::vector<char *> command;
  for (const auto &argument : arguments) command.push_back(const_cast<char *>(argument.c_str()));
  command.push_back(nullptr);
  pid_t child;
  if (posix_spawn(&child, command[0], nullptr, nullptr, command.data(), environ))
    throw std::runtime_error("cannot launch the SDK consumer linker");
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
    throw std::runtime_error("SDK consumer could not link the Host OpenBLAS object");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: host-blas LINKER OPENBLAS OUTPUT_DIRECTORY");
    std::vector<Case> cases;
    std::string source = "module {\n";
    for (int batches : {1, 2}) for (int rows : {0, 1, 3}) for (int columns : {0, 1, 5})
      for (int inner : {0, 1, 7}) for (bool ta : {false, true}) for (bool tb : {false, true})
        for (bool fused : {false, true}) {
          cases.push_back({batches, rows, columns, inner, ta, tb, fused});
          source += graph(cases.back(), "matrix_" + std::to_string(cases.size() - 1));
        }
    source += "}";
    auto directory = std::filesystem::path(argv[3]);
    std::filesystem::create_directories(directory);
    auto object = (directory / "matrix.o").string(), library = (directory / "matrix.so").string();
    ask::compiler::CodegenOptions options;
    options.target = ask::compiler::CodegenTarget::HostObject;
    options.outputPath = object;
    options.optimize = true;
    auto compiled = ask::compiler::codegen(source, options);
    if (compiled.abiJSON->find("\"link_libraries\":[\"openblas\"]") == std::string::npos)
      throw std::runtime_error("Host object omitted its OpenBLAS link contract");
    link({argv[1], "-shared", object, "-o", library, argv[2],
          "-Wl,-rpath," + std::filesystem::path(argv[2]).parent_path().string(), "-lm"});
    void *loaded = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!loaded) throw std::runtime_error(dlerror());
    for (std::size_t index = 0; index < cases.size(); ++index) {
      const auto &c = cases[index];
      auto function = reinterpret_cast<void (*)(const float *, const float *, float *)>(
          dlsym(loaded, ("matrix_" + std::to_string(index)).c_str()));
      if (!function) throw std::runtime_error("Host object lost a generated entry point");
      std::vector<float> a(std::max(1, c.batches * c.rows * c.inner));
      std::vector<float> b(std::max(1, c.batches * c.inner * c.columns));
      std::vector<float> out(std::max(1, c.batches * c.rows * c.columns), std::numeric_limits<float>::quiet_NaN());
      for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i * 17 % 31) - 15) / 8;
      for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i * 13 % 29) - 14) / 16;
      function(a.data(), b.data(), out.data());
      for (int batch = 0; batch < c.batches; ++batch) for (int row = 0; row < c.rows; ++row)
        for (int column = 0; column < c.columns; ++column) {
          double expected = 0;
          for (int k = 0; k < c.inner; ++k) {
            const auto ai = batch * c.rows * c.inner + (c.transposeA ? k * c.rows + row : row * c.inner + k);
            const auto bi = batch * c.inner * c.columns + (c.transposeB ? column * c.inner + k : k * c.columns + column);
            expected += double(a[ai]) * double(b[bi]);
          }
          if (c.fused) expected *= 4;
          if (out[(batch * c.rows + row) * c.columns + column] != expected)
            throw std::runtime_error("OpenBLAS SDK matrix differs from the independent scalar reference");
        }
      if ((!c.rows || !c.columns) && !std::isnan(out.front()))
        throw std::runtime_error("empty Host matrix touched its output");
    }
    dlclose(loaded);
    std::cout << "Host OpenBLAS link and numerical matrix: " << cases.size() << " cases\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
