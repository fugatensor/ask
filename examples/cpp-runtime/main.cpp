#include <ask/Runtime/CUDA.h>

#include <array>
#include <iostream>

// Runtime-only example: no Python, LLVM SDK or local CUDA compiler is needed.
// The driver compiles this PTX when the resident kernel is first loaded.
int main() {
  try {
    using namespace ask::runtime;
    const std::string ptx = R"(.version 8.0
.target sm_80
.address_size 64
.visible .entry double_i32(.param .u64 src, .param .u64 dst) {
  .reg .pred %p;
  .reg .b32 %i, %v;
  .reg .b64 %x, %y, %offset;
  mov.u32 %i, %tid.x;
  setp.ge.u32 %p, %i, 4;
  @%p bra done;
  ld.param.u64 %x, [src];
  ld.param.u64 %y, [dst];
  mul.wide.u32 %offset, %i, 4;
  add.u64 %x, %x, %offset;
  add.u64 %y, %y, %offset;
  ld.global.u32 %v, [%x];
  add.u32 %v, %v, %v;
  st.global.u32 [%y], %v;
done:
  ret;
}
)";
    auto session = Session::create();
    auto kernel = session->load(ptx, "double_i32", {"i32"}, 80, {0, 0});
    std::vector<PlanStep> steps;
    for (std::size_t i = 0; i < 2; ++i) {
      PlanStep step;
      step.kernel = kernel;
      step.arguments = {{static_cast<std::int64_t>(i), {}}, {static_cast<std::int64_t>(i + 1), {}}};
      step.grid = {1, 1, 1}; step.block = {32, 1, 1};
      step.inputs = {i}; step.outputs = {i + 1};
      steps.push_back(std::move(step));
    }
    auto program = session->prepareProgram({16, 16, 16}, {0}, {2}, std::move(steps), true);
    for (int iteration = 0; iteration < 2; ++iteration) {
      std::array<std::int32_t, 4> input = {iteration + 1, 2, -3, 4}, output{};
      auto source = session->allocate(sizeof(input));
      source->copyFromHost(input.data(), sizeof(input));
      auto [results, event] = program->run({source});
      event->wait();
      results.front()->copyToHost(output.data(), sizeof(output));
      for (std::size_t i = 0; i < input.size(); ++i)
        if (output[i] != input[i] * 4) return 1;
    }
    const auto stats = session->stats();
    std::cout << "graph instances: " << stats.at("graph_instantiations")
              << ", graph launches: " << stats.at("graph_launches") << '\n';
    return stats.at("graph_instantiations") == 1 && stats.at("graph_launches") == 2 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
