#include <ask/Runtime/CUDA.h>

#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

// Runtime contract test: no Python, LLVM SDK or local CUDA compiler is needed.
// The driver compiles this PTX when the resident kernel is first loaded.
int main() {
  try {
    using namespace ask::runtime;
    std::ifstream source(ASK_RUNTIME_FIXTURE);
    if (!source) throw std::runtime_error("cannot open runtime PTX fixture");
    const std::string ptx((std::istreambuf_iterator<char>(source)), {});
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
    {
      // Both logical buffers are eventually produced; the first step still
      // reads buffer 1 before its producer, which the Rin proof must reject.
      auto early = steps[0];
      early.inputs = {1}; early.outputs = {2};
      early.arguments = {{1, {}}, {2, {}}};
      auto producer = steps[1];
      producer.inputs = {0}; producer.outputs = {1};
      producer.arguments = {{0, {}}, {1, {}}};
      bool rejected = false;
      try { session->prepareProgram({16, 16, 16}, {0}, {2}, {early, producer}, false); }
      catch (const CUDAError &error) {
        rejected = std::string(error.what()).find("ASK execution plan proof failed") != std::string::npos;
      }
      if (!rejected) return 1;
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
    if (stats.at("graph_instantiations") != 1 || stats.at("graph_launches") != 2) return 1;

    // Four serial kernels have three internal buffers. The first and third
    // lifetimes are disjoint, while adjacent buffers overlap at one launch.
    // A verified allocation plan must therefore use exactly two slots.
    const auto verifySlots = [&](bool graph) {
      auto local = Session::create();
      auto localKernel = local->load(ptx, "double_i32", {"i32"}, 80, {0, 0});
      std::vector<PlanStep> chain;
      for (std::size_t i = 0; i < 4; ++i) {
        PlanStep step;
        step.kernel = localKernel;
        step.arguments = {{static_cast<std::int64_t>(i), {}},
                          {static_cast<std::int64_t>(i + 1), {}}};
        step.grid = {1, 1, 1}; step.block = {32, 1, 1};
        step.inputs = {i}; step.outputs = {i + 1};
        chain.push_back(std::move(step));
      }
      auto prepared = local->prepareProgram({16, 16, 16, 16, 16}, {0}, {4},
                                            std::move(chain), graph);
      std::vector<std::shared_ptr<DeviceBuffer>> heldInputs, heldOutputs;
      for (int iteration = 0; iteration < 2; ++iteration) {
        std::array<std::int32_t, 4> input = {iteration + 1, 2, -3, 4}, output{};
        auto source = local->allocate(sizeof(input));
        source->copyFromHost(input.data(), sizeof(input));
        auto before = local->stats().at("allocations");
        auto [results, event] = prepared->run({source});
        if (iteration == 0 && local->stats().at("allocations") - before != 3)
          return false; // One public output plus two internal slots.
        event->wait();
        results.front()->copyToHost(output.data(), sizeof(output));
        for (std::size_t i = 0; i < input.size(); ++i)
          if (output[i] != input[i] * 16) return false;
        heldInputs.push_back(source);
        heldOutputs.push_back(results.front());
      }
      auto result = local->stats();
      if (result.at("workspace_reuses") != 1) return false;
      if (graph && (result.at("graph_instantiations") != 1 ||
                    result.at("graph_launches") != 2 ||
                    result.at("graph_node_updates") == 0)) return false;

      // Consecutive submissions retain caller-owned outputs while events may
      // still be in flight. Their numerical results must stay independent.
      std::array<std::int32_t, 4> a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
      auto sourceA = local->allocate(sizeof(a));
      auto sourceB = local->allocate(sizeof(b));
      sourceA->copyFromHost(a.data(), sizeof(a));
      sourceB->copyFromHost(b.data(), sizeof(b));
      auto [resultA, eventA] = prepared->run({sourceA});
      auto [resultB, eventB] = prepared->run({sourceB});
      eventA->wait();eventB->wait();
      std::array<std::int32_t, 4> actualA{}, actualB{};
      resultA.front()->copyToHost(actualA.data(), sizeof(actualA));
      resultB.front()->copyToHost(actualB.data(), sizeof(actualB));
      for (std::size_t i = 0; i < a.size(); ++i)
        if (actualA[i] != a[i] * 16 || actualB[i] != b[i] * 16) return false;
      return true;
    };
    if (!verifySlots(false) || !verifySlots(true)) return 1;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
