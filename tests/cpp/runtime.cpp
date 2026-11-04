#include <ask/Runtime/CUDA.h>
#include "support/graph_matrix.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
using namespace ask::runtime;
namespace matrix = ask::test::graph_matrix;

std::size_t deepForkMatrix(const std::shared_ptr<Session> &session, const std::string &ptx) {
  auto kernel = session->load(ptx, "map_i32", {"i32"}, 80, {0, 0, 4});
  auto input = session->allocate(0);
  std::size_t cases = 0;
  for (std::size_t prefix : {256, 1024})
    for (std::size_t branches : {16, 64})
      for (bool unused : {false, true})
        for (bool graph : {false, true}) {
          std::vector<PlanStep> steps;
          const auto append = [&](std::size_t source) {
            const auto output = steps.size() + 1;
            PlanStep step;step.kernel = kernel;
            step.inputs = {source};step.outputs = {output};
            step.arguments = {{static_cast<std::int64_t>(source), {}},
                              {static_cast<std::int64_t>(output), {}}, {-1, std::string(4, '\0')}};
            step.grid = {1, 1, 1};step.block = {128, 1, 1};
            steps.push_back(std::move(step));return output;
          };
          if (unused) append(0);
          std::size_t root = 0;
          for (std::size_t i = 0; i < prefix; ++i) root = append(root);
          std::vector<std::size_t> outputs;
          for (std::size_t i = 0; i < branches; ++i) outputs.push_back(append(append(root)));
          const auto before = session->stats();
          auto program = session->prepareProgram(std::vector<std::size_t>(steps.size() + 1, 0),
                                                 {0}, outputs, steps, graph);
          auto [result, event] = program->run({input});event->wait();
          const auto after = session->stats();
          const auto slots = graph ? branches + 1 + unused : 2;
          if (result.size() != branches ||
              after.at("prepared_internal_slots") - before.at("prepared_internal_slots") != slots ||
              after.at("graph_nodes") - before.at("graph_nodes") != (graph ? steps.size() : 0) ||
              after.at("graph_edges") - before.at("graph_edges") != (graph ? prefix - 1 + 2 * branches : 0))
            throw std::runtime_error("large graph prefix/branch structure");
          session->reap();++cases;
        }
  return cases;
}

std::size_t graphMatrix(const std::shared_ptr<Session> &session, const std::string &ptx) {
  auto map = session->load(ptx, "map_i32", {"i32"}, 80, {0, 0, 4});
  auto add = session->load(ptx, "add_i32", {"i32"}, 80, {0, 0, 0, 4});
  std::size_t cases = 0;
  for (auto topology : matrix::topologies)
    for (std::size_t depth : {2, 4})
      for (std::size_t width : {2, 3})
        for (std::uint32_t count : {0, 1, 33, 1025})
          for (bool graph : {false, true})
            for (bool changing : {false, true})
              for (bool inflight : {false, true}) {
                auto shape = matrix::make(topology, depth, width);
                const std::size_t bytes = count * sizeof(std::int32_t);
                std::vector<PlanStep> steps;
                for (const auto &node : shape.steps) {
                  PlanStep step;
                  step.kernel = node.inputs.size() == 1 ? map : add;
                  step.inputs = node.inputs;step.outputs = node.outputs;
                  for (auto buffer : node.inputs) step.arguments.push_back({static_cast<std::int64_t>(buffer), {}});
                  for (auto buffer : node.outputs) step.arguments.push_back({static_cast<std::int64_t>(buffer), {}});
                  step.arguments.push_back({-1, std::string(reinterpret_cast<const char *>(&count), sizeof(count))});
                  step.grid = {std::max(1u, (count + 127) / 128), 1, 1};step.block = {128, 1, 1};
                  steps.push_back(std::move(step));
                }
                const auto before = session->stats();
                auto program = session->prepareProgram(std::vector<std::size_t>(shape.factors.size(), bytes),
                                                       {0}, shape.outputs, std::move(steps), graph);
                auto prepared = session->stats();
                const auto slots = prepared.at("prepared_internal_slots") - before.at("prepared_internal_slots");
                if (prepared.at("prepared_internal_bytes") - before.at("prepared_internal_bytes") != slots * bytes)
                  throw std::runtime_error("graph matrix workspace byte count");
                std::size_t expected = slots;
                if (topology == matrix::Topology::Chain) expected = std::min<std::size_t>(2, depth - 1);
                if (topology == matrix::Topology::Unused) expected = graph ? shape.steps.size() : 1;
                if (graph && topology == matrix::Topology::Fork) expected = 1 + width * std::min<std::size_t>(2, depth - 1);
                if (slots != expected) throw std::runtime_error("graph matrix workspace slot count");
                std::array<std::vector<std::int32_t>, 2> values;
                std::array<std::shared_ptr<DeviceBuffer>, 2> inputs;
                std::array<std::vector<std::shared_ptr<DeviceBuffer>>, 2> outputs;
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                  values[i].resize(count);
                  for (std::size_t j = 0; j < count; ++j) values[i][j] = (std::int32_t(j % 7) - 3) * std::int32_t(i + 1);
                  inputs[i] = session->allocate(bytes);inputs[i]->copyFromHost(values[i].data(), bytes);
                  for (std::size_t port = 0; port < shape.outputs.size(); ++port) outputs[i].push_back(session->allocate(bytes));
                }
                const auto verify = [&](std::size_t input, const auto &actual) {
                  std::vector<std::int32_t> result(count);
                  for (std::size_t port = 0; port < actual.size(); ++port) {
                    actual[port]->copyToHost(result.data(), bytes);
                    for (std::size_t j = 0; j < count; ++j)
                      if (result[j] != values[input][j] * shape.factors[shape.outputs[port]])
                        throw std::runtime_error("graph matrix numerical result");
                  }
                };
                for (std::size_t batch = 0; batch < 2; ++batch) {
                  std::array<std::shared_ptr<Event>, 2> events;
                  std::array<std::vector<std::shared_ptr<DeviceBuffer>>, 2> results;
                  std::array<std::size_t, 2> source;
                  for (std::size_t lane = 0; lane < 2; ++lane) {
                    source[lane] = changing ? (batch + lane) % 2 : 0;
                    results[lane] = outputs[inflight ? lane : changing ? (batch + lane) % 2 : 0];
                    for (std::size_t port = 0; port < shape.outputs.size(); ++port)
                      if (shape.outputs[port] == 0) results[lane][port] = inputs[source[lane]];
                    events[lane] = program->submit({inputs[source[lane]]}, results[lane]);
                    if (!inflight) { events[lane]->wait();verify(source[lane], results[lane]); }
                  }
                  for (std::size_t lane = 0; lane < 2; ++lane) {
                    events[lane]->wait();
                    if (inflight) verify(source[lane], results[lane]);
                  }
                }
                const auto after = session->stats();
                const auto instances = after.at("graph_instantiations") - before.at("graph_instantiations");
                if (after.at("graph_nodes") - before.at("graph_nodes") != instances * shape.steps.size() ||
                    after.at("graph_edges") - before.at("graph_edges") != instances * matrix::edges(shape))
                  throw std::runtime_error("graph matrix dependency structure");
                if (graph && (!instances || after.at("graph_launches") - before.at("graph_launches") != 4))
                  throw std::runtime_error("graph matrix replay count");
                session->reap();++cases;
              }
  return cases;
}
} // namespace

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
      // reads buffer 1 before its producer, which plan validation must reject.
      auto early = steps[0];
      early.inputs = {1}; early.outputs = {2};
      early.arguments = {{1, {}}, {2, {}}};
      auto producer = steps[1];
      producer.inputs = {0}; producer.outputs = {1};
      producer.arguments = {{0, {}}, {1, {}}};
      bool rejected = false;
      try { session->prepareProgram({16, 16, 16}, {0}, {2}, {early, producer}, false); }
      catch (const CUDAError &error) {
        rejected = std::string(error.what()).find("ASK execution plan validation failed") != std::string::npos;
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
    std::cout << "graph matrix: " << graphMatrix(session, ptx) << '\n';
    std::cout << "deep fork matrix: " << deepForkMatrix(session, ptx) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
