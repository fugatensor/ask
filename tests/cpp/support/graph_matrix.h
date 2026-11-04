#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ask::test::graph_matrix {
enum class Topology { Chain, Fork, Join, Diamond, Unused };
inline constexpr std::array topologies{Topology::Chain, Topology::Fork, Topology::Join,
                                       Topology::Diamond, Topology::Unused};
struct Step { std::vector<std::size_t> inputs, outputs; };
struct Plan {
  std::vector<Step> steps;
  std::vector<std::size_t> outputs;
  // Each generated kernel doubles one input or adds two inputs. Factors form
  // an independent scalar reference for every logical tensor.
  std::vector<std::int32_t> factors{1};
};
inline Plan make(Topology topology, std::size_t depth, std::size_t width) {
  Plan plan;
  const auto append = [&](std::vector<std::size_t> inputs) {
    const auto output = plan.factors.size();
    const auto value = inputs.size() == 1 ? 2 * plan.factors[inputs[0]]
                                         : plan.factors[inputs[0]] + plan.factors[inputs[1]];
    plan.steps.push_back({std::move(inputs), {output}});
    plan.factors.push_back(value);
    return output;
  };
  if (topology == Topology::Chain) {
    std::size_t value = 0;
    for (std::size_t i = 0; i < depth; ++i) value = append({value});
    plan.outputs = {value};
  } else if (topology == Topology::Unused) {
    for (std::size_t i = 0; i < depth * width; ++i) append({0});
    plan.outputs = {0};
  } else {
    auto root = topology == Topology::Diamond || topology == Topology::Fork ? append({0}) : 0;
    for (std::size_t branch = 0; branch < width; ++branch) {
      auto value = root;
      for (std::size_t i = 0; i < depth; ++i) value = append({value});
      plan.outputs.push_back(value);
    }
    if (topology != Topology::Fork) {
      auto value = plan.outputs.front();
      for (std::size_t i = 1; i < plan.outputs.size(); ++i) value = append({value, plan.outputs[i]});
      plan.outputs = {value};
    }
  }
  return plan;
}
inline std::size_t edges(const Plan &plan) {
  std::size_t edges = 0;
  // The generator has one output per step and input buffer 0 is public.
  for (const auto &step : plan.steps)
    for (auto input : step.inputs) edges += input != 0;
  return edges;
}
} // namespace ask::test::graph_matrix
