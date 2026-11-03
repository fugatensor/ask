#pragma once

#include "Pass.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace ask::test::plan_matrix {
using pass::AllocationPlan;
using pass::Plan;

enum class Topology { Chain, Wide, Skip };
inline constexpr std::array<Topology, 3> topologies{
    Topology::Chain, Topology::Wide, Topology::Skip};

inline Plan plan(Topology topology, std::size_t steps) {
  Plan result;
  result.buffer_count = steps + 1;
  result.public_inputs = {0};
  result.public_outputs = {steps};
  for (std::size_t index = 0; index < steps; ++index) {
    pass::PlanStep step{{topology == Topology::Wide ? 0 : index}, {index + 1}};
    if (topology == Topology::Skip && index >= 2) step.inputs.push_back(index - 2);
    result.steps.push_back(std::move(step));
  }
  return result;
}

inline AllocationPlan cyclic(Topology topology, std::size_t steps, std::size_t bytes) {
  const std::size_t period = topology == Topology::Wide ? 1 : topology == Topology::Chain ? 2 : 4;
  AllocationPlan result{plan(topology, steps), std::vector<std::size_t>(steps + 1, bytes), {},
                        std::vector<std::size_t>(std::min(period, steps - 1), bytes)};
  for (std::size_t buffer = 1; buffer < steps; ++buffer)
    result.assignments.push_back({buffer, (buffer - 1) % result.slot_capacity_bytes.size()});
  return result;
}

inline AllocationPlan mapping(Plan plan, std::size_t bytes, std::size_t capacity, std::size_t mask) {
  AllocationPlan result{std::move(plan), {}, {}, {}};
  result.buffer_sizes.assign(result.plan.buffer_count, bytes);
  std::array<bool, 2> used{};
  for (std::size_t buffer = 1; buffer + 1 < result.plan.buffer_count; ++buffer)
    used[(mask >> (buffer - 1)) & 1] = true;
  std::array<std::size_t, 2> slots{};
  for (std::size_t slot = 0; slot < used.size(); ++slot)
    if (used[slot]) {
      slots[slot] = result.slot_capacity_bytes.size();
      result.slot_capacity_bytes.push_back(capacity);
    }
  for (std::size_t buffer = 1; buffer + 1 < result.plan.buffer_count; ++buffer)
    result.assignments.push_back({buffer, slots[(mask >> (buffer - 1)) & 1]});
  return result;
}

// The reference consumes ordered read/write events, then checks every pair of
// complete access intervals. It does not use production's adjacent-slot check.
inline bool valid(const AllocationPlan &allocation) {
  const auto &plan = allocation.plan;
  if (allocation.buffer_sizes.size() != plan.buffer_count || plan.public_outputs.empty()) return false;
  std::vector<bool> available(plan.buffer_count), publicPort(plan.buffer_count);
  std::vector<std::vector<std::size_t>> accesses(plan.buffer_count);
  for (auto buffer : plan.public_inputs) {
    if (buffer >= plan.buffer_count || available[buffer]) return false;
    available[buffer] = publicPort[buffer] = true;
  }
  for (std::size_t event = 0; event < plan.steps.size(); ++event) {
    for (auto buffer : plan.steps[event].inputs) {
      if (buffer >= plan.buffer_count || !available[buffer]) return false;
      accesses[buffer].push_back(event);
    }
    for (auto buffer : plan.steps[event].outputs) {
      if (buffer >= plan.buffer_count || available[buffer]) return false;
      available[buffer] = true;
      accesses[buffer].push_back(event);
    }
  }
  if (std::find(available.begin(), available.end(), false) != available.end()) return false;
  for (auto buffer : plan.public_outputs) {
    if (buffer >= plan.buffer_count || !available[buffer]) return false;
    publicPort[buffer] = true;
  }
  std::vector<bool> assigned(plan.buffer_count), used(allocation.slot_capacity_bytes.size());
  for (const auto &entry : allocation.assignments) {
    if (entry.buffer >= plan.buffer_count || entry.slot >= used.size() ||
        publicPort[entry.buffer] || assigned[entry.buffer]) return false;
    if (allocation.buffer_sizes[entry.buffer] > allocation.slot_capacity_bytes[entry.slot]) return false;
    assigned[entry.buffer] = used[entry.slot] = true;
  }
  for (std::size_t buffer = 0; buffer < plan.buffer_count; ++buffer)
    if (!publicPort[buffer] && !assigned[buffer]) return false;
  if (std::find(used.begin(), used.end(), false) != used.end()) return false;
  std::vector<std::pair<std::size_t, std::size_t>> intervals(plan.buffer_count);
  for (const auto &entry : allocation.assignments) {
    const auto edges = std::minmax_element(accesses[entry.buffer].begin(), accesses[entry.buffer].end());
    intervals[entry.buffer] = {*edges.first, *edges.second};
  }
  for (std::size_t a = 0; a < allocation.assignments.size(); ++a)
    for (std::size_t b = a + 1; b < allocation.assignments.size(); ++b) {
      const auto &left = allocation.assignments[a], &right = allocation.assignments[b];
      if (left.slot != right.slot) continue;
      const auto x = intervals[left.buffer], y = intervals[right.buffer];
      if (std::max(x.first, y.first) <= std::min(x.second, y.second)) return false;
    }
  return true;
}

enum class Mutation {
  InputIndex, DuplicateInput, OutputIndex, NoOutputs, RepeatOutput,
  ReadIndex, EarlyRead, RepeatRead, DuplicateWriter, MissingWriter,
  MissingSize, MissingAssignment, DuplicateAssignment, AssignmentBuffer,
  AssignmentSlot, PublicAssignment, UnusedSlot, ShortCapacity, LargerBuffer,
  OverlappingSlots, LaterRead, ReverseAssignments
};
inline constexpr std::array<Mutation, 22> mutations{
    Mutation::InputIndex, Mutation::DuplicateInput, Mutation::OutputIndex,
    Mutation::NoOutputs, Mutation::RepeatOutput, Mutation::ReadIndex,
    Mutation::EarlyRead, Mutation::RepeatRead, Mutation::DuplicateWriter,
    Mutation::MissingWriter, Mutation::MissingSize, Mutation::MissingAssignment,
    Mutation::DuplicateAssignment, Mutation::AssignmentBuffer, Mutation::AssignmentSlot,
    Mutation::PublicAssignment, Mutation::UnusedSlot, Mutation::ShortCapacity,
    Mutation::LargerBuffer, Mutation::OverlappingSlots, Mutation::LaterRead,
    Mutation::ReverseAssignments};

inline void mutate(AllocationPlan &allocation, Mutation mutation) {
  auto &plan = allocation.plan;
  switch (mutation) {
  case Mutation::InputIndex: plan.public_inputs[0] = plan.buffer_count; break;
  case Mutation::DuplicateInput: plan.public_inputs.push_back(0); break;
  case Mutation::OutputIndex: plan.public_outputs[0] = plan.buffer_count; break;
  case Mutation::NoOutputs: plan.public_outputs.clear(); break;
  case Mutation::RepeatOutput: plan.public_outputs.push_back(plan.public_outputs[0]); break;
  case Mutation::ReadIndex: plan.steps[0].inputs[0] = plan.buffer_count; break;
  case Mutation::EarlyRead: plan.steps[0].inputs[0] = 1; break;
  case Mutation::RepeatRead: plan.steps[0].inputs.push_back(0); break;
  case Mutation::DuplicateWriter: plan.steps.back().outputs[0] = 1; break;
  case Mutation::MissingWriter: plan.steps.back().outputs.clear(); break;
  case Mutation::MissingSize: allocation.buffer_sizes.pop_back(); break;
  case Mutation::MissingAssignment: allocation.assignments.pop_back(); break;
  case Mutation::DuplicateAssignment: allocation.assignments.push_back(allocation.assignments[0]); break;
  case Mutation::AssignmentBuffer: allocation.assignments[0].buffer = plan.buffer_count; break;
  case Mutation::AssignmentSlot: allocation.assignments[0].slot = allocation.slot_capacity_bytes.size(); break;
  case Mutation::PublicAssignment: allocation.assignments[0].buffer = 0; break;
  case Mutation::UnusedSlot: allocation.slot_capacity_bytes.push_back(allocation.buffer_sizes[0]); break;
  case Mutation::ShortCapacity: --allocation.slot_capacity_bytes[0]; break;
  case Mutation::LargerBuffer: ++allocation.buffer_sizes[allocation.assignments[0].buffer]; break;
  case Mutation::OverlappingSlots:
    for (auto &entry : allocation.assignments) entry.slot = 0;
    allocation.slot_capacity_bytes.resize(1);
    plan.steps[2].inputs.push_back(1);
    break;
  case Mutation::LaterRead: plan.steps.back().inputs.push_back(1); break;
  case Mutation::ReverseAssignments: std::reverse(allocation.assignments.begin(), allocation.assignments.end()); break;
  }
}
} // namespace ask::test::plan_matrix
