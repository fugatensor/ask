#include "Pass.h"
#include "support/plan_matrix.h"
#include "support/graph_matrix.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ask::pass::AllocationPlan;
using ask::pass::Decision;
using ask::pass::Plan;
using ask::pass::PlanStep;
using ask::pass::Verdict;
namespace matrix = ask::test::plan_matrix;
namespace graph = ask::test::graph_matrix;

void expect(const Decision &decision, Verdict verdict, const char *caseName) {
  if (decision.verdict != verdict)
    throw std::runtime_error(std::string(caseName) + ": " + decision.explanation);
}

Plan chain(std::size_t steps) {
  return matrix::plan(matrix::Topology::Chain, steps);
}

std::size_t allocationMatrix() {
  std::size_t cases = 0;
  const std::size_t sizes[] = {0, 1, 16, std::numeric_limits<std::size_t>::max()};
  for (auto topology : matrix::topologies)
    for (std::size_t count : {2, 4, 6})
      for (auto bytes : sizes)
        for (bool delayedRead : {false, true})
          for (bool shortCapacity : {false, true}) {
            auto plan = matrix::plan(topology, count);
            if (delayedRead) plan.steps.back().inputs.push_back(1);
            const auto capacity = shortCapacity && bytes ? bytes - 1 : bytes;
            for (std::size_t mask = 0; mask < (std::size_t{1} << (count - 1)); ++mask) {
              const auto allocation = matrix::mapping(plan, bytes, capacity, mask);
              const auto valid = matrix::valid(allocation);
              expect(ask::pass::checkAllocationPlan(allocation), valid ? Verdict::Pass : Verdict::Fail,
                     "generated allocation matrix");
              if (valid) expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "cached plan");
              ++cases;
            }
          }
  return cases;
}

std::size_t cacheMutationMatrix() {
  std::size_t cases = 0;
  const std::size_t sizes[] = {1, 16, std::numeric_limits<std::size_t>::max() - 1};
  for (auto topology : matrix::topologies)
    for (std::size_t count : {4, 8})
      for (auto bytes : sizes) {
        const auto original = matrix::cyclic(topology, count, bytes);
        if (!matrix::valid(original)) throw std::runtime_error("invalid cache matrix source");
        expect(ask::pass::checkAllocationPlan(original), Verdict::Pass, "cache matrix source");
        for (auto mutation : matrix::mutations) {
          auto candidate = original;
          matrix::mutate(candidate, mutation);
          expect(ask::pass::checkAllocationPlan(candidate),
                 matrix::valid(candidate) ? Verdict::Pass : Verdict::Fail, "cached-plan mutation");
          expect(ask::pass::checkAllocationPlan(original), Verdict::Pass, "source survives mutation");
          ++cases;
        }
      }
  return cases;
}

std::size_t largePlanMatrix() {
  std::size_t cases = 0;
  const std::size_t sizes[] = {0, 16, std::numeric_limits<std::size_t>::max()};
  for (auto topology : matrix::topologies)
    for (std::size_t count : {128, 1024, 4096})
      for (auto bytes : sizes) {
        const auto allocation = matrix::cyclic(topology, count, bytes);
        if (!matrix::valid(allocation)) throw std::runtime_error("invalid large matrix source");
        expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "large allocation plan");
        expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "repeated large allocation plan");
        ++cases;
      }
  return cases;
}

// Exhaustive closure and pairwise accesses are deliberately separate from
// the validator's ancestor search and adjacent-slot obligations.
bool validGraph(const AllocationPlan &allocation) {
  if (!matrix::valid(allocation)) return false;
  if (!allocation.graph) return allocation.dependencies.empty();
  const auto &plan = allocation.plan;
  const auto count = plan.steps.size();
  if (allocation.dependencies.size() != count) return false;
  std::vector<std::vector<bool>> before(count, std::vector<bool>(count));
  for (std::size_t step = 0; step < count; ++step) {
    std::vector<bool> unique(count);
    for (auto predecessor : allocation.dependencies[step]) {
      if (predecessor >= step || unique[predecessor]) return false;
      unique[predecessor] = true;before[predecessor][step] = true;
      for (std::size_t ancestor = 0; ancestor < predecessor; ++ancestor)
        before[ancestor][step] = before[ancestor][step] || before[ancestor][predecessor];
    }
  }
  std::vector<std::size_t> producer(plan.buffer_count, count);
  std::vector<std::vector<std::size_t>> accesses(plan.buffer_count);
  for (std::size_t step = 0; step < count; ++step) {
    for (auto buffer : plan.steps[step].outputs) {
      producer[buffer] = step;accesses[buffer].push_back(step);
    }
    for (auto buffer : plan.steps[step].inputs) {
      accesses[buffer].push_back(step);
      if (producer[buffer] != count && !before[producer[buffer]][step]) return false;
    }
  }
  for (const auto &a : allocation.assignments)
    for (const auto &b : allocation.assignments) {
      if (a.buffer == b.buffer || a.slot != b.slot || producer[a.buffer] > producer[b.buffer]) continue;
      for (auto access : accesses[a.buffer])
        if (!before[access][producer[b.buffer]]) return false;
    }
  return true;
}

AllocationPlan graphAllocation(const graph::Plan &source, bool scratch, std::size_t bytes,
                               std::size_t period) {
  AllocationPlan allocation;
  auto &plan = allocation.plan;
  plan.buffer_count = source.factors.size();plan.public_inputs = {0};plan.public_outputs = source.outputs;
  allocation.graph = true;
  for (const auto &step : source.steps) {
    plan.steps.push_back({step.inputs, step.outputs});
    if (scratch) plan.steps.back().outputs.push_back(plan.buffer_count++);
    auto &parents = allocation.dependencies.emplace_back();
    for (auto input : step.inputs) if (input) parents.push_back(input - 1);
  }
  allocation.buffer_sizes.assign(plan.buffer_count, bytes);
  std::vector<bool> publicPort(plan.buffer_count);
  publicPort[0] = true;
  for (auto output : plan.public_outputs) publicPort[output] = true;
  for (std::size_t buffer = 0; buffer < plan.buffer_count; ++buffer)
    if (!publicPort[buffer]) {
      auto slot = allocation.assignments.size() % period;
      if (slot == allocation.slot_capacity_bytes.size()) allocation.slot_capacity_bytes.push_back(bytes);
      allocation.assignments.push_back({buffer, slot});
    }
  return allocation;
}

std::size_t graphMatrix() {
  std::size_t cases = 0;
  for (auto topology : graph::topologies)
    for (std::size_t depth : {2, 4})
      for (std::size_t width : {2, 3})
        for (bool scratch : {false, true})
          for (std::size_t bytes : {0, 1, 16}) {
            auto source = graph::make(topology, depth, width);
            for (std::size_t period = 1; period <= source.steps.size() * (scratch ? 2 : 1); ++period) {
              auto allocation = graphAllocation(source, scratch, bytes, period);
              for (bool shortCapacity : {false, true}) {
                auto candidate = allocation;
                if (shortCapacity && bytes && !candidate.slot_capacity_bytes.empty())
                  --candidate.slot_capacity_bytes[0];
                expect(ask::pass::checkAllocationPlan(candidate), validGraph(candidate) ? Verdict::Pass : Verdict::Fail,
                       "generated graph allocation");
                ++cases;
              }
            }
            auto original = graphAllocation(source, scratch, bytes, source.steps.size() * 2);
            expect(ask::pass::checkAllocationPlan(original), Verdict::Pass, "graph cache source");
            // Mutate every dependency in a cached valid plan, including missing,
            // repeated and backwards edges; cache keys must retain the full DAG.
            for (std::size_t step = 0; step < original.dependencies.size(); ++step)
              for (std::size_t port = 0; port < original.dependencies[step].size(); ++port)
                for (int mutation = 0; mutation < 3; ++mutation) {
                  auto candidate = original;
                  auto &parents = candidate.dependencies[step];
                  if (mutation == 0) parents.erase(parents.begin() + port);
                  if (mutation == 1) parents.push_back(parents[port]);
                  if (mutation == 2) parents[port] = step;
                  expect(ask::pass::checkAllocationPlan(candidate), validGraph(candidate) ? Verdict::Pass : Verdict::Fail,
                         "graph cache edge mutation");
                  ++cases;
                }
            auto candidate = original;candidate.graph = false;
            expect(ask::pass::checkAllocationPlan(candidate), Verdict::Fail, "graph cache schedule mutation");
            candidate = original;candidate.dependencies.pop_back();
            expect(ask::pass::checkAllocationPlan(candidate), Verdict::Fail, "graph cache dependency count mutation");
          }
  // Scratch-like unused writes have lifetimes too. Adding a data-independent
  // ordering edge permits reuse, and removing it must invalidate that plan.
  auto unused = graphAllocation(graph::make(graph::Topology::Unused, 3, 2), true, 16, 2);
  for (std::size_t i = 1; i < unused.dependencies.size(); ++i) unused.dependencies[i] = {i - 1};
  // Buffer-index cyclic assignment may place simultaneous scratch/tensor writes
  // together; assign the two outputs of every step to their own reused slots.
  unused.assignments.clear();
  for (const auto &step : unused.plan.steps)
    for (std::size_t port = 0; port < step.outputs.size(); ++port)
      unused.assignments.push_back({step.outputs[port], port});
  expect(ask::pass::checkAllocationPlan(unused), Verdict::Pass, "ordered unused producers");
  unused.dependencies[1].clear();
  expect(ask::pass::checkAllocationPlan(unused), Verdict::Fail, "unordered unused producers");
  for (std::size_t count : {128, 1024, 4096}) {
    auto allocation = matrix::cyclic(matrix::Topology::Chain, count, 16);
    allocation.graph = true;allocation.dependencies.resize(count);
    for (std::size_t i = 1; i < count; ++i) allocation.dependencies[i] = {i - 1};
    expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "large graph chain");
  }
  return cases;
}
} // namespace

int main() {
  try {
    const auto smallCases = allocationMatrix();
    const auto cacheCases = cacheMutationMatrix();
    const auto largeCases = largePlanMatrix();
    const auto graphCases = graphMatrix();
    auto plan = chain(4);
    AllocationPlan allocation{plan, {16, 16, 16, 16, 16},
                              {{1, 0}, {2, 1}, {3, 0}}, {16, 16}};
    expect(ask::pass::checkPlan(plan), Verdict::Pass, "valid plan");
    expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "two slots");

    Plan longPassthrough{1, {0}, {0}, std::vector<PlanStep>(300)};
    longPassthrough.steps.back().inputs = {0};
    expect(ask::pass::checkAllocationPlan(
               {longPassthrough, {16}, {}, {}}), Verdict::Pass,
           "long public passthrough");

    auto longChain = chain(150);
    AllocationPlan scaled{longChain, std::vector<std::size_t>(151, 16), {}, {16, 16}};
    for (std::size_t index = 1; index < 150; ++index)
      scaled.assignments.push_back({index, (index - 1) % 2});
    expect(ask::pass::checkAllocationPlan(scaled), Verdict::Pass,
           "150-step plan with linear slot obligations");

    AllocationPlan passthrough{Plan{1, {0}, {0}, {}}, {16}, {}, {}};
    expect(ask::pass::checkAllocationPlan(passthrough), Verdict::Pass,
           "public passthrough without workspace slots");
    auto zeroBytes = allocation;
    zeroBytes.buffer_sizes = {16, 0, 0, 0, 16};
    zeroBytes.slot_capacity_bytes = {0, 0};
    expect(ask::pass::checkAllocationPlan(zeroBytes), Verdict::Pass,
           "zero-byte internal slots");

    auto sameSlot = allocation;
    sameSlot.assignments[1].slot = 0;
    sameSlot.assignments[2].slot = 0;
    sameSlot.slot_capacity_bytes = {16};
    auto overlapDecision = ask::pass::checkAllocationPlan(sameSlot);
    expect(overlapDecision, Verdict::Fail, "overlapping adjacent lifetimes");
    if (overlapDecision.explanation.find("ADJACENT_1_2 FAIL") == std::string::npos)
      throw std::runtime_error("missing adjacent-lifetime diagnostic");

    auto smallSlot = allocation;
    smallSlot.slot_capacity_bytes[0] = 15;
    auto capacityDecision = ask::pass::checkAllocationPlan(smallSlot);
    expect(capacityDecision, Verdict::Fail, "insufficient capacity");
    if (capacityDecision.explanation.find("CAPACITY_1 FAIL") == std::string::npos)
      throw std::runtime_error("missing capacity diagnostic");

    auto multipleFailures = sameSlot;
    multipleFailures.plan.steps[0].inputs = {1};
    multipleFailures.slot_capacity_bytes[0] = 15;
    const auto multipleDecision = ask::pass::checkAllocationPlan(multipleFailures);
    expect(multipleDecision, Verdict::Fail, "all violated constraints are reported");
    if (multipleDecision.explanation !=
        "verify READ_0_0 FAIL\n"
        "verify CAPACITY_1 FAIL\n"
        "verify CAPACITY_2 FAIL\n"
        "verify CAPACITY_3 FAIL\n"
        "verify ADJACENT_1_2 FAIL\n"
        "verify ADJACENT_2_3 FAIL\n")
      throw std::runtime_error("combined validation diagnostics differ");

    auto incomplete = allocation;
    incomplete.assignments.pop_back();
    expect(ask::pass::checkAllocationPlan(incomplete), Verdict::Fail,
           "missing buffer assignment");

    auto duplicate = allocation;
    duplicate.assignments.push_back({1, 1});
    expect(ask::pass::checkAllocationPlan(duplicate), Verdict::Fail,
           "duplicate buffer assignment");
    auto invalidSlot = allocation;
    invalidSlot.assignments[0].slot = 2;
    expect(ask::pass::checkAllocationPlan(invalidSlot), Verdict::Fail,
           "out-of-range slot assignment");

    auto publicAlias = allocation;
    publicAlias.assignments.push_back({4, 0});
    expect(ask::pass::checkAllocationPlan(publicAlias), Verdict::Fail,
           "public output alias");

    Plan simultaneous{4, {0}, {3}, {{{0}, {1, 2}}, {{1, 2}, {3}}}};
    AllocationPlan sameStep{simultaneous, {16, 16, 16, 16},
                            {{1, 0}, {2, 0}}, {16}};
    expect(ask::pass::checkAllocationPlan(sameStep), Verdict::Fail,
           "same-step writes cannot share");

    auto threeStepChain = chain(6);
    AllocationPlan threeUsers{threeStepChain, std::vector<std::size_t>(7, 16),
                              {{1, 0}, {2, 1}, {3, 0}, {4, 1}, {5, 0}}, {16, 16}};
    expect(ask::pass::checkAllocationPlan(threeUsers), Verdict::Pass,
           "three lifetimes in one slot");
    // A long-lived first buffer overlaps the third. Sorting by producer
    // forces an adjacent failed obligation even though buffers 1 and 5 are
    // not adjacent in the original assignment order.
    auto overlap = threeUsers;
    overlap.plan.steps[4].inputs.push_back(1);
    expect(ask::pass::checkAllocationPlan(overlap), Verdict::Fail,
           "nonadjacent overlap detected");

    auto early = plan;
    early.steps[0].inputs = {2};
    expect(ask::pass::checkPlan(early), Verdict::Fail, "read before write");

    std::cout << "allocation matrix: " << smallCases << ", cache mutations: " << cacheCases
              << ", large plans: " << largeCases << ", graph cases: " << graphCases << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
