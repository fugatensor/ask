#include "Pass.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ask::pass::AllocationPlan;
using ask::pass::Decision;
using ask::pass::Plan;
using ask::pass::PlanStep;
using ask::pass::Verdict;

void expect(const Decision &decision, Verdict verdict, const char *caseName) {
  if (decision.verdict != verdict)
    throw std::runtime_error(std::string(caseName) + ": " + decision.explanation);
}

Plan chain(std::size_t steps) {
  Plan plan;
  plan.buffer_count = steps + 1;
  plan.public_inputs = {0};
  plan.public_outputs = {steps};
  for (std::size_t index = 0; index < steps; ++index)
    plan.steps.push_back(PlanStep{{index}, {index + 1}});
  return plan;
}

void allocationMatrix() {
  // Enumerate assignments independently of the runtime allocator. The oracle
  // checks every interval pair, while production proves adjacent intervals.
  for (std::size_t count : {2, 4, 6})
    for (std::size_t bytes : {0, 1, 16})
      for (bool delayedRead : {false, true})
        for (bool shortCapacity : {false, true}) {
          auto plan = chain(count);
          if (delayedRead) plan.steps.back().inputs.push_back(1);
          for (std::size_t mask = 0; mask < (std::size_t{1} << (count - 1)); ++mask) {
            AllocationPlan allocation{plan, std::vector<std::size_t>(count + 1, bytes), {}, {}};
            const auto capacity = shortCapacity && bytes ? bytes - 1 : bytes;
            bool used[2] = {};
            for (std::size_t b = 1; b < count; ++b) used[(mask >> (b - 1)) & 1] = true;
            std::size_t slots[2] = {}, slotCount = 0;
            for (int i = 0; i < 2; ++i) if (used[i]) slots[i] = slotCount++;
            allocation.slot_capacity_bytes.assign(slotCount, capacity);
            for (std::size_t b = 1; b < count; ++b)
              allocation.assignments.push_back({b, slots[(mask >> (b - 1)) & 1]});
            bool valid = capacity >= bytes;
            for (std::size_t a = 1; a < count; ++a)
              for (std::size_t b = a + 1; b < count; ++b) {
                if (allocation.assignments[a - 1].slot != allocation.assignments[b - 1].slot) continue;
                auto end = delayedRead && a == 1 ? count - 1 : a;
                if (end >= b - 1) valid = false;
              }
            expect(ask::pass::checkAllocationPlan(allocation), valid ? Verdict::Pass : Verdict::Fail,
                   "generated allocation matrix");
            if (valid) expect(ask::pass::checkAllocationPlan(allocation), Verdict::Pass, "cached plan");
          }
        }
}
} // namespace

int main() {
  try {
    allocationMatrix();
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
      throw std::runtime_error("missing adjacent-lifetime proof obligation");

    auto smallSlot = allocation;
    smallSlot.slot_capacity_bytes[0] = 15;
    auto capacityDecision = ask::pass::checkAllocationPlan(smallSlot);
    expect(capacityDecision, Verdict::Fail, "insufficient capacity");
    if (capacityDecision.explanation.find("CAPACITY_1 FAIL") == std::string::npos)
      throw std::runtime_error("missing capacity proof obligation");

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

    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
