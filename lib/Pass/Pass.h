#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ask::pass {

enum class Verdict { Pass, Fail, Unknown, Error };

struct Decision {
  Verdict verdict = Verdict::Error;
  std::string explanation;
};

struct PlanStep {
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
};

struct Plan {
  std::size_t buffer_count = 0;
  std::vector<std::size_t> public_inputs;
  std::vector<std::size_t> public_outputs;
  std::vector<PlanStep> steps;
};

struct SlotAssignment {
  std::size_t buffer = 0;
  std::size_t slot = 0;
};

// Every nonpublic buffer has exactly one assignment. Public ports stay on
// caller-owned storage. Slot capacity is the allocated byte count.
struct AllocationPlan {
  Plan plan;
  std::vector<std::size_t> buffer_sizes;
  std::vector<SlotAssignment> assignments;
  std::vector<std::size_t> slot_capacity_bytes;
};

// Rejects a plan unless every read and public output is available in the
// declared single-stream order. FAIL means an obligation was not proved.
Decision checkPlan(const Plan &plan);

// Includes all checkPlan obligations, mapping completeness and capacity.
// It proves consecutive lifetimes in producer order for each complete slot;
// strict separation of adjacent lifetimes implies every pair is disjoint.
// UNKNOWN and ERROR also reject the candidate.
Decision checkAllocationPlan(const AllocationPlan &allocation);

} // namespace ask::pass
