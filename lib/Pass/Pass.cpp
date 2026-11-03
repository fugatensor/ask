#include "Pass.h"
#include "Rules.h"
#include "suka/pass.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <set>
#include <utility>

namespace ask::pass {
namespace {
using suka::pass::Fact;
using suka::pass::Obligation;

Decision invalid(std::string reason) { return {Verdict::Fail, std::move(reason)}; }

Fact fact(const char *name, std::initializer_list<std::size_t> values) {
  Fact result{name, {}};
  for (auto value : values) result.arguments.emplace_back(std::uint64_t(value));
  return result;
}

struct Evidence {
  std::vector<Fact> facts;
  std::vector<Obligation> obligations;
  void add(const char *name, std::initializer_list<std::size_t> values) {
    facts.push_back(fact(name, values));
  }
  void require(const char *name, std::initializer_list<std::size_t> values,
               std::string label) {
    obligations.push_back({fact(name, values), std::move(label)});
  }
};

Decision prove(const Evidence &evidence) {
  static const suka::pass::Program rules(
      std::string(kPlanAvailabilityRules) + kAllocationPlanRules);
  auto result = rules.check(evidence.facts, evidence.obligations);
  switch (result.verdict) {
  case suka::pass::Verdict::Pass: return {Verdict::Pass, std::move(result.explanation)};
  case suka::pass::Verdict::Fail: return {Verdict::Fail, std::move(result.explanation)};
  case suka::pass::Verdict::Unknown: return {Verdict::Unknown, std::move(result.explanation)};
  case suka::pass::Verdict::Error: return {Verdict::Error, std::move(result.explanation)};
  }
  return {Verdict::Error, "unknown SUKA proof verdict"};
}

std::string appendPlan(const Plan &plan, Evidence &evidence) {
  std::vector<bool> produced(plan.buffer_count, false);
  for (auto index : plan.public_inputs) {
    if (index >= plan.buffer_count) return "public input buffer index is out of range";
    if (produced[index]) return "public input buffer is repeated";
    produced[index] = true;
  }
  for (const auto &operation : plan.steps) {
    for (auto index : operation.inputs)
      if (index >= plan.buffer_count) return "step input buffer index is out of range";
    for (auto index : operation.outputs) {
      if (index >= plan.buffer_count) return "step output buffer index is out of range";
      if (produced[index]) return "buffer has multiple writers or aliases a public input";
      produced[index] = true;
    }
  }
  if (plan.public_outputs.empty()) return "plan must have a public output";
  for (auto index : plan.public_outputs)
    if (index >= plan.buffer_count) return "public output buffer index is out of range";
  for (bool hasWriter : produced)
    if (!hasWriter) return "plan contains a buffer without a writer";

  for (std::size_t index = 0; index < plan.buffer_count; ++index)
    evidence.add("buffer", {index});
  for (std::size_t index = 0; index <= plan.steps.size(); ++index)
    evidence.add("time", {index});
  for (auto index : plan.public_inputs) evidence.add("born", {index, 0});
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    for (auto output : plan.steps[index].outputs)
      evidence.add("born", {output, index + 1});
    for (std::size_t port = 0; port < plan.steps[index].inputs.size(); ++port) {
      auto input = plan.steps[index].inputs[port];
      evidence.add("needed", {input, index});
      evidence.require("available", {input, index},
                       "READ_" + std::to_string(index) + "_" + std::to_string(port));
    }
  }
  for (std::size_t port = 0; port < plan.public_outputs.size(); ++port) {
    auto output = plan.public_outputs[port];
    evidence.add("needed", {output, plan.steps.size()});
    evidence.require("available", {output, plan.steps.size()},
                     "OUTPUT_" + std::to_string(port));
  }
  return {};
}

// Keys contain every proof input with sequence lengths. Equality checks the
// complete value, and the bounded process-local cache stores only success.
using Key = std::vector<std::size_t>;
Key keyFor(const AllocationPlan &allocation) {
  const auto &plan = allocation.plan;
  Key key{plan.buffer_count};
  const auto append = [&](const auto &values) {
    key.push_back(values.size());key.insert(key.end(), values.begin(), values.end());
  };
  append(plan.public_inputs);append(plan.public_outputs);
  key.push_back(plan.steps.size());
  for (const auto &step : plan.steps) { append(step.inputs);append(step.outputs); }
  append(allocation.buffer_sizes);append(allocation.slot_capacity_bytes);
  key.push_back(allocation.assignments.size());
  for (const auto &entry : allocation.assignments) {
    key.push_back(entry.buffer);key.push_back(entry.slot);
  }
  return key;
}
struct ProofCache {
  std::mutex mutex;
  std::set<Key> entries;
  std::deque<Key> order;
  std::size_t words = 0;
  static constexpr std::size_t maxEntries = 64, maxWords = 65536;
  bool contains(const Key &key) {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.count(key) != 0;
  }
  void insert(Key key) {
    if (key.size() > maxWords) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (entries.count(key)) return;
    while (entries.size() >= maxEntries || words + key.size() > maxWords) {
      words -= order.front().size();entries.erase(order.front());order.pop_front();
    }
    words += key.size();entries.insert(key);order.push_back(std::move(key));
  }
};

Decision checkAllocation(const AllocationPlan &allocation) {
  const Plan &plan = allocation.plan;
  Evidence evidence;
  if (auto error = appendPlan(plan, evidence); !error.empty()) return invalid(error);
  if (allocation.buffer_sizes.size() != plan.buffer_count)
    return invalid("allocation buffer sizes differ from the plan");

  std::vector<bool> publicPort(plan.buffer_count, false);
  for (auto index : plan.public_inputs) publicPort[index] = true;
  for (auto index : plan.public_outputs) publicPort[index] = true;
  std::vector<bool> assigned(plan.buffer_count, false);
  std::vector<std::vector<std::size_t>> slotUsers(allocation.slot_capacity_bytes.size());
  for (const auto &entry : allocation.assignments) {
    if (entry.buffer >= plan.buffer_count || entry.slot >= slotUsers.size())
      return invalid("allocation assignment index is out of range");
    if (publicPort[entry.buffer]) return invalid("public port cannot use an internal slot");
    if (assigned[entry.buffer]) return invalid("internal buffer has multiple slot assignments");
    assigned[entry.buffer] = true;
    slotUsers[entry.slot].push_back(entry.buffer);
  }
  for (std::size_t index = 0; index < plan.buffer_count; ++index)
    if (!publicPort[index] && !assigned[index])
      return invalid("internal buffer has no slot assignment");
  for (const auto &users : slotUsers)
    if (users.empty()) return invalid("allocation contains an unused slot");

  std::vector<std::size_t> producer(plan.buffer_count, plan.steps.size());
  std::vector<std::size_t> lastUse(plan.buffer_count, 0);
  for (std::size_t index = 0; index < plan.steps.size(); ++index)
    for (auto output : plan.steps[index].outputs) {
      producer[output] = index;
      lastUse[output] = index;
    }
  for (std::size_t index = 0; index < plan.steps.size(); ++index)
    for (auto input : plan.steps[index].inputs)
      lastUse[input] = std::max(lastUse[input], index);

  for (std::size_t index = 0; index < slotUsers.size(); ++index) {
    auto capacity = allocation.slot_capacity_bytes[index];
    evidence.add("slot", {index});evidence.add("byte_count", {capacity});
    evidence.add("capacity", {index, capacity});
  }
  for (const auto &entry : allocation.assignments) {
    auto b = entry.buffer, s = entry.slot, size = allocation.buffer_sizes[b];
    evidence.add("internal", {b});evidence.add("byte_count", {size});
    evidence.add("assigned", {b, s});evidence.add("size", {b, size});
    evidence.add("lifetime", {b, producer[b], lastUse[b]});
    evidence.require("fits", {b, s}, "CAPACITY_" + std::to_string(b));
  }
  // Disjoint adjacent intervals in producer order imply pairwise separation.
  for (auto &users : slotUsers) {
    std::sort(users.begin(), users.end(), [&](auto a, auto b) {
      return std::make_pair(producer[a], a) < std::make_pair(producer[b], b);
    });
    for (std::size_t i = 1; i < users.size(); ++i) {
      auto a = users[i - 1], b = users[i];
      evidence.add("adjacent", {a, b});
      evidence.require("compatible", {a, b},
                       "ADJACENT_" + std::to_string(a) + "_" + std::to_string(b));
    }
  }
  return prove(evidence);
}
} // namespace

Decision checkPlan(const Plan &plan) {
  Evidence evidence;
  if (auto error = appendPlan(plan, evidence); !error.empty()) return invalid(error);
  return prove(evidence);
}

Decision checkAllocationPlan(const AllocationPlan &allocation) {
  static ProofCache cache;
  auto key = keyFor(allocation);
  if (cache.contains(key)) return {Verdict::Pass, {}};
  auto result = checkAllocation(allocation);
  if (result.verdict == Verdict::Pass) cache.insert(std::move(key));
  return result;
}
} // namespace ask::pass
