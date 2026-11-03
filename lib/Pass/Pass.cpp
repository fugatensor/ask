#include "Pass.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <set>
#include <utility>

namespace ask::pass {
namespace {
Decision invalid(std::string reason) { return {Verdict::Fail, std::move(reason)}; }

std::string describePlan(const Plan &plan, std::vector<std::size_t> &birth) {
  const auto absent = plan.steps.size() + 1;
  birth.assign(plan.buffer_count, absent);
  for (auto index : plan.public_inputs) {
    if (index >= plan.buffer_count) return "public input buffer index is out of range";
    if (birth[index] != absent) return "public input buffer is repeated";
    birth[index] = 0;
  }
  for (std::size_t step = 0; step < plan.steps.size(); ++step) {
    for (auto index : plan.steps[step].inputs)
      if (index >= plan.buffer_count) return "step input buffer index is out of range";
    for (auto index : plan.steps[step].outputs) {
      if (index >= plan.buffer_count) return "step output buffer index is out of range";
      if (birth[index] != absent) return "buffer has multiple writers or aliases a public input";
      birth[index] = step + 1;
    }
  }
  if (plan.public_outputs.empty()) return "plan must have a public output";
  for (auto index : plan.public_outputs)
    if (index >= plan.buffer_count) return "public output buffer index is out of range";
  // Every declared buffer, including each public output, is available by
  // the final boundary once it has a unique producer or is a public input.
  for (auto value : birth)
    if (value == absent) return "plan contains a buffer without a writer";
  return {};
}

void failed(std::string &output, const std::string &name) {
  output += "verify "; output += name; output += " FAIL\n";
}

void checkAvailability(const Plan &plan, const std::vector<std::size_t> &birth,
                       std::string &failures) {
  for (std::size_t step = 0; step < plan.steps.size(); ++step)
    for (std::size_t port = 0; port < plan.steps[step].inputs.size(); ++port)
      if (birth[plan.steps[step].inputs[port]] > step)
        failed(failures, "READ_" + std::to_string(step) + "_" + std::to_string(port));
}

// Keys contain every validation input with sequence lengths. Equality checks the
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
struct ValidationCache {
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
  std::vector<std::size_t> birth;
  if (auto error = describePlan(plan, birth); !error.empty()) return invalid(error);
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

  std::vector<std::size_t> lastUse(plan.buffer_count, 0);
  for (std::size_t step = 0; step < plan.steps.size(); ++step) {
    for (auto buffer : plan.steps[step].outputs)
      lastUse[buffer] = std::max(lastUse[buffer], step);
    for (auto buffer : plan.steps[step].inputs)
      lastUse[buffer] = std::max(lastUse[buffer], step);
  }
  std::string failures;
  checkAvailability(plan, birth, failures);
  for (const auto &entry : allocation.assignments)
    if (allocation.buffer_sizes[entry.buffer] > allocation.slot_capacity_bytes[entry.slot])
      failed(failures, "CAPACITY_" + std::to_string(entry.buffer));
  // Disjoint adjacent intervals in producer order imply pairwise separation.
  for (auto &users : slotUsers) {
    std::sort(users.begin(), users.end(), [&](auto a, auto b) {
      return std::make_pair(birth[a], a) < std::make_pair(birth[b], b);
    });
    for (std::size_t i = 1; i < users.size(); ++i) {
      auto a = users[i - 1], b = users[i];
      if (lastUse[a] >= birth[b] - 1)
        failed(failures, "ADJACENT_" + std::to_string(a) + "_" + std::to_string(b));
    }
  }
  return {failures.empty() ? Verdict::Pass : Verdict::Fail, std::move(failures)};
}
} // namespace

Decision checkPlan(const Plan &plan) {
  std::vector<std::size_t> birth;
  if (auto error = describePlan(plan, birth); !error.empty()) return invalid(error);
  std::string failures;
  checkAvailability(plan, birth, failures);
  return {failures.empty() ? Verdict::Pass : Verdict::Fail, std::move(failures)};
}

Decision checkAllocationPlan(const AllocationPlan &allocation) {
  static ValidationCache cache;
  auto key = keyFor(allocation);
  if (cache.contains(key)) return {Verdict::Pass, {}};
  auto result = checkAllocation(allocation);
  if (result.verdict == Verdict::Pass) cache.insert(std::move(key));
  return result;
}
} // namespace ask::pass
