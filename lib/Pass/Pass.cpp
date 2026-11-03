#include "Pass.h"
#include "Rules.h"
#include "suka/pass.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ask::pass {
namespace {

std::string buffer(std::size_t index) { return "b" + std::to_string(index); }
std::string slot(std::size_t index) { return "slot" + std::to_string(index); }
std::string step(std::size_t index) { return "s" + std::to_string(index); }

std::string instantiate(std::size_t bufferIndex, std::size_t from) {
  std::string rule(kPlanAvailabilityRule);
  const auto replace = [&](const char *token, const std::string &value) {
    const std::string key(token);
    std::size_t position = 0;
    while ((position = rule.find(key, position)) != std::string::npos) {
      rule.replace(position, key.size(), value);
      position += value.size();
    }
  };
  replace("@TYPE@", "buffer_" + std::to_string(bufferIndex));
  replace("@FROM@", step(from));
  replace("@TO@", step(from + 1));
  return rule;
}

Decision invalid(std::string reason) { return {Verdict::Fail, std::move(reason)}; }

Decision prove(const std::string &source) {
  auto result = suka::pass::run(source);
  switch (result.verdict) {
  case suka::pass::Verdict::Pass: return {Verdict::Pass, std::move(result.explanation)};
  case suka::pass::Verdict::Fail: return {Verdict::Fail, std::move(result.explanation)};
  case suka::pass::Verdict::Unknown: return {Verdict::Unknown, std::move(result.explanation)};
  case suka::pass::Verdict::Error: return {Verdict::Error, std::move(result.explanation)};
  }
  return {Verdict::Error, "unknown SUKA proof verdict"};
}

std::string appendPlanSource(const Plan &plan, std::string &source) {
  // Structural conditions are checked here; the positive Horn rules prove
  // availability at each read and at the final public-output boundary.
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

  std::vector<std::size_t> firstBoundary(plan.buffer_count, 0);
  for (std::size_t index = 0; index < plan.buffer_count; ++index)
    source += buffer(index) + ": buffer_" + std::to_string(index) + "\n";
  for (auto index : plan.public_inputs)
    source += "available(" + buffer(index) + ",s0)\n";
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    for (auto output : plan.steps[index].outputs) {
      firstBoundary[output] = index + 1;
      source += "available(" + buffer(output) + "," + step(index + 1) + ")\n";
    }
  }
  std::vector<std::size_t> lastNeeded = firstBoundary;
  for (std::size_t index = 0; index < plan.steps.size(); ++index)
    for (auto input : plan.steps[index].inputs)
      lastNeeded[input] = std::max(lastNeeded[input], index);
  for (auto output : plan.public_outputs)
    lastNeeded[output] = std::max(lastNeeded[output], plan.steps.size());
  for (std::size_t index = 0; index < plan.buffer_count; ++index)
    for (std::size_t boundary = firstBoundary[index]; boundary < lastNeeded[index]; ++boundary)
      source += instantiate(index, boundary);
  for (std::size_t index = 0; index < plan.steps.size(); ++index)
    for (std::size_t port = 0; port < plan.steps[index].inputs.size(); ++port)
      source += "verify READ_" + std::to_string(index) + "_" + std::to_string(port) +
                " = available(" + buffer(plan.steps[index].inputs[port]) + "," + step(index) + ")\n";
  for (std::size_t port = 0; port < plan.public_outputs.size(); ++port)
    source += "verify OUTPUT_" + std::to_string(port) + " = available(" +
              buffer(plan.public_outputs[port]) + "," + step(plan.steps.size()) + ")\n";
  return {};
}

void emit(std::string &source, const char *predicate, bool present) {
  if (present) source += std::string(predicate) + "(candidate)\n";
}

} // namespace

Decision checkPlan(const Plan &plan) {
  std::string source;
  if (auto error = appendPlanSource(plan, source); !error.empty()) return invalid(error);
  return prove(source);
}

Decision checkAllocationPlan(const AllocationPlan &allocation) {
  const Plan &plan = allocation.plan;
  std::string source;
  if (auto error = appendPlanSource(plan, source); !error.empty()) return invalid(error);
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

  source += kAllocationPlanRules;
  for (std::size_t index = 0; index < slotUsers.size(); ++index)
    source += slot(index) + ": slot\n";
  for (const auto &entry : allocation.assignments) {
    const auto b = buffer(entry.buffer), s = slot(entry.slot);
    source += b + ": internal\n";
    source += "assigned(" + b + "," + s + ")\n";
    if (allocation.slot_capacity_bytes[entry.slot] >= allocation.buffer_sizes[entry.buffer])
      source += "capacity_sufficient(" + b + "," + s + ")\n";
    source += "verify CAPACITY_" + std::to_string(entry.buffer) + " = fits(" + b + "," + s + ")\n";
  }

  // Sort each complete slot assignment by producer step. Strict separation
  // of each adjacent pair implies separation of every earlier/later pair;
  // equal producer steps and read/write in one step never share storage.
  for (auto users : slotUsers) {
    std::sort(users.begin(), users.end(), [&](std::size_t a, std::size_t b) {
      if (producer[a] != producer[b]) return producer[a] < producer[b];
      return a < b;
    });
    for (std::size_t i = 1; i < users.size(); ++i) {
      const auto first = users[i - 1], second = users[i];
      const auto a = buffer(first), b = buffer(second);
      source += "same_slot(" + a + "," + b + ")\n";
      if (lastUse[first] < producer[second])
        source += "ordered_nonoverlap(" + a + "," + b + ")\n";
      source += "verify ADJACENT_" + std::to_string(first) + "_" + std::to_string(second) +
                " = compatible(" + a + "," + b + ")\n";
    }
  }
  return prove(source);
}

Decision checkReuse(const ReuseCandidate &candidate) {
  std::string source(kBufferReuseRules);
  source += '\n';
  emit(source, "same_submission", candidate.same_submission);
  emit(source, "different_submission", !candidate.same_submission);
  emit(source, "strictly_after_last_use",
       candidate.prior_last_use_step < candidate.next_first_use_step);
  emit(source, "prior_completion_finished", candidate.prior_completion_finished);
  emit(source, "same_allocation_domain", candidate.same_allocation_domain);
  emit(source, "allocator_owned", candidate.allocator_owned);
  emit(source, "same_device", candidate.same_device);
  emit(source, "capacity_sufficient",
       candidate.prior_capacity_bytes >= candidate.next_size_bytes);
  emit(source, "output_released", candidate.output_released);
  return prove(source);
}

} // namespace ask::pass
