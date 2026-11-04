#include <ask/Runtime/CUDA.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

using namespace ask::runtime;

namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

// Matrix inputs and their reference results are exactly representable dyadics.
std::string encode(const std::vector<float> &values, const std::string &type) {
  const std::size_t width = type == "f32" ? 4 : 2;
  std::string bytes(values.size() * width, '\0');
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, &values[i], 4);
    if (width == 4) std::memcpy(bytes.data() + i * width, &bits, 4);
    else {
      const std::uint16_t encoded = type == "bf16" ? bits >> 16 : values[i] == 0 ? 0 :
          ((bits >> 16) & 0x8000) | ((((bits >> 23) & 255) - 127 + 15) << 10) | ((bits >> 13) & 1023);
      std::memcpy(bytes.data() + i * width, &encoded, 2);
    }
  }
  return bytes;
}

std::vector<float> decode(const std::string &bytes, const std::string &type) {
  const std::size_t width = type == "f32" ? 4 : 2;
  std::vector<float> values(bytes.size() / width);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint32_t bits = 0;
    if (width == 4) std::memcpy(&bits, bytes.data() + i * width, 4);
    else {
      std::uint16_t encoded;
      std::memcpy(&encoded, bytes.data() + i * width, 2);
      bits = type == "bf16" ? std::uint32_t(encoded) << 16 : (encoded & 32767) == 0 ?
          std::uint32_t(encoded) << 16 : ((std::uint32_t(encoded) & 0x8000) << 16) |
          ((((encoded >> 10) & 31) - 15 + 127) << 23) | ((std::uint32_t(encoded) & 1023) << 13);
    }
    std::memcpy(&values[i], &bits, 4);
  }
  return values;
}

std::shared_ptr<DeviceBuffer> upload(const std::shared_ptr<Session> &session,
                                    const std::vector<float> &values, const std::string &type) {
  const auto bytes = encode(values, type);
  auto buffer = session->allocate(bytes.size());
  buffer->copyFromHost(bytes.data(), bytes.size());
  return buffer;
}

void check(const std::shared_ptr<DeviceBuffer> &buffer, const std::string &type,
           const std::vector<float> &expected, const std::string &context) {
  const auto actual = decode(buffer->copyToHost(), type);
  require(actual == expected, "matmul value mismatch: " + context);
}

std::vector<float> reference(const MatmulDescription &d, const std::vector<float> &a,
                             const std::vector<float> &b) {
  std::vector<float> result(d.batchCount * d.m * d.n, 0);
  for (std::size_t batch = 0; batch < d.batchCount; ++batch)
    for (std::size_t row = 0; row < d.m; ++row)
      for (std::size_t column = 0; column < d.n; ++column)
        for (std::size_t k = 0; k < d.k; ++k)
          result[batch * d.m * d.n + row * d.n + column] +=
              a[batch * d.m * d.k + (d.transposeA ? k * d.m + row : row * d.k + k)] *
              b[batch * d.k * d.n + (d.transposeB ? column * d.k + k : k * d.n + column)];
  return result;
}

std::shared_ptr<PreparedProgram> prepare(const std::shared_ptr<Session> &session,
                                       const MatmulDescription &d, bool graph) {
  const std::size_t width = d.elementType == "f32" ? 4 : 2;
  PlanStep step;
  step.inputs = {0, 1}; step.outputs = {2}; step.matmul = d;
  return session->prepareProgram({std::size_t(d.batchCount * d.m * d.k * width),
      std::size_t(d.batchCount * d.k * d.n * width), std::size_t(d.batchCount * d.m * d.n * width)},
      {0, 1}, {2}, {step}, graph);
}

void matrix() {
  auto session = Session::create();
  std::size_t cases = 0;
  for (const std::string type : {"f32", "f16", "bf16"})
    for (const std::array<std::uint64_t, 3> shape : {
        std::array<std::uint64_t, 3>{2, 3, 4}, {5, 7, 3}, {0, 3, 4}, {2, 0, 4}, {2, 3, 0}})
      for (std::uint64_t batch : {0, 1, 2})
        for (bool transposeA : {false, true})
          for (bool transposeB : {false, true})
            for (bool graph : {false, true}) {
              const auto [m, n, k] = shape;
              MatmulDescription d{type, m, n, k, batch, transposeA, transposeB};
              auto program = prepare(session, d, graph);
              const auto preparations = session->stats().at("matmul_preparations");
              const auto context = type + ":" + std::to_string(m) + "x" + std::to_string(n) +
                  "x" + std::to_string(k) + ":batch=" + std::to_string(batch) +
                  ":ta=" + std::to_string(transposeA) + ":tb=" + std::to_string(transposeB) +
                  ":graph=" + std::to_string(graph);
              std::vector<std::shared_ptr<DeviceBuffer>> heldInputs, heldOutputs;
              std::vector<std::vector<float>> expectedOutputs;
              for (int iteration = 0; iteration < 3; ++iteration) {
                std::vector<float> a(batch * m * k), b(batch * k * n);
                for (std::size_t i = 0; i < a.size(); ++i) a[i] = (int(i % 5) - 2 + iteration) * 0.5f;
                for (std::size_t i = 0; i < b.size(); ++i) b[i] = (int(i % 7) - 3) * 0.5f;
                auto da = upload(session, a, type), db = upload(session, b, type);
                auto [outputs, event] = program->run({da, db});
                event->wait();
                expectedOutputs.push_back(reference(d, a, b));
                heldInputs.push_back(da); heldInputs.push_back(db); heldOutputs.push_back(outputs[0]);
              }
              for (std::size_t i = 0; i < heldOutputs.size(); ++i)
                check(heldOutputs[i], type, expectedOutputs[i], context);
              require(session->stats().at("matmul_preparations") == preparations,
                      "submission repeated cuBLASLt preparation");
              ++cases;
            }
  const auto stats = session->stats();
  require(stats.at("graph_node_updates") > 0, "matrix did not exercise graph address updates");
  require(stats.at("resident_device_bytes") > 0, "pool residency was not counted");
  session->close();
  require(session->stats().at("resident_device_bytes") == 0, "closed pool retained device memory");
  std::cout << "matmul matrix: " << cases << " cases\n";
}

PlanStep doubling(const std::shared_ptr<Kernel> &kernel, std::size_t input,
                  std::size_t output, std::uint32_t count) {
  PlanStep step;
  step.kernel = kernel; step.inputs = {input}; step.outputs = {output};
  step.grid = {(count + 31) / 32, 1, 1}; step.block = {32, 1, 1};
  std::string scalar(sizeof(count), '\0');
  std::memcpy(scalar.data(), &count, sizeof(count));
  step.arguments = {{std::int64_t(input), {}}, {std::int64_t(output), {}}, {-1, scalar}};
  return step;
}

void mixedPlan(bool graph, const std::string &ptx) {
  auto session = Session::create();
  auto kernel = session->load(ptx, "double_f32", {"f32"}, 80, {0, 0, 4});
  PlanStep matmul;
  matmul.inputs = {2, 1}; matmul.outputs = {3};
  matmul.matmul = MatmulDescription{"f32", 2, 3, 4};
  auto program = session->prepareProgram({32, 48, 32, 24, 24}, {0, 1}, {4},
      {doubling(kernel, 0, 2, 8), matmul, doubling(kernel, 3, 4, 6)}, graph);
  std::vector<float> a{1, 2, 3, 4, -1, 1, -2, 2}, b{1, 2, 3, 4, 5, 6, 1, 0, -1, 2, 1, 0};
  auto expectedA = reference(*matmul.matmul, a, b);
  for (auto &value : expectedA) value *= 4;
  auto da = upload(session, a, "f32"), db = upload(session, b, "f32");
  for (auto &value : a) value += 1;
  auto expectedB = reference(*matmul.matmul, a, b);
  for (auto &value : expectedB) value *= 4;
  auto otherA = upload(session, a, "f32");
  auto [first, firstEvent] = program->run({da, db});
  auto [second, secondEvent] = program->run({otherA, db});
  firstEvent->wait(); secondEvent->wait();
  check(first[0], "f32", expectedA, "mixed first in-flight result");
  check(second[0], "f32", expectedB, "mixed second in-flight result");
  auto [third, thirdEvent] = program->run({da, db});
  thirdEvent->wait();
  check(third[0], "f32", expectedA, "mixed graph replay");
  check(first[0], "f32", expectedA, "retained mixed first result");
  check(second[0], "f32", expectedB, "retained mixed second result");
  require(session->stats().at("matmul_preparations") == 1, "mixed plan repeated preparation");
}

void precisionAndResidency() {
  auto session = Session::create();
  {
    auto buffer = session->allocate(16);
    require(session->stats().at("resident_device_bytes") == 16, "allocation residency mismatch");
    auto imported = session->importBuffer(buffer->pointer(), 16);
    require(session->stats().at("resident_device_bytes") == 16, "import counted as allocation");
  }
  require(session->stats().at("resident_device_bytes") == 16, "pool residency mismatch");
  auto program = prepare(session, MatmulDescription{"f32", 1, 1, 1}, false);
  const float precise = 1.0f + 1.0f / 4096.0f;
  auto a = upload(session, {precise}, "f32"), b = upload(session, {1}, "f32");
  auto [outputs, event] = program->run({a, b});
  event->wait();
  const auto actual = decode(outputs[0]->copyToHost(), "f32");
  require(actual.size() == 1 && std::abs(actual[0] - precise) <= std::ldexp(precise, -11),
          "f32 multiplication exceeded TF32 rounding error");
  require(session->stats().at("peak_resident_device_bytes") >=
          session->stats().at("resident_device_bytes"), "resident peak mismatch");
}

void float32Accuracy() {
  auto session = Session::create();
  for (const auto shape : {std::array<std::uint64_t, 3>{17, 9, 65}, {64, 32, 128}})
    for (bool graph : {false, true}) {
      const auto [m, n, k] = shape;
      MatmulDescription description{"f32", m, n, k};
      std::vector<float> a(m * k), b(k * n);
      for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(int(i * 7 % 31) - 15) / 37;
      for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(int(i * 11 % 29) - 14) / 41;
      auto program = prepare(session, description, graph);
      auto [outputs, event] = program->run({upload(session, a, "f32"), upload(session, b, "f32")});
      event->wait();
      const auto actual = decode(outputs[0]->copyToHost(), "f32");
      // Bound input rounding and f32 accumulation against a double reference.
      const double u = std::ldexp(1.0, -11), accumulation = k * std::ldexp(1.0, -24);
      const double relative = 2 * u + u * u + accumulation / (1 - accumulation) * (1 + u) * (1 + u);
      for (std::size_t row = 0; row < m; ++row)
        for (std::size_t column = 0; column < n; ++column) {
          double expected = 0, magnitude = 0;
          for (std::size_t inner = 0; inner < k; ++inner) {
            const double term = double(a[row * k + inner]) * b[inner * n + column];
            expected += term; magnitude += std::abs(term);
          }
          require(std::abs(actual[row * n + column] - expected) <= relative * magnitude + 1e-7,
                  "f32 matmul exceeded TF32 input and f32 accumulation error");
        }
    }
}

void importedAddresses() {
  for (const std::string type : {"f32", "f16", "bf16"})
    for (bool graph : {false, true}) {
      auto session = Session::create();
      const MatmulDescription d{type, 2, 3, 4};
      auto program = prepare(session, d, graph);
      const std::vector<float> a{1, 2, 3, 4, -1, 1, -2, 2}, b{1, 2, 3, 4, 5, 6, 1, 0, -1, 2, 1, 0};
      const auto expected = reference(d, a, b);
      const std::size_t width = type == "f32" ? 4 : 2;
      std::vector<std::shared_ptr<DeviceBuffer>> owners, outputs;
      for (std::size_t offset : {std::size_t(0), width}) {
        const auto imported = [&](const std::vector<float> &values) {
          const auto bytes = encode(values, type);
          const auto padded = std::string(offset, '\0') + bytes;
          auto owner = session->allocate(padded.size());
          owner->copyFromHost(padded.data(), padded.size());
          owners.push_back(owner);
          return session->importBuffer(owner->pointer() + offset, bytes.size());
        };
        auto da = imported(a), db = imported(b), output = imported(std::vector<float>(expected.size(), 0));
        auto event = program->submit({da, db}, {output});
        event->wait(); outputs.push_back(output);
      }
      for (const auto &output : outputs) check(output, type, expected, "natural alignment import");
    }
}

void scratchPlan(bool graph, std::size_t limit) {
  auto session = Session::create();
  constexpr std::size_t side = 512, elements = side * side, bytes = elements * 2;
  MatmulDescription description{"f16", side, side, side};
  description.workspaceLimit = limit;
  PlanStep first, second;
  first.inputs = {0, 1}; first.outputs = {2}; first.matmul = description;
  second.inputs = {2, 1}; second.outputs = {3}; second.matmul = description;
  auto program = session->prepareProgram({bytes, bytes, bytes, bytes}, {0, 1}, {3}, {first, second}, graph);
  std::vector<float> ones(elements, 1), identity(elements, 0);
  for (std::size_t i = 0; i < side; ++i) identity[i * side + i] = 1;
  auto a = upload(session, ones, "f16"), b = upload(session, identity, "f16");
  const auto before = session->stats();
  auto [firstOutput, firstEvent] = program->run({a, b});
  firstEvent->wait();
  const auto after = session->stats();
  const auto extraBytes = after.at("resident_device_bytes") - before.at("resident_device_bytes");
  require(extraBytes >= 2 * bytes && extraBytes <= 2 * bytes + limit,
          "library scratch capacity or lifetime reuse mismatch");
  const auto scratch = extraBytes - 2 * bytes;
  require(after.at("allocations") - before.at("allocations") == 2 + std::uint64_t(scratch != 0),
          "zero workspace occupied a slot or serial scratch was not reused");
  check(firstOutput[0], "f16", ones, "scratch chain");
  auto [secondOutput, secondEvent] = program->run({a, b});
  secondEvent->wait();
  check(secondOutput[0], "f16", ones, "scratch graph update");
  check(firstOutput[0], "f16", ones, "retained scratch result");
  // Each half of K exceeds f16's range before exact cancellation in f32.
  // Split-K must preserve the accumulator type in its intermediate storage.
  std::vector<float> large(elements, 256), cancelling(elements);
  for (std::size_t i = 0; i < elements; ++i) cancelling[i] = i / side < side / 2 ? 256 : -256;
  auto largeA = upload(session, large, "f16"), cancellingB = upload(session, cancelling, "f16");
  auto [cancelled, cancelledEvent] = program->run({largeA, cancellingB});
  cancelledEvent->wait();
  check(cancelled[0], "f16", std::vector<float>(elements, 0), "f32 split-K intermediates");
  require(session->stats().at("matmul_preparations") == 2, "scratch chain repeated preparation");
  std::cout << "matmul scratch: " << scratch << " bytes, limit " << limit << ", graph " << graph << '\n';
}

void branchMatrix() {
  std::size_t cases = 0;
  for (const std::string type : {"f32", "f16", "bf16"})
    for (std::size_t side : {8, 512})
      for (std::size_t branches : {2, 3})
        for (std::size_t limit : {std::size_t(0), std::size_t(4 * 1024 * 1024)}) {
          std::uint64_t serialBytes = 0;
          for (bool graph : {false, true}) {
            auto session = Session::create();
            const auto elements = side * side, bytes = elements * (type == "f32" ? 4 : 2);
            MatmulDescription description{type, side, side, side};
            description.workspaceLimit = limit;
            std::vector<PlanStep> steps;
            std::vector<std::size_t> outputs;
            for (std::size_t branch = 0; branch < branches; ++branch) {
              const auto middle = 2 + 2 * branch, output = middle + 1;
              PlanStep first, second;
              first.inputs = {0, 1}; first.outputs = {middle}; first.matmul = description;
              second.inputs = {middle, 1}; second.outputs = {output}; second.matmul = description;
              steps.push_back(first); steps.push_back(second); outputs.push_back(output);
            }
            auto program = session->prepareProgram(std::vector<std::size_t>(2 + 2 * branches, bytes),
                                                    {0, 1}, outputs, steps, graph);
            const auto internalBytes = session->stats().at("prepared_internal_bytes");
            if (!graph) serialBytes = internalBytes;
            else require(internalBytes == branches * serialBytes,
                         "independent matmul branches shared live storage");
            std::vector<float> identity(elements, 0);
            for (std::size_t row = 0; row < side; ++row) identity[row * side + row] = 1;
            auto b = upload(session, identity, type);
            std::vector<std::vector<std::shared_ptr<DeviceBuffer>>> held;
            for (int iteration = 1; iteration <= 2; ++iteration) {
              auto a = upload(session, std::vector<float>(elements, float(iteration)), type);
              auto [result, completion] = program->run({a, b});
              completion->wait();
              held.push_back(result);
              const auto updates = session->stats().at("graph_node_updates");
              program->submit({a, b}, result)->wait();
              require(session->stats().at("graph_node_updates") == updates,
                      "fixed matmul ports updated graph children");
            }
            for (std::size_t iteration = 0; iteration < held.size(); ++iteration)
              for (const auto &output : held[iteration])
                check(output, type, std::vector<float>(elements, float(iteration + 1)),
                      "parallel matmul retained output");
            const auto stats = session->stats();
            require(stats.at("matmul_preparations") == branches * 2, "branch replay repeated preparation");
            require(stats.at("graph_nodes") == (graph ? branches * 2 : 0), "matmul graph node count");
            require(stats.at("graph_edges") == (graph ? branches : 0), "matmul graph dependency count");
            ++cases;
          }
        }
  std::cout << "matmul branch matrix: " << cases << " cases\n";
}
} // namespace

int main() {
  try {
    std::ifstream fixture(ASK_MATMUL_FIXTURE);
    require(bool(fixture), "cannot open matmul PTX fixture");
    const std::string ptx((std::istreambuf_iterator<char>(fixture)), {});
    matrix();
    for (bool graph : {false, true}) mixedPlan(graph, ptx);
    precisionAndResidency();
    float32Accuracy();
    importedAddresses();
    branchMatrix();
    for (bool graph : {false, true})
      for (std::size_t limit : {std::size_t(0), std::size_t(4 * 1024 * 1024)}) scratchPlan(graph, limit);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
