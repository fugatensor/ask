#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ask::runtime {
class CUDAError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
class DeviceBuffer;
class Kernel;
class Event;
class PreparedProgram;
struct SessionState;

struct KernelArgument {
  std::shared_ptr<DeviceBuffer> buffer;
  // Empty scalar storage denotes a buffer (including a null device pointer).
  std::string scalar;
};

struct PlanArgument {
  // A negative buffer index denotes null unless scalar contains ABI bytes.
  std::int64_t buffer = -1;
  std::string scalar;
};

// Contiguous row-major matrices. Transpose flags apply to the last two axes;
// batchCount is the product of matching leading dimensions. Accumulation is f32.
struct MatmulDescription {
  std::string elementType = "f32";
  std::uint64_t m = 0, n = 0, k = 0, batchCount = 1;
  bool transposeA = false, transposeB = false;
  std::size_t workspaceLimit = 4 * 1024 * 1024;
};

struct PlanStep {
  std::shared_ptr<Kernel> kernel;
  std::vector<PlanArgument> arguments;
  std::array<unsigned, 3> grid{}, block{};
  unsigned sharedBytes = 0;
  std::vector<std::size_t> inputs, outputs;
  // A library step binds inputs={A,B}, outputs={C}; kernel is empty.
  std::optional<MatmulDescription> matmul;
};

// The primary context is retained, never reset. Each session owns one
// nonblocking stream; every driver operation pushes/pops its thread's context.
class Session : public std::enable_shared_from_this<Session> {
public:
  static std::shared_ptr<Session> create(int device = 0);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  std::shared_ptr<DeviceBuffer> allocate(std::size_t bytes);
  std::shared_ptr<DeviceBuffer> importBuffer(std::uint64_t pointer, std::size_t bytes,
                                            bool readonly = false);
  std::shared_ptr<Kernel> load(const std::string &cubin, const std::string &entry,
                               const std::vector<std::string> &requiredTypes,
                               int arch, const std::vector<unsigned> &parameterWidths);
  std::shared_ptr<Event> submit(const std::shared_ptr<Kernel> &kernel,
      const std::vector<KernelArgument> &arguments,
      std::array<unsigned, 3> grid, std::array<unsigned, 3> block,
      unsigned sharedBytes = 0);
  std::shared_ptr<PreparedProgram> prepareProgram(
      std::vector<std::size_t> bufferSizes, std::vector<std::size_t> inputs,
      std::vector<std::size_t> outputs, std::vector<PlanStep> steps,
      bool graph = false);
  // CUDA Array Interface stream values 1/2 mean legacy/per-thread default.
  void waitForExternalStream(std::uint64_t stream);
  void checkTypes(const std::vector<std::string> &types) const;
  void synchronize();
  void reap();
  void close();
  bool closed() const;
  std::uint64_t stream() const;
  int device() const;
  std::array<int, 2> capability() const;
  std::string deviceName() const;
  std::map<std::string, std::uint64_t> stats() const;
private:
  friend class PreparedProgram;
  void validateLaunch(const std::shared_ptr<Kernel> &kernel,
      const std::array<unsigned, 3> &grid, const std::array<unsigned, 3> &block,
      unsigned sharedBytes) const;
  std::shared_ptr<Event> completion();
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

// Immutable bindings and per-in-flight workspaces. Every submission validates
// all public ports before enqueueing anything; a single event owns the plan's
// resources until the entire program completes.
class PreparedProgram : public std::enable_shared_from_this<PreparedProgram> {
public:
  ~PreparedProgram();
  std::shared_ptr<Event> submit(
      const std::vector<std::shared_ptr<DeviceBuffer>> &inputs,
      const std::vector<std::shared_ptr<DeviceBuffer>> &outputs);
  std::pair<std::vector<std::shared_ptr<DeviceBuffer>>, std::shared_ptr<Event>> run(
      const std::vector<std::shared_ptr<DeviceBuffer>> &inputs);
private:
  friend class Session;
  struct Impl;
  explicit PreparedProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

class DeviceBuffer : public std::enable_shared_from_this<DeviceBuffer> {
public:
  ~DeviceBuffer();
  std::uint64_t pointer() const;
  std::size_t size() const;
  bool readonly() const;
  void copyFromHost(const void *data, std::size_t bytes);
  std::string copyToHost() const;
  void copyToHost(void *data, std::size_t bytes) const;
private:
  friend class Session;
  friend class PreparedProgram;
  struct Impl;
  explicit DeviceBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

class Kernel : public std::enable_shared_from_this<Kernel> {
public:
  ~Kernel();
  std::string entry() const;
  unsigned maxThreads() const;
private:
  friend class Session;
  friend class PreparedProgram;
  struct Impl;
  explicit Kernel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

class Event : public std::enable_shared_from_this<Event> {
public:
  ~Event();
  bool query();
  void wait();
private:
  friend class Session;
  friend class PreparedProgram;
  struct Impl;
  explicit Event(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};
} // namespace ask::runtime
