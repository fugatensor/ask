#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ask::runtime {
class CUDAError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
class DeviceBuffer;
class Kernel;
class Event;
struct SessionState;

// The primary context is retained, never reset. Each session owns one
// nonblocking stream; every driver operation pushes/pops its thread's context.
class Session {
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
                               int arch, unsigned expectedParameters = 4);
  std::shared_ptr<Event> submit(const std::shared_ptr<Kernel> &kernel,
      const std::vector<std::shared_ptr<DeviceBuffer>> &buffers,
      std::array<unsigned, 3> grid, std::array<unsigned, 3> block,
      unsigned sharedBytes = 0);
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
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl);
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
private:
  friend class Session;
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
  struct Impl;
  explicit Event(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};
} // namespace ask::runtime
