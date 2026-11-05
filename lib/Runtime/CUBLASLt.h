#pragma once

#include "ask/Runtime/CUDA.h"
#include <functional>

namespace ask::runtime::detail {

struct CUBLASLtContext;

struct MatmulTuningBridge {
  void *stream = nullptr;
  std::uint64_t scratch = 0;
  std::size_t scratchBytes = 0;
  std::function<float(const std::function<void()> &)> measure = nullptr;
};

// Descriptors and the selected algorithm are immutable after preparation.
class CUBLASLtMatmul {
public:
  static std::shared_ptr<CUBLASLtMatmul> prepare(
      const MatmulDescription &, std::shared_ptr<CUBLASLtContext> &,
      const MatmulTuningBridge *bridge = nullptr);
  ~CUBLASLtMatmul();
  std::size_t workspaceBytes() const;
  void launch(const std::uint64_t *arguments, void *stream) const;
private:
  struct Impl;
  explicit CUBLASLtMatmul(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl;
};

// A, B and C byte counts, including all batches; rejects overflow and dtype.
std::array<std::size_t, 3> matmulBufferBytes(const MatmulDescription &);
unsigned matmulElementBytes(const std::string &elementType);

} // namespace ask::runtime::detail
