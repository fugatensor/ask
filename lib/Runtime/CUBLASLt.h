#pragma once

#include "ask/Runtime/CUDA.h"

namespace ask::runtime::detail {

struct CUBLASLtContext;

// Descriptors and the selected algorithm are immutable after preparation.
class CUBLASLtMatmul {
public:
  static std::shared_ptr<CUBLASLtMatmul> prepare(
      const MatmulDescription &, std::shared_ptr<CUBLASLtContext> &);
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
