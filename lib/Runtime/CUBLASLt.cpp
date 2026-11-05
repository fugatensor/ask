#include "CUBLASLt.h"

#include <limits>
#include <map>
#include <mutex>
#include <tuple>

#if ASK_WITH_CUBLASLT
#include <cublasLt.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace ask::runtime::detail {

unsigned matmulElementBytes(const std::string &type) {
  if (type == "f32") return 4;
  if (type == "f16" || type == "bf16") return 2;
  throw CUDAError("matmul requires f32, f16 or bf16 elements");
}

std::array<std::size_t, 3> matmulBufferBytes(const MatmulDescription &d) {
  auto width = matmulElementBytes(d.elementType);
  const auto multiply = [](std::uint64_t a, std::uint64_t b) {
    constexpr auto limit = std::uint64_t(std::numeric_limits<std::int64_t>::max());
    if (a > limit || b > limit || (b && a > limit / b))
      throw CUDAError("matmul dimensions exceed the signed 64-bit byte range");
    return a * b;
  };
  if (d.batchCount > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
    throw CUDAError("matmul batch count exceeds the cuBLASLt limit");
  const auto bytes = [&](std::uint64_t a, std::uint64_t b) {
    return std::size_t(multiply(multiply(multiply(a, b), d.batchCount), width));
  };
  return {bytes(d.m, d.k), bytes(d.k, d.n), bytes(d.m, d.n)};
}

#if ASK_WITH_CUBLASLT
namespace {
void check(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw CUDAError(std::string(operation) + " failed (cuBLASLt status " +
                    std::to_string(int(status)) + ")");
}

struct CachedAlgo {
  cublasLtMatmulAlgo_t algo;
  std::size_t workspace = 0;
};
using AlgoKey = std::tuple<std::string, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, std::size_t>;
static std::mutex globalAlgoMutex;
static std::map<AlgoKey, CachedAlgo> globalAlgoCache;

struct Library {
  void *library = nullptr;
#define LT_FUNCTION(name) decltype(&::name) name = nullptr
  LT_FUNCTION(cublasLtCreate);
  LT_FUNCTION(cublasLtDestroy);
  LT_FUNCTION(cublasLtMatmulDescCreate);
  LT_FUNCTION(cublasLtMatmulDescDestroy);
  LT_FUNCTION(cublasLtMatmulDescSetAttribute);
  LT_FUNCTION(cublasLtMatrixLayoutCreate);
  LT_FUNCTION(cublasLtMatrixLayoutDestroy);
  LT_FUNCTION(cublasLtMatrixLayoutSetAttribute);
  LT_FUNCTION(cublasLtMatmulPreferenceCreate);
  LT_FUNCTION(cublasLtMatmulPreferenceDestroy);
  LT_FUNCTION(cublasLtMatmulPreferenceSetAttribute);
  LT_FUNCTION(cublasLtMatmulAlgoGetHeuristic);
  LT_FUNCTION(cublasLtMatmul);
#undef LT_FUNCTION
  template<class T> void bind(T &function, const char *name) {
#ifdef _WIN32
    function = reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    function = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (!function) throw CUDAError(std::string("cuBLASLt library is missing ") + name);
  }
  Library() {
#ifdef _WIN32
    library = LoadLibraryA("cublasLt64_13.dll");
#else
    for (const char *name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
      library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (library) break;
    }
#endif
    if (!library) throw CUDAError("cuBLASLt shared library is unavailable");
    try {
#define B(name) bind(name, #name)
      B(cublasLtCreate); B(cublasLtDestroy);
      B(cublasLtMatmulDescCreate); B(cublasLtMatmulDescDestroy); B(cublasLtMatmulDescSetAttribute);
      B(cublasLtMatrixLayoutCreate); B(cublasLtMatrixLayoutDestroy); B(cublasLtMatrixLayoutSetAttribute);
      B(cublasLtMatmulPreferenceCreate); B(cublasLtMatmulPreferenceDestroy); B(cublasLtMatmulPreferenceSetAttribute);
      B(cublasLtMatmulAlgoGetHeuristic); B(cublasLtMatmul);
#undef B
    } catch (...) { unload(); throw; }
  }
  void unload() noexcept {
    if (!library) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
    library = nullptr;
  }
  ~Library() { unload(); }
};

std::shared_ptr<Library> getLibrary() {
  static std::mutex mutex;
  static std::weak_ptr<Library> cached;
  std::lock_guard<std::mutex> lock(mutex);
  auto library = cached.lock();
  if (!library) { library = std::make_shared<Library>(); cached = library; }
  return library;
}
} // namespace

struct CUBLASLtContext {
  std::shared_ptr<Library> library = getLibrary();
  cublasLtHandle_t handle = nullptr;
  CUBLASLtContext() { check(library->cublasLtCreate(&handle), "cublasLtCreate"); }
  ~CUBLASLtContext() { if (handle) library->cublasLtDestroy(handle); }
};

struct CUBLASLtMatmul::Impl {
  std::shared_ptr<CUBLASLtContext> context;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace = 0;
  ~Impl() {
    auto &library = *context->library;
    if (c) library.cublasLtMatrixLayoutDestroy(c);
    if (b) library.cublasLtMatrixLayoutDestroy(b);
    if (a) library.cublasLtMatrixLayoutDestroy(a);
    if (operation) library.cublasLtMatmulDescDestroy(operation);
  }
};

std::shared_ptr<CUBLASLtMatmul> CUBLASLtMatmul::prepare(
    const MatmulDescription &d, std::shared_ptr<CUBLASLtContext> &context,
    const MatmulTuningBridge *bridge) {
  if (!context) context = std::make_shared<CUBLASLtContext>();
  auto p = std::make_unique<Impl>(); p->context = context;
  auto &library = *context->library;
  // FP32 storage permits TF32 multiplication; all input types accumulate in FP32.
  const auto compute = d.elementType == "f32" ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  const auto type = d.elementType == "f32" ? CUDA_R_32F : d.elementType == "f16" ? CUDA_R_16F : CUDA_R_16BF;
  check(library.cublasLtMatmulDescCreate(&p->operation, compute, CUDA_R_32F), "cublasLtMatmulDescCreate");
  const auto transposeA = d.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto transposeB = d.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
  check(library.cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSA,
      &transposeA, sizeof(transposeA)), "cublasLtMatmulDescSetAttribute(transpose A)");
  check(library.cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSB,
      &transposeB, sizeof(transposeB)), "cublasLtMatmulDescSetAttribute(transpose B)");
  const auto layout = [&](cublasLtMatrixLayout_t &target, std::uint64_t rows, std::uint64_t columns) {
    check(library.cublasLtMatrixLayoutCreate(&target, type, rows, columns, std::int64_t(columns)),
          "cublasLtMatrixLayoutCreate");
    const auto order = CUBLASLT_ORDER_ROW;
    const auto batches = std::int32_t(d.batchCount);
    const auto stride = std::int64_t(rows * columns);
    check(library.cublasLtMatrixLayoutSetAttribute(target, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order, sizeof(order)), "cublasLtMatrixLayoutSetAttribute(order)");
    check(library.cublasLtMatrixLayoutSetAttribute(target, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batches, sizeof(batches)), "cublasLtMatrixLayoutSetAttribute(batch)");
    check(library.cublasLtMatrixLayoutSetAttribute(target, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride, sizeof(stride)), "cublasLtMatrixLayoutSetAttribute(stride)");
  };
  layout(p->a, d.transposeA ? d.k : d.m, d.transposeA ? d.m : d.k);
  layout(p->b, d.transposeB ? d.n : d.k, d.transposeB ? d.k : d.n);
  layout(p->c, d.m, d.n);
  const auto key = std::make_tuple(d.elementType, d.m, d.n, d.k, d.batchCount, d.transposeA, d.transposeB, d.workspaceLimit);
  {
    std::lock_guard<std::mutex> lock(globalAlgoMutex);
    auto it = globalAlgoCache.find(key);
    if (it != globalAlgoCache.end()) {
      p->algorithm = it->second.algo;
      p->workspace = it->second.workspace;
      return std::shared_ptr<CUBLASLtMatmul>(new CUBLASLtMatmul(std::move(p)));
    }
  }
  cublasLtMatmulPreference_t preference = nullptr;
  check(library.cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
  try {
    check(library.cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &d.workspaceLimit, sizeof(d.workspaceLimit)), "cublasLtMatmulPreferenceSetAttribute(workspace)");
    const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_MASK;
    check(library.cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        &reduction, sizeof(reduction)), "cublasLtMatmulPreferenceSetAttribute(reduction)");

    constexpr int maxAlgos = 16;
    cublasLtMatmulHeuristicResult_t results[maxAlgos]{};
    int found = 0;
    check(library.cublasLtMatmulAlgoGetHeuristic(context->handle, p->operation, p->a, p->b, p->c, p->c,
        preference, maxAlgos, results, &found), "cublasLtMatmulAlgoGetHeuristic");
    if (!found)
      throw CUDAError("cuBLASLt has no algorithm for the requested shape, precision and workspace limit");

    int bestIdx = -1;
    for (int i = 0; i < found; ++i) {
      if (results[i].state == CUBLAS_STATUS_SUCCESS) {
        bestIdx = i;
        break;
      }
    }
    if (bestIdx < 0)
      throw CUDAError("cuBLASLt returned no successful algorithm for the requested configuration");

    if (found > 1 && bridge && bridge->measure && bridge->scratch &&
        d.m >= 64 && d.n >= 64 && d.k >= 64) {
      float bestTime = 1e9f;
      void *matPtr = reinterpret_cast<void *>(bridge->scratch);
      const float alpha = 1.0f, beta = 0.0f;
      if (results[bestIdx].workspaceSize <= bridge->scratchBytes) {
        try {
          void *ws0 = results[bestIdx].workspaceSize
                          ? reinterpret_cast<void *>(bridge->scratch + (bridge->scratchBytes - results[bestIdx].workspaceSize))
                          : nullptr;
          float t0 = bridge->measure([&]() {
            check(library.cublasLtMatmul(context->handle, p->operation,
                &alpha, matPtr, p->a, matPtr, p->b, &beta, matPtr, p->c, matPtr, p->c,
                &results[bestIdx].algo, ws0, results[bestIdx].workspaceSize,
                static_cast<cudaStream_t>(bridge->stream)), "cublasLtMatmul (tune)");
          });
          if (t0 > 0.0f) bestTime = t0;
        } catch (...) {}
      }

      for (int i = 0; i < found; ++i) {
        if (i == bestIdx || results[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (results[i].workspaceSize > bridge->scratchBytes) continue;
        void *wsPtr = results[i].workspaceSize
                          ? reinterpret_cast<void *>(bridge->scratch + (bridge->scratchBytes - results[i].workspaceSize))
                          : nullptr;
        try {
          float elapsed = bridge->measure([&]() {
            check(library.cublasLtMatmul(context->handle, p->operation,
                &alpha, matPtr, p->a, matPtr, p->b, &beta, matPtr, p->c, matPtr, p->c,
                &results[i].algo, wsPtr, results[i].workspaceSize,
                static_cast<cudaStream_t>(bridge->stream)), "cublasLtMatmul (tune)");
          });
          if (elapsed <= 0.0f) continue;
          bool better = false;
          if (elapsed < bestTime * 0.99f) {
            better = true;
          } else if (elapsed <= bestTime * 1.01f) {
            if (results[i].workspaceSize < results[bestIdx].workspaceSize) {
              better = true;
            }
          }
          if (better) {
            bestTime = elapsed;
            bestIdx = i;
          }
        } catch (...) {}
      }
    }

    p->algorithm = results[bestIdx].algo;
    p->workspace = results[bestIdx].workspaceSize;
    {
      std::lock_guard<std::mutex> lock(globalAlgoMutex);
      globalAlgoCache[key] = {p->algorithm, p->workspace};
    }
  } catch (...) { library.cublasLtMatmulPreferenceDestroy(preference); throw; }
  library.cublasLtMatmulPreferenceDestroy(preference);
  return std::shared_ptr<CUBLASLtMatmul>(new CUBLASLtMatmul(std::move(p)));
}

void CUBLASLtMatmul::launch(const std::uint64_t *arguments, void *stream) const {
  const float alpha = 1.0f, beta = 0.0f;
  check(impl->context->library->cublasLtMatmul(impl->context->handle, impl->operation,
      &alpha, reinterpret_cast<const void *>(arguments[0]), impl->a,
      reinterpret_cast<const void *>(arguments[1]), impl->b, &beta,
      reinterpret_cast<const void *>(arguments[2]), impl->c,
      reinterpret_cast<void *>(arguments[2]), impl->c, &impl->algorithm,
      reinterpret_cast<void *>(arguments[3]), impl->workspace, static_cast<cudaStream_t>(stream)), "cublasLtMatmul");
}
#else
struct CUBLASLtContext {};
struct CUBLASLtMatmul::Impl { std::size_t workspace = 0; };
std::shared_ptr<CUBLASLtMatmul> CUBLASLtMatmul::prepare(
    const MatmulDescription &, std::shared_ptr<CUBLASLtContext> &,
    const MatmulTuningBridge *) {
  throw CUDAError("ASK Runtime was built without cuBLASLt headers");
}
void CUBLASLtMatmul::launch(const std::uint64_t *, void *) const {}
#endif

CUBLASLtMatmul::CUBLASLtMatmul(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
CUBLASLtMatmul::~CUBLASLtMatmul() = default;
std::size_t CUBLASLtMatmul::workspaceBytes() const { return impl->workspace; }

} // namespace ask::runtime::detail
