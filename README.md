<img src="docs/assets/ask-logo-inverse.svg" alt="ASK logo" width="100%">

---

<p align="center">
  <b><a href="README.md">English</a></b> | <b><a href="README.zh.md">简体中文</a></b>
</p>

<p align="center">
  <a href="docs/en/install.md#requirements"><img src="https://img.shields.io/badge/C%2B%2B-17-00599c" alt="C++17"></a>
  <a href="docs/en/build.md#version-contract"><img src="https://img.shields.io/badge/LLVM-23.1.x-64748b" alt="LLVM 23.1.x"></a>
  <a href="docs/en/concepts.md#ask-owned-codegen"><img src="https://img.shields.io/badge/backend-CPU%20%7C%20CUDA-76b900" alt="CPU / CUDA"></a>
</p>

ASK is a C++ compiler and runtime for tensor frameworks. It ingests static MLIR `ask` graphs, generates CPU/CUDA machine code, and orchestrates native as well as third-party cubin/PTX kernels.

- **Types and Numerics**: Host supports `bool`, `i32`, `i64`, `f32`, `f16`; CUDA additionally supports `bf16` (`sm_80+`). Floating-point constants are rounded using RNE; additions and multiplications require identical shapes and element types.
- **Compilation & Fusion**: Constant folding, DCE, multi-output elementwise loop fusion; Host generates `.o`, CUDA generates LLVM IR followed by `llc` / `ptxas` cubin generation.
- **Execution Plans**: The ASK Pass statically verifies buffer slot mappings and caches valid plans at prepare time; execution reuses argument templates, scratch workspaces, and completion events with full CUDA Graph replay support.
- **External Kernels**: Connected via device buffers as decoupled steps, supporting multiple inputs, multiple outputs, pointer arguments, and by-value scalars.

The project provides a C++ SDK and the `ask-opt` utility. The Compiler SDK features named compilation options and stage-segregated results, returning execution plans, ABIs, artifacts, and diagnostics independently. [FugaTensor](../fugatensor/README.md) builds on ASK to deliver Python APIs, automatic graph construction, differentiation, and external compiler adapters.

For interfaces, see [User API](docs/en/api.md); for architectural principles, see [Core Concepts](docs/en/concepts.md). For end-to-end integration examples, refer to FugaTensor's [Tensor API](../fugatensor/docs/en/API/README.md) and [Lowering, Native Bindings & Execution](../fugatensor/docs/en/architecture/backend.md).

## Build, Install and Verification

Building requires C++17, CMake 3.28+, Ninja, and matching LLVM/MLIR 23.1.x SDKs. By default, the pre-built Linux x86-64 release for LLVM 23.1.4 is fetched; see [Dependency Locks](cmake/llvm.lock.json) for versions and checksums. Execution plans are verified ahead-of-time by ASK's native C++ static verifier; see [Plan Verification](docs/en/verification.md). Dependency download scripts use the Python 3 standard library and build a locked native OpenBLAS SDK via C compiler and Make for Host FP32 matrix multiplication.

```bash
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
cmake --install build/core --prefix /path/to/ask-sdk
```

Consumers link against `ASK::Compiler` and `ASK::Runtime`; code manipulating MLIR/LLVM directly links `ASK::Core` with matching LLVM/MLIR SDKs. See [Installation Guide](docs/en/install.md) for components and dependencies.

```cmake
find_package(ASK CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

## Testing

```bash
build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --partition \
  -o build/partitioned.mlir --report-json build/plan-report.json

# Enable physical GPU tests for native runtime, resource reuse, and CUDA Graph replay
cmake --preset core -DASK_TEST_CUDA=ON
cmake --build --preset core
ctest --preset core --output-on-failure
```

Default CTest runs `ask-core`, `ask-compiler`, `ask-pass`, `ask-primitives`, and `ask-host-blas`; enabling GPU adds `ask-runtime`, and enabling cuBLASLt adds `ask-matmul`. C++ source and SDK consumer tests reside in `tests/`; see [Testing Infrastructure](docs/en/testing.md).

FugaTensor's [End-to-End Tests](../fugatensor/docs/en/testing.md) validate graph lowering, numerics, autograd, and external kernel scheduling.

## License

This project is licensed under the [Apache-2.0](LICENSE) License.

[Documentation Index](docs/en/README.md) · [External Worker Protocol](docs/en/compilers.md) · [Execution Plans](docs/en/partitioning.md)
