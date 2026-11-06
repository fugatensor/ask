Back to Core Concepts: [Testing and Verification](concepts.md#ask-testing-roadmap) · Related details: [Frontend and Build](build.md) · [Numerical Semantics](ir.md) · [External Artifacts and Execution Interface](artifacts.md)

# Project Structure and Testing Suite

<a id="repo-layout"></a>

## 1. Directory Responsibilities

```text
ask/
├── include/ask/           # Public interfaces for Compiler facade, MLIR/LLVM, and native Runtime
├── lib/
│   ├── IR/               # ask dialect and operation verification
│   ├── Frontend/         # Preparation, capabilities reporting, graph partitioning, and compiler facade
│   ├── Transforms/       # Constant folding, DCE, native elementwise fusion
│   ├── Lowering/         # Native Host/CUDA LLVM lowering
│   ├── LLIR/             # LLVM import, cleanup, and numerical helpers
│   ├── Pass/             # Dedicated static verification for execution plans and memory slots
│   └── Runtime/          # CUDA plans, buffers, events, and Graph replay
├── schemas/              # JSON schemas for execution plans and external requests/responses
├── cmake/                # LLVM dependency locks, component export, and installation configs
├── tools/                # ask-opt, dependency retrieval, and project sanity checks
├── tests/ir/             # MLIR inputs used by CLI and frontend tests
├── tests/fixtures/       # PTX inputs used by Runtime
├── tests/cpp/            # Compiler/Core/Pass/Runtime assertions and standalone SDK consumer tests
└── docs/                 # Documentation for interfaces, architecture, and engineering
```

FugaTensor's `cpp/` bindings link against ASK; graph transformations, worker invocations, disk caching, and compilation package validations reside in `src/fugatensor/_ask/`.

`.deps/`, `build/`, `dist/`, virtual environments, and caches are excluded from version control; `cmake/*.lock.json` and dependency fetching scripts are retained.

<a id="ctest-targets"></a>

## 2. Native CTest

`ctest --preset core --output-on-failure` runs CPU checks by default; GPU tests are enabled via `ASK_TEST_CUDA=ON`. For interface examples, see [User API](api.md).

| CTest Name | Entry Point | Coverage Scope |
| --- | --- | --- |
| `ask-core` | `tests/cpp/frontend.cpp` | Core static graph preparation, shapes, and optimizations. |
| `ask-compiler` | `tests/cpp/compiler.cpp` | Standard C++ compilation facade, Host/CUDA code generation, rejection of invalid inputs, numerical literals, and independent concurrent contexts. |
| `ask-primitives` | `tests/cpp/primitives.cpp` | dtype × shape elementwise fusion, index transformations, reduction materialization, matmul partitioning, and CUDA warp maximum dispatch; reuses `support/primitive_graphs.h` to generate graphs. |
| `ask-host-blas` | `tests/cpp/host_blas.cpp` | Generates batch × M/N/K (including 0/1) × transpose × pre/post fusion matrices, linking OpenBLAS from the installed SDK against independent double references; Compiler consumer tests reuse this same entry point. |
| `ask-pass` | `tests/cpp/pass.cpp` | Generates matrices across topology, step count, byte size, slot allocation, deferred reads, and capacity, reusing `support/plan_matrix.h` with exhaustive pairwise interval comparison as independent reference; also verifies large plans, complete cache keys, public port isolation, and invalid candidates. |
| `ask-runtime` | `tests/cpp/runtime.cpp` | Multi-step PTX execution on physical GPUs, rejection of write-before-read plans, internal slot allocation, dynamic inputs, and workspace/Graph instance reuse; registered only when `ASK_TEST_CUDA=ON`. |
| `ask-matmul` | `tests/cpp/matmul.cpp` | dtype × shape × batch × transpose × stream/Graph, totaling 360 suites; also verifies address changes, inflight outputs, mixed steps, precision, and scratch space. Requires GPU and enabled cuBLASLt. |

```bash
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure

# Execute additionally in environments with a compatible GPU.
cmake --preset core -DASK_TEST_CUDA=ON
cmake --build --preset core
ctest --preset core -R '^ask-(runtime|matmul)$' --output-on-failure
```

GPU tests require an available driver and device; missing devices when enabled are treated as failures.

<a id="matrix-and-reference"></a>

## 3. FugaTensor Integration and End-to-End Verification

[FugaTensor](../../../fugatensor/README.md) serves as the recommended integration example: constructing graphs from the [Tensor API](../../../fugatensor/docs/en/API/README.md), differentiating in Fuga IR, and passing down to ASK for compilation and dispatch via [Native Bindings](../../../fugatensor/docs/en/architecture/backend.md).

FugaTensor's end-to-end tests cover graph semantics, caching, CPU/CUDA numerical correctness, external kernels, and multi-step plans. Base API tests generate operator combinations from the registry crossed with precision, shape, and differentiation order; CAI contracts independently cross dtypes, shapes, and layouts; for execution and filtering, see [Testing Matrix](../../../fugatensor/docs/en/testing.md).

Numerical references use Python arbitrary-precision integers, covering single-step RNE, subnormals, integer wraparound, saturating casts, large integers, negative zero, infinity, NaN, and non-full tail tiles. New combinations reuse this reference and `plan_matrix`. External algorithmic formulas are tested by producers, while the ASK side verifies ABIs, resource lifetimes, and step connectivity.

<a id="continuous-integration"></a>

## 4. SDK Distribution Verification

SDK distribution checks include configuration builds, default CTest, installation into a standalone directory, and independently consuming Compiler, Runtime, and Core from that installation directory. Compiler/Runtime projects link against ASK shared libraries via public headers; Core additionally configures an LLVM/MLIR SDK matching ASK's build version.

`python3 tools/run_ci_tests.py --build-dir build/core` runs the base suites and the Host-configured OpenBLAS link and numerical suite, verifying result reports; missing suites or skipped tests trigger errors. Adding `--cuda` requires the GPU Runtime suite; adding `--cublaslt` requires both GPU Runtime and matmul suites.

```bash
cmake --install build/core --prefix /path/to/ask-sdk
cmake -S tests/cpp -B /path/to/runtime-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH=/path/to/ask-sdk -DASK_TEST_COMPONENT=Runtime
cmake --build /path/to/runtime-consumer
```

`ASK_TEST_COMPONENT` accepts `Compiler` (default), `Runtime`, or `Core`. Runtime can be built in CPU environments, requiring a GPU only during test execution; `ASK_TEST_CUBLASLT=ON` adds matrix multiplication verification for installed packages. FugaTensor independently verifies wheel bindings, shared library packaging, and isolated installations.