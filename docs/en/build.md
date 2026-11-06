Back to Core Concepts: [Build and In-Process Integration](concepts.md#ask-frontend-build) · Related details: [User API](api.md) · [ASK IR and Numerical Semantics](ir.md) · [Project Structure and Testing Suite](testing.md)

# Frontend Integration and Build Configuration

ASK provides the C++ components `ASK::Compiler`, `ASK::Runtime`, `ASK::Core`, and the command-line tool `ask-opt`. FugaTensor's native bindings link directly against ASK shared libraries to expose Python APIs to users.

<a id="build-workflow"></a>

## 1. Dependency Acquisition, Build, and Installation

Building requires C++17, CMake 3.28+, Ninja, and LLVM/MLIR 23.1.x stable release from the same installation. `cmake/llvm.lock.json` records the source and digest for the Linux x86-64 SDK 23.1.4. ASK's execution plan verification is implemented in `lib/Pass/`. Dependency fetching and project verification scripts use the Python standard library.

```bash
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
cmake --install build/core --prefix /path/to/ask-install
```

The fetch script supports `--only llvm`, verifies SHA-256 digests, archive paths, and symlinks, and records fetching metadata in `.deps/<dir>/.ask-sdk.json`. CMake automatically discovers SDKs in `.deps/`; a compatible stable release can also be specified via `-DLLVM_DIR` / `-DMLIR_DIR`. LLVM and MLIR must come from the same version and the same installation. For installation layout, see [Installation Guide](install.md).

<a id="compiler-selection"></a>

## 2. Core Configuration and External Compilers

The `core` preset builds CPU/CUDA lowering and the device runtime. `ASK_TARGET_BACKENDS=host` enables Host object code generation, while `none` disables it; both configurations preserve CUDA LLVM lowering. `ASK_ENABLE_CUBLASLT` controls the Runtime library provider; for build and dynamic loading requirements, see [CUDA Tooling](install.md#cuda-and-adapters).

Default CTest runs Compiler, Core, Pass, Primitives, and Host OpenBLAS-linked numerical tests. Configuring `-DASK_TEST_CUDA=ON` adds GPU runtime tests, and adds matmul matrices when cuBLASLt is enabled; execution requires available drivers and devices.

Callers manage third-party compiler workers, runtime environments, and artifact caches. ASK inspects `ask.external`, generates partitioned execution plans, and schedules cubin/PTX. FugaTensor provides a Triton worker; see [External Artifacts and Execution Interface](artifacts.md) for the protocol.

<a id="native-compiler"></a>
<a id="python-sdk"></a>

## 3. Compiler and Runtime Interfaces

```cmake
find_package(ASK 0.1.0 EXACT CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

The public header for `ASK::Compiler` is `ask/Frontend/CompilerAPI.h`. `ask::compiler::prepare`, `partition`, and `codegen` accept an ASK MLIR string and named options, returning `PrepareResult`, `PartitionResult`, and `CodegenResult` respectively. Results provide IR, shape snapshots, external requests, execution plans, step sources, and target ABI/artifacts via distinct fields; `diagnosticsJSON` carries stage, provenance, and statistical information. The interface uses only C++17 standard library types; see [Compilation Interface](api.md#compiler-api) for concrete field descriptions.

Each request uses an independent MLIR context, within which parsing, optimization, and lowering take place. Failures throw `std::runtime_error`. `prepareJSON`, `partitionJSON`, and `codegenJSON` in `namespace ask` return `{ir, reportJSON}`; the CLI likewise provides complete JSON reports.

The public header for `ASK::Runtime` is `ask/Runtime/CUDA.h`, providing `Session`, `PreparedProgram`, `DeviceBuffer`, `Kernel`, and `Event`, supporting both sequential submission and CUDA Graph replay. For invocation examples, see [User API](api.md).

<a id="cpp-sdk"></a>

## 4. Core Interface for Direct MLIR Manipulation

Hosts requiring custom MLIR graph construction or transformations link against `ASK::Core` and provide an LLVM/MLIR SDK matching ASK's actual build version:

```cmake
find_package(ASK CONFIG REQUIRED COMPONENTS Core)
target_link_libraries(my_frontend PRIVATE ASK::Core)
```

`ask/Frontend/Compiler.h` provides `prepareSource`, `partitionSource`, and `codegenSource`, returning a `PreparedSource` containing LLVM JSON objects. Existing MLIR modules can directly invoke `prepareModule` / `optimizeModule`; these two interfaces mutate the module in place, so callers needing transactional semantics must clone the module prior to submission. Static shapes, preparation pipelines, and optimization behavior are verified by [Core tests](../../tests/cpp/frontend.cpp).

Reports are serialized using `ask::reportJSON(result.report, result.floatPaths)`, preserving lexical floating-point types and negative zero in external configurations. `floatPaths` is consumed by the serializer. Compiler and CLI share this entry point.

<a id="cli-driver"></a>

## 5. `ask-opt` Command-Line Options

| Option | Usage |
| --- | --- |
| `--capabilities` | Output build version, target backends, and supported types per backend |
| `--prepare` | Validate, infer static shapes, extract and erase provenance markers |
| `--optimize` | Perform constant folding, DCE, and native elementwise fusion grouping after preparation |
| `--no-constant-fold` / `--no-dce` / `--no-fusion` | Disable corresponding graph optimizations |
| `--partition` | Generate native/external/library steps, logical buffers, dependencies, and `step_sources`; supports `--no-optimize` |
| `--emit-llvm` / `--emit-object` | Generate Host LLVM text or object file; object files require specifying path via `-o` |
| `--emit-cuda-llvm` | Generate CUDA LLVM text in conjunction with `--cuda-arch`, `--cuda-block-size`, `--cuda-ptx-version` |
| `--check-llir` / `--optimize-llir` | Validate or sanitize native LLVM IR |
| `--report-json FILE` | Write preparation, partitioning, or ABI report to a file |

FugaTensor invokes shared libraries by default to perform compilation and execution; callers convert CUDA LLVM IR into PTX/cubin using `llc` / `ptxas`.

<a id="version-contract"></a>

## 6. Version and Shared Library Contracts

| Field | Source and Requirements |
| --- | --- |
| ASK Release Version | Root `VERSION`, currently `0.1.0`; shared across CMake, shared library versions, and capabilities |
| C++ Package Compatibility | `ASKConfigVersion.cmake` uses `ExactVersion`; callers must use matching versions and compatible C++ toolchains |
| Core Dependencies | LLVM/MLIR must match ASK's actual build version and originate from the same installation |
| Build Identity | `build_id` records build directory, ASK and LLVM build info, Host switches, and OpenBLAS lock records; compilation caches differentiate builds by actual shared library content digests |
| Data Protocols | ASK IR / execution plans are v1; external worker / normalized CUDA invocation is v2; versions evolve independently |

Installation artifacts include `libask_compiler.so`, `libask_runtime.so`, `libask_core.a`, public headers, CMake targets, `ask-opt`, and dependency lock records; Host compilation components also bundle the OpenBLAS library and license. Compilation caches should record the actual digest of the code generation library to distinguish different builds of the same version; `ask::compilerLibraryPath()` returns the path to the loaded Compiler shared library.

Shared libraries follow standard C++ library ABI conventions; callers must use compatible toolchains. For verification methods, see [CI and Installation Acceptance](testing.md#continuous-integration).
