Back to Core Concepts: [Division of Labor](concepts.md#ask-positioning) · [Test Verification](concepts.md#ask-testing-roadmap) · Related details: [ASK IR](ir.md) · [External Execution Interface](artifacts.md) · [FugaTensor Architecture](../../../fugatensor/docs/en/architecture/README.md)

# Responsibilities of FugaTensor and ASK

<a id="division-of-labor"></a>

## 1. Framework and Compiler Runtime

[FugaTensor](../../../fugatensor/README.md) is recommended as the reference integration example for ASK. It provides a [Python API](../../../fugatensor/docs/en/API/README.md), its native extension links against ASK C++ libraries, and its wheel bundles the compiler and runtime shared libraries. For build steps, see [ASK Installation](install.md#framework-install) and [FugaTensor Installation](../../../fugatensor/docs/en/install.md).

| Concern | FugaTensor | ASK |
| --- | --- | --- |
| User API and Differentiation | Tensor, automatic graph construction, functional `ft.grad` pure IR transformations, `ft.fun` forward and VJP rules, evaluation triggering. | Static graph interface, compilation, and execution. |
| Control Flow and IR | Fuga IR expresses regions and function boundaries; submits ASK graphs after static unrolling or on-demand execution. | Single basic-block, shape-concrete MLIR `ask` graphs; native computation lowered to LLVM. |
| External Compilers | Preserves source code and compilation configurations, runs workers, validates and caches artifacts. | Generates compilation requests, verifies artifacts and launch parameters. |
| Execution | Selects backends, manages Tensor references, triggers evaluation, and submits plans. | Generates native code, partitions plans, manages device memory, events, and CUDA Graphs. |
| Provenance | Outputs `fuga-to-ask.json` mapping; see [Lowering to ASK](../../../fugatensor/docs/en/ir/lowering.md). | Preserves stable node/value identities, extracts `ask.mark`, and correlates with partition reports. |

Gradient stoppage is expressed by `fuga.stop_gradient` in Fuga IR, where JVP/VJP terminates propagation; when [lowering to ASK](../../../fugatensor/docs/en/ir/lowering.md#type-and-op-mapping), it directly uses the input value, and ASK executes the retained numerical dependency.

<a id="implemented-scope"></a>

## 2. Compilation and Execution Capabilities

- Six explicit types, single-step RNE constants, and 6×6 conversions; Host execution uses `bool/i32/i64/f32/f16`, `bf16` execution uses CUDA `sm_80+`.
- C++ `ASK::Compiler`, `ASK::Runtime`, `ASK::Core`, and `ask-opt`; shape inference, provenance extraction, native constant folding, DCE, and elementwise fusion.
- Native Host shared-loop `.o` and CUDA cubin code generation; multi-port, 0-D tensors, and empty tensors.
- Broadcast, transpose, reduction, conditional select, and floating-point functions; CUDA matrix multiplication uses cuBLASLt, CPU f32 uses OpenBLAS, and CPU f16 uses native code generation.
- External cubin/PTX integration and protocol v2; callers run workers, and each external node is an independent plan step.
- Native resident session, device buffers, whole-plan submission, asynchronous events and resource reuse, CUDA Graph replay; FugaTensor builds on top of this with CAI borrowing, disk caching, and relocatable multi-kernel compilation packages.

[attention](../../../fugatensor/docs/en/API/attention.md) is an example: FugaTensor provides algorithms and backward rules, while ASK schedules corresponding kernels.

<a id="roadmap"></a>

## 3. Build Dependency Directions

| Project | Provided Interface | Build-Time Consumption |
| --- | --- | --- |
| FugaTensor | Python Tensor API, Fuga IR, automatic differentiation, third-party workers | `Compiler` and `Runtime` from ASK installation package, plus nanobind |
| ASK | C++ compiler, device runtime, execution plan verification | LLVM/MLIR, Host OpenBLAS SDK; optional cuBLASLt headers for CUDA matmul |

FugaTensor first completes graph transformations via Fuga IR before submitting static graphs to ASK. ASK constructs plans and verifies buffers and storage slots using native C++ validators. Tensors and differentiation rules are maintained by FugaTensor, while execution plans and slot constraints are maintained by ASK.

Install the ASK SDK first, then build FugaTensor using `CMAKE_PREFIX_PATH`. For interface and dependency details, see [Frontend Integration](build.md#native-compiler) and [Plan Verification Build](verification.md#build-dependency).
