Back to Core Concepts: [Owned Code Generation](concepts.md#ask-owned-codegen) · [Graph Partitioning and Execution Plans](concepts.md#execution-plan) · Related details: [Primitive Semantics](ir.md#tensor-primitives) · [Matrix Multiplication API](api.md#matmul-api) · [FugaTensor Basic Operations](../../../fugatensor/docs/en/API/ops.md)

# Primitives, Fusion, and Matrix Multiplication Verification Report

Performance tables record sampled results from the primitives and matrix multiplication delivery phase; for the current execution plan verification implementation, see [Dedicated Static Verification](verification.md#plan-proof).

This delivery adds 10 orthogonal primitives and matrix multiplication, bridging Fuga's type inference, automatic differentiation, compilation packages, and native execution. Elementwise computation remains fused, while CUDA matrix multiplication utilizes NVIDIA cuBLASLt. CPU tensors and code generation support five types (`bool/i32/i64/f32/f16`), while BF16 uses CUDA.

<a id="implementation"></a>

## Implementation and Boundaries

| Responsibility | ASK Primitive | Fuga User Interface |
| --- | --- | --- |
| Dimension Mapping | `broadcast_in_dim`, `transpose` | `broadcast_in_dim`, `broadcast_to`, `transpose`, `.T` |
| Reduction | `reduce_sum`, `reduce_max` | `sum`, `amax`, supporting axes and `keepdims` |
| Conditional Selection | `compare`, `select` | Comparison operators, `compare`, `where` |
| Floating-Point Computation | `div`, `exp`, `log`, `sqrt` | `/`, `exp`, `log`, `sqrt` |
| Matrix Multiplication | `matmul` | `matmul`, `@`, supporting batches and transposition of final two axes |

Fuga's JVP/VJP continues to generate the same set of primitives, with conditional inputs remaining non-differentiable; equal extrema in max split gradients equally, and NaN rows propagate NaN. Type casting and broadcasting are expressed explicitly. Sum accumulates stepwise according to input type, requiring an explicit cast when f32 accumulation is needed; matmul uses f32 accumulation.

CPU employs a five-type whitelist: ASK verifies prior to graph optimization, and Fuga verifies during CPU storage allocation or import. Host implementations, public compilation options, and build configurations have purged BF16-specific paths. GPU `from_buffer` uploads directly following general byte validation, and factory functions populate device buffers using scalar encodings; these paths remain independent of CPU tensor storage. BF16 weights are loaded onto CUDA.

Host and CUDA share index expression lowering. Intermediate values of contiguous elementwise chains reside in SSA registers, while broadcasting and transposition map index reads. True reduction results are precomputed and reused when expanded again; empty axes and single-element reductions continue to participate in fusion. CUDA `mul → matmul → add` forms three steps, with elementwise regions on both sides fused independently.

cuBLASLt descriptors, algorithms, and actual scratch sizes are fixed in `prepareProgram`, and scratch space joins the same slot verification in ASK PASS. Successful verifications are cached per complete plan; execution functions per step are selected ahead of time. The submission loop executes prepared steps, centralizing dynamic public buffer checks at the submission entry point.

Library steps are captured as subgraphs in Graph mode. Direct replay occurs when addresses remain unchanged; subgraphs are recaptured and updated when addresses change, keeping inflight workspaces independent. `workspaceLimit` defaults to 4 MiB, with actual allocation governed by algorithmic requirements; scratch spaces of sequential library steps can share slots.

cuBLASLt enters only the private implementation of Runtime, keeping public headers standard C++17. When the provider is enabled, the build consumes CUDA headers and dynamically loads libraries at execution time; base Runtime builds with `ASK_ENABLE_CUBLASLT=OFF` are verified. Consuming Compiler/Runtime installations does not require LLVM development SDKs; Core uses matching LLVM/MLIR. Execution and slot constraints of library steps are checked by ASK's native static verifier.

f32 matmul uses `CUBLAS_COMPUTE_32F_PEDANTIC`; low-precision inputs use f32 accumulation and restrict split-K intermediate reduction types to f32. For rationale regarding algorithm and scratch selection, see the [NVIDIA cuBLASLt Documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul). CUDA compilation of exp/log links libdevice, with Fuga cache recording `llvm-link` and libdevice digests.

<a id="fusion-profile"></a>

## Elementwise Fusion Measurements

Environment: WSL/Linux, GCC 14.2, Release, LLVM/MLIR 23.1.4, RTX 3080, Driver 615.65.06. cuBLASLt originates from system CUDA Toolkit 13.3.1; native kernels use configured ptxas 12.9.86 and bundled libdevice. Measurement code is separated from production code; execution runs were kept distinct from GPU numerical test suites.

Across identical f32 chains `add → mul → div → sqrt → exp`, native plans comparing fused single-kernel vs. five independent kernels were evaluated. Inputs and outputs were resident in advance, with 25 warmup runs and 51 samples; elapsed time begins at `submit` and ends when completion event waits finish. The five-step plan utilizes two reusable temporary slots.

| Element Count | Baseline Plan: 5 Kernels / μs | Fused: 1 Kernel / μs | Scratch VRAM: Independent → Fused |
| ---: | ---: | ---: | ---: |
| 256 | 84.86 | 45.55 | 2 KiB → 0 |
| 4,096 | 86.25 | 46.35 | 32 KiB → 0 |
| 65,536 | 88.02 | 46.61 | 512 KiB → 0 |
| 1,048,576 | 120.21 | 57.75 | 8 MiB → 0 |

The values above represent medians. Normal submission overhead itself dropped from approximately 42–47 μs to 13 μs; end-to-end latency for one million elements decreased by roughly 52.0%. In Graph mode, latency for one million elements fell from 110.23 μs to 57.65 μs, with CPU submission averaging around 12 μs; for 65,536 elements, latency shifted from 43.64 μs → 42.75 μs, where small graph fluctuations approach scheduling noise. On CPU for the same one-million-element chain, latency decreased from 7,846.48 μs to 7,524.41 μs (~4.1% gain), with transcendental functions remaining the primary bottleneck.

Independent numerical references, slot bytes, and launch counts were used to verify correctness. Each pair of scenarios was sampled alternately; for full CPU/CUDA datasets and P95 metrics, see the [Elementwise Fusion Measurement Table](../assets/逐元素融合测量.csv). Time measured reflects submit-to-complete duration.

<a id="cpu-reduction"></a>

## CPU Reduction Reuse

Audits revealed that whole-graph Host lowering previously embedded reductions within broadcast output loops, causing repeated input reads. Following the fix, complete reductions are retained as SSA values outside loops, and partial reductions write first to statically planned scratch space.

Xeon E5-2696 v4, f32, native C++ calling generated functions directly, 10 warmups, 21 pairs of alternating samples; using identical HostObject options and `cc -shared -lm`. The gains below correspond to two instances where redundant reductions were eliminated:

| Graph | Before Fix / μs | After Fix / μs | Input Loads: Before → After | Added Scratch Space |
| --- | ---: | ---: | ---: | ---: |
| 4096-element full sum, then splat to 4096 elements | 18,374.73 | 6.08 | 16,777,216 → 4,096 | 0 |
| `[64,256]` sum along axis 1, then broadcast back to original shape | 4,289.31 | 26.51 | 4,194,304 → 16,384 | 256 B |

The [CPU Reduction Measurement Table](../assets/CPU归约复用测量.csv) preserves the summary above. In addition, 338 independent CPU numerical checks cover dtypes, shapes, reduction axes, subsequent expressions, and scratch alignment for mixed types. Empty outputs halt tracing of numerical dependencies; zero-capacity workspaces generate neither allocations nor failure branches.

<a id="matmul-profile"></a>

## Matrix Multiplication and Component Overhead

Direct cuBLASLt and ASK use identical data types, compute precision, natural alignment constraints, workspace limits, and heuristic selections. Across 24 paired configurations generated by f32/f16 × 128/512/1024 square matrices × stream/Graph × fixed/alternating addresses, each sample measured both execution paths alternately. Preparation phase ran 31 pairs; warm submission ran 101 pairs after 15 warmups; device time measured another 51 pairs.

Device timing placed a gate before the stream, releasing it after all commands were queued to eliminate host queuing gaps using CUDA events. Host submission and submit-to-complete durations were measured independently. ASK's `resident_device_bytes` accounts for actual owned allocations, including allocation pools; driver and library internal storage are managed by their respective owners.

Medians for fixed-address Graph execution are shown below in μs; for the full 24 scenario groups and P95 metrics, see the [Measurement Table](../assets/矩阵乘法测量.csv).

| dtype / N | CPU Submission: Direct / ASK | Device Time: Direct / ASK | Submit-to-Complete: Direct / ASK |
| --- | ---: | ---: | ---: |
| f32 / 128 | 9.81 / 13.10 | 8.90 / 8.19 | 51.06 / 51.19 |
| f32 / 512 | 9.65 / 12.44 | 26.62 / 26.62 | 59.18 / 70.27 |
| f32 / 1024 | 9.94 / 13.35 | 126.98 / 126.98 | 159.52 / 168.17 |
| f16 / 128 | 9.90 / 12.98 | 6.98 / 6.14 | 41.04 / 46.43 |
| f16 / 512 | 9.85 / 12.67 | 20.48 / 20.48 | 52.51 / 62.51 |
| f16 / 1024 | 9.58 / 12.52 | 72.70 / 72.70 | 104.34 / 114.48 |

Relative to direct invocation, paired median CPU submission deltas across scenarios were: baseline stream **4.87 μs**, fixed-address Graph **3.17 μs**, and alternating-address Graph **4.73 μs**. Warm preparation was approximately 2.38–2.54 μs for direct calls vs. 5.84–6.58 μs for ASK. Eliminating temporary message strings on successful checks reduced paired median excess submission overhead by roughly 0.73 μs; runtime continues to validate actual public buffers.

In this round, ASK fixed-address Graph submission took ~12.44–13.35 μs, while alternating addresses took ~29.48–42.38 μs. The latter includes recapturing and updating library subgraphs; preserving buffer addresses mitigates this overhead. Device execution times closely match direct cuBLASLt, with ASK's primary additional overhead lying in host preparation and dispatch.

Owned VRAM and peak usage were identical across all paired scenarios. The benchmark preallocated two sets of A/B/C for alternating addresses; f16 512 square matrices occupied 3 MiB of port storage, plus 4 MiB of actual scratch space, totaling 7 MiB; chosen algorithms for the other five dtype/size configurations required zero scratch. Two sequential matmuls reused the same scratch slot; zero-workspace plans selected with `workspaceLimit=0` were also verified independently.

<a id="fuga-profile"></a>

## Fuga Invocation Hierarchy

Hierarchical profiling measured native binding `submit`, prepared object `submit_many`, user `a @ b`, `evaluate`, and completion waits. Each layer ran 20 warmups across 101 retained samples with inputs pre-resident; the user layer also encompasses graph construction, cache lookup, result allocation, and Python object management.

Medians for fixed-address Graph "submit / submit-to-complete" are listed below in μs. The user layer operated via automatic output allocation, including corresponding Graph address updates.

| dtype / N | Native Bindings | Python Prepared Object | User `a @ b` + `evaluate` |
| --- | ---: | ---: | ---: |
| f32 / 128 | 16.66 / 52.53 | 38.15 / 68.75 | 186.54 / 200.68 |
| f32 / 512 | 16.24 / 69.33 | 37.41 / 86.84 | 189.96 / 221.22 |
| f16 / 128 | 16.29 / 49.84 | 37.23 / 68.60 | 183.54 / 214.04 |
| f16 / 512 | 16.36 / 64.16 | 37.28 / 83.38 | 192.69 / 223.28 |

For this set of pure library graphs, median ASK partitioning took ~0.76–0.81 ms, and Fuga compilation package construction took ~1.91–3.54 ms; within an existing session, Python `prepare` took ~127–135 μs, covering plan verification, Python descriptor conversion, and native preparation. Warm user invocations took ~0.20–0.22 ms. Inter-layer deltas encompass resource handling at each layer; the [Layered Measurement Table](../assets/Fuga调用分层测量.csv) preserves stream, Graph, alternating address, and P95 data.

Cold starts were evaluated using six independent processes, three runs per configuration. The table below lists median durations in ms; totals were computed per run before taking medians. Timing spans imports, Engine, context, input upload, graph compilation/preparation, and initial device completion, excluding measurement probes, test data generation, and result readbacks.

| Configuration | Import | Engine | Context / Stream | Input Upload | Compile, Prepare, First Execution | Segment Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 / 128 | 82.86 | 1.02 | 313.16 | 8.41 | 212.58 | 615.13 |
| f16 / 512 | 86.72 | 1.04 | 322.43 | 8.64 | 473.90 | 879.13 |

VRAM was queried via identical driver interfaces after session creation and following initial run completion: for f32 128, net device consumption increased by **4 MiB**, with ASK owned current/peak both at **192 KiB**; for f16 512, net consumption increased by **14 MiB**, with ASK owned current/peak both at **5.5 MiB** (three ports totaling 1.5 MiB, scratch 4 MiB). Net driver increase also reflects library/Graph storage and allocation granularity; three independent processes yielded identical net increases. The [Cold Start Measurement Table](../assets/Fuga冷启动测量.csv) stores segmented results across six runs.

<a id="validation"></a>

## Testing and Delivery

Permanent coverage generates combinations along orthogonal axes, with numerical references and input generators located in `tests/` support modules. ASK validates IR, fusion, and execution contracts; Fuga verifies user APIs, numerical correctness, and automatic differentiation.

| Check | Result |
| --- | --- |
| ASK default compile/primitives/proofs & physical GPU | Final CTest 6/6; matmul base matrix 360 suites |
| Installed SDK standalone consumption | Compiler 2/2, Runtime 2/2, Core 1/1 |
| Optional library dependency | `ASK_ENABLE_CUBLASLT=OFF` build passed; existing owned/Graph Runtime tests passed |
| Project scripts | 5/5; CI entry point fully executes 4 base suites; `--cublaslt` verifies all 6 suites |
| Execution plan schema | 24 generated plans passed, 120 invalid variants rejected |
| Fuga full API, CUDA/Triton enabled | 172 items: 171 passed on first run; after updating 1 legacy comparison rejection assertion, all 5 dedicated items in its group passed; 0 skip |
| Fuga new primitives CPU/GPU numerical matrix | 15/15, 94.339 s, 0 skip; using tightened low-precision tolerances |
| Fuga backend protocol/caching | 64/64 |
| Fuga final GPU integration | 9/9, 7.883 s, 0 skip; also includes 12 public scalar configs across 6 dtypes × identity/constant |
| Fuga types, storage, and weight regressions | API 34/34, Host backend 3/3, CPU unsupported types dedicated 1/1, 0 skip |
| Creation and data exchange matrix | CPU: 120 valid combinations, 28 unsupported type rejections; GPU: 144 valid combinations, plus verification of BF16 4-shape computation, 6 raw byte transfers, and weight roundtrips |

Existing operator combination matrices cover 1,900 CPU / 2,350 CUDA configurations and 7,300 / 10,450 forward and derivative outputs. New geometry matrices cover 30 CPU / 36 CUDA configurations traversing axis subsets and permutations; matmul covers 160 CPU / 240 CUDA configurations crossing dtypes, shapes, batching, and transposition. CPU numerical matrices and Host capabilities lists employ five types, while BF16 numerical verification uses CUDA.

Native library tests further cover zero-dimensional scenarios, alternating-address Graphs, inflight output preservation, naturally aligned offset imports, owned/library mixed steps, 0/4 MiB scratch spaces, and low-precision intermediate overflow cancellations. Permanent compilation tests verify single-kernel fusion, temporary buffer counts, reduction materialization, public scalars, empty outputs, and single-element reductions.

The final SDK resides in `build/sdk`, with FugaTensor rebuilt against it; byte-for-byte SHA-256 consistency was verified between installed Compiler/Runtime and SDK artifacts. ASK implementation and final contracts were committed as `8a6d5b6`, with preceding commits recording primitives, Runtime, partition integration, reduction reuse, and CPU type boundaries. The final FugaTensor commit is `ea33f41`.

New API documentation elaborates on minimal invocation, shape rules, and foldable parameter details. Engineering documentation synchronizes three execution steps, optional dependencies, libdevice, precision, Graph updates, and VRAM tracking scopes. Temporary validation, benchmark code, and transient consumers were removed upon completion; measurement summaries are preserved alongside reports, and build artifacts remain in ignored directories.