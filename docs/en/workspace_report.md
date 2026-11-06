Back to Core Concepts: [Resident Runtime and Caching](concepts.md#ask-runtime-cache) · Related details: [Static Plan Verification](verification.md#plan-proof) · [Host Code Generation](codegen.md#host-codegen) · [Fuga Integration](../../../fugatensor/docs/en/architecture/backend.md#native-plan-replay)

# CUDA Graph Dependency Scheduling and CPU Scratch Space Reuse Verification Report

Scratch space for CPU eight-layer reduction-expansion chains is reduced by 75%, with overall latency remaining essentially flat. CUDA branch concurrency was validated via device timestamps; overall latency for local short graphs showed no stable improvement, four-branch small workloads experienced regression, and internal VRAM increased for some plans. Both gains and overheads are reported below.

<a id="implementation"></a>

## Implementation and Interfaces

CUDA Graph establishes data dependencies from read/write ports of each step, preserving opportunities for independent branches to execute in parallel. CPU materialized tensors reuse scratch space based on actual reading phases. Both optimizations are accepted via dedicated C++ static verifications: GPU verification occurs during `prepareProgram`, while CPU verification occurs during code generation.

`Session::prepareProgram`, Fuga user APIs, and the Host C ABI retain their existing usage patterns. Added Session diagnostic counters report prepared internal capacities as well as actually instantiated nodes and edges; these update upon preparation or instantiation. Warm Graph replay follows existing port bindings and launch paths, and generated CPU loops also preserve their original branching.

ASK continues to deliver Compiler/Runtime SDKs independently, and Fuga consumes the installed packages. CPU types are bool, i32, i64, f32, and f16.

<a id="method"></a>

## Measurement Methodology

The baseline is the installed SDK of ASK `98a8bc6`. The environment comprises Xeon E5-2696 v4, RTX 3080, WSL2, GCC 14.2, and LLVM/MLIR 23.1.4. Both library versions and Fuga installations were snapshotted separately, running identical workloads in independent processes; builds and tests were halted during formal measurements.

A total of 7 rounds were conducted, alternating version execution order between rounds. Each cell first took the intra-process sample median, then aggregated the median across the 7 rounds; raw tables preserve per-round results, while summary tables record ranges and paired changes simultaneously. CPU timing was pinned to CPU 2, with input generation, module loading, and independent reference computation excluded from native execution timing.

- Native GPU: chain/branch-merge × 1,024/32,768 elements × 2/4 branches × stream/Graph × fixed/alternating public addresses. Each work kernel executes 512 rounds of integer recurrence per thread, with results verified elementwise. Prepare, initial submit-to-complete, warm submit, and warm submit-to-complete were timed separately.
- Device execution window: Each block of an independently instrumented kernel logs `%globaltimer` at start and finish, aggregating earliest start and latest end times and inspecting temporal interval overlaps across branches. This cross-check consists of 32 paired cells, each taking the median of 21 runs; normal end-to-end latency comes from the seven uninstrumented rounds.
- Native CPU: 32/8,192 rows × 4/16 columns × 3/8 layers of reduction-expansion × whether first-layer results are additionally returned. HostObject compilation, first invocation, and warm function calls were measured separately, verifying ABI workspaces and numerical results.
- Fuga: CPU reuses the aforementioned 16-column workload; GPU evaluates reduction-expansion with independent input branches as well as single-kernel fusion controls. Cold paths encompass graph construction, compilation, and first completion; warm paths create new expressions each time, timing graph construction, evaluate, and synchronous completion separately. Each cell asserts 25 actual executions and 1 compilation.

ASK VRAM counters track bytes of owned allocation requests, including allocation pools; Graph and driver internal storage are managed by the driver. Precise CPU scratch space was independently checked via sampled malloc/free, with sampling processes excluded from normal timing. Process RSS high-water marks include LLVM, Python, inputs, and reference data, reported separately.

Device timing initially evaluated host callback gates and GPU gates. Both methods introduced notable perturbations into parallel Graph event intervals; raw values are preserved in diagnostic tables. This report adopts gate-free device timestamp cross-evidence. Medians in data summaries are computed independently; sum of medians for graph construction, evaluate, and complete time may differ from the complete invocation median.

<a id="cuda-graph"></a>

## CUDA Graph

Slot reuse requires that an old value's producer and all its reader nodes complete via data dependencies prior to the new producer. Unconsumed outputs and cuBLASLt scratch spaces also incorporate write accesses. Independent branches retain their respective active storage; library steps continue to participate in scheduling and alternating-address updates as subgraphs.

The dedicated verifier independently validates topology, read availability, capacity, and the full access ordering of shared slots. Successful cache keys contain scheduling modes and complete dependency edges. Ancestor searches restrict scope based on the production location of reusable candidates, excluding candidates whose final accesses lack successors.

The table below shows seven-round medians for fixed public addresses in Graph mode, with latency in μs and VRAM in KiB of internal slots:

| Branches × Elements | Warm Prepare: Before → After | Host Submit: Before → After | Submit-to-Complete: Before → After | Internal VRAM: Before → After |
| --- | --- | --- | --- | --- |
| 2 × 32,768 | 6.507 → 9.565 | 12.112 → 15.624 | 62.818 → 59.485 | 384 → 512 |
| 4 × 1,024 | 12.336 → 18.453 | 12.017 → 27.037 | 66.643 → 76.910 | 20 → 32 |
| 4 × 32,768 | 12.059 → 17.969 | 12.354 → 27.108 | 79.989 → 79.342 | 640 → 1,024 |

The 4-branch large workload contains 11 steps with data dependencies forming 10 edges. In a single instrumented paired set, the device window dropped from 32.768 to 27.648 μs, with overlapping execution of independent branches detected. Normal host submit overhead increases offset this device gain. ASK owned storage including preallocated public ports grew from 896 to 1,280 KiB.

In the same example, first submit-to-complete was 236.992 → 407.350 μs, including workspace allocation and Graph instantiation; cold prepare in an independent fresh process was 60.388 → 65.050 μs. Both costs occur prior to warm replay.

Multi-root launch overhead aligns with NVIDIA's [multi-root CUDA Graph measurements](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/). A common empty-root prototype was tested separately: across 7 rounds and 448 numerically valid records, 4-branch large workload submit was 28.335 → 28.301 μs, and complete duration was 84.810 → 85.881 μs. Consequently, the production implementation retains direct data DAGs.

Scale audits covered deep chains followed by branching as well as prefix unconsumed nodes. For an 8,000-step deeply branched plan, prepare dropped from 138.594 ms in initial searches to 12.525 ms; the final regular chain measured 13.025 ms. This suite represents engineering diagnostics over three medians, not blended with the formal seven-round silent measurements; permanent tests assert structure and capacity only.

<a id="host-workspace"></a>

## CPU Materialization Scratch Space

ScalarEmitter records scratch loads actually emitted during compilation. Materialization stages, precomputed scalars, and public output batches jointly determine final read locations; uniform SSA and same-batch cache hits retain their original behaviors. Reads and writes in the same stage use different slots; subsequent stages select free slots with smaller capacity deltas, satisfying alignment across all members.

Following independent verification, the generator writes fixed capacities and offsets into LLVM instructions. Functions retain a single malloc/free; empty workspaces preserve pure computation paths. 650 compilation matrix test suites verify capacities, materialization counts, allocation counts, and precise conditional branches.

The table below shows results for an 8,192 × 16, eight-layer reduction-expansion chain:

| Metric | Baseline | Candidate |
| --- | --- | --- |
| Scratch Space, Final Output | 256 KiB | 64 KiB (75% reduction) |
| Scratch Space, Retaining First-Layer Output | 256 KiB | 96 KiB (62.5% reduction) |
| Single malloc/free | 1 / 1 | 1 / 1 |
| Native Warm Invocation, Final Output | 2,042.088 μs | 2,054.584 μs |
| Native Warm Invocation, Retaining First-Layer Output | 2,117.026 μs | 2,120.743 μs |

All 32 records sampled via independent malloc satisfy actual requested bytes and peaks matching ABI workspaces, with 0 surviving bytes upon function return. Materialization counts and computation loops remain unchanged; gains stem entirely from space reuse.

<a id="fuga-profile"></a>

## Overall Fuga Execution Path

Timing was evaluated separately across user interface graph construction, evaluate after cache hits, and synchronous completion. Units are in μs; cold paths include initial compilation:

| Path | Cold Path: Before → After | Warm Construction: Before → After | Warm Evaluate: Before → After | Construction to Sync Completion: Before → After |
| --- | --- | --- | --- | --- |
| CPU 8,192 × 16, 8 Layers | 245,687 → 249,647 | 490.786 → 490.241 | 2,238.752 → 2,227.109 | 2,735.123 → 2,746.924 |
| GPU 512 × 256, 4 Branches | 378,811 → 386,317 | 358.771 → 367.379 | 197.029 → 197.811 | 731.383 → 762.170 |
| GPU Same-Size Elementwise Fusion | 44,468 → 46,834 | 71.329 → 72.482 | 107.446 → 113.055 | 200.337 → 209.353 |

The 4-branch plan maintains 5 compute steps, with ASK owned VRAM shifting from 4,100 to 4,610 KiB; fusion control maintains 1 kernel and 1,536 KiB. Overall latency includes frontend and driver overhead, against which actual improvements should be judged. In this round, CPU primarily achieved memory savings, while GPU data dependency scheduling provided genuine concurrency opportunities alongside additional active storage and launch overhead.

Across seven rounds of whole-process RSS high-water records, native CPU was 21,288 → 31,328 KiB and Fuga CPU was 57,288 → 67,188 KiB. Subsequent retesting with identical original binaries, library digests, and complete wrappers yielded 21,284 → 21,276 KiB for native CPU; anonymous memory after 16 consecutive compilations was 3,432 → 3,412 KiB. The ~10 MiB delta in original records did not reproduce; original tables and per-mapping audit results have been archived. Workspace savings are jointly corroborated by independent malloc counts and ABI byte counts.

<a id="validation"></a>

## Build and Verification

| Verification Item | Result |
| --- | --- |
| ASK Core and physical GPU | 6/6 CTest suites passed, 0 skipped |
| Static plans | Original 2,439 master matrices + 4,752 Graph allocation/dependency variations; independent closures and pairwise access references; ASan/UBSan passed |
| CUDA Runtime | 640 topology, size, address, and inflight state combinations + 16 deep-chain branching structures |
| cuBLASLt | Original 360 suites + 48 dtype × size × branch × scratch cap × scheduling mode suites; 4 MiB actual scratch path passed |
| Host compilation | 650 dtype, shape, reduction, layout, lifetime preservation, fusion, and mixed-type combinations |
| Fuga full integration | 60/60 methods passed, 0 skipped; added 192 CPU reuse, 50 mixed-type, and 8 native branch combinations |
| Final Runtime integration audit | 9/9 GPU methods passed after reinstall, 0 skipped |
| Installed SDK standalone consumption | Compiler 2/2, Runtime 2/2; LLVM/MLIR package search disabled |
| Engineering scripts and package dependencies | 5/5; Fuga `pip check` passed |

Test combinations were crossed across axes via generators, with numerical references computed independently; code resides in ASK `tests/cpp/` and Fuga `tests/`. Verification incorporates public output lifetime preservation, alternating-address subgraphs, inflight workspaces, empty tensors, single-element SSA values, and elementwise fusion.

ASK was built and installed to `build/sdk`, and Fuga was rebuilt against this SDK. Shared library SHA-256 digests across build directory, SDK, and Fuga installation directory match identically:

| Library | SHA-256 |
| --- | --- |
| Compiler | `5b30943de8845711e9cdb84cf03a97ab5ea2c010e0bba6fbb73687825a8ea208` |
| Runtime | `bd09c738c2553675d2358383b3665bec72f746deec6753d200b9f3cf332ebcac` |

<a id="evidence"></a>

## Measurement Records and Cleanup

- [Full Summary](../assets/graph-workspace/summary.csv), [Detailed Methodology](../assets/graph-workspace/method.md), [Environment and Hashes](../assets/graph-workspace/profile-metadata.json). Summaries preserve per-version medians, min/max bounds, and round-by-round paired changes.
- [CPU Raw Data](../assets/graph-workspace/host-raw.csv), [Baseline malloc](../assets/graph-workspace/baseline-host-memory.csv), [Candidate malloc](../assets/graph-workspace/candidate-host-memory.csv).
- [GPU Raw Data](../assets/graph-workspace/gpu-raw.csv), [Cold Preparation](../assets/graph-workspace/cold-raw.csv), [Device Timestamps](../assets/graph-workspace/trace-raw.csv), [Fuga CPU](../assets/graph-workspace/fuga-cpu-raw.csv), [Fuga GPU](../assets/graph-workspace/fuga-gpu-raw.csv).
- [Scaling Audit](../assets/graph-workspace/scaling.csv), [Empty-Root Control](../assets/graph-workspace/empty-root-summary.csv), [RSS Raw Records](../assets/graph-workspace/process-rss.csv), [RSS Attribution Audit](../assets/graph-workspace/rss-attribution.csv), [Installed Library Consistency](../assets/graph-workspace/installed-libraries.json).

Temporary measurement runners, standalone prototypes, instrumented kernels, memory probes, generated artifacts, and baseline snapshots were purged upon archiving. The repository preserves production code, generative test matrices, and reporting datasets.