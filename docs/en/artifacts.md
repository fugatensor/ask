Back to Core Concepts: [External Artifacts and Execution Interface](concepts.md#external-execution) · Related details: [External Operator Declaration](ir.md#external-declaration) · [Worker Protocol](compilers.md#worker-protocol) · [Graph Partitioning and Scheduling](partitioning.md)

# External Artifacts and Execution Interface

Each `ask.external` corresponds to an independent kernel. Callers generate cubin or PTX using third-party compilers, verify artifacts, and submit the binary and launch configurations to the ASK C++ Runtime.

<a id="external-execution"></a>

## 1. Graph Boundaries and Responsibilities

`ask.semantics` defines input/output types and shape rules, while `ask.implementation` stores source code, compilation configuration, and port bindings. The compiler inspects the graph, exports requests, partitions native regions on both sides of external nodes, and then performs constant folding, DCE, and elementwise fusion on native regions.

For example, `P: t = x + 1 → F: external kernel → Q: y = u + 3` forms three steps. Intermediate results `t` and `u` remain in device buffers, accessed via pointers in subsequent steps. CUDA Graph replay is used to reduce submission overhead for these steps.

ASK is responsible for request and artifact protocols, graph partitioning, ABI checks, and device scheduling. Callers are responsible for launching workers, selecting compiler versions, parsing artifacts, and disk caching. FugaTensor's corresponding code resides in `src/fugatensor/_ask/`, and its Triton compilation code is in [`_triton_worker.py`](../../../fugatensor/src/fugatensor/_triton_worker.py).

<a id="binary-artifact"></a>

## 2. Executable Artifacts and Ports

The worker returns base64-encoded cubin or PTX text along with the entry point, target architecture, producer, launch configuration, parameter order, and required element types; see [Artifact Contract](compilers.md#artifact-contract) for format details.

Callers construct invocation v2 according to `signature`, `source_bindings`, and `port_shapes` in the request, verify file digests and tensor ports, and then hand over artifact bytes to `Session::load`. PTX is JIT-compiled by the CUDA driver upon loading.

Ports use static, contiguous, row-major tensors, supporting all six ASK element types and multiple outputs. Kernels accept 1–64 physical parameters; pointers bind to `input:N`, `output:N`, or `null`, while scalars are encoded into bytes according to their ABI type. The runtime verifies parameter count, offsets, and widths against driver reports, and checks thread counts, shared memory, device residency, and storage aliasing prior to launch.

`PlanArgument` stores logical buffer indices or frozen scalar bytes, while `PlanStep` stores kernel, grid, block, shared memory, and input/output sets. Callers verify tensor shapes and dtypes before converting them into byte sizes used by native interfaces. When preparing plans, scalar bytes and parameter locations are baked into templates; repeated executions update device addresses of public ports.

The launch interface uses standard pointers and standard CUDA kernels, with empty scratchpad memory. External artifacts adhere to the producer's numerical policies, while native computations follow ASK's rounding and conversion rules.

<a id="worker-interface"></a>

## 3. Caller Worker Configuration

`ask.implementation.compiler` serves as the key for callers to select a worker. Workers can be standalone executables or scripts executed by a designated Python interpreter.

FugaTensor provides a Triton worker; for its configuration, see [Upstream Execution Guide](../../../fugatensor/docs/en/architecture/backend.md#cuda-and-triton). Other frameworks can implement the same [Request/Response Protocol](compilers.md).

External steps use the worker's compilation environment; owned steps in hybrid graphs are compiled by the caller running `llc` / `ptxas`. Missing workers, invalid requests, compilation failures, corrupted artifacts, and ABI mismatches must all trigger errors.
