Back to Core Concepts: [Resident Runtime and Caching](concepts.md#ask-runtime-cache) · Related details: [External Artifacts and Execution Interface](artifacts.md) · [Graph Optimization and Native Code Generation](codegen.md) · [FugaTensor Multi-Backend Execution](../../../fugatensor/docs/en/architecture/backend.md)

# Resident Runtime and Multi-Level Caching

`ASK::Runtime` manages execution plans, device buffers, streams, completion events, and Graph replay. FugaTensor is responsible for Python APIs, tensor descriptors, lifetimes of external Python objects, and on-disk compilation caching.

<a id="execution-session"></a>

## 1. Dynamic Driver Loading and Resident Session

The C++ interface is located in [CUDA.h](../../include/ask/Runtime/CUDA.h), with implementation in `lib/Runtime/CUDA.cpp`. The driver library `libcuda.so.1` is loaded dynamically upon creating a `Session`.

`Session::create(device)` retains the target device's original primary context and creates a non-blocking stream. Each driver operation sets the current thread's context; the session protects driver state with a mutex, supporting submissions from multiple host threads.

`Session::load` accepts cubin/PTX bytes, entry point, architecture, and parameter width, validating target device and kernel resource limits. The same session reuses modules based on artifact content and entry point. PTX is compiled by the driver upon first loading, whereas cubin loads directly.

For C++ invocation examples, see [Native Session and Plan Execution](api.md#execution-api).

<a id="device-tensor-and-cai"></a>

## 2. Native Buffers and External Storage

`Session::allocate(nbytes)` returns an owned `DeviceBuffer`, and `importBuffer(pointer, nbytes, readonly)` borrows existing device memory. Buffers record device, byte count, and read-only status; empty buffers also possess valid handles with zero length.

`DeviceBuffer::copyFromHost(pointer, nbytes)` and `copyToHost(pointer, nbytes)` perform batch copies of contiguous memory; `copyToHost()` returning a byte string is also provided. Copies proceed on raw bytes with lengths matching strictly, waiting for session stream completion before returning.

When borrowing external memory, callers must retain the actual storage owner until GPU usage finishes. FugaTensor's binding layer parses CUDA Array Interface, verifies shape/dtype/contiguity, preserves Python owners, and invokes `waitForExternalStream` to await producer streams. For exchange interfaces, see [FugaTensor Tensor and Data Exchange](../../../fugatensor/docs/en/API/tensor.md).

Before external producers re-access the same storage, they must await ASK's completion event or establish equivalent stream dependencies.

<a id="async-events"></a>

## 3. Whole-Plan Submission, Resource Lifetime Management, and Reuse

`Session::prepareProgram(bufferSizes, inputs, outputs, steps, graph)` binds buffer indices, physical parameters, scalar bytes, and launch configurations, returning a `PreparedProgram`. Callers verify tensor shape/dtype and compilation packages prior to preparation; the native layer validates step references and kernel ABIs, while [ASK PASS](verification.md#plan-proof) directly validates read ports, public outputs, and internal slot mappings; identical plans reuse the in-process cache of successful verifications. cuBLASLt descriptors, algorithms, and actual scratch capacity are fixed during preparation, with scratch space joining the same slot verification; execution functions per step are chosen at this time, invoked directly by the submission loop.

Each execution invokes `PreparedProgram::submit(inputs, outputs)`, or uses `run(inputs)` where the runtime allocates outputs. The preparation phase generates allocation requirements for outputs and port aliasing constraint tables; after verifying inputs, `run` allocates storage only for independent outputs, while duplicate outputs and input aliases reuse original buffers. `submit` validates caller-supplied ports; both entry points share the submission implementation of the bound plan. Public port device ownership, byte sizes, writability, and aliasing are verified at submission boundaries, followed by address updates and execution of all steps. The entire plan records a single completion event covering all buffers, kernels, and workspaces.

- Each inflight invocation occupies an independent workspace; internal buffers with disjoint lifetimes within the same workspace share physical slots, and workspaces are reused upon completion.
- Workspaces allocate parameter storage upon initial creation, writing frozen scalars, nulls, and internal addresses, and constructing CUDA launch descriptors. Subsequent submissions reuse these storages and descriptors.
- Once owned allocations are released, they enter the session cache after related tasks complete; results still held by users continue to occupy original storage.
- Native handles of completion events are reusable, while old `Event` objects remain completed.
- `Event::query` returns status immediately; `wait`, `Session::synchronize`, and host copies await completion. Closing a session completes inflight work before tearing down resources.
- Single-kernel and multi-kernel executions use the identical plan mechanism. On launch failure, the runtime waits for queued steps to finish before propagating errors; device writes are not rolled back.

Querying, waiting, and session operations reclaim completed resources. Events retain C++ objects, while FugaTensor bindings separately retain external Python owners until completion.

<a id="cuda-graph"></a>

## 4. CUDA Graph Replay

`prepareProgram(..., graph=true)` uses CUDA Graph, while `false` uses C++ sequential launches. Both modes share port, parameter, and resource checks.

When preparing Graphs, ASK derives producer dependencies from each step's `inputs` / `outputs`. Independent branches are scheduled by CUDA according to dependencies; slots are reused only when an earlier producer and all its reader steps complete before the new producer. The preparation phase statically verifies topology, data availability, and slot access order, freezing node predecessors thereafter. Normal stream mode reuses expired lifetimes in step order.

Each workspace creates and instantiates a graph upon initial execution; upon reuse, it updates node input/output addresses and invokes `cuGraphLaunch`. Inflight invocations each use independent instances, allowing node parameters to be modified only after completion. Topology, kernels, shapes, launch configurations, and frozen scalars remain fixed, while public port addresses may vary. cuBLASLt calls are captured as subgraphs; when addresses change, the subgraph is recaptured and the executable graph updated, whereas identical addresses trigger direct replay.

Zero-step aliased plans record only stream completion events. Exceptions are thrown if the driver lacks required Graph APIs or if instantiation, updates, or launches fail.

`Session::stats()` provides counters including `program_submissions`, `workspace_reuses`, `allocation_reuses`, `event_creations/event_reuses`, and `graph_instantiations/graph_launches`. `launches` denotes submitted plan steps (a library step may encompass multiple CUDA kernels internally), while Graph submissions are tallied separately. FugaTensor's CUDA JIT uses graph mode via its native bridge. `matmul_preparations` tracks library algorithm preparations; `resident_device_bytes` / `peak_resident_device_bytes` track current and peak VRAM actually allocated by ASK including allocation pools, while external borrowings and driver/library internal storage are accounted for by their respective owners.

`graph_nodes` / `graph_edges` accumulate top-level nodes and dependency edges of instantiated Graphs, with cuBLASLt subgraphs counted as a single top-level node. `prepared_internal_slots` / `prepared_internal_bytes` accumulate slot counts and capacities of a single internal workspace across successfully prepared plans, including library scratch; actual resident volume varies with inflight workspaces and allocation pools.

For Graph replay, TF32, and submission mechanisms of matrix multiplication, see [FugaTensor Arithmetic Operations](../../../fugatensor/docs/en/API/ops.md#matmul).

<a id="two-tier-cache"></a>

## 5. Caller Compilation Cache and Native Runtime Cache

FugaTensor's `_ask` integration layer maintains the following on-disk caches, delivering bytes and plans to ASK after verifying artifacts:

| Cache | Skipped on Hit | Primary Inputs to Key |
| --- | --- | --- |
| Owned LLVM IR | Graph optimization and CUDA LLVM lowering | Graph text, shapes, compilation library identity, target/options, and orchestration implementation digest |
| Owned cubin | `llc` and `ptxas` | LLVM IR, artifact descriptor, tooling and environment digest |
| External Artifact | Worker compilation | Normalized request, probe result, worker/interpreter, dependencies and environment digest |

External caches store cubin or raw PTX, probing producers before lookup and re-checking dependencies after compilation. Caching uses locks, temporary directories, and atomic publication, checking manifests, lengths, and SHA-256 upon reading. For invalidation rules, see [FugaTensor Cache Division](../../../fugatensor/docs/en/architecture/backend.md#cache-hierarchy) and [Worker Cache Identity](compilers.md#cache-identity).

ASK caches successful verifications during plan preparation, while sessions cache resident modules, workspaces, device memory, and events. For keys and capacities of verification caches, see [Successful Verification Cache](verification.md#proof-cache).