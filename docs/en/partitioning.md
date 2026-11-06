Back to Core Concepts: [Graph Partitioning and Execution Plans](concepts.md#execution-plan) · Related details: [Owned Code Generation](codegen.md) · [External Artifacts and Execution Interface](artifacts.md) · [Resident Runtime](runtime.md)

# Graph Partitioning and Multi-Kernel Execution Plans

<a id="partition-ir"></a>

## 1. Partition Representation

An ASK graph describes data dependencies, while an execution plan determines kernels and intermediate device buffers between them. The partitioner first validates types, infers shapes, and optimizes the graph in MLIR before partitioning kernels. Each partition uses `func.func` and `ask` operations, recording identity and kind via `ask.kernel_id` and `ask.kernel_kind` attributes; the module's `ask.execution_plan` string attribute stores the JSON plan.

`ask::compiler::partition(source, options)` returns `PartitionResult`: `ir` is the partitioned MLIR, `executionPlanJSON` stores the plan, `stepSources` stores independently compilable MLIR per step ID including semantics and implementation declarations referenced by that step, and `externalRequestsJSON` provides external compilation requests. Original `ask.id`, value identities, and tracing records in `diagnosticsJSON` correlate graphs before and after partitioning. For interface fields, see [User API](api.md#compiler-api).

CLI usage is `ask-opt --partition --report-json plan-report.json graph.mlir`. Full reports from CLI and `ask::partitionJSON` preserve the `execution_plan`, `step_sources`, and `external_requests` fields.

Steps are categorized into three kinds: `owned` contains ASK's native computation, which can be optimized and fused internally; `external` corresponds to a single external kernel; `library` corresponds to cuBLASLt matrix multiplication.

<a id="plan-v1"></a>

## 2. Execution Plan JSON v1

Format definitions are found in [JSON Schema](../../schemas/execution-plan-v1.schema.json). The following plan executes native computation first before invoking an external operator:

```json
{
  "schema_version": 1,
  "kind": "ask.cuda.execution_plan",
  "inputs": ["x"],
  "outputs": ["y"],
  "buffers": [
    {"id": "x", "shape": [128], "element_type": "f32"},
    {"id": "tmp", "shape": [128], "element_type": "f32"},
    {"id": "y", "shape": [128], "element_type": "f32"}
  ],
  "steps": [
    {"id": "k0", "kind": "owned", "function": "kernel_0",
     "inputs": ["x"], "outputs": ["tmp"], "depends_on": []},
    {"id": "k1", "kind": "external", "function": "kernel_1",
     "inputs": ["tmp"], "outputs": ["y"], "depends_on": ["k0"],
     "external_id": "graph.external0"}
  ],
  "scheduling": {"policy": "sequential", "stream": "session"}
}
```

- `buffers` contains unique IDs, concrete non-negative integer dimensions for shapes, and types selected from the six basic ASK types. Buffers are contiguous and row-major, with byte sizes derived from shape and type.
- Both public scalars and cross-step scalars are passed as single-element device buffers with `shape=[]`, preserving native CUDA's pointer ABI; returned scalar constants are also written out by owned steps.
- The order of top-level `inputs` and `outputs` defines the public tensor ABI. Outputs may reference inputs or repeatedly reference the same buffer; these aliased ports must bind to the same storage.
- `steps` are ordered by execution sequence. Each result has a single producer; inputs originate from graph inputs or earlier steps, and `depends_on` must list precisely the producers of those inputs. Internal constants remain in subgraphs.
- `function` points to a function in the partitioned MLIR; shapes and types of step inputs and outputs must match ports of the actual compiled kernel item by item. `external_id` of external steps points back to the original operator, facilitating correlation with source code and failure diagnosis.
- `library` steps set `library = "cublaslt"` and `matmul` descriptions (`element_type`, `m/n/k`, `batch_count`, `transpose_a/b`), with inputs A, B and output C.
- v1 sequentially submits static device computations on a single stream within the same `Session`. Validation failures raise errors.

The runtime delegates read port, public output, and internal slot mappings to [ASK PASS Rule Proof](verification.md#plan-proof) checks; indices, single producers, and kernel ABIs are validated by ASK.

<a id="partition-boundaries"></a>

## 3. Materialization Boundaries and Owned Regions

The partitioner divides native regions along external nodes. For example, `P → F → Q` generates `owned → external → owned`. Each native region may contain multiple elementwise operators. Matmul likewise forms independent library steps; for instance, `mul → matmul → add` generates `owned → library → owned`.

Broadcasting and transposition participate in native fusion via indexing. Reductions and same-sized consumers can execute in the same kernel; when reduction results are broadcast to larger shapes, they are materialized first to avoid repeating reductions for each output. Cross-step reduction scalars use device buffers with `shape=[]`.

`external` steps are handed to caller workers for compilation into cubin/PTX; `owned` steps are compiled through ASK's LLVM pipeline; `library` steps select cuBLASLt algorithms during Runtime preparation. Invalid MLIR, missing workers, export failures, corrupted artifacts, and codegen errors all fail the compilation.

External steps support multiple inputs, multiple outputs, and ports of differing shapes and types. `source_bindings` connects pointer arguments to logical ports and binds scalar arguments to fixed launch values; callers construct execution ABIs based on the requested `signature` and `port_shapes`. An external kernel may write multiple results.

<a id="program-runtime"></a>

## 4. Compilation Packages and Resident Scheduling

FugaTensor saves plans and step artifacts as a `CUDAProgram`. Relative paths are used within the package: `execution_plan.json` stores the logical plan, while `program.json` records subdirectory artifacts and digests corresponding to steps. Path scopes, digests, plans, and kernel ABIs are validated upon saving and loading, with native and external steps utilizing their respective disk caches; library matrix descriptors are preserved in the plan. Following validation, the in-memory plan descriptor is delivered to ASK Runtime.

Callers load owned/external kernels, populate `MatmulDescription` for library steps, construct byte size tables, public port indices, and `PlanStep` instances from logical buffers, and pass them to `Session::prepareProgram(..., graph)`. `PreparedProgram::submit(inputs, outputs)` executes all steps in a single call; C++ first verifies device ownership, byte sizes, writability, and aliasing of public ports, then updates public port addresses and submits, recording only a single completion event. `run(inputs)` additionally handles allocation for current outputs, called directly by FugaTensor's thin native bridge.

Each inflight invocation occupies an independent workspace, reusable only after its event completes. Physical slots are shared within a workspace according to verified step lifetimes, with public ports bound independently. Events retain programs and buffers, while sessions retain pending events, ensuring inflight resources remain valid even after callers release objects. On launch failure, the runtime waits for already-queued steps to finish before propagating exceptions; device writes are not rolled back. Zero-step aliased plans record completion events, covering prior stream dependencies to maintain asynchronous submission.

For the division of labor between plan proofs and VRAM reuse, see [Rule Proofs and Plan Verification](verification.md#reuse-boundary).

When `graph=true` is passed, [CUDA Graph Replay](runtime.md#cuda-graph) is used. The preparation phase derives dependencies from step read/write ports, freezes node predecessors, and verifies internal slot reuse according to dependency reachability; independent branches can execute concurrently. Each workspace caches a graph instance, updating public port addresses upon reuse; native and external steps share this identical mechanism.

FugaTensor submits forward or differentiated static graphs to ASK, where ASK partitions at external nodes and optimizes native regions. Backward rules for external operators are provided by the upper-level framework.

<a id="complex-external"></a>

## 5. Integrating Complex External Operators

External kernels can utilize multiple inputs, multiple outputs, differing shapes and storage types, internal FP32 accumulation, and dynamic shared memory. Workers are responsible for generating artifacts and reporting required resources; launch constraints are detailed in [External Execution Interface](artifacts.md#binary-artifact).

For example, FugaTensor's [Attention API](../../../fugatensor/docs/en/API/attention.md) uses FlashAttention's Triton kernel and defines pure cotangent rules via `ft.fun.vjp`. ASK partitions forward, summary statistics, dQ, dK, and dV into separate steps scheduled on the same stream; numerical validation of algorithms and gradients is maintained by FugaTensor.

For integrating new DSLs, see [Adapter Specification](compilers.md#adapter-contract); for request and artifact schemas, see [schemas/](../../schemas/).