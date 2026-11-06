Back to Core Concepts: [Build and In-Process Integration](concepts.md#ask-frontend-build) · Related details: [Installation](install.md) · [External Compiler Integration](compilers.md) · [Resident Runtime](runtime.md)

# ASK User API

ASK's C++ interface is divided into Compiler and Runtime. For an integration example, see [FugaTensor](../../../fugatensor/README.md), whose native extension links against Compiler and Runtime; for Python usage, see [FugaTensor API](../../../fugatensor/docs/en/API/README.md).

<a id="api-entrypoints"></a>

## 1. Integration and Responsibilities

```cmake
find_package(ASK 0.1.0 EXACT CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

| Use Case | Headers and Components |
| --- | --- |
| Graph verification, optimization, partitioning, native code generation | `ask/Frontend/CompilerAPI.h`, links `ASK::Compiler` |
| Device buffers, kernels, plans, CUDA Graph, events | `ask/Runtime/CUDA.h`, links `ASK::Runtime` |
| Direct MLIR construction or transformation | `ask/Frontend/Compiler.h` etc., links `ASK::Core` and requires LLVM/MLIR SDK matching ASK's actual build version |

Compiler and Runtime use standard C++ types; compilation and GPU submission are completed in-process. The code generation library calls LLVM internally, and the CUDA runtime loads the driver on demand.

<a id="graph-api"></a>

## 2. Submitting ASK Graphs

Compiler accepts ASK MLIR text; callers can generate text directly or construct graphs via Core's MLIR API. FugaTensor lowers Fuga IR to ASK IR before submitting it for compilation.

The following is a static graph for `y = x + 1`:

```mlir
module attributes {ask.schema_version = 1 : i64} {
  func.func @increment(%x: tensor<4xi32>) -> tensor<4xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 4>} : (i32) -> tensor<4xi32>
    %y = "ask.add"(%x, %ones) {overflow = "wrap"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %y : tensor<4xi32>
  }
}
```

IR types include `bool/i32/i64/f32/f16/bf16`; `bool` in MLIR is written as `i1`. Host uses the first five types; `bf16` targets CUDA `sm_80+`. Type conversion uses `ask.cast`, scalar splatting uses `ask.splat`, and all shapes must be determined before execution. `tensor<f32>` is a single-element 0-D tensor passed via pointer; Host scalar `f32` inputs are passed by value. For rounding and provenance rules, see [ASK IR](ir.md).

Shape changes use `ask.reshape`, for example `"ask.reshape"(%x) {shape = array<i64: 2, 2>} : (tensor<4xi32>) -> tensor<2x2xi32>`. Inputs and outputs maintain the same element type, total element count, and contiguous order; any `-1` in the target shape must be resolved by the caller beforehand. External kernels will receive the reshaped `2×2` ports.

<a id="compiler-api"></a>

## 3. Compilation Interface

First, save the above graph as `std::string source` to generate a Host object file:

```cpp
#include <ask/Frontend/CompilerAPI.h>

ask::compiler::CodegenOptions options;
options.target = ask::compiler::CodegenTarget::HostObject;
options.outputPath = "increment.o";
auto result = ask::compiler::codegen(source, options);
```

`result.outputPath` is the object file path, and `result.abiJSON` describes entry points and parameters. The caller links and loads this file. Compilation failures throw `std::runtime_error` with diagnostics.

Select the compilation stage as needed:

| Objective | Invocation | Output Read |
| --- | --- | --- |
| Verify graph and infer shapes | `prepare(source)` | `ir`, `shapeSnapshotJSON` |
| Obtain CUDA execution plan | `partition(source)` | `executionPlanJSON`, `stepSources` indexed by step ID |
| Generate native code | `codegen(source, options)` | `ir` or `outputPath`, and target ABI |

All three results provide `externalRequestsJSON` for the caller to pass to third-party workers; `diagnosticsJSON` preserves provenance and optimization statistics. Each request uses an independent MLIR context, supporting concurrent calls.

<details>
<summary>CUDA Code Generation and Compilation Options</summary>

```cpp
ask::compiler::CodegenOptions options;
options.target = ask::compiler::CodegenTarget::CUDALLVM;
options.architecture = 86;
auto result = ask::compiler::codegen(source, options);
```

`result.ir` is CUDA LLVM text, and `result.cudaArtifactJSON` describes the launch ABI. Callers continue to generate PTX/cubin using `llc -fp-contract=off` and `ptxas`.

| Option | Usage |
| --- | --- |
| `target` | `HostLLVM` (default) returns Host LLVM; `HostObject` writes to an object file; `CUDALLVM` returns CUDA LLVM |
| `outputPath` | Required output path for `HostObject`; LLVM text targets use default empty string |
| `architecture` | CUDA architecture number, e.g. `86` |
| `blockSize` / `ptxVersion` | CUDA defaults are `128` and `80`, respectively |
| `optimize` | `prepare` defaults to `false`; `partition` and `codegen` default to `true` |
| `optimization` | `foldConstants`, `eliminateDeadCode`, and `fuseElementwise` all default to `true` |

`abiJSON` and `cudaArtifactJSON` are optional strings returning the corresponding fields depending on the target. For complete type definitions, see [CompilerAPI.h](../../include/ask/Frontend/CompilerAPI.h).

</details>

For build info, CLI, and JSON report interfaces, see [Frontend Integration](build.md#native-compiler).

<a id="execution-api"></a>

## 4. Native Sessions and Plan Execution

Runtime integration consists of three steps: loading kernels, preparing plans, and binding buffers for execution. The following assumes `binary` is a cubin/PTX string targeting `sm_86`, whose `increment` entry point reads four `i32` elements and writes four `i32` elements:

```cpp
#include <ask/Runtime/CUDA.h>

using namespace ask::runtime;
auto session = Session::create(0);
auto kernel = session->load(binary, "increment", {"i32"}, 86, {0, 0});

PlanStep step;
step.kernel = kernel;
step.arguments = {{0, {}}, {1, {}}};
step.grid = {1, 1, 1};
step.block = {128, 1, 1};
step.inputs = {0};
step.outputs = {1};
auto program = session->prepareProgram({16, 16}, {0}, {1}, {step});

std::int32_t values[4] = {1, 2, 3, 4};
auto x = session->allocate(sizeof(values));
x->copyFromHost(values, sizeof(values));
auto [outputs, done] = program->run({x});
done->wait();
std::int32_t result[4];
outputs.front()->copyToHost(result, sizeof(result));
```

`prepareProgram(bufferSizes, inputs, outputs, steps, graph=false)` accepts logical buffer byte sizes, public input/output indices, and a step list. It verifies the plan, generates internal slot mappings, and returns a `PreparedProgram` ready for repeated execution. `run(inputs)` automatically allocates public outputs; when output buffers already exist, use `submit(inputs, outputs)` to obtain a completion event.

Pass `graph=true` to enable CUDA Graph replay. The plan fixes kernels, shapes, and launch configurations, while allowing new public buffer addresses to be passed on each run. For workspace and event reuse mechanisms, see [Resident Runtime](runtime.md#cuda-graph).

<details>
<summary>Kernel Parameters and Buffer Contracts</summary>

`load(binary, entry, requiredTypes, arch, parameterWidths)` reuses modules based on artifact content and entry point. `requiredTypes` declares the underlying primitive types used, and `arch` is the compilation target. `parameterWidths` is filled in physical parameter order: pointers (including null) are `0`, while pass-by-value scalars are ABI byte widths `1/2/4/8`. Upon loading, parameter counts, widths, and offsets reported by the driver are checked.

`PlanStep` stores `kernel`, `arguments`, 3D `grid/block`, `sharedBytes`, and logical input/output indices. `PlanArgument.buffer` points to the plan's buffer table, with `-1` denoting null; `scalar` holds frozen scalar ABI bytes. Prepare a new program when modifying scalars or launch configurations.

Public inputs are permitted to share storage. Distinct logical outputs must be separated from other public ports; duplicate outputs or input aliases must bind to the same storage. Each submission verifies the actual buffers' device, byte count, writability, and aliasing. Capacities and lifetimes of internal slots are verified during the preparation stage by [ASK PASS](verification.md).

</details>

<a id="matmul-api"></a>

## Matrix Multiplication Plans

Writing `ask.matmul` in a graph causes the partitioner to generate a `library = "cublaslt"` step. When using Runtime directly, simply fill in the matrix description and the three buffer indices:

```cpp
PlanStep mm;
mm.matmul = MatmulDescription{"f16", 32, 48, 64}; // M, N, K
mm.inputs = {0, 1};
mm.outputs = {2};
auto program = session->prepareProgram(
    {32 * 64 * 2, 64 * 48 * 2, 32 * 48 * 2}, {0, 1}, {2}, {mm});
```

Preparation creates cuBLASLt descriptors and algorithms, while execution binds the A, B, and C addresses. Matrices are stored in contiguous row-major order; for batching, transpositions, and numerical conventions, see [Matrix Semantics](ir.md#matmul-semantics). For build dependencies, see [CUDA Tooling](install.md#cuda-and-adapters).

<details>
<summary>Batching and Workspace</summary>

`batchCount` defaults to 1, representing the product of identical leading batch dimensions. `transposeA` / `transposeB` default to false, altering the logical direction of the last two axes. Public buffer addresses must be aligned to element widths.

`workspaceLimit` defaults to 4 MiB, serving as the upper bound for algorithm selection; the runtime allocates only the byte size actually requested by the algorithm. Scratchpad allocations participate in the same slot verification, and sequential steps can reuse slots with disjoint lifetimes. Setting it to 0 selects zero-workspace algorithms. Create a new plan when altering these preparation parameters.

</details>

<a id="external-api"></a>

## 5. Third-Party Kernels and External Storage

Adapters provide cubin/PTX, entry points, ports, and launch configurations, integrated via `load` and `prepareProgram`. The compiler checks `ask.external` declarations, and `ask::compiler::partition` splits each external node into an independent step, while native regions continue fusing internally.

For worker requests, artifact formats, and scalar types, see [External Adapter Specification](compilers.md). Callers are responsible for workers, tooling environments, disk caches, and native program preparation; FugaTensor already provides corresponding implementations.

When borrowing external device pointers, callers must retain ownership of the underlying storage until the completion event finishes; `importBuffer` merely holds an ASK wrapper object. Use `session->waitForExternalStream(producerStream)` to wait for producers; external frameworks must likewise wait for ASK completion or establish equivalent dependencies before accessing buffers again.

<a id="artifact-api"></a>

## 6. Raw Bytes, Resource Lifetimes, and Caching

Sessions and resource objects are managed by `std::shared_ptr`. A session holds a non-blocking CUDA stream, and steps execute in plan order.

| Interface | Purpose |
| --- | --- |
| `DeviceBuffer::pointer()` / `size()` / `readonly()` | Read device address, byte count, and read-only attribute |
| `copyFromHost(pointer, bytes)` / `copyToHost(pointer, bytes)` | Copy full buffer; waits for session work before returning |
| `copyToHost()` | Copy buffer as `std::string` bytes |
| `Event::query()` / `wait()` | Query completion status or wait for completion |
| `Session::reap()` / `synchronize()` / `close()` | Reap completed events, synchronize stream, synchronously close session |
| `Session::stats()` | Inspect counts of modules, allocations, workspaces, events, and graphs, as well as current and peak native device memory |

Sessions retain pending completion events, and events retain programs, argument buffers, and workspaces until device work completes. Once completed, workspaces and event handles can be reused; old event objects remain completed, and old results held by callers continue using their original storage.

Callers persist compilation artifacts along with their targets, ports, parameter ABIs, launch configurations, and content digests, verifying them upon loading. For Execution Plan JSON v1, see [Graph Partitioning and Plans](partitioning.md#plan-v1).

<a id="errors-and-limits"></a>

## 7. Error Handling

Compiler interfaces throw `std::runtime_error`, while the device layer throws `ask::runtime::CUDAError`. On launch failure, preceding steps may have already been submitted; the runtime waits for those operations to finish before releasing references and propagating errors; device writes are not rolled back.

Plans use static contiguous buffers to execute kernels and cuBLASLt steps. Standard streams launch sequentially, while CUDA Graph schedules and replays based on port dependencies. FugaTensor handles the Python API and framework-level control flow.
