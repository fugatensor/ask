Back to Core Concepts: [Graph Partitioning and Execution Plans](concepts.md#execution-plan) · Related details: [Execution Plan JSON v1](partitioning.md#plan-v1) · [Resident Runtime](runtime.md#async-events)

# Static Verification of ASK PASS Execution Plans

<a id="plan-proof"></a>

## 1. Plan Verification during Preparation

`ASK::Pass` is ASK's native C++ static verification library, with interface and implementation located in [`lib/Pass/`](../../lib/Pass/). The verifier directly inspects execution plans, buffer sizes, and slot mappings, checking structural constraints and integer bounds.

`Session::prepareProgram` verifies port bindings and kernel ABIs, generates slot mappings for internal buffers, and calls `checkAllocationPlan`. The verifier checks indices, single producers, public ports, mapping completeness, and the following numerical relationships:

| Objective | Plan Data | Verification Condition |
| --- | --- | --- |
| Each read available | Buffer production boundary, read boundary | `production boundary <= read boundary` |
| Public outputs available | Output indices and producers of respective buffers | Valid index; buffer has a unique producer or belongs to public inputs |
| Slot capacity sufficient | Buffer byte count, slot capacity | `byte count <= capacity` |
| Disjoint lifetimes within same slot | Slot assignment, production step, last use step | Adjacent buffers satisfy `former last use < latter production` |

Graph mode also checks that predecessor indices are strictly less than the current step, that predecessors are unique, and confirms via dependency reachability that each read occurs after producer completion. Adjacent buffers sharing a slot require that the earlier producer and all its read steps can reach the later producer along dependency edges; unread outputs and library scratch spaces still incorporate producer write accesses.

The production boundary of public inputs is `0`; outputs of step `i` are produced at boundary `i + 1`, and reads occur prior to the corresponding step. Public outputs are available at the final boundary, guaranteed by valid indexing and producer completeness checks. Buffer sizes, slot capacities, and boundaries use `size_t`; the verifier directly compares raw numerical values from the plan.

Verification executes outside the Session lock, creating an immutable `PreparedProgram` upon passing. Internal results are `Pass` or `Fail`; failure of numerical conditions lists diagnostics across reads, capacities, and adjacent lifetimes. The preparation interface throws `CUDAError` prefixed with `ASK execution plan validation failed`.

<a id="reuse-boundary"></a>

## 2. Slot Allocation and Execution

ASK traverses internal buffers in production step order, setting slot capacity to the maximum byte size among all its buffers. Normal stream mode reuses slots whose lifetimes have ended; Graph mode reuses only slots whose full set of accesses have completed among ancestors of the current node. Independent branches maintain active storage separately. Public inputs and outputs use independent storage, and inputs and outputs within the same step occupy distinct slots.

`checkAllocationPlan` independently checks that each internal buffer is allocated exactly once, slot indices are valid, and each slot has users. It sorts buffers within each slot by production step: stream mode verifies that adjacent lifetimes are strictly disjoint, while Graph mode verifies dependency paths from all accesses of the earlier buffer to the later producer. Ordered adjacent occupants guarantee ordered access throughout the entire slot.

Each inflight submission holds an independent workspace, with each slot corresponding to a `DeviceBuffer`. Completion events retain workspaces, which are reused upon completion. The preparation phase categorizes parameters into frozen values, internal addresses, and public port addresses; workspaces write frozen values and internal addresses upon creation, updating public port addresses on each submission. The execution loop directly utilizes prepared parameters and launch descriptors.

Kernel producers must adhere to declared port contracts: reading `inputs`, writing `outputs`, restricting accesses within declared buffers, with port address usage terminating upon step completion. The runtime submits sequentially via a single stream, or constructs CUDA Graph nodes following frozen dependencies.

Slot allocation reduces internal VRAM requirements, and the static verifier certifies candidate mappings as legal. Graph-level constant folding, DCE, fusion grouping, and step partitioning are performed by ASK's compilation optimization modules, as detailed in [Graph Optimization](codegen.md#graph-passes).

<a id="proof-cache"></a>

## 3. Successful Verification Cache

New plans check production and read boundaries via linear traversal, collect internal buffers by slot, and sort them to check adjacent lifetimes. Graph verification searches backward from nodes being validated to check ordering requirements of data and shared slots, terminating once all requirements are met. Verification uses temporary vectors for the current call, freed upon completion.

Successful results are stored in a bounded process cache. Keys contain full plan structures, sequence lengths, buffer sizes, slot capacities, all allocations, scheduling modes, and dependency edges, compared by full value. Identical plans reuse successful results when prepared again; the cache holds up to 64 keys with a cumulative key length cap of 65,536 `size_t` elements, evicted in write order.

Both cache lookup and static verification take place during the plan preparation phase. `PreparedProgram` reuses determined mappings and parameters during execution; public buffers passed on each run are checked at submission entry for device ownership, size, writability, and aliasing.

<a id="build-dependency"></a>

## 4. Build Dependencies

`ASK::Pass` is built from `lib/Pass/Pass.cpp` in this repository, and `ASK::Runtime` privately links this static library. Packaged SDK users consume verified native execution plans via `ASK::Runtime`. Builds use standard presets:

```bash
cmake --preset core
cmake --build --preset core
```