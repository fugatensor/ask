Back to Core Concepts: [Graph Optimization and Native Code Generation](concepts.md#ask-owned-codegen) · Related details: [ASK IR and Numerical Semantics](ir.md) · [External Artifacts and Execution Interface](artifacts.md) · [Resident Runtime and Multi-Level Caching](runtime.md)

# Graph Optimization and Native Code Generation

ASK performs graph optimization on native computations, then generates CPU object files or CUDA LLVM IR.

<a id="graph-passes"></a>

## 1. High-Level Graph Optimization (`lib/Transforms/Optimize.cpp`)

When `PrepareOptions.optimize = true`, `ask::compiler::prepare` runs three transformations in sequence after graph validation. Statistics are stored in the `optimization` field of `diagnosticsJSON`; the corresponding CLI flag is `ask-opt --optimize`:

1. **Constant and Type Folding (`fold_constants`)**:
   - Folds `ask.add`, `ask.mul`, and `ask.cast` on scalar `ask.constant`, as well as `div`, `compare`, `select`, and elementwise constant operations where inputs are `ask.splat(c)` (reduced to `ask.splat(folded_c)`); uniform constants can also pass through reshape, broadcast, and transpose to participate in folding.
   - Integers follow 32/64-bit two's complement wraparound; floating-point operations are evaluated in the target format (`IEEEsingle` / `IEEEhalf` / `BFloat`) using `llvm::APFloat` under single-step RNE; floating-point to integer casts perform saturating truncation.
   - Floating-point expressions containing runtime tensors preserve operation order, such as `(x + c1) + c2` and `x * 0.0`. This preserves step-by-step rounding as well as `NaN` and `-0.0` semantics; the specific payload of NaN varies with the target implementation.
2. **Dead Code Elimination (`eliminate_dead_code`)**:
   - Walks the single basic block in reverse order to eliminate native pure operations whose results have no users (including shape transformations, reductions, elementwise operations, and matmul), pruning predecessors that become unused as a consequence. `ask.external` is conservatively retained, even if declared with `effects = "none"`; other side-effect declarations are currently rejected during validation.
   - `ask.mark` is extracted and erased during the preceding `prepare` phase.
3. **Multi-Output Elementwise Fusion Grouping (`fuse_elementwise`)**:
   - Following basic block order, partitions adjacent native tensor operations with identical result shapes into groups, tagging `ask.fusion_group` only when a group contains at least two operations; pure scalar operations may be interleaved, while shape changes, matmul, or external operations terminate groups. Reshape and downstream operations with the same target shape can share the same group. Operation attributes and `fusion_groups` in optimization statistics record results; intermediate values that serve both as function returns and downstream operands (e.g. `return t, t * 2`) can be written out within the shared loop.

Consecutive transpositions of the trailing two axes preceding a matmul are absorbed into `transpose_a` / `transpose_b` when fusion is enabled; transpositions consumed solely by that node are removed by DCE.

<a id="host-codegen"></a>

## 2. Host CPU Shared-Loop Code Generation (`lib/Lowering/Host.cpp`)

`ask::compiler::codegen(source, options)` lowers native graphs to host CPU machine code, optimizing the graph by default. `options.target = CodegenTarget::HostLLVM` returns LLVM text; `HostObject` writes `.o` to `options.outputPath`. Both modes return port contracts via the result's `abiJSON`, and callers are responsible for linking and loading. The [C++ Compilation Interface](api.md#compiler-api) and `ask-opt --emit-llvm` / `--emit-object` share `ask::codegenSource`:

- **Shared Loop Generation**: Merges elementwise chains with identical shapes into a single `i64` linear-index loop according to `fusion_groups`, keeping elementwise intermediate values in LLVM SSA registers; 0-D tensors have a single element, and empty outputs immediately terminate the corresponding computation.
- **Indexing and Reduction**: Reshape reuses linear indices, while broadcast and transpose map to input indices; reductions generate inner loops. The shared `ScalarEmitter` generates these expressions for Host/CUDA; cross-kernel boundaries or public outputs are written to buffers.
- **C ABI**: Generated functions share the same name as `func.func` in MLIR. Parameters are ordered as all inputs followed by all outputs: scalar inputs are passed by value, while tensor inputs and all outputs use pointers. Output pointers carry `noalias` and must not overlap with any input or other output buffers; inputs may share storage. For each function in `abiJSON.functions`, `inputs` and `outputs` specify `kind`, `shape`, `dtype`, `element_bytes`, and physical parameter index. Upstream layers (such as [FugaTensor's `HostBackend`](../../../fugatensor/docs/en/architecture/backend.md#host-backend)) allocate buffers accordingly, retrieve library paths from `host_link_libraries` in capabilities according to `abiJSON.link_libraries`, pass them to the system linker, configure runtime load paths, and invoke the exported C ABI functions.
- **Host Types**: CPU code generation supports `bool/i32/i64/f32/f16`, with static type validation prior to graph optimization. Available types are reported by `host_supported_element_types` in `--capabilities`.
- **f32 Matrix Multiplication**: Invokes single-threaded OpenBLAS SGEMM per batch. M/N/K, transpositions, leading dimensions, alpha=1, and beta=0 are fixed at compile time, and matrix dimensions are validated against LP64 ranges. Contiguous inputs pass pointers directly, trailing two-axis transpositions convert to library parameters, and input-side elementwise operations are fused and materialized; results are written to public outputs or existing scratch slots, with subsequent elementwise chains continuing to fuse. K=0 generates zero-fill, and empty outputs bypass computation. Library algorithms employ f32 arithmetic and govern accumulation order.
- **f16 Matrix Multiplication**: Generates f32 vector accumulation in 4 rows × 16 columns per batch; four rows share reads of B, and each output performs independent multiplication and addition along the K sequence, finally rounding to f16. Trailing rows and columns are determined by static shapes.
- **Machine Code Optimization**: `HostObject` runs a target-dependent LLVM O3 pipeline before emitting native object code, executing loop and vector optimizations; floating-point multiply-accumulate contraction remains disabled. `HostLLVM` returns lowering-stage LLVM IR, and statistics with `statistics_stage = "host_lowering"` in ABI also describe this phase. [Generic LLVM IR Cleanup](#llir-cleanup) uses separate cleanup rules.

`lowering.loop_count` records the number of non-empty output batches, and `reduction_loop_count` records the count of K loops and reduction loops emitted during the lowering phase. `blas_matmul_count` records emitted SGEMM call sites. Actual machine loops are determined by subsequent target optimizations; performance profiling uses final machine code and independent timing.

Scalar reductions are evaluated prior to output loops. When partial reduction results are expanded or reused by matrix multiplications, they are first materialized into function-level scratch space. The code generator tracks actual scratch reads emitted across materialization phases and output batches, reusing capacity and suitably aligned slots based on last-read positions; reads and writes within the same phase occupy distinct slots. Layouts are validated at compile time and emitted as fixed offsets, allocated via a single `malloc` at execution and freed upon completion. Single-element results are typically kept as SSA values; SGEMM inputs and intermediate results requiring memory addresses share the same materialized scratch space. The function's `lowering.temporary_workspace_bytes` reports the reused byte count, while `materialized_tensor_count` reports the count of materialized values; when capacity is zero, generated code retains a pure compute path. The established void C ABI calls `abort` if scratch space allocation fails.

<a id="owned-cuda"></a>

## 3. Native CUDA Lowering (`lib/Lowering/CUDA.cpp`)

Native GPU graphs are lowered via `ask::compiler::codegen` by specifying `CodegenTarget::CUDALLVM` and the target `architecture`, e.g. `86`. The result's `ir` returns LLVM text, and `cudaArtifactJSON` returns the CUDA artifact contract. Callers generate PTX and cubin using `llc` / `ptxas`, recording the file digest of the Compiler shared library or selected CLI in the compilation cache. exp/log emit `__nv_expf` / `__nv_logf` references; callers first link against `nvvm/libdevice/libdevice.10.bc` from the CUDA Toolkit using matching `llvm-link`. FugaTensor incorporates linker tools and libdevice digests into its artifact cache.

- **NVPTX Native LLVM IR**: Generates `target triple = "nvptx64-nvidia-cuda"` modules with `define ptx_kernel void @<entry>(...)`, using `addrspace(1)` global memory pointers.
- **64-bit Grid-Stride Loops**: Reads `@llvm.nvvm.read.ptx.sreg.{ctaid,tid}.x` and extends them to `i64`, computing starting indices based on a fixed block size. The grid's x dimension is capped at 65535; the stride for thread scheduling is `grid.x * block_size`, while warp scheduling divides this stride by 32. Mixed output groups share a grid satisfying all group requirements.
- **Warp-Cooperative Reductions**: On `sm_70+` with block size a multiple of 32, independent `reduce_max` and `reduce_sum` with reduction extent of at least 32 can be cooperatively computed by one warp per output. Selection also requires the reduction result to match that output group's element count; lanes read with stride 32 (large sizes enable 4-way ILP unrolling with independent accumulation streams), perform 5 shuffle-XOR merges, and lane 0 writes out the result. Trailing reads generate predicates only for extents less than 32, after which all lanes converge. Floating-point summations use the `reassoc` fast-math flag to exploit hardware parallel pipelines, floating-point maximums use `llvm.maximum` (propagating NaNs and distinguishing signed zeros), and i64 preserves full bit patterns across two 32-bit shuffles. Surrounding elementwise expressions continue fusing in the same kernel.
- **Multi-Port and Boundary Handling**: Supports multi-tensor/scalar inputs, multi-output ports with heterogeneous shapes and dtypes, non-aligned tail-element boundary masks (`idx < N`), and empty tensors.
- **Floating-Point and Architecture Checks**:
  - Floating-point instructions omit `contract` / `fast` flags, pairing with `llc -fp-contract=off` to guarantee that `a * b + c` executes as two separate RNE operations (multiplication then addition).
  - When inputs, outputs, or intermediate values in the graph contain `bf16`, compilation requires `arch >= 80` and records `"bf16"` in the artifact's `required_element_types` for the [CUDA Runtime](runtime.md#execution-session) to verify physical GPU capabilities upon loading.

Chained elementwise expressions execute within a single kernel, retaining intermediate values in registers. When reduction results are expanded again, the partitioner writes out the reduction value first, and subsequent kernels reuse it. `ask.matmul` is partitioned and dispatched to cuBLASLt; see [Library Steps](partitioning.md#partition-boundaries). In reports, `reduction_loop_count` records the count of generated reduction loops; `warp_reduction_groups` records the number of warp-cooperative output groups, with each group's `threads_per_output` being 1 or 32.

<a id="llir-cleanup"></a>

## 4. Native LLVM IR Cleanup (`lib/LLIR/Optimize.cpp`)

`ask-opt --optimize-llir` cleans up input LLVM IR independently. Optimization after native Host/CUDA lowering is handled by the respective LLVM backends.

- Runs InstCombine, SimplifyCFG, and ADCE in sequence, skipping functions marked with `optnone`; does not enable generic inlining, loop transformations, vectorization, or module-level unreferenced function elimination.
- Strips stale ASK provenance metadata and instruction-level ASK attachments, preserving module interface metadata such as `ask.abi` and `ask.shape_snapshot`, while listing unverified ASK-named metadata in reports.
- Validates the LLVM module before and after optimization, verifying that target info and external function ABIs remain unchanged. Inline PTX participates in optimization according to its LLVM `sideeffect`, operands, and constraint semantics; assembly instructions without side effects whose results are unused may be eliminated.
