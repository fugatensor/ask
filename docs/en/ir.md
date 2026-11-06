Back to Core Concepts: [Two-Level IR Model](concepts.md#ask-ir-model) · Related details: [Frontend Integration and Build Configuration](build.md) · [Graph Optimization and Native Code Generation](codegen.md) · [Fuga IR Lowering to ASK](../../../fugatensor/docs/en/ir/lowering.md)

# ASK IR and Numerical Semantics

ASK IR uses the MLIR `ask` dialect to describe tensor types, shapes, operations, tracing markers, and external operators. Upper-level frameworks submit computation graphs to ASK through it, with definitions located in `include/ask/IR/` and `lib/IR/`.

<a id="ir-structure"></a>

## 1. Module Structure and Supported Operation Set

`prepare` validates and populates the following module attributes:

- `ask.schema_version = 1 : i64`: Schema version of ASK IR, with a value of `1`.
- `ask.semantic_revision` / `ask.ir_revision`: Non-negative `i64` mutation counters, starting at `0` by default and must be less than `INT64_MAX`. Type refinement increments both counters; mutating operations such as identity completion or marker erasure increment `ir_revision`.

Only two classes of symbol declarations and single-block `func.func` computation graph functions are permitted at the module top level:

| Operation | Level | Role |
| --- | --- | --- |
| `ask.semantics` | Module top-level | Declares input/output type signatures and shape inference rules (`same_as` or `fixed`) for external operators; uses `effects = "none"`. |
| `ask.implementation` | Module top-level | Binds external compiler ID (e.g. `"triton"`), DSL source code snapshot, entry function name, pointer/scalar signatures, `source_bindings`, `constants`, `options`, `target`, and `grid`. |
| `ask.constant` | Inside `func.func` | Generates a scalar constant of the specified type (`bool` / `i32` / `i64` / `f32` / `f16` / `bf16`). |
| `ask.splat` | Inside `func.func` | Replicates and fills a scalar into a tensor of the same element type according to static `shape`. |
| `ask.add` / `ask.mul` | Inside `func.func` | Elementwise addition/multiplication of identical shapes and numerical element types; integers wrap around using two's complement (`overflow = "wrap"`). |
| `ask.cast` | Inside `func.func` | Converts element type while preserving shape; supports all 6×6 directions. |
| `ask.reshape` | Inside `func.func` | Changes tensor shape while preserving element type, total count, and contiguous row-major order; supports 0-D and empty tensors. |
| `ask.broadcast_in_dim` / `ask.transpose` | Inside `func.func` | Explicitly expands dimensions or permutes axes while preserving element type. |
| `ask.reduce_sum` / `ask.reduce_max` | Inside `func.func` | Reduces along specified axes, removing them. |
| `ask.div` / `ask.exp` / `ask.log` / `ask.sqrt` | Inside `func.func` | Elementwise floating-point computation on identical types. |
| `ask.compare` / `ask.select` | Inside `func.func` | Comparison produces a bool tensor of identical shape; select picks values of identical shape and type based on a bool condition. |
| `ask.matmul` | Inside `func.func` | Multiplies contiguous matrices or batched matrices; CUDA uses cuBLASLt. |
| `ask.mark` | Inside `func.func` | Tracing passthrough marker recording logical value identity (`value_id`) or port binding (`port_id`), erased after extraction in `prepare`. |
| `ask.external` | Inside `func.func` | References a pair of `@semantics` and `@implementation` symbols, representing an external operator node handled by a third-party compiler. |

Each graph function uses a single basic block; upper-level frameworks unroll function calls and control flow into static graphs. Upon entering `prepare`, input dimensions must be concrete values, and the total tensor byte size must fall within the range of a signed 64-bit integer.

`ask.reshape` specifies the target concrete shape via the `shape` attribute; the result type may temporarily contain `?` and will be refined by `prepare`. For example, rearranging four elements into a two-dimensional layout:

```mlir
%matrix = "ask.reshape"(%flat) {shape = array<i64: 2, 2>} : (tensor<4xf32>) -> tensor<2x2xf32>
```

`shape = array<i64>` represents a 0-D tensor; dimension inferences such as `-1` are resolved by callers, as seen in [FugaTensor reshape](../../../fugatensor/docs/en/API/tensor.md#reshape). Reshape preserves element types and linear order; external kernels receive the transformed shape. Native elementwise loops reuse corresponding input values, while public outputs use independent buffers.

<a id="tensor-primitives"></a>

## Explicit Shape Transformations and Reductions

`broadcast_in_dim`'s `shape` specifies the target shape, and `dimensions` maps each input axis to a target axis in strictly ascending order. The input dimension of a mapped axis must equal the target dimension or be 1; scalars use an empty mapping. For example:

```mlir
%wide = "ask.broadcast_in_dim"(%row) {shape = array<i64: 2, 3>, dimensions = array<i64: 0, 1>} : (tensor<1x3xf32>) -> tensor<2x3xf32>
%swapped = "ask.transpose"(%wide) {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
%sum = "ask.reduce_sum"(%swapped) {axes = array<i64: 1>} : (tensor<3x2xf32>) -> tensor<3xf32>
```

`transpose`'s `permutation` lists the input axes corresponding to each output axis. Reduction `axes` are strictly ascending and unique, and the specified axes are removed in the result; when all axes are removed, the result is a scalar, which can be expanded using `splat` or `broadcast_in_dim`. An empty axis list for a positive-rank tensor preserves original values.

`reduce_sum` supports warp shuffles and instruction-parallel tree-reduction accumulations without strictly enforcing sequential IEEE-754 accumulation order; integers wrap around, and empty domains evaluate to zero. When f32 accumulation is needed, perform `cast` first. `reduce_max` accepts integers and floats, propagates NaN, distinguishes positive and negative zero, and requires non-empty reduction domains.

<a id="elementwise-primitives"></a>

## Floating-Point Arithmetic and Conditional Selection

`div`, `exp`, `log`, and `sqrt` accept f32/f16/bf16; low-precision transcendental functions are first widened to f32, then rounded back to their original type using RNE. CUDA exp/log uses NVIDIA libdevice. Comparison `predicate` values are `eq`, `ne`, `lt`, `le`, `gt`, or `ge`; floating-point NaN satisfies only `ne`. All three operands of `select(condition, yes, no)` share identical shapes, the condition is bool, and both branches share identical types.

Adjacent elementwise operations are executed by shared loops, with broadcast and transpose mappings reading indices. For concrete materialization boundaries, see [Graph Partitioning](partitioning.md#partition-boundaries).

<a id="matmul-semantics"></a>

## Matrix Multiplication

```mlir
%c = "ask.matmul"(%a, %b) : (tensor<32x64xf16>, tensor<64x48xf16>) -> tensor<32x48xf16>
```

Inputs must share identical f32, f16, or bf16 element types, have a rank of at least 2, and have matching leading batch dimensions element-by-element. `transpose_a` and `transpose_b` are boolean attributes defaulting to false that operate on the final two axes; logical shapes satisfy `[..., M, K] × [..., K, N] → [..., M, N]`. The upper bound of batch counts for CUDA library steps is a signed 32-bit integer. `K=0` produces zeros; empty batches or empty outputs remain empty tensors.

Results preserve input types, while accumulation uses f32. CUDA cuBLASLt uses `CUBLAS_COMPUTE_32F_FAST_TF32` for f32 inputs, permitting TF32 multiplication; f16/bf16 uses f32 accumulation with f32 segmented reduction intermediates. Host f32 invokes single-threaded OpenBLAS SGEMM, with accumulation order determined by the library; Host f16 uses f32 stepwise multiply-add and rounds finally to f16. Transpositions on the final two axes can be absorbed into matmul flags during optimization.

Library matrix multiplication validates numerical values against permitted input precisions and reduction error bounds. Elementwise operations and native reductions continue to adhere to their respective rounding rules.

<a id="types-and-precision"></a>

## 2. Six Element Types and Single-Step RNE Constants

| Element Type Name | MLIR Type | Memory Size | LLVM Value Type | Numerical and Hardware Conventions |
| --- | --- | --- | --- | --- |
| `"bool"` | `i1` | 1 Byte | `i1` (memory access as `i8` `0`/`1`) | Stores truth values only; requires explicit `ask.cast` to integer or floating-point before participating in arithmetic. |
| `"i32"` | `i32` | 4 Bytes | `i32` | 32-bit signed two's complement integer; addition and multiplication wrap modulo $2^{32}$. |
| `"i64"` | `i64` | 8 Bytes | `i64` | 64-bit signed two's complement integer. |
| `"f32"` | `f32` | 4 Bytes | `float` | IEEE-754 binary32; native code generation disables floating-point contraction by default (`fp-contract=off`). |
| `"f16"` | `f16` | 2 Bytes | `half` | IEEE-754 binary16 (5-bit exponent + 10-bit mantissa). |
| `"bf16"` | `bf16` | 2 Bytes | `bfloat` | bfloat16 (8-bit exponent + 7-bit mantissa); execution targets CUDA `sm_80+`. |

`ask.constant` literals for `f16`, `bf16`, and `f32` adopt single-step round-to-nearest-even (RNE), rounding directly to the target bitwidth. This avoids double-rounding errors from first converting to `f32` and then truncating to `f16`/`bf16`, while preserving `-0.0`, `±inf`, and `NaN`. FugaTensor's [numerical encoding](../../../fugatensor/src/fugatensor/_ask/numerics.py) converts Python inputs into the target format following the same rule, which is then emitted as ASK IR by the internal graph bridge layer.

<a id="cast-semantics"></a>

## 3. `ask.cast` Explicit Conversions and Mixed Precision

When combining low-precision storage with `f32` computation, insert `f16 -> f32` and `f32 -> f16` `ask.cast` operations into the graph. CPU and GPU use the identical set of conversion rules:

- **Same-type conversion**: Values remain unchanged; constant inputs can be folded at high level, and runtime inputs directly reuse the original value during LLVM lowering.
- **Float ↔ Float**: Narrowing (e.g. `f32 -> f16`, `f32 -> bf16`) and cross-format (`f16 <-> bf16`) follow IEEE RNE rounding; values exceeding the finite representation range overflow to signed `±inf`, preserving `-0.0` and `NaN`.
- **Integer → Float**: Rounds signed integers to the target floating-point format according to IEEE RNE.
- **Float → Signed Integer (`i32` / `i64`)**: Uses `llvm.fptosi.sat` saturating truncation, truncating fractional parts toward zero; `NaN` maps to `0`, and out-of-range values (including `±inf`) clamp to `[INT_MIN, INT_MAX]`.
- **Integer ↔ Integer**: `i32 -> i64` sign-extends; `i64 -> i32` truncates, keeping the lower 32 bits; `bool -> i32/i64` zero-extends to `0` or `1`.
- **Any Type → `bool`**: Non-zero values (including `NaN`, `±inf`) evaluate to `true` (written as `1` in memory); `0` / `+0.0` / `-0.0` evaluate to `false` (written as `0` in memory). `bool -> float` maps to `0.0` or `1.0`.

<a id="shape-and-markers"></a>

## 4. Shape Inference, Stable Identities, and `ask.mark` Erasure

`ask::compiler::prepare` executes the following steps in sequence via `lib/Frontend/Prepare.cpp`:

1. **Validate symbols and operation IDs**: `ask.id` within a module must not duplicate; `@semantics` and `@implementation` referenced by `ask.external` must exist, with matching port counts and element types.
2. **Infer shape snapshots**: Verify that input and output shapes of elementwise operations match and that element counts before and after reshape are identical; determine output shapes of `ask.external` based on `same_as` (matching specified input operand shapes) or `fixed` (using static shapes from semantic signatures); output tensor value shapes in `shape_bindings` of `shapeSnapshotJSON`, recording input specializations, semantic revisions, and snapshot digests.
3. **Extract and erase `ask.mark`**: Collect `mark_id`, `value_id`, `role` (`value` / `input` / `output`), and optional `port_id` into the `markers` field of `diagnosticsJSON`, then remove `ask.mark` from IR and connect original values directly to downstream operations. Running `--prepare` again on prepared IR leaves the IR and revision counts unchanged; new reports will no longer contain erased markers.

The CLI and full JSON interface store shape snapshots in the `shape_snapshot` field of full reports, and markers in the `markers` field. For named results, see [Compilation Interfaces](api.md#compiler-api).

<a id="external-declaration"></a>

## 5. `ask.external` with Separation of Semantics and Implementation

External operators are split into two top-level declarations:

- `ask.semantics` describes the graph-level interface (input/output ranks and dtypes, shape rules, and `effects = "none"`), used for graph verification and port connections. High-level DCE retains `ask.external`.
- `ask.implementation` describes concrete compilation configurations (`compiler = "triton"`, DSL source code, `entry`, `signature`, `source_bindings`, `constants`, `options`, `target`, `grid`).

`prepare` exports protocol v2 requests via `externalRequestsJSON` of the result; callers select workers based on `compiler`; CLI and full JSON interfaces correspond to the `external_requests` field of the full report. Compiled artifacts are passed to the runtime via the [External Execution Interface](artifacts.md).