返回核心思想：[双层 IR 模型](核心思想.md#ask-ir-model) · 相关细节：[前端接入与构建配置](前端接入与构建配置.md) · [图优化与自有代码生成](图优化与自有代码生成.md) · [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/降级到ASK.md)

# ASK IR 与数值语义

ASK IR 使用 MLIR `ask` dialect 描述张量类型、形状、运算、溯源标记和外部算子。上层框架通过它向 ASK 提交计算图，定义位于 `include/ask/IR/` 和 `lib/IR/`。

<a id="ir-structure"></a>

## 1. 模块结构与支持的操作集

`prepare` 检查并补齐以下模块属性：

- `ask.schema_version = 1 : i64`：ASK IR 的格式版本，取值为 `1`。
- `ask.semantic_revision` / `ask.ir_revision`：非负 `i64` 变更计数，默认从 `0` 开始，须小于 `INT64_MAX`。类型细化增加两项计数；补齐身份或擦除标记等修改增加 `ir_revision`。

模块顶层仅允许两类符号声明和单基本块 `func.func` 计算图函数：

| 操作 | 所在层级 | 作用 |
| --- | --- | --- |
| `ask.semantics` | Module 顶层 | 声明外部算子的输入/输出类型签名、形状推导规则（`same_as` 或 `fixed`）；使用 `effects = "none"`。 |
| `ask.implementation` | Module 顶层 | 绑定外部编译器 ID（如 `"triton"`）、DSL 源码快照、入口函数名、指针/标量签名、`source_bindings`、`constants`、`options`、`target` 和 `grid`。 |
| `ask.constant` | `func.func` 内 | 生成指定类型的标量常量（`bool` / `i32` / `i64` / `f32` / `f16` / `bf16`）。 |
| `ask.splat` | `func.func` 内 | 将标量按静态 `shape` 复制填充为同元素类型的张量。 |
| `ask.add` / `ask.mul` | `func.func` 内 | 同形状、同数值元素类型的逐元素加/乘；整数采用二补数回绕（`overflow = "wrap"`）。 |
| `ask.cast` | `func.func` 内 | 转换元素类型，保持形状；支持全部 6×6 方向。 |
| `ask.reshape` | `func.func` 内 | 改变张量形状，保持元素类型、总数和连续行主序次序；支持 0-D 和空张量。 |
| `ask.broadcast_in_dim` / `ask.transpose` | `func.func` 内 | 显式扩展维度或重排轴，保持元素类型。 |
| `ask.reduce_sum` / `ask.reduce_max` | `func.func` 内 | 按指定轴归约，移除这些轴。 |
| `ask.div` / `ask.exp` / `ask.log` / `ask.sqrt` | `func.func` 内 | 同类型浮点逐元素计算。 |
| `ask.compare` / `ask.select` | `func.func` 内 | 比较产生同形状 bool；按 bool 条件选择同形状、同类型的值。 |
| `ask.matmul` | `func.func` 内 | 连续矩阵或同批次矩阵相乘，CUDA 使用 cuBLASLt。 |
| `ask.mark` | `func.func` 内 | 追踪透传标记，记录逻辑值身份（`value_id`）或端口绑定（`port_id`），在 `prepare` 提取报告后擦除。 |
| `ask.external` | `func.func` 内 | 引用一对 `@semantics` 与 `@implementation` 符号，表示第三方编译器负责的外部算子节点。 |

每个图函数使用一个基本块，上层将函数调用和控制流展开为静态图。进入 `prepare` 时，输入维度须为具体值，张量总字节数须在有符号 64 位整数范围内。

`ask.reshape` 用 `shape` 属性给出目标具体形状，结果类型可暂含 `?`，由 `prepare` 细化。例如将四个元素改为二维排列：

```mlir
%matrix = "ask.reshape"(%flat) {shape = array<i64: 2, 2>} : (tensor<4xf32>) -> tensor<2x2xf32>
```

`shape = array<i64>` 表示 0-D 张量；`-1` 等维度推断由调用方完成，见 [FugaTensor reshape](../../fugatensor/docs/API/张量与数据交换.md#reshape)。reshape 保持元素类型和线性次序，外部 kernel 接收变换后的形状。自有逐元素循环复用对应的输入值，公开输出使用独立缓冲区。

<a id="tensor-primitives"></a>

## 显式形状变换与归约

`broadcast_in_dim` 的 `shape` 给出目标形状，`dimensions` 将每个输入轴映射到目标轴，按严格递增顺序排列。映射轴的输入维度须等于目标维度或为 1；标量使用空映射。例如：

```mlir
%wide = "ask.broadcast_in_dim"(%row) {shape = array<i64: 2, 3>, dimensions = array<i64: 0, 1>} : (tensor<1x3xf32>) -> tensor<2x3xf32>
%swapped = "ask.transpose"(%wide) {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
%sum = "ask.reduce_sum"(%swapped) {axes = array<i64: 1>} : (tensor<3x2xf32>) -> tensor<3xf32>
```

`transpose` 的 `permutation` 列出各输出轴对应的输入轴。归约的 `axes` 严格递增且唯一，结果移除指定轴；全部轴被移除时结果为标量，可用 `splat` 或 `broadcast_in_dim` 扩展。正秩张量的空轴列表保持原值。

`reduce_sum` 按归约轴的行主序累计，每步使用输入类型舍入，整数回绕，空域结果为零。需要 f32 累加时先 `cast`。`reduce_max` 接受整数和浮点，传播 NaN，区分正负零，要求归约域非空。

<a id="elementwise-primitives"></a>

## 浮点计算与条件选择

`div`、`exp`、`log`、`sqrt` 接受 f32/f16/bf16；低精度超越函数先扩展至 f32，再 RNE 回原类型。CUDA 的 exp/log 使用 NVIDIA libdevice。比较的 `predicate` 为 `eq`、`ne`、`lt`、`le`、`gt` 或 `ge`；浮点 NaN 仅使 `ne` 成立。`select(condition, yes, no)` 三个操作数形状相同，条件为 bool，两个分支类型相同。

相邻逐元素操作由共享循环执行，广播和转置映射读取下标。具体物化边界见[图切分](图切分与多Kernel执行计划.md#partition-boundaries)。

<a id="matmul-semantics"></a>

## 矩阵乘法

```mlir
%c = "ask.matmul"(%a, %b) : (tensor<32x64xf16>, tensor<64x48xf16>) -> tensor<32x48xf16>
```

输入使用相同的 f32、f16 或 bf16，秩至少为 2，前导批次维度逐项相同。`transpose_a`、`transpose_b` 是默认 false 的布尔属性，作用于最后两轴；逻辑形状满足 `[..., M, K] × [..., K, N] → [..., M, N]`。CUDA 库步骤的批次数上限为有符号 32 位整数。`K=0` 产生零，空批次或空输出保持空张量。

结果保持输入类型，累加使用 f32。CUDA cuBLASLt 的 f32 输入采用 `CUBLAS_COMPUTE_32F_PEDANTIC`；f16/bf16 使用 f32 累加及 f32 分段归约中间值。库算法决定归约顺序。Host 通过行主序循环计算，f32 乘法和加法分别舍入。末两轴的转置可在优化时吸收到 matmul 标志中。

<a id="types-and-precision"></a>

## 2. 六种元素类型与单步 RNE 常量

| 元素类型名 | MLIR 类型 | 内存字节数 | LLVM 值类型 | 数值与硬件约定 |
| --- | --- | --- | --- | --- |
| `"bool"` | `i1` | 1 Byte | `i1`（内存存取为 `i8` `0`/`1`） | 仅保存真值；参与算术前需显式 `ask.cast` 到整数或浮点。 |
| `"i32"` | `i32` | 4 Bytes | `i32` | 32 位有符号二补数整数，加乘按模 $2^{32}$ 回绕。 |
| `"i64"` | `i64` | 8 Bytes | `i64` | 64 位有符号二补数整数。 |
| `"f32"` | `f32` | 4 Bytes | `float` | IEEE-754 binary32，自有代码生成默认关闭浮点乘加收缩（`fp-contract=off`）。 |
| `"f16"` | `f16` | 2 Bytes | `half` | IEEE-754 binary16（5 位指数 + 10 位尾数）。 |
| `"bf16"` | `bf16` | 2 Bytes | `bfloat` | bfloat16（8 位指数 + 7 位尾数）；执行目标为 CUDA `sm_80+`。Host 代码生成拒绝含 `bf16` 的图。 |

`ask.constant` 的 `f16`、`bf16`、`f32` 字面量采用单步最近偶数舍入（RNE），直接舍入到目标位宽，避免先转 `f32` 再截断到 `f16`/`bf16` 的双重舍入误差，并保留 `-0.0`、`±inf` 和 `NaN`。FugaTensor 的[数值编码](../../fugatensor/src/fugatensor/_ask/numerics.py)按同一规则将 Python 输入转换为目标格式，再由内部图桥接层输出 ASK IR。

<a id="cast-semantics"></a>

## 3. `ask.cast` 显式转换与混合精度

低精度存储配合 `f32` 计算时，在图中插入 `f16 -> f32` 与 `f32 -> f16` 的 `ask.cast`。CPU 与 GPU 使用同一组转换规则：

- **同类型转换**：数值保持不变；常量输入可被高层折叠，运行时输入在 LLVM 降级时直接复用原值。
- **浮点 ↔ 浮点**：窄化（如 `f32 -> f16`、`f32 -> bf16`）与跨格式（`f16 <-> bf16`）按 IEEE RNE 舍入，超出有限表示范围时溢出为同号 `±inf`，保留 `-0.0` 与 `NaN`。
- **整数 → 浮点**：将有符号整数按 IEEE RNE 舍入到目标浮点格式。
- **浮点 → 有符号整数（`i32` / `i64`）**：采用 `llvm.fptosi.sat` 饱和截断，向零截断小数，`NaN` 映射为 `0`，超出范围的值（含 `±inf`）钳制到 `[INT_MIN, INT_MAX]`。
- **整数 ↔ 整数**：`i32 -> i64` 符号扩展；`i64 -> i32` 截断保留低 32 位；`bool -> i32/i64` 零扩展为 `0` 或 `1`。
- **任意类型 → `bool`**：非零值（含 `NaN`、`±inf`）为 `true`（内存写 `1`），`0` / `+0.0` / `-0.0` 为 `false`（内存写 `0`）。`bool -> 浮点` 映射为 `0.0` 或 `1.0`。

<a id="shape-and-markers"></a>

## 4. 形状推导、稳定身份与 `ask.mark` 擦除

`ask::compiler::prepare` 通过 `lib/Frontend/Prepare.cpp` 依次执行以下步骤：

1. **校验符号与操作 ID**：模块内的 `ask.id` 不可重复；`ask.external` 引用的 `@semantics` 和 `@implementation` 必须存在，且端口数量与元素类型一致。
2. **推导形状快照**：检查逐元素算子的输入输出形状一致、reshape 前后元素数相同，按 `same_as`（跟随指定输入操作数形状）或 `fixed`（使用语义签名中的静态形状）确定 `ask.external` 的输出形状，在 `shapeSnapshotJSON` 的 `shape_bindings` 中输出张量值的形状，并记录输入特化、语义修订与快照摘要。
3. **提取并擦除 `ask.mark`**：收集 `mark_id`、`value_id`、`role`（`value` / `input` / `output`）与可选 `port_id` 到 `diagnosticsJSON` 的 `markers` 字段，随后从 IR 中移除 `ask.mark` 并将原值直连下游算子。对准备后的 IR 再次运行 `--prepare`，IR 与修订计数保持不变；新的报告不再含已擦除的标记。

CLI 和完整 JSON 接口将形状快照保存在完整报告的 `shape_snapshot` 字段，标记保存在 `markers` 字段。具名结果见[编译接口](用户API.md#compiler-api)。

<a id="external-declaration"></a>

## 5. 语义与实现分离的 `ask.external`

外部算子拆分为两条顶层声明：

- `ask.semantics` 描述图级接口（输入输出秩与 dtype、形状规则及 `effects = "none"`），供图校验和端口连接使用。高层 DCE 保留 `ask.external`。
- `ask.implementation` 描述具体编译配置（`compiler = "triton"`、DSL 源码、`entry`、`signature`、`source_bindings`、`constants`、`options`、`target`、`grid`）。

`prepare` 通过结果的 `externalRequestsJSON` 导出协议 v2 请求，调用方按 `compiler` 选择 worker；CLI 和完整 JSON 接口对应完整报告的 `external_requests` 字段。编译后的产物通过[外部执行接口](外部产物与执行接口.md)交给运行时。
