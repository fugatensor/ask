# 01. ASK IR 与数值语义

返回核心思想：[双层 IR 模型](核心思想.md#ask-ir-model) · 相关细节：[前端接入与构建配置](02-前端接入与构建配置.md) · [图优化与自有代码生成](03-图优化与自有代码生成.md) · [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)

高层 ASK IR 是上层框架向 ASK 提交计算图的唯一契约，基于 MLIR `ask` dialect 实现（定义见 `include/ask/IR/` 与 `lib/IR/`）。它的职责很纯粹：**把张量形状、每个值的精确元素类型、显式类型转换、追踪标记和外部 DSL 边界一次性描述清楚**，不掺杂设备指针、stream 句柄或隐式广播。

<a id="ir-structure"></a>

## 1. 模块结构与支持的操作集

每个提交给 ASK 的 MLIR `module` 携带两个模块级版本属性（缺省时由 `prepare` 自动补齐，显式写错则直接拒绝）：

- `ask.schema_version = 1 : i64`
- `ask.ir_revision = "ask-ir-0.1"`

模块顶层只允许两类符号声明和单基本块 `func.func` 计算图函数：

| 操作 | 所在层级 | 作用 |
| --- | --- | --- |
| `ask.semantics` | Module 顶层 | 声明外部算子的输入/输出类型签名、形状推导规则（`same_as` 或 `fixed`）与副作用属性（如 `effects = "none"`）。 |
| `ask.implementation` | Module 顶层 | 绑定具体外部编译器 ID（如 `"triton"`）、DSL 源码快照、入口函数名、指针签名、`source_bindings`、`constants`、`options`、`target` 和 `grid`。 |
| `ask.constant` | `func.func` 内 | 生成具体精度的标量常量（`bool` / `i32` / `i64` / `f32` / `f16` / `bf16`）。 |
| `ask.splat` | `func.func` 内 | 将标量按静态 `shape` 广播为同元素类型的张量。 |
| `ask.add` / `ask.mul` | `func.func` 内 | 同形状、同元素类型的逐元素加/乘；不支持 `bool` 直接算术，整数默认采用补码回绕（`overflow = "wrap"`）。 |
| `ask.cast` | `func.func` 内 | 在保持标量或张量形状不变的前提下，显式转换元素类型（覆盖全部 6×6 方向）。 |
| `ask.mark` | `func.func` 内 | 纯追踪透传标记，记录逻辑值身份（`value_id`）或端口绑定（`port_id`），在 `prepare` 收集报告后擦除。 |
| `ask.external` | `func.func` 内 | 引用一对 `@semantics` 与 `@implementation` 符号，在计算图中表示一段由第三方编译器负责的外部算子区域。 |

图内不支持 `func.call` 跨函数调用或动态控制流 region；函数输入形状在进入 `prepare` 时必须是具体静态维度（非 `?`），且张量总字节数不得超过有符号 64 位整数上限。

<a id="types-and-precision"></a>

## 2. 六种元素类型与单步 RNE 常量

ASK 全链路支持以下 6 种元素类型：

| ASK SDK 名称 | MLIR 类型 | 内存字节数 | LLVM 值类型 | 数值与硬件约定 |
| --- | --- | --- | --- | --- |
| `"bool"` | `i1` | 1 Byte | `i1`（内存存取为 `i8` `0`/`1`） | 仅保存真值；参与算术前必须显式 `ask.cast` 到整数或浮点。 |
| `"i32"` | `i32` | 4 Bytes | `i32` | 32 位有符号二补数整数，加乘按模 $2^{32}$ 回绕。 |
| `"i64"` | `i64` | 8 Bytes | `i64` | 完整 64 位有符号二补数整数，不经过 `float64` 中转，低位精度不丢失。 |
| `"f32"` | `f32` | 4 Bytes | `float` | IEEE-754 binary32，自有代码生成默认关闭浮点乘加收缩（`fp-contract=off`）。 |
| `"f16"` | `f16` | 2 Bytes | `half` | IEEE-754 binary16（5 位指数 + 10 位尾数）。 |
| `"bf16"` | `bf16` | 2 Bytes | `bfloat` | bfloat16（8 位指数 + 7 位尾数）；要求实际执行硬件支持（CUDA `sm_80+` 或支持原生 `bf16` 的 CPU），不支持时直接拒绝编译或加载。 |

> [!IMPORTANT]
> Python SDK（`python/ask/ir.py`）在编码 `f16`、`bf16`、`f32` 标量常量时采用**单步最近偶数舍入（RNE）**：把 Python `int` 或 `float` 的精确二进制定点/浮点尾数一次性舍入到目标位宽，避免“先舍入到 `f32` 再截断到 `f16`/`bf16`”带来的双舍入误差，并完整保留 `-0.0`、`±inf` 和 `NaN`。

<a id="cast-semantics"></a>

## 3. `ask.cast` 显式转换与混合精度

ASK 不会替用户猜测混合精度策略。如果输入输出是 `f16` / `bf16`，但希望中间累加用 `f32` 保住低位有效数字，上层框架就在图中显式插入 `f16 -> f32` 和 `f32 -> f16` 的 `ask.cast`。

6×6 类型转换遵循统一、跨 CPU/GPU 一致的精确定义：

- **同类型转换**：恒等传递，且会在高层常量/类型折叠中被简化。
- **浮点 ↔ 浮点**：窄化（如 `f32 -> f16`、`f32 -> bf16`）与跨格式（`f16 <-> bf16`）均按 IEEE RNE 舍入，超出有限表示范围时溢出为同号 `±inf`，保留 `-0.0` 与 `NaN`。
- **整数 → 浮点**：将有符号整数按 IEEE RNE 舍入到目标浮点格式。
- **浮点 → 有符号整数（`i32` / `i64`）**：采用**安全饱和截断**（与 LLVM `llvm.fptosi.sat` 一致）——向零截断小数部分，`NaN` 映射为 `0`，超出目标整数上下界的值（含 `±inf`）饱和到 `[INT_MIN, INT_MAX]`，彻底消除 C/LLVM 原始 `fptosi` 溢出时的 `poison` 未定义行为。
- **整数 ↔ 整数**：`i32 -> i64` 做符号扩展；`i64 -> i32` 截断保留低 32 位；`bool -> i32/i64` 零扩展为 `0` 或 `1`。
- **任意类型 → `bool`**：数值非零（含 `-0.0` 以外的任意非零数及 `NaN`、`±inf`）为 `true`（内存写 `1`），`0` / `+0.0` / `-0.0` 为 `false`（内存写 `0`）。`bool -> 浮点` 映射为 `0.0` 或 `1.0`。

<a id="shape-and-markers"></a>

## 4. 形状推导、稳定身份与 `ask.mark` 擦除

运行 `compiler.prepare(module)`（见 `lib/Frontend/Prepare.cpp`）时，ASK 依次执行以下前端准备步骤：

1. **校验符号与操作唯一性**：模块内显式声明的 `ask.id` 不能重复；`ask.external` 引用的 `@semantics` 和 `@implementation` 必须存在且端口数量、元素类型完全一致。
2. **推导并冻结形状快照**：验证所有逐元素算子的输入输出形状一致，按 `same_as`（跟随指定输入操作数形状）或 `fixed`（使用语义签名中的静态结果形状）落定 `ask.external` 的具体输出形状，并在准备报告 `report["shapes"]` 中输出全部端口的冻结形状。
3. **提取并擦除 `ask.mark`**：收集 `ask.mark` 记录的 `mark_id`、`value_id`、`role`（`value` / `input` / `output`）和 `port_id` 到 `report["trace_records"]`，随后把所有 `ask.mark` 从 IR 中直接擦除，把输入 SSA 值直连给下游消费者。这样后续优化和代码生成完全不受追踪探针干扰，而且对准备后的 IR 再次运行 `--prepare` 是严格幂等的。

<a id="external-declaration"></a>

## 5. 语义与实现分离的 `ask.external`

为什么要把外部算子拆成 `ask.semantics` 和 `ask.implementation` 两条顶层声明？

- `ask.semantics` 只描述**数学与图接口**（输入输出张量秩与 dtype、形状规则、有无副作用），供高层图验证、死代码消除和端口连接使用。
- `ask.implementation` 才描述**怎么编译这段外部代码**（`compiler = "triton"`、DSL 源码、`entry`、`signature`、`source_bindings`、`constants`、`options`、`target`、`grid`）。

这种分离带来两个直接好处：第一，`core`（未启用任何第三方适配器）构建照样可以完整验证和准备包含 `ask.external` 的图，并在 `report["external_requests"]` 中生成标准化导出请求；第二，同一套外部算子语义将来可以挂接不同架构或不同编译器的实现，无需改动图结构。

下一步可继续阅读：[前端接入与构建配置](02-前端接入与构建配置.md) 或 [外部编译器适配与边界内联](04-外部编译器适配与边界内联.md)。
