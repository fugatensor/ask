# 01. ASK IR 与数值语义

返回核心思想：[双层 IR 模型](核心思想.md#ask-ir-model) · 相关细节：[前端接入与构建配置](02-前端接入与构建配置.md) · [图优化与自有代码生成](03-图优化与自有代码生成.md) · [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)

高层 ASK IR 是上层框架向 ASK 提交计算图的接口，基于 MLIR `ask` dialect 实现（`include/ask/IR/`、`lib/IR/`），用于表达张量形状、精确元素类型、显式类型转换、追踪标记与外部 DSL 边界，不包含设备指针、stream 句柄或隐式广播。

<a id="ir-structure"></a>

## 1. 模块结构与支持的操作集

每个 MLIR `module` 携带两个模块级版本属性（缺省时由 `prepare` 补齐，不匹配则报错）：

- `ask.schema_version = 1 : i64`
- `ask.ir_revision = "ask-ir-0.1"`

模块顶层仅允许两类符号声明和单基本块 `func.func` 计算图函数：

| 操作 | 所在层级 | 作用 |
| --- | --- | --- |
| `ask.semantics` | Module 顶层 | 声明外部算子的输入/输出类型签名、形状推导规则（`same_as` 或 `fixed`）与副作用属性（如 `effects = "none"`）。 |
| `ask.implementation` | Module 顶层 | 绑定外部编译器 ID（如 `"triton"`）、DSL 源码快照、入口函数名、指针签名、`source_bindings`、`constants`、`options`、`target` 和 `grid`。 |
| `ask.constant` | `func.func` 内 | 生成指定类型的标量常量（`bool` / `i32` / `i64` / `f32` / `f16` / `bf16`）。 |
| `ask.splat` | `func.func` 内 | 将标量按静态 `shape` 复制填充为同元素类型的张量。 |
| `ask.add` / `ask.mul` | `func.func` 内 | 同形状、同元素类型的逐元素加/乘；不支持 `bool` 直接算术，整数采用二补数回绕（`overflow = "wrap"`）。 |
| `ask.cast` | `func.func` 内 | 保持标量或张量形状不变，显式转换元素类型（支持全部 6×6 方向）。 |
| `ask.mark` | `func.func` 内 | 追踪透传标记，记录逻辑值身份（`value_id`）或端口绑定（`port_id`），在 `prepare` 提取报告后擦除。 |
| `ask.external` | `func.func` 内 | 引用一对 `@semantics` 与 `@implementation` 符号，表示第三方编译器负责的外部算子节点。 |

图内不支持 `func.call` 跨函数调用或动态控制流 region；函数输入形状在进入 `prepare` 时须为静态具体维度（非 `?`），张量总字节数不超过有符号 64 位整数上限。

<a id="types-and-precision"></a>

## 2. 六种元素类型与单步 RNE 常量

| ASK SDK 名称 | MLIR 类型 | 内存字节数 | LLVM 值类型 | 数值与硬件约定 |
| --- | --- | --- | --- | --- |
| `"bool"` | `i1` | 1 Byte | `i1`（内存存取为 `i8` `0`/`1`） | 仅保存真值；参与算术前需显式 `ask.cast` 到整数或浮点。 |
| `"i32"` | `i32` | 4 Bytes | `i32` | 32 位有符号二补数整数，加乘按模 $2^{32}$ 回绕。 |
| `"i64"` | `i64` | 8 Bytes | `i64` | 64 位有符号二补数整数，不经过 `float64` 中转。 |
| `"f32"` | `f32` | 4 Bytes | `float` | IEEE-754 binary32，自有代码生成默认关闭浮点乘加收缩（`fp-contract=off`）。 |
| `"f16"` | `f16` | 2 Bytes | `half` | IEEE-754 binary16（5 位指数 + 10 位尾数）。 |
| `"bf16"` | `bf16` | 2 Bytes | `bfloat` | bfloat16（8 位指数 + 7 位尾数）；要求执行硬件支持（CUDA `sm_80+` 或支持原生 `bf16` 的 CPU），不支持时拒绝编译或加载。 |

Python SDK（`python/ask/ir.py`）在编码 `f16`、`bf16`、`f32` 标量常量时采用单步最近偶数舍入（RNE）：直接从输入数值的精确二进制表示一次性舍入到目标位宽，避免先转 `f32` 再截断到 `f16`/`bf16` 的双重舍入误差，并保留 `-0.0`、`±inf` 和 `NaN`。

<a id="cast-semantics"></a>

## 3. `ask.cast` 显式转换与混合精度

需要低精度存储配 `f32` 中间计算时，上层框架在图中显式插入 `f16 -> f32` 与 `f32 -> f16` 的 `ask.cast`。6×6 转换规则在 CPU 与 GPU 上保持一致：

- **同类型转换**：恒等传递，在高层折叠中简化。
- **浮点 ↔ 浮点**：窄化（如 `f32 -> f16`、`f32 -> bf16`）与跨格式（`f16 <-> bf16`）按 IEEE RNE 舍入，超出有限表示范围时溢出为同号 `±inf`，保留 `-0.0` 与 `NaN`。
- **整数 → 浮点**：将有符号整数按 IEEE RNE 舍入到目标浮点格式。
- **浮点 → 有符号整数（`i32` / `i64`）**：采用饱和截断（对应 `llvm.fptosi.sat`）——向零截断小数部分，`NaN` 映射为 `0`，超出目标整数上下界的值（含 `±inf`）饱和到 `[INT_MIN, INT_MAX]`，避免原始 `fptosi` 溢出产生 `poison`。
- **整数 ↔ 整数**：`i32 -> i64` 符号扩展；`i64 -> i32` 截断保留低 32 位；`bool -> i32/i64` 零扩展为 `0` 或 `1`。
- **任意类型 → `bool`**：非零值（含 `NaN`、`±inf`）为 `true`（内存写 `1`），`0` / `+0.0` / `-0.0` 为 `false`（内存写 `0`）。`bool -> 浮点` 映射为 `0.0` 或 `1.0`。

<a id="shape-and-markers"></a>

## 4. 形状推导、稳定身份与 `ask.mark` 擦除

`compiler.prepare(module)`（`lib/Frontend/Prepare.cpp`）依次执行以下步骤：

1. **校验符号与操作 ID**：模块内显式声明的 `ask.id` 不可重复；`ask.external` 引用的 `@semantics` 和 `@implementation` 必须存在，且端口数量与元素类型一致。
2. **推导形状快照**：检查逐元素算子的输入输出形状一致，按 `same_as`（跟随指定输入操作数形状）或 `fixed`（使用语义签名中的静态形状）确定 `ask.external` 的输出形状，并在 `report["shapes"]` 中输出各端口形状。
3. **提取并擦除 `ask.mark`**：收集 `mark_id`、`value_id`、`role`（`value` / `input` / `output`）与 `port_id` 到 `report["trace_records"]`，随后从 IR 中移除 `ask.mark` 并将原值直连下游算子。对准备后的 IR 再次运行 `--prepare` 结果保持不变。

<a id="external-declaration"></a>

## 5. 语义与实现分离的 `ask.external`

外部算子拆分为两条顶层声明：

- `ask.semantics` 描述图级接口（输入输出秩与 dtype、形状规则、有无副作用），供图校验、死代码消除和端口连接使用。
- `ask.implementation` 描述具体编译配置（`compiler = "triton"`、DSL 源码、`entry`、`signature`、`source_bindings`、`constants`、`options`、`target`、`grid`）。

拆分后，未启用第三方适配器的 `core` 构建也能验证和准备含 `ask.external` 的图，并在 `report["external_requests"]` 中导出标准化请求；更换目标配置时也无需改动外层图语义。
