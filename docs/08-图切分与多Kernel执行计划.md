# 08. 图切分与多 Kernel 执行计划

返回核心思想：[图切分与执行计划](核心思想.md#execution-plan) · 相关细节：[自有代码生成](03-图优化与自有代码生成.md) · [外部编译器与边界内联](04-外部编译器适配与边界内联.md) · [常驻运行时](05-常驻运行时与多级缓存.md)

<a id="partition-ir"></a>

## 1. 切分表示

ASK 图描述数据依赖，执行计划描述哪些依赖需要落到设备缓冲区。切分发生在高层 MLIR，先验证类型、推导静态形状和执行图优化，再决定 kernel 边界。每个分区仍使用普通 `func.func` 和 `ask` 算子，函数增加 `ask.kernel_id`、`ask.kernel_kind` 属性；模块的 `ask.execution_plan` 字符串属性记录同一份 JSON 契约。无需在低层 LLVM IR 中猜测原始张量图。

`Compiler.partition(...)` 返回切分后的 MLIR 与报告；命令行入口是 `ask-opt --partition --report-json plan-report.json graph.mlir`。报告的 `execution_plan` 保存逻辑计划，`step_sources` 按步骤 ID 保存可以单独编译的完整 MLIR。每份子模块带上所引用的语义和实现声明。原始 `ask.id`、值身份及溯源标记继续用于关联切分前后的图。

一个步骤有三种类型：`owned` 是 ASK 自有图；`fused` 是待验证的单外部算子及其逐元素前后计算；`external` 是独立外部 kernel。`fused` 表示融合候选，只有外部编译器契约与 LLVM 边界证明通过后才生成融合产物。

<a id="plan-v1"></a>

## 2. 执行计划 JSON v1

[JSON Schema](../schemas/execution-plan-v1.schema.json) 与运行时校验共同约束此契约。以下示意一个先执行自有计算、再调用外部算子的计划：

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

- `buffers` 中 ID 唯一，形状为具体的非负整数维度，类型为 ASK 六种基础类型。缓冲区为连续、行优先存储；字节数由形状和类型推导，不重复保存一份可冲突的长度。
- 顶层 `inputs`、`outputs` 的顺序就是公开张量 ABI。输出可以引用输入，也可以重复引用同一缓冲区，表示别名；此时提交方必须提供同一存储，不隐式插入复制。
- `steps` 按执行顺序排列。每个结果只能有一个生产者，输入必须已经由图输入或更早步骤定义，`depends_on` 必须恰为步骤输入的生产者集合，禁止前向引用和环。内部常量留在子图，不占设备缓冲区端口。
- `function` 指向切分 MLIR 中的函数，步骤输入输出的形状、类型必须逐项匹配实际编译 kernel 的端口。外部步骤的 `external_id` 指回原始算子，便于重切分和报告失败原因。
- v1 在同一 `ExecutionSession` 的同一 stream 上顺序提交，不表达多 stream、主机计算、动态形状、循环或条件分支。静态图验证失败直接报错。

<a id="fusion-fallback"></a>

## 3. 融合失败时重切分

切分器优先保留可证明的逐元素融合区域；多个外部算子、扇出、异构端口或显式形状边界可以形成多个步骤。对于实际编译时才发现不能建立融合证明的外部算子，编译层记录其原始 ID 和诊断，要求切分器在该边界拆开，分别编译前图、外部 kernel 和后图。`fuse_external=False` 可关闭外部融合，`unfused_externals=[...]` 可指定必须隔离的外部 ID。

需要在切分阶段区分的两类情况：

- **结构上不可融合**：多输入/多输出外部算子、扇出、异构端口或显式形状边界在切分时直接形成 `external`/`owned` 步骤，不先尝试融合。例如 FlashAttention 前向有三个输入，计划中就是一个独立 `external` 步骤，不需要等编译失败。
- **编译时融合证明失败**：只有逐元素融合候选在适配器边界检查中返回明确的 `UnsupportedFusion`，编译层才用原始外部 ID 重跑切分，拆成前图、独立外部 kernel 和后图。其它导出、worker、LLVM 或代码生成错误仍直接失败。

重切分仅用于能力或融合证明不足。非法 MLIR、未启用适配器、worker 启动错误、损坏产物和代码生成错误仍然失败。各步骤均编译成机器码，执行过程中不通过 Python 计算张量元素。

独立外部导出不要求逐元素一进一出。源码参数通过 `source_bindings` 绑定到多个输入端口，静态形状和类型经 `port_map` 进入通用 CUDA invocation v2；共享内存等启动信息来自外部编译器产物。原生 MLIR 可描述多结果外部节点；当前 Python ASK SDK 和 FugaTensor 的源码注册接口仍为单输出，完整图可以有多个输出。

<a id="program-runtime"></a>

## 4. 编译包与常驻调度

`CUDAProgram` 把逻辑计划与各步骤的编译产物组合起来。`execution_plan.json` 不包含主机绝对路径；`program.json` 保存步骤到子目录产物的关联及摘要，保存和重新加载时验证路径范围、内容摘要、计划与 kernel ABI。每个子 kernel 继续复用 ASK 的 LLVM IR 和机器码缓存。

`program.prepare(session)` 在执行前校验全部步骤并加载 kernel。`submit_many(inputs, outputs)` 先检查所有公开端口、设备归属、可写性与别名，再分配中间设备缓冲区、按步骤提交。最后一个事件保留本次执行的中间缓冲区直到完成。设备资源检查或实际提交失败可能已有前序步骤排队，此时已提交事件继续保留其缓冲区；调度不提供 GPU 事务回滚。不同批次拥有不同中间存储；v1 不进行跨步骤存储复用或原地覆盖。零步骤计划只表示别名，确认当前 stream 完成后返回已完成事件。

FugaTensor 在自动组图后直接提交完整静态图，由 ASK 决定融合和切分；读取数据、条件判断、反向传播与显式求值的触发方式不变。多 kernel 不改变梯度定义，未知外部算子的求导仍需显式规则。

<a id="complex-external"></a>

## 5. 复杂外部算子接入

ASK 只定义外部 kernel 的导出、端口、缓存与调度接口，不在仓内实现注意力、矩阵乘、归约等算法。独立外部步骤不要求逐元素边界，因此可以使用多个输入、不同输入/输出形状或存储类型、FP32 中间累加、`tl.dot` 与动态共享内存；适配器负责把生产者的 LLVM IR 和端口映射校验后交给通用 CUDA 执行契约。

复杂算子的算法语义和训练接口属于上层框架。例如 FugaTensor 维护 [FlashAttention-2 前向与反向实现](../../fugatensor/examples/flash_attention2.py)及其 [Triton 源码](../../fugatensor/examples/kernels/flash_attention2.py)，通过 `ft.fun` 的显式反向规则注册为多个 `ask.external`；ASK 将其切分为前向、统计量、dQ、dK、dV 等独立步骤并在同一会话 stream 上调度。该示例的数值验证、反向公式和适用边界由上层文档定义，ASK 不依赖或内置该算法。

适配器接入新 DSL 的规范见[外部编译器适配接入规范](09-外部编译器适配接入规范.md#adapter-contract)，公开的 worker 请求与 artifact 结构见 [schemas/](../schemas/)。
