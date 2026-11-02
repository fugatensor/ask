# 08. 图切分与多 Kernel 执行计划

返回核心思想：[图切分与执行计划](核心思想.md#execution-plan) · 相关细节：[自有代码生成](03-图优化与自有代码生成.md) · [外部产物与执行接口](04-外部产物与执行接口.md) · [常驻运行时](05-常驻运行时与多级缓存.md)

<a id="partition-ir"></a>

## 1. 切分表示

ASK 图描述数据依赖，执行计划描述哪些依赖需要落到设备缓冲区。切分发生在高层 MLIR，先验证类型、推导静态形状和执行图优化，再决定 kernel 边界。每个分区仍使用普通 `func.func` 和 `ask` 算子，函数增加 `ask.kernel_id`、`ask.kernel_kind` 属性；模块的 `ask.execution_plan` 字符串属性记录同一份 JSON 契约。无需在低层 LLVM IR 中猜测原始张量图。

`Compiler.partition(...)` 返回切分后的 MLIR 与报告；命令行入口是 `ask-opt --partition --report-json plan-report.json graph.mlir`。报告的 `execution_plan` 保存逻辑计划，`step_sources` 按步骤 ID 保存可以单独编译的完整 MLIR。每份子模块带上所引用的语义和实现声明。原始 `ask.id`、值身份及溯源标记继续用于关联切分前后的图。

步骤只有两种类型：`owned` 是 ASK 自有图，`external` 是单个外部 kernel。自有区域内部继续优化和融合；每个外部节点始终单独成为一步。

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
- `function` 指向切分 MLIR 中的函数，步骤输入输出的形状、类型必须逐项匹配实际编译 kernel 的端口。外部步骤的 `external_id` 指回原始算子，便于关联源码和报告失败原因。
- v1 在同一 `ExecutionSession` 的同一 stream 上顺序提交，不表达多 stream、主机计算、动态形状、循环或条件分支。静态图验证失败直接报错。

<a id="partition-boundaries"></a>

## 3. 外部边界与自有区域

切分器按图中的外部节点划分自有区域。例如 `P → F → Q` 直接形成 `owned → external → owned`，不会尝试第三方融合，也没有融合失败后重新切分的路径。自有区域可包含多个逐元素算子，并继续使用 ASK 的代码生成优化。

每个 external 步骤的请求交给调用方 worker，取得 cubin/PTX；owned 步骤经 ASK 的 LLVM 管线编译。任何非法 MLIR、缺失 worker、导出失败、损坏产物或代码生成错误都使编译失败，不能用其它计算路径掩盖。

外部步骤允许多输入、多输出与不同端口形状、类型。指针参数通过 `source_bindings` 连接逻辑端口，标量参数在同一字典中绑定按值传递的启动值；ASK 从 `signature` 与 `port_shapes` 推导 ABI。原生 MLIR、Python ASK SDK 和 FugaTensor 注册接口均支持多结果节点，多输出仍由一个外部 kernel 写出。

<a id="program-runtime"></a>

## 4. 编译包与常驻调度

`CUDAProgram` 把逻辑计划与各步骤的编译产物组合起来。`execution_plan.json` 不包含主机绝对路径；`program.json` 保存步骤到子目录产物的关联及摘要，保存和重新加载时验证路径范围、内容摘要、计划与 kernel ABI。自有步骤复用 IR/机器码缓存，外部步骤复用 worker 产物缓存。

`program.prepare(session)` 在执行前校验全部步骤并加载 kernel。`submit_many(inputs, outputs)` 先检查所有公开端口、设备归属、可写性与别名，再分配中间设备缓冲区、按步骤提交。最后一个事件保留本次执行的中间缓冲区直到完成。设备资源检查或实际提交失败可能已有前序步骤排队，此时已提交事件继续保留其缓冲区；调度不提供 GPU 事务回滚。不同批次拥有不同中间存储；v1 不进行跨步骤存储复用或原地覆盖。零步骤计划只表示别名，确认当前 stream 完成后返回已完成事件。

FugaTensor 在自动组图后直接提交完整静态图，由 ASK 在外部边界切分并优化自有区域；读取数据、条件判断、反向传播与显式求值的触发方式不变。多 kernel 不改变梯度定义，未知外部算子的求导仍需显式规则。

<a id="complex-external"></a>

## 5. 复杂外部算子接入

ASK 只定义外部 kernel 的产物、端口、缓存与调度接口，不在仓内实现注意力、矩阵乘或归约算法。外部步骤可使用多输入、多输出、不同形状或存储类型、内部 FP32 累加和动态共享内存；调用方 worker 负责生成可执行产物并声明实际资源需求。支持范围受通用参数 ABI 与普通 CUDA 启动契约约束，见[外部执行接口](04-外部产物与执行接口.md#binary-artifact)。

复杂算子的算法语义和训练接口属于上层框架。例如 FugaTensor 维护 [FlashAttention-2 前向与反向实现](../../fugatensor/examples/flash_attention2.py)及其 [Triton 源码](../../fugatensor/examples/kernels/flash_attention2.py)，通过 `ft.fun` 的显式反向规则注册为多个 `ask.external`；ASK 将其切分为前向、统计量、dQ、dK、dV 等独立步骤并在同一会话 stream 上调度。该示例的数值验证、反向公式和适用边界由上层文档定义，ASK 不依赖或内置该算法。

适配器接入新 DSL 的规范见[外部编译器适配接入规范](09-外部编译器适配接入规范.md#adapter-contract)，公开的 worker 请求与 artifact 结构见 [schemas/](../schemas/)。
