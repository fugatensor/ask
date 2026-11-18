返回核心思想：[图切分与执行计划](concepts.md#execution-plan) · 相关细节：[自有代码生成](codegen.md) · [外部产物与执行接口](artifacts.md) · [常驻运行时](runtime.md)

# 图切分与多 Kernel 执行计划

<a id="partition-ir"></a>

## 1. 切分表示

ASK 图描述数据依赖，执行计划确定 kernel 及其间的设备缓冲区。切分器先在 MLIR 中检查类型、推导形状并优化图，再划分 kernel。每个分区使用 `func.func` 和 `ask` 算子，以 `ask.kernel_id`、`ask.kernel_kind` 属性记录身份和类型；模块的 `ask.execution_plan` 字符串属性保存 JSON 计划。

`ask::compiler::partition(source, options)` 返回 `PartitionResult`：`ir` 是切分后的 MLIR，`executionPlanJSON` 保存计划，`stepSources` 按步骤 ID 保存可单独编译的 MLIR，包含该步骤引用的语义和实现声明；`externalRequestsJSON` 提供外部编译请求。原始 `ask.id`、值身份及 `diagnosticsJSON` 中的溯源记录用于关联切分前后的图。接口字段见[用户 API](api.md#compiler-api)。

CLI 用法是 `ask-opt --partition --report-json plan-report.json graph.mlir`。CLI 与 `ask::partitionJSON` 的完整报告保留 `execution_plan`、`step_sources` 和 `external_requests` 字段。

步骤分为三类：`owned` 包含 ASK 自有计算，可在内部优化和融合；`external` 对应单个外部 kernel；`library` 对应 cuBLASLt 矩阵乘法。

<a id="plan-v1"></a>

## 2. 执行计划 JSON v1

格式定义见 [JSON Schema](../../schemas/execution-plan-v1.schema.json)。下面的计划先执行自有计算，再调用外部算子：

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

- `buffers` 中 ID 唯一，形状为具体的非负整数维度，类型取 ASK 六种基础类型。缓冲区连续、行优先，字节数由形状和类型推导。
- 公开标量与跨步骤标量均以 `shape=[]` 的单元素设备缓冲区传递，保持自有 CUDA 的指针 ABI；返回的标量常量也由 owned 步骤写出。
- 顶层 `inputs`、`outputs` 的顺序就是公开张量 ABI。输出可引用输入，也可重复引用同一缓冲区；这些别名端口须绑定同一存储。
- `steps` 按执行顺序排列。每个结果只有一个生产者，输入来自图输入或更早步骤，`depends_on` 须恰好列出这些输入的生产者。内部常量保留在子图中。
- `function` 指向切分 MLIR 中的函数，步骤输入输出的形状、类型必须逐项匹配实际编译 kernel 的端口。外部步骤的 `external_id` 指回原始算子，便于关联源码和报告失败原因。
- `library` 步骤设置 `library = "cublaslt"` 和 `matmul` 描述（`element_type`、`m/n/k`、`batch_count`、`transpose_a/b`），输入为 A、B，输出为 C。
- v1 在同一 `Session` 的单条 stream 上顺序提交静态设备计算。校验失败会报错。

运行时将读端口、公开输出和内部槽位映射交由 [ASK PASS 规则证明](verification.md#plan-proof)检查；索引、单一生产者和 kernel ABI 由 ASK 检查。

<a id="partition-boundaries"></a>

## 3. 物化边界与自有区域

切分器按外部节点划分自有区域。例如 `P → F → Q` 生成 `owned → external → owned`。各自有区域可包含多个逐元素算子。matmul 同样形成独立 library 步骤，例如 `mul → matmul → add` 生成 `owned → library → owned`。

广播、转置、切片、填充、拼接与下标读取通过索引参与自有融合。`ask.scatter_add` 切分为独立的 `owned` 步骤，使 `base` 先通过步骤前设备拷贝初始化再执行原子累加。归约与同尺寸消费者可同核执行；归约结果在被广播、转置、填充、聚集扩展或供 `matmul`/`gather`/`scatter_add` 消费时先物化，以免重复归约。跨步骤的归约标量使用 `shape=[]` 的设备缓冲区。

external 步骤交给调用方 worker 编译为 cubin/PTX，owned 步骤经 ASK 的 LLVM 管线编译，library 步骤由 Runtime 在准备时选择 cuBLASLt 算法。非法 MLIR、缺失 worker、导出失败、损坏产物和代码生成错误都会使编译失败。

外部步骤支持多输入、多输出，以及不同形状和类型的端口。`source_bindings` 将指针参数连到逻辑端口，将标量参数绑定到固定启动值；调用方根据请求的 `signature` 和 `port_shapes` 构造执行 ABI。一个外部 kernel 可写出多个结果。

<a id="program-runtime"></a>

## 4. 编译包与常驻调度

调用方可持久化保存导出的 `executionPlanJSON` 及各步骤编译产物与摘要。保存和加载时应检查路径范围、摘要、计划和 kernel ABI，自有与外部步骤分别使用各自的磁盘缓存，library 的矩阵描述保存在计划中，校验后再交给 ASK Runtime。

调用方加载 owned/external kernel，并为 library 填写 `MatmulDescription`，按逻辑缓冲区构造字节数表、公开端口索引与 `PlanStep`，交给 `Session::prepareProgram(..., graph)`。`PreparedProgram::submit(inputs, outputs)` 一次完成所有步骤；C++ 先校验公开端口的设备归属、字节数、可写性和别名，再更新公开端口地址并提交，只记录一个完成事件。`run(inputs)` 还负责分配本次输出。

每个在途调用占用独立工作区，事件完成后才能复用。工作区内按已验证的步骤生存期共享物理槽位，公开端口单独绑定。事件保留程序和缓冲区，会话保留待完成事件，因此调用方释放对象后，在途资源仍然有效。启动失败时，运行时等待已排队的步骤结束再传播异常；设备写入不回滚。零步骤别名计划记录完成事件，覆盖此前的 stream 依赖，保持异步提交。

计划证明与显存复用的分工见[规则证明与计划校验](verification.md#reuse-boundary)。

传入 `graph=true` 时使用 [CUDA Graph replay](runtime.md#cuda-graph)。准备阶段从步骤读写端口推导依赖，冻结节点前驱，并按依赖可达性校验内部槽位复用；独立分支可并行执行。每个工作区缓存 graph 实例，复用时更新公开端口地址，自有与外部步骤使用同一机制。

上层框架将前向或求导后的静态图提交给 ASK，由 ASK 在外部节点处分区并优化自有区域。外部算子的反向规则由上层框架提供。

<a id="complex-external"></a>

## 5. 复杂外部算子接入

外部 kernel 可使用多输入、多输出、不同形状和存储类型、内部 FP32 累加及动态共享内存。worker 负责生成产物并报告所需资源，启动限制见[外部执行接口](artifacts.md#binary-artifact)。

当外部算子包含多个关联 kernel（例如前向与拆分后的多个反向 kernel）时，ASK 将每个 `ask.external` 划分为独立步骤并按缓冲区依赖调度；算法和梯度的数值验证由调用方维护。

接入新 DSL 见[适配规范](compilers.md#adapter-contract)，请求和产物格式见 [schemas/](../../schemas/)。
