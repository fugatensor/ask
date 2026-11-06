返回核心思想：[外部产物与执行接口](concepts.md#external-execution) · 相关细节：[外部算子声明](ir.md#external-declaration) · [Worker 协议](compilers.md#worker-protocol) · [图切分与调度](partitioning.md)

# 外部产物与执行接口

每个 `ask.external` 对应一个独立 kernel。调用方用第三方编译器生成 cubin 或 PTX，校验产物后，将二进制和启动配置交给 ASK C++ Runtime。

<a id="external-execution"></a>

## 1. 图边界与职责

`ask.semantics` 定义输入输出类型和形状规则，`ask.implementation` 保存源码、编译配置和端口绑定。编译器检查图并导出请求，在外部节点两侧切分自有区域，再对自有区域做常量折叠、DCE 和逐元素融合。

例如 `P: t = x + 1 → F: 外部 kernel → Q: y = u + 3` 形成三个步骤。中间结果 `t`、`u` 留在设备缓冲区，下一步通过指针访问。Graph replay 用于减少这些步骤的提交开销。

ASK 负责请求与产物协议、图分区、ABI 检查和设备调度。调用方负责启动 worker、选择编译器版本、解析产物和磁盘缓存。FugaTensor 的相关代码位于 `src/fugatensor/_ask/`，Triton 编译代码位于 [`_triton_worker.py`](../../../fugatensor/src/fugatensor/_triton_worker.py)。

<a id="binary-artifact"></a>

## 2. 可执行产物与端口

Worker 返回 base64 编码的 cubin 或 PTX 文本，以及入口、目标架构、生产者、启动配置、参数顺序和所需元素类型，格式见[产物协议](compilers.md#artifact-contract)。

调用方根据请求中的 `signature`、`source_bindings`、`port_shapes` 构造 invocation v2，核对文件摘要和张量端口，再将产物字节交给 `Session::load`。PTX 在加载时由 CUDA 驱动 JIT。

端口使用静态、连续、行主序张量，支持六种 ASK 元素类型和多输出。kernel 接受 1–64 个物理参数；指针绑定 `input:N`、`output:N` 或 `null`，标量按 ABI 类型编码为字节。运行时按驱动报告检查参数数量、偏移和宽度，启动前检查线程数、共享内存、设备归属和存储别名。

`PlanArgument` 保存逻辑缓冲区索引或冻结的标量字节，`PlanStep` 保存 kernel、grid、block、共享内存和输入输出集合。调用方核对张量形状与 dtype，再换算为原生接口使用的字节数。准备计划时将标量字节和参数位置编入模板；重复执行更新公开端口的设备地址。

启动接口使用普通指针和普通 CUDA kernel，scratch 为空。外部产物使用生产者的数值策略，自有计算遵循 ASK 的舍入和转换规则。

<a id="worker-interface"></a>

## 3. 调用方配置 Worker

`ask.implementation.compiler` 是调用方选择 worker 的键。worker 可以是可执行文件，也可以由指定的 Python 解释器运行脚本。

FugaTensor 提供 Triton worker，配置方式见[上层执行说明](../../../fugatensor/docs/zh/architecture/backend.md#cuda-and-triton)。其他框架可实现同一[请求/响应协议](compilers.md)。

外部步骤使用 worker 的编译环境；混合图中的 owned 步骤由调用方运行 `llc` / `ptxas`。缺失 worker、非法请求、编译失败、损坏产物和 ABI 不匹配均应报错。
