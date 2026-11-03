# FugaTensor 协同与演进路线

返回核心思想：[框架分工](核心思想.md#ask-positioning) · [验证与支持边界](核心思想.md#ask-testing-roadmap) · 相关细节：[ASK IR](ASK-IR与数值语义.md) · [外部执行接口](外部产物与执行接口.md) · [FugaTensor 架构](../../fugatensor/docs/architecture/README.md)

<a id="division-of-labor"></a>

## 1. 两个仓库的职责

推荐以 [FugaTensor](../../fugatensor/README.md) 作为 ASK 接入示例。它提供 [Python API](../../fugatensor/docs/API/README.md)，扩展链接 ASK C++ 库，wheel 携带编译器和运行时共享库。构建步骤见 [ASK 安装](安装.md#framework-install)和 [FugaTensor 安装](../../fugatensor/docs/安装.md)。

| 关注点 | FugaTensor | ASK |
| --- | --- | --- |
| 用户 API 与微分 | Tensor、自动组图、`ft.grad` 纯 IR 变换、`ft.fun` 前向与 VJP 规则、求值触发。 | 静态图接口与编译执行。 |
| 控制流与 IR | Fuga IR 表达区域和函数边界；静态展开或按需执行后提交 ASK 图。 | 单基本块、形状具体的 MLIR `ask` 图，自有计算降级到 LLVM。 |
| 外部编译器 | 保存源码和编译配置，运行 worker，校验并缓存产物。 | 生成编译请求，检查产物和启动参数。 |
| 执行 | 选择后端，管理 Tensor 引用，触发求值并提交计划。 | 生成自有代码，切分计划，管理显存、事件和 CUDA Graph。 |
| 溯源 | 输出 `fuga-to-ask.json` 映射，见 [降级到 ASK](../../fugatensor/docs/ir/降级到ASK.md)。 | 保留稳定节点/值身份，提取 `ask.mark` 并关联分区报告。 |

停止梯度由 Fuga IR 的 `fuga.stop_gradient` 表达，JVP/VJP 在该节点停止传播；[降级到 ASK](../../fugatensor/docs/ir/降级到ASK.md#type-and-op-mapping)时直接使用输入值，ASK 执行保留的数值依赖。

<a id="implemented-scope"></a>

## 2. 当前已实现范围

- 六种显式类型、单步 RNE 常量和 6×6 转换；CPU/GPU 目标支持时使用原生 `bf16`。
- C++ `ASK::Compiler`、`ASK::Runtime`、`ASK::Core` 与 `ask-opt`；形状推导、溯源提取、自有常量折叠、DCE 和逐元素融合。
- 自有 Host 共享循环 `.o` 与 CUDA cubin 代码生成；多端口、0-D 张量和空张量。
- 外部 cubin/PTX 接入与协议 v2；调用方运行 worker，每个外部节点是独立计划步骤。
- 原生常驻会话、设备缓冲区、整计划提交、异步事件与资源复用、CUDA Graph replay；FugaTensor 在此基础上提供 CAI 借用、磁盘缓存和可迁移多 kernel 编译包。

[attention](../../fugatensor/docs/API/attention.md) 是一个例子：FugaTensor 提供算法和反向规则，ASK 调度对应的 kernel。

<a id="roadmap"></a>

## 3. 当前限制

| 边界 | 当前行为 |
| --- | --- |
| 第三方融合 | 每个外部 kernel 独立执行，跨步骤通过显存连接。 |
| 调度 | 静态计划、单 stream 顺序提交或 Graph replay；各未完成提交拥有独立 workspace，没有图内动态条件/循环或多 stream 并发计划。 |
| 操作集 | 自有算子没有归约、轴广播或矩阵乘；`splat` 仅将标量填充到指定静态形状。 |
| 外部 ABI | 静态连续张量、六种张量类型、指针与按值标量、固定普通 launch；没有非空 scratch 或描述符启动支持。 |
| 副作用 | `ask.semantics` 只接受 `effects = "none"`，没有状态 token。 |
| 互操作 | CAI 不承载 `bf16`，没有 DLPack 接口。 |
