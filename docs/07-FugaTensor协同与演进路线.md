# 07. FugaTensor 协同与演进路线

返回核心思想：[框架分工](核心思想.md#ask-positioning) · [验证与支持边界](核心思想.md#ask-testing-roadmap) · 相关细节：[ASK IR](01-ASK-IR与数值语义.md) · [外部执行接口](04-外部产物与执行接口.md) · [FugaTensor 架构](../../fugatensor/docs/architecture/README.md)

<a id="division-of-labor"></a>

## 1. 两个仓库的职责

ASK 独立交付 C++ SDK，FugaTensor 构建时链接 `ASK::Compiler` 与 `ASK::Runtime`，并将共享库装入自己的 wheel。Python 用户 API 和原生绑定只有 FugaTensor 一份；ASK 不依赖 FugaTensor。消费这两个 SDK 组件不读取 ASK 源码或 LLVM SDK。安装方式见 [ASK 安装](安装.md#framework-install)与 [FugaTensor 安装](../../fugatensor/docs/安装.md)。

| 关注点 | FugaTensor | ASK |
| --- | --- | --- |
| 用户 API 与微分 | Tensor、自动组图、`ft.fun`、前向/反向规则与求值触发。 | 静态图接口与编译执行。 |
| 控制流与 IR | Fuga IR 表达区域和函数边界；静态展开或按需执行后提交 ASK 图。 | 单基本块、形状具体的 MLIR `ask` 图，自有计算降级到 LLVM。 |
| 外部编译器 | 保存 DSL 源码，维护 Triton worker、依赖锁和编译选项；调用 worker、校验产物与维护磁盘缓存。 | 维护请求/产物协议，生成标准化请求，检查原生启动 ABI。 |
| 执行 | 选择后端、管理上层 Tensor 引用与求值边界，经薄原生桥提交完整计划。 | 生成 CPU/CUDA 自有代码、切分计划、管理显存与事件、单 stream 提交和 CUDA Graph replay。 |
| 溯源 | 输出 `fuga-to-ask.json` 映射，见 [降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)。 | 保留稳定节点/值身份，提取 `ask.mark` 并关联分区报告。 |

<a id="implemented-scope"></a>

## 2. 当前已实现范围

- 六种显式类型、单步 RNE 常量和 6×6 转换；CPU/GPU 目标支持时使用原生 `bf16`。
- C++ `ASK::Compiler`、`ASK::Runtime`、`ASK::Core` 与 `ask-opt`；形状推导、溯源提取、自有常量折叠、DCE 和逐元素融合。
- 自有 Host 共享循环 `.o` 与 CUDA cubin 代码生成；多端口、0-D 张量和空张量。
- 外部 cubin/PTX 接入与协议 v2；调用方运行 worker，每个外部节点是独立计划步骤。
- 原生常驻会话、设备缓冲区、整计划提交、异步事件与资源复用、CUDA Graph replay；FugaTensor 在此基础上提供 CAI 借用、磁盘缓存和可迁移多 kernel 编译包。

复杂外部算法由上层实现。例如 [FlashAttention-2](../../fugatensor/examples/flash_attention2.py) 的算法、梯度和适用条件由 FugaTensor 维护，ASK 只处理其端口、编译产物和步骤依赖。

<a id="roadmap"></a>

## 3. 当前限制

| 边界 | 当前行为 |
| --- | --- |
| 第三方融合 | 不合并第三方与 ASK 自有代码，也不合并多个外部 kernel；跨步骤通过显存连接。 |
| 调度 | 静态计划、单 stream 顺序提交或 Graph replay；各未完成提交拥有独立 workspace，没有图内动态条件/循环或多 stream 并发计划。 |
| 操作集 | 自有算子没有归约、轴广播或矩阵乘；`splat` 仅将标量填充到指定静态形状。 |
| 外部 ABI | 静态连续张量、六种张量类型、指针与按值标量、固定普通 launch；没有非空 scratch 或描述符启动支持。 |
| 副作用 | `ask.semantics` 只接受 `effects = "none"`，没有状态 token。 |
| 互操作 | CAI 不承载 `bf16`，没有 DLPack 接口。 |

这些是实现边界，不代表已排定的发布计划。
