# 07. FugaTensor 协同与演进路线

返回核心思想：[与上层框架分工](核心思想.md#ask-positioning) · [验证与演进边界](核心思想.md#ask-testing-roadmap) · 相关细节：[ASK IR 与数值语义](01-ASK-IR与数值语义.md) · [外部编译器适配与边界内联](04-外部编译器适配与边界内联.md) · [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)

ASK 并不是孤立存在的玩具编译器，它已经作为唯一算术执行后端驱动上层张量框架 [FugaTensor](../../fugatensor/docs/README.md) 运行端到端的前向 JIT、动态控制流切分、一阶与高阶自动微分以及用户手写 Triton 融合算子。本章总结两者的协同界面、当前已经跑通的能力清单，以及接下来明确规划的演进方向。

<a id="division-of-labor"></a>

## 1. `FugaTensor` 与 `ASK` 的协同界面

两个仓库遵循**单向依赖、职责正交**的设计（`FugaTensor` 依赖安装后的 `ask`，`ask` 不依赖也不感知 `FugaTensor`）：

| 关注点 | FugaTensor（上层张量框架） | ASK（底层统一编译与运行时引擎） |
| --- | --- | --- |
| **用户 API 与张量语义** | 提供 `ft.tensor`、`@ft.jit`、`ft.fun`、`ft.triton.kernel`、`requires_grad` / `.backward()` / `.grad`。 | 不暴露机器学习层面的自动微分或模型 API，专注接收静态计算图并高效执行。 |
| **中间表示（IR）** | 拥有完全独立于 MLIR/ASK 的 [Fuga IR](../../fugatensor/docs/ir/README.md)，表达结构化区域（`if`/`for`/`while`）、自定义函数边界（`fuga.call`）与前向/反向微分变换。 | 拥有 MLIR `ask` dialect（高层静态图）与原生 `llvm::Module`（低层代码生成与跨编译器内联）。 |
| **控制流处理** | 静态 `if`/`for` 在 Fuga IR 层展开；运行时标量 `bool` 决定的动态 `if`/`while` 通过 [Graph Break](../../fugatensor/docs/architecture/01-张量模型与追踪机制.md#graph-break) 切成多个静态片段。 | 每次接收一个已展开、形状具体的单基本块静态计算图，配合双层缓存毫秒级复用已编译片段。 |
| **跨层溯源与外部 DSL** | 原样保存用户 `@triton.jit` 源码快照，降级时输出 `fuga-to-ask.json` 映射表（见 [降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)），并延迟到实际 GPU 编译前才绑定物理显卡 `sm_xx` 架构号。 | 校验 `ask.external` 语义与形状，驱动 Triton worker 导出原生 LLVM IR，执行 `P → F → Q` 值边界融合或自有 CPU/CUDA 降级。 |
| **算术执行铁律** | Python 层与 C++ `_native` 图分析层**绝不执行任何张量元素算术**（哪怕 `a + b` 也走 JIT 编译）。 | 承担 100% 的 CPU 目标文件执行与常驻 CUDA `ExecutionSession` kernel 执行。 |

<a id="implemented-scope"></a>

## 2. 当前已落地的能力边界

为了确保文档与代码完全一致，下面明确列出当前仓库已经实现并通过自动化测试验证的能力：

- **数据类型与转换**：`bool`、`i32`、`i64`、`f32`、`f16`、`bf16` 六种标量与连续行主序张量；单步 RNE 常量编码；全部 6×6 `ask.cast` 方向（含浮点到整数饱和截断与硬件 `bf16` 能力门禁）。
- **前端与高层优化**：进程内 `nanobind` `ask.Compiler()`、C++ `ASK::Core` 与 `ask-opt` CLI；形状推导与 `ask.mark` 提取擦除；严格保精度的常量/类型折叠、死代码消除与多输出逐元素融合组规划。
- **自有代码生成**：Host CPU 共享循环 LLVM IR / `.o` 代码生成；自有 CUDA `nvptx64` 多输入多输出异构端口代码生成（支持零维标量、零元素空张量与 32/64 位自适应索引，`core` 构建开箱可用）。
- **外部编译器融合**：插件式 `adapters/<id>/` 注册机制；Triton 3.7.1 导出协议 v1 与执行契约 ABI v2；通用 C++ LLVM 值边界内联核心（支持多参数、多站点、定长向量与失败回滚）；单输入单输出静态逐元素 `ASK → Triton → ASK` 跨编译器融合及指令级 FMA `contract` 隔离。
- **运行时与缓存**：动态加载 `libcuda.so.1` 的常驻 `ExecutionSession`、非阻塞 stream、异步 `Event`、`__cuda_array_interface__` v2/v3 零拷贝接入（含跨流同步与宿主对象保活），以及跨进程持久化的 IR / `cubin` 双层缓存。

<a id="roadmap"></a>

## 3. 后续演进方向

当前架构已经把最难啃的“双层 IR + 原生 LLVM 值边界内联 + 常驻异步会话”主干打通，后续能力可以在不推翻现有边界的前提下渐进扩展：

1. **多区域混合 DAG 图调度**：目前单张 CUDA 编译图要么是纯自有算子图（支持任意多输入多输出），要么是围绕单个 `ask.external` 的单输入单输出融合链。下一步是在运行时层引入多 kernel `RegionPlan` 执行计划，使一张大图里包含多个无法合并的外部算子（如连续两个 MatMul 或带屏障的规约）时，能由 ASK 自动分配中间显存并在同一 `ExecutionSession` stream 上按拓扑依赖连续派发。
2. **归约、广播与矩阵乘基础算子**：在 `ask` dialect 与自有 CPU/CUDA 降级中补充 `sum` / `max` 归约与通用轴广播，使上层自动微分的标量广播反向（`splat` VJP）和常用归一化层可以直接落到自有后端。
3. **符号维度（Symbolic Shapes）**：将准备阶段的静态形状检查扩展为带约束的符号维度传递，减少上层框架在 batch size 变化时的重编译次数。
4. **更多互操作协议与外部适配器**：在 `ExecutionSession` 中补齐含 `bf16` 在内的 DLPack 零拷贝导入导出；在 `adapters/` 下按同一套 `CompilerAdapter` + ABI v2 契约接入更多能够导出原生 LLVM IR 的 DSL 或 MLIR 编译器。
