# 07. FugaTensor 协同与演进路线

返回核心思想：[与上层框架分工](核心思想.md#ask-positioning) · [验证与演进边界](核心思想.md#ask-testing-roadmap) · 相关细节：[ASK IR 与数值语义](01-ASK-IR与数值语义.md) · [外部编译器适配与边界内联](04-外部编译器适配与边界内联.md) · [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)

ASK 作为底层算术执行后端驱动上层张量框架 [FugaTensor](../../fugatensor/docs/README.md) 运行前向 JIT、动态控制流切分、一阶与高阶自动微分以及用户手写 Triton 融合算子。

<a id="division-of-labor"></a>

## 1. `FugaTensor` 与 `ASK` 的职责划分

两个仓库保持单向依赖：`FugaTensor` 依赖安装后的 `ask`，`ask` 不依赖 `FugaTensor`。

| 关注点 | FugaTensor（上层张量框架） | ASK（底层编译与运行时引擎） |
| --- | --- | --- |
| **用户 API** | 提供 `ft.tensor`、`@ft.jit`、`ft.fun`、`ft.triton.kernel`、`requires_grad` / `.backward()` / `.grad`。 | 接收静态计算图并编译执行，不涉及自动微分或模型层 API。 |
| **中间表示（IR）** | 使用独立于 MLIR/ASK 的 [Fuga IR](../../fugatensor/docs/ir/README.md)，表达结构化区域（`if`/`for`/`while`）、自定义函数边界（`fuga.call`）与前向/反向微分变换。 | 使用 MLIR `ask` dialect（高层静态图）与原生 `llvm::Module`（低层代码生成与跨编译器内联）。 |
| **控制流** | 静态 `if`/`for` 在 Fuga IR 层化简展开；依赖运行时标量 `bool` 的 `if`/`while` 通过 [Graph Break](../../fugatensor/docs/architecture/01-张量模型与追踪机制.md#graph-break) 切分为多个静态片段。 | 每次接收已展开、形状具体的单基本块静态计算图，通过双层缓存复用已编译片段。 |
| **跨层溯源与外部 DSL** | 保存用户 `@triton.jit` 源码快照，降级时输出 `fuga-to-ask.json` 映射表（见 [降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)），在实际 GPU 编译前绑定物理显卡 `sm_xx` 架构号。 | 校验 `ask.external` 语义与形状，调用 Triton worker 导出原生 LLVM IR，执行 `P → F → Q` 值边界融合或自有 CPU/CUDA 降级。 |
| **算术执行** | Python 层与 C++ `_native` 图分析层不执行张量元素算术。 | 执行全部 CPU 目标代码与常驻 CUDA `ExecutionSession` kernel。 |

<a id="implemented-scope"></a>

## 2. 当前已实现范围

- **数据类型与转换**：`bool`、`i32`、`i64`、`f32`、`f16`、`bf16` 六种标量与连续行主序张量；单步 RNE 常量编码；6×6 `ask.cast`（含浮点到整数饱和截断与硬件 `bf16` 能力检查）。
- **前端与高层优化**：进程内 `nanobind` `ask.Compiler()`、C++ `ASK::Core` 与 `ask-opt` CLI；形状推导与 `ask.mark` 提取擦除；常量/类型折叠、死代码消除与多输出逐元素融合组规划。
- **自有代码生成**：Host CPU 共享循环 LLVM IR / `.o` 代码生成；自有 CUDA `nvptx64` 多输入多输出异构端口代码生成（支持 0-D 标量、空张量与 32/64 位索引，包含在 `core` 构建中）。
- **外部编译器融合**：`adapters/<id>/` 注册机制；Triton 3.7.1 导出协议 v1 与执行契约 ABI v2；通用 C++ LLVM 值边界内联核心（支持多参数、多站点、定长向量与失败回滚）；单输入单输出静态逐元素 `ASK → Triton → ASK` 融合及指令级 FMA `contract` 隔离。
- **运行时与缓存**：动态加载 `libcuda.so.1` 的常驻 `ExecutionSession`、非阻塞 stream、异步 `Event`、`__cuda_array_interface__` v2/v3 零拷贝接入（含跨流同步与宿主对象保活），以及跨进程持久化的 IR / `cubin` 双层缓存。

<a id="roadmap"></a>

## 3. 后续演进方向

1. **多区域混合 DAG 图调度**：当前单张 CUDA 编译图仅支持纯自有算子图或围绕单个 `ask.external` 的单输入单输出融合链。后续可在运行时引入多 kernel 执行计划，对包含多个不可融合外部算子的大图自动分配中间显存并在同一 stream 上按拓扑序派发。
2. **归约、广播与矩阵乘算子**：在 `ask` dialect 与自有 CPU/CUDA 降级中增加 `sum` / `max` 归约与轴广播，支撑上层广播反向（`splat` VJP）与常见归一化计算。
3. **符号维度（Symbolic Shapes）**：将准备阶段的静态形状检查扩展为带约束的符号维度传递，减少动态 batch size 场景的重编译次数。
4. **互操作协议与外部适配器扩展**：在 `ExecutionSession` 中补齐含 `bf16` 的 DLPack 零拷贝导入导出；在 `adapters/` 下按 `CompilerAdapter` + ABI v2 契约接入更多可导出原生 LLVM IR 的编译器。
