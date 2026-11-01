# 07. FugaTensor 协同与演进路线

返回核心思想：[与上层框架分工](核心思想.md#ask-positioning) · [验证与支持边界](核心思想.md#ask-testing-roadmap) · 相关细节：[ASK IR 与数值语义](01-ASK-IR与数值语义.md) · [外部编译器适配与边界内联](04-外部编译器适配与边界内联.md) · [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)

ASK 是上层张量框架 [FugaTensor](../../fugatensor/docs/README.md) 使用的编译与算术执行后端。Tensor 自动组图、Python 标量条件的执行边界和自动微分属于 FugaTensor；读取结果、条件判断、反向传播或显式求值时，FugaTensor 将所需依赖合并为静态图交给 ASK。本文记录当前接入方式与支持边界，保留原文件名和 `roadmap` 锚点供已有链接使用。

<a id="division-of-labor"></a>

## 1. `FugaTensor` 与 `ASK` 的职责划分

两个仓库保持单向依赖：`FugaTensor` 必须依赖安装后的 `ask`，`ask` 不依赖 `FugaTensor`。构建边界也分开：ASK 独立产出编译器与运行时包，FugaTensor 独立构建自己的原生图分析扩展，再在运行环境中安装两个兼容的包；FugaTensor 构建不读取 ASK 源码或 LLVM SDK。安装步骤见 [ASK 独立安装](安装.md#framework-install) 与 [FugaTensor 独立安装](../../fugatensor/docs/安装.md)。

| 关注点 | FugaTensor（上层张量框架） | ASK（底层编译与运行时引擎） |
| --- | --- | --- |
| **用户 API** | 提供 `ft.tensor`、`ft.evaluate`、`ft.fun`、`ft.triton.kernel`、`requires_grad` / `.backward()` / `.grad`。 | 接收静态计算图并编译执行，不涉及自动微分或模型层 API。 |
| **中间表示（IR）** | 使用独立于 MLIR/ASK 的 [Fuga IR](../../fugatensor/docs/ir/README.md)，表达结构化区域（`if`/`for`/`while`）、自定义函数边界（`fuga.call`）与前向/反向微分变换。 | 使用 MLIR `ask` dialect（高层静态图）与原生 `llvm::Module`（低层代码生成与跨编译器内联）。 |
| **控制流** | 静态 `if`/`for` 在 Fuga IR 层化简展开；普通 Python 的 `if`/`while` 在读取 0-D Tensor 真值时触发[按需执行](../../fugatensor/docs/architecture/01-张量模型与追踪机制.md#graph-break)，再按实际分支继续组图；用户无需 JIT 装饰器。 | 每次接收已展开、形状具体的单基本块静态计算图，通过双层缓存复用已编译片段。 |
| **跨层溯源与外部 DSL** | 保存用户 `@triton.jit` 源码快照，降级时输出 `fuga-to-ask.json` 映射表（见 [降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)），在实际 GPU 编译前绑定物理显卡 `sm_xx` 架构号。 | 校验 `ask.external` 语义与形状，调用 Triton worker 导出原生 LLVM IR，执行 `P → F → Q` 值边界融合或自有 CPU/CUDA 降级。 |
| **算术执行** | Python 层与 C++ `_native` 图分析层不执行张量元素算术。 | 执行全部 CPU 目标代码与常驻 CUDA `ExecutionSession` kernel。 |

<a id="implemented-scope"></a>

## 2. 当前已实现范围

- **数据类型与转换**：`bool`、`i32`、`i64`、`f32`、`f16`、`bf16` 六种标量与连续行主序张量；单步 RNE 常量编码；6×6 `ask.cast`（含浮点到整数饱和截断与硬件 `bf16` 能力检查）。
- **前端与高层优化**：进程内 `nanobind` `ask.Compiler()`、C++ `ASK::Core` 与 `ask-opt` CLI；形状推导与 `ask.mark` 提取擦除；常量/类型折叠、死代码消除与多输出逐元素融合组规划。
- **自有代码生成**：Host CPU 共享循环 LLVM IR / `.o` 代码生成；自有 CUDA `nvptx64` 多输入多输出异构端口代码生成（支持 0-D 标量、空张量与 64 位 grid-stride 循环，包含在 `core` 构建中）。
- **外部编译器融合**：`adapters/<id>/` 注册机制；Triton 3.7.1 导出协议 v1 与执行契约 ABI v2；通用 C++ LLVM 值边界内联核心（支持多参数、多站点、定长向量与失败回滚）；单输入单输出静态逐元素 `ASK → Triton → ASK` 融合及指令级 FMA `contract` 隔离。
- **运行时与缓存**：动态加载 `libcuda.so.1` 的常驻 `ExecutionSession`、非阻塞 stream、异步 `Event`、五种非 `bf16` 类型的 `__cuda_array_interface__` v2/v3 零拷贝接入（含生产者流依赖与宿主对象保活），以及跨进程持久化的 IR / `cubin` 双层缓存。

<a id="roadmap"></a>

## 3. 当前未支持范围

| 边界 | 当前行为 |
| --- | --- |
| 多外部节点或不可融合混合图 | CUDA 图编译要求纯自有图，或一个外部节点构成的受支持单输入单输出融合图。其它混合图返回错误，不生成多 kernel 执行计划，也不自动分配跨 kernel 中间缓冲区。 |
| 归约、轴广播与矩阵乘 | `ask` 操作集中没有 `sum`、`max`、轴广播或矩阵乘。`ask.splat` 只将标量填充成给定静态形状，不是通用广播。 |
| 符号形状与图内动态控制流 | 函数输入须提供具体形状；输出的未定维度只能由现有规则细化为静态值。ASK 不执行运行时张量条件分支、循环或跨函数调用，上层须先静态展开或分段。 |
| 外部算子副作用 | 语义声明仅接受 `effects = "none"`，没有状态 token 或有副作用节点的调度语义。 |
| 互操作与适配器 | 没有 DLPack 接口；CAI 不承载 `bf16`。注册表当前只有 Triton，添加其它编译器仍需实现并注册相应适配器。 |

这些是现有实现的限制，不代表已排定的发布计划。
