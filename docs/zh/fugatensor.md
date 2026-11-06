返回核心思想：[框架分工](concepts.md#ask-positioning) · [测试验证](concepts.md#ask-testing-roadmap) · 相关细节：[ASK IR](ir.md) · [外部执行接口](artifacts.md) · [FugaTensor 架构](../../../fugatensor/docs/zh/architecture/README.md)

# FugaTensor 与 ASK 的职责

<a id="division-of-labor"></a>

## 1. 框架与编译运行时

推荐以 [FugaTensor](../../../fugatensor/README.zh.md) 作为 ASK 接入示例。它提供 [Python API](../../../fugatensor/docs/zh/API/README.md)，扩展链接 ASK C++ 库，wheel 携带编译器和运行时共享库。构建步骤见 [ASK 安装](install.md#framework-install)和 [FugaTensor 安装](../../../fugatensor/docs/zh/install.md)。

| 关注点 | FugaTensor | ASK |
| --- | --- | --- |
| 用户 API 与微分 | Tensor、自动组图、`ft.grad` 纯 IR 变换、`ft.fun` 前向与 VJP 规则、求值触发。 | 静态图接口与编译执行。 |
| 控制流与 IR | Fuga IR 表达区域和函数边界；静态展开或按需执行后提交 ASK 图。 | 单基本块、形状具体的 MLIR `ask` 图，自有计算降级到 LLVM。 |
| 外部编译器 | 保存源码和编译配置，运行 worker，校验并缓存产物。 | 生成编译请求，检查产物和启动参数。 |
| 执行 | 选择后端，管理 Tensor 引用，触发求值并提交计划。 | 生成自有代码，切分计划，管理显存、事件和 CUDA Graph。 |
| 溯源 | 输出 `fuga-to-ask.json` 映射，见 [降级到 ASK](../../../fugatensor/docs/zh/ir/lowering.md)。 | 保留稳定节点/值身份，提取 `ask.mark` 并关联分区报告。 |

停止梯度由 Fuga IR 的 `fuga.stop_gradient` 表达，JVP/VJP 在该节点停止传播；[降级到 ASK](../../../fugatensor/docs/zh/ir/lowering.md#type-and-op-mapping)时直接使用输入值，ASK 执行保留的数值依赖。

<a id="implemented-scope"></a>

## 2. 编译与执行能力

- 六种显式类型、单步 RNE 常量和 6×6 转换；Host 执行使用 `bool/i32/i64/f32/f16`，`bf16` 执行使用 CUDA `sm_80+`。
- C++ `ASK::Compiler`、`ASK::Runtime`、`ASK::Core` 与 `ask-opt`；形状推导、溯源提取、自有常量折叠、DCE 和逐元素融合。
- 自有 Host 共享循环 `.o` 与 CUDA cubin 代码生成；多端口、0-D 张量和空张量。
- 广播、转置、归约、条件选择及浮点函数；CUDA 矩阵乘法使用 cuBLASLt，CPU f32 使用 OpenBLAS，f16 使用自有代码生成。
- 外部 cubin/PTX 接入与协议 v2；调用方运行 worker，每个外部节点是独立计划步骤。
- 原生常驻会话、设备缓冲区、整计划提交、异步事件与资源复用、CUDA Graph replay；FugaTensor 在此基础上提供 CAI 借用、磁盘缓存和可迁移多 kernel 编译包。

[attention](../../../fugatensor/docs/zh/API/attention.md) 是一个例子：FugaTensor 提供算法和反向规则，ASK 调度对应的 kernel。

<a id="roadmap"></a>

## 3. 构建依赖方向

| 项目 | 提供的接口 | 构建时消费 |
| --- | --- | --- |
| FugaTensor | Python Tensor API、Fuga IR、自动微分、第三方 worker | ASK 安装包中的 `Compiler`、`Runtime`，以及 nanobind |
| ASK | C++ 编译器、设备运行时、执行计划校验 | LLVM/MLIR、Host OpenBLAS SDK；CUDA matmul 可选 cuBLASLt 头文件 |

FugaTensor 先通过 Fuga IR 完成图变换，再向 ASK 提交静态图。ASK 构建计划，通过自有 C++ 校验器检查缓冲区与存储槽位。Tensor 与微分规则由 FugaTensor 维护，执行计划与槽位约束由 ASK 维护。

先安装 ASK SDK，再用 `CMAKE_PREFIX_PATH` 构建 FugaTensor。接口和依赖详情见[前端接入](build.md#native-compiler)与[计划校验构建](verification.md#build-dependency)。
