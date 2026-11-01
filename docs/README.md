# ASK 文档目录

ASK 是面向张量框架的底层编译与运行时引擎：上层以 MLIR `ask` dialect 提交静态计算图和可选的外部 DSL（如 Triton），ASK 完成语义与形状检查、基础图优化、原生 LLVM IR 降级与受支持的跨编译器值边界融合。CPU 目标文件由宿主链接调用，CUDA `cubin` 由常驻设备会话执行。

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| [独立安装](安装.md) | wheel 安装、任意路径源码构建、独立 CUDA 工具与包内 worker；不要求相邻源码仓库 | `pyproject.toml`、`tools/fetch_deps.py` |
| [核心思想](核心思想.md) | ASK 定位、双层 IR 分工、端到端编译执行流水线 | 全局架构 |
| [01. ASK IR 与数值语义](01-ASK-IR与数值语义.md) | `ask` dialect 结构、6 种标量/张量类型、单步 RNE 常量、`ask.cast` 转换规则、形状推导与 `ask.mark` / `ask.external` | `include/ask/IR/`、`lib/IR/`、`lib/Frontend/Prepare.cpp` |
| [02. 前端接入与构建配置](02-前端接入与构建配置.md) | 依赖拉取、CMake Preset、可选适配器白名单、Python `nanobind` SDK、C++ `ASK::Core` 与 `ask-opt` CLI | `tools/fetch_deps.py`、`CMakeLists.txt`、`python/ask/`、`bindings/` |
| [03. 图优化与自有代码生成](03-图优化与自有代码生成.md) | 常量/类型折叠、死代码消除、多输出逐元素融合组、Host 共享循环降级、自有 CUDA 降级与 LLVM IR 清理 | `lib/Transforms/`、`lib/Lowering/`、`lib/LLIR/Optimize.cpp` |
| [04. 外部编译器适配与边界内联](04-外部编译器适配与边界内联.md) | `adapters/<id>/` 插件机制、导出协议与 ABI v2、通用 LLVM 值边界内联器、Triton `P → F → Q` 融合与 FMA 隔离 | `adapters/`、`lib/Adapters/`、`lib/LLIR/Inline.cpp`、`lib/LLIR/Fuse.cpp` |
| [05. 常驻运行时与多级缓存](05-常驻运行时与多级缓存.md) | 动态加载 CUDA Driver、`ExecutionSession`、设备张量与 `__cuda_array_interface__` 零拷贝、异步事件、IR / `cubin` 双层持久缓存 | `lib/Runtime/CUDA.cpp`、`bindings/runtime.cpp`、`python/ask/gpu.py` |
| [06. 工程结构与测试体系](06-工程结构与测试体系.md) | 仓库目录职责、5 个 CTest 测试集、共享多精度测试矩阵与独立数值参考实现 | `tests/`、`adapters/triton/tests/` |
| [07. FugaTensor 协同与演进路线](07-FugaTensor协同与演进路线.md) | 与上层框架 [FugaTensor](../../fugatensor/docs/README.md) 的职责边界、已实现能力清单与当前限制 | `ask` ↔ `fugatensor` |

## 关联仓库文档

以下相对链接用于联合浏览源码文档，安装和执行 ASK 不依赖这些目录。

- [FugaTensor 文档目录](../../fugatensor/docs/README.md)
- [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)
- [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)
