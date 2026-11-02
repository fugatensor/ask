# ASK 文档目录

ASK 接收静态张量图，编译自有 CPU/CUDA 计算，并调度调用方提供的外部 cubin/PTX。自有区域内部执行逐元素融合，外部节点保持独立。

| 文档 | 内容 | 对应实现 |
| --- | --- | --- |
| [独立安装](安装.md) | wheel、源码构建、独立 CUDA 工具和调用方 worker | `pyproject.toml`、`tools/fetch_deps.py` |
| [用户 API](用户API.md) | Python 构图、编译、会话执行、外部标量/多输出和产物保存 | `python/ask/` |
| [核心思想](核心思想.md) | 框架分工、IR 边界和执行流程 | 全局架构 |
| [01. ASK IR 与数值语义](01-ASK-IR与数值语义.md) | 显式类型、RNE、cast、形状与外部声明 | `lib/IR/`、`lib/Frontend/Prepare.cpp` |
| [02. 前端接入与构建配置](02-前端接入与构建配置.md) | core 构建、Python/C++/CLI、版本与分发 | `CMakeLists.txt`、`bindings/`、`python/ask/compiler.py` |
| [03. 图优化与自有代码生成](03-图优化与自有代码生成.md) | 折叠、DCE、自有融合、Host/CUDA 降级 | `lib/Transforms/`、`lib/Lowering/` |
| [04. 外部产物与执行接口](04-外部产物与执行接口.md) | cubin/PTX、端口 ABI 和 worker 配置 | `python/ask/adapters.py`、`python/ask/cuda.py` |
| [05. 常驻运行时与多级缓存](05-常驻运行时与多级缓存.md) | 会话、CAI、异步事件与缓存 | `lib/Runtime/`、`python/ask/runtime.py`、`python/ask/cache.py` |
| [06. 工程结构与测试体系](06-工程结构与测试体系.md) | 目录职责、两个 CTest 目标、数值参考和 CI | `tests/`、`.github/workflows/ci.yml` |
| [07. FugaTensor 协同与演进路线](07-FugaTensor协同与演进路线.md) | 两仓分工和当前支持边界 | ASK 与 FugaTensor |
| [08. 图切分与多 Kernel 执行计划](08-图切分与多Kernel执行计划.md) | owned/external 分区、JSON v1、编译包和调度 | `lib/Frontend/Partition.cpp`、`python/ask/plan.py` |
| [09. 外部编译器适配接入规范](09-外部编译器适配接入规范.md) | 通用 worker 请求/响应协议 v2 与缓存身份 | `python/ask/adapters.py`、`schemas/` |

## 关联仓库文档

以下相对链接用于联合浏览源码文档，ASK 构建和运行不依赖这些目录。

- [FugaTensor 文档目录](../../fugatensor/docs/README.md)
- [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)
- [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)
