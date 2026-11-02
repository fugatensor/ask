# ASK 文档目录

ASK 提供 C++ 编译 SDK、原生 CUDA 执行计划与 CLI。上层框架负责用户 API、图语义、第三方 worker 调用和磁盘缓存；自有区域内部融合，外部 kernel 独立调度。

| 文档 | 内容 | 对应实现 |
| --- | --- | --- |
| [独立安装](安装.md) | CMake SDK 安装、组件消费与 CUDA 工具 | `CMakeLists.txt`、`tools/fetch_deps.py` |
| [用户 API](用户API.md) | C++ 编译门面、CLI、原生计划与缓冲区 | `include/ask/Frontend/`、`include/ask/Runtime/` |
| [核心思想](核心思想.md) | 框架分工、IR 边界和执行流程 | 全局架构 |
| [01. ASK IR 与数值语义](01-ASK-IR与数值语义.md) | 显式类型、RNE、cast、形状与外部声明 | `lib/IR/`、`lib/Frontend/Prepare.cpp` |
| [02. 前端接入与构建配置](02-前端接入与构建配置.md) | Compiler/Runtime/Core 组件、构建与版本契约 | `CMakeLists.txt`、`cmake/` |
| [03. 图优化与自有代码生成](03-图优化与自有代码生成.md) | 折叠、DCE、自有融合、Host/CUDA 降级 | `lib/Transforms/`、`lib/Lowering/` |
| [04. 外部产物与执行接口](04-外部产物与执行接口.md) | cubin/PTX、物理 ABI 和 worker 职责 | `schemas/`、`lib/Runtime/` |
| [05. 常驻运行时与多级缓存](05-常驻运行时与多级缓存.md) | Session、原生计划、资源复用、Graph replay 与调用方缓存 | `lib/Runtime/` |
| [06. 工程结构与测试体系](06-工程结构与测试体系.md) | C++ 工程、CTest 和上层集成测试边界 | `tests/cpp/`、`tests/ir/` |
| [07. FugaTensor 协同与演进路线](07-FugaTensor协同与演进路线.md) | 两仓职责和当前支持边界 | ASK 与 FugaTensor |
| [08. 图切分与多 Kernel 执行计划](08-图切分与多Kernel执行计划.md) | owned/external 分区、JSON v1 与原生调度 | `lib/Frontend/Partition.cpp`、`lib/Runtime/` |
| [09. 外部编译器适配接入规范](09-外部编译器适配接入规范.md) | 通用 worker 请求/响应协议 v2 与缓存身份 | `schemas/` |

## 关联仓库文档

以下相对链接用于联合浏览源码文档，不构成 ASK 的构建依赖。

- [FugaTensor 文档目录](../../fugatensor/docs/README.md)
- [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)
- [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/03-降级到ASK.md)
