# ASK 文档目录

按安装、API、编译和运行时查找文档。

| 文档 | 内容 | 对应实现 |
| --- | --- | --- |
| [独立安装](安装.md) | CMake SDK 安装、组件消费与 CUDA 工具 | `CMakeLists.txt`、`tools/fetch_deps.py` |
| [用户 API](用户API.md) | C++ 编译接口、CLI、执行计划与缓冲区 | `include/ask/Frontend/`、`include/ask/Runtime/` |
| [核心思想](核心思想.md) | 框架分工、IR 和执行流程 | 全局架构 |
| [ASK IR 与数值语义](ASK-IR与数值语义.md) | 显式类型、RNE、cast、形状与外部声明 | `lib/IR/`、`lib/Frontend/Prepare.cpp` |
| [前端接入与构建配置](前端接入与构建配置.md) | Compiler/Runtime/Core 组件、构建与版本契约 | `CMakeLists.txt`、`cmake/` |
| [图优化与自有代码生成](图优化与自有代码生成.md) | 折叠、DCE、自有融合、Host/CUDA 降级 | `lib/Transforms/`、`lib/Lowering/` |
| [外部产物与执行接口](外部产物与执行接口.md) | cubin/PTX、物理 ABI 和 worker 职责 | `schemas/`、`lib/Runtime/` |
| [常驻运行时与多级缓存](常驻运行时与多级缓存.md) | Session、原生计划、资源复用、Graph replay 与调用方缓存 | `lib/Runtime/` |
| [工程结构与测试体系](工程结构与测试体系.md) | C++ 工程、CTest 和上层集成测试边界 | `tests/cpp/`、`tests/ir/` |
| [FugaTensor 协同与演进路线](FugaTensor协同与演进路线.md) | 两仓职责和当前限制 | ASK 与 FugaTensor |
| [图切分与多 Kernel 执行计划](图切分与多Kernel执行计划.md) | owned/external 分区、JSON v1 与原生调度 | `lib/Frontend/Partition.cpp`、`lib/Runtime/` |
| [SUKA 规则证明与计划校验](规则证明与计划校验.md) | 计划可用性证明、显存复用边界与构建依赖 | `lib/Runtime/`、`cmake/suka.lock.json` |
| [外部编译器适配接入规范](外部编译器适配接入规范.md) | 通用 worker 请求/响应协议 v2 与缓存身份 | `schemas/` |

## 关联仓库文档

- [FugaTensor 文档目录](../../fugatensor/docs/README.md)
- [FugaTensor 架构总览](../../fugatensor/docs/architecture/README.md)
- [Fuga IR 降级到 ASK](../../fugatensor/docs/ir/降级到ASK.md)
