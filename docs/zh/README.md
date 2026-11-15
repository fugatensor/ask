# ASK 文档目录

按安装、API、编译和运行时查找文档。

| 文档 | 内容 | 对应实现 |
| --- | --- | --- |
| [独立安装](install.md) | CMake SDK 安装、组件消费与 CUDA 工具 | `CMakeLists.txt`、`tools/fetch_deps.py` |
| [用户 API](api.md) | C++ 编译接口、CLI、执行计划与缓冲区 | `include/ask/Frontend/`、`include/ask/Runtime/` |
| [核心思想](concepts.md) | 框架分工、IR 和执行流程 | 全局架构 |
| [ASK IR 与数值语义](ir.md) | 显式类型、RNE、cast、形状与外部声明 | `lib/IR/`、`lib/Frontend/Prepare.cpp` |
| [前端接入与构建配置](build.md) | Compiler/Runtime/Core 组件、构建与版本契约 | `CMakeLists.txt`、`cmake/` |
| [图优化与自有代码生成](codegen.md) | 折叠、DCE、自有融合、Host/CUDA 降级 | `lib/Transforms/`、`lib/Lowering/` |
| [外部产物与执行接口](artifacts.md) | cubin/PTX、物理 ABI 和 worker 职责 | `schemas/`、`lib/Runtime/` |
| [常驻运行时与多级缓存](runtime.md) | Session、原生计划、资源复用、Graph replay 与调用方缓存 | `lib/Runtime/` |
| [工程结构与测试体系](testing.md) | C++ 工程、CTest 与 SDK 验证 | `tests/cpp/`、`tests/ir/` |
| [图切分与多 Kernel 执行计划](partitioning.md) | owned/external/library 分区、JSON v1 与原生调度 | `lib/Frontend/Partition.cpp`、`lib/Runtime/` |
| [ASK PASS 执行计划静态校验](verification.md) | 计划可用性、整份槽位映射与构建依赖 | `lib/Pass/` |
| [外部编译器适配接入规范](compilers.md) | 通用 worker 请求/响应协议 v2 与缓存身份 | `schemas/` |
| [原语与矩阵乘法验证报告](matmul_report.md) | 原语、融合、cuBLASLt、数值矩阵与分层性能测量 | `lib/`、`tests/cpp/` |
| [依赖调度与临时区复用验证报告](workspace_report.md) | CUDA Graph 分支并行与 CPU 物化临时区复用实测 | `lib/Pass/`、`lib/Runtime/`、`lib/Lowering/` |
