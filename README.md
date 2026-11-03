# ASK

[![Version](https://img.shields.io/badge/version-0.1.0-2563eb)](VERSION)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599c)](docs/安装.md#requirements)
[![LLVM 23.1.x](https://img.shields.io/badge/LLVM-23.1.x-64748b)](docs/前端接入与构建配置.md#version-contract)
[![CPU / CUDA](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA-76b900)](docs/核心思想.md#ask-owned-codegen)

ASK 是张量框架的 C++ 编译器和运行时。它接收 MLIR `ask` 静态图，生成 CPU/CUDA 代码，并调度自有及第三方 cubin/PTX。

- 类型与数值：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`，常量按 RNE 舍入，加乘输入须同形状、同类型。
- 编译与融合：常量折叠、DCE、多输出逐元素融合；Host 生成 `.o`，CUDA 生成 LLVM IR，再由 `llc` / `ptxas` 生成 cubin。
- 执行计划：一次提交整段计算，按完成状态复用缓冲区和事件，支持 CUDA Graph replay。
- 外部 kernel：通过设备缓冲区连接独立步骤，支持多输入、多输出、指针和按值标量。

项目提供 C++ SDK 与 `ask-opt`。[FugaTensor](../fugatensor/README.md) 在此基础上实现 Python API、自动组图、求导和第三方编译器适配。

接口见[用户 API](docs/用户API.md)，设计见[核心思想](docs/核心思想.md)。接入示例推荐看 FugaTensor 的 [Tensor API](../fugatensor/docs/API/README.md) 和[图降级、原生绑定与执行](../fugatensor/docs/architecture/ASK桥接与多后端执行.md)。

## 构建、安装与验证

构建需要 C++17、CMake 3.28+、Ninja 和同一套 LLVM/MLIR 23.1.x SDK。默认下载 Linux x86-64 的 23.1.4 发行版，版本和摘要见[依赖记录](cmake/llvm.lock.json)。下载脚本使用 Python 3 标准库。

```bash
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
cmake --install build/core --prefix /path/to/ask-sdk
```

调用方链接 `ASK::Compiler` 和 `ASK::Runtime`；直接操作 MLIR/LLVM 时使用 `ASK::Core`，并提供匹配的 LLVM/MLIR SDK。组件和依赖见[安装说明](docs/安装.md)。

```cmake
find_package(ASK CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

## 测试

```bash
build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --partition \
  -o build/partitioned.mlir --report-json build/plan-report.json

# 物理 GPU 上启用原生运行时、资源复用与 Graph replay 验证。
cmake --preset core -DASK_TEST_CUDA=ON
cmake --build --preset core
ctest --preset core --output-on-failure
```

默认 CTest 运行 `ask-core` 和 `ask-compiler`，启用 GPU 后增加 `ask-runtime`。源码与 SDK 消费测试位于 `tests/`，覆盖范围见[测试体系](docs/工程结构与测试体系.md)。

FugaTensor 的[端到端测试](../fugatensor/docs/测试.md)覆盖图降级、数值、求导和外部 kernel 调度。

[文档目录](docs/README.md) · [外部 Worker 协议](docs/外部编译器适配接入规范.md) · [执行计划](docs/图切分与多Kernel执行计划.md)
