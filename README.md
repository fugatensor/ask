# ASK

ASK 是面向张量框架的 C++ 编译与运行时引擎。它接收 MLIR `ask` 静态图，检查类型与形状，优化自有 CPU/CUDA 计算，并用原生执行计划调度自有及第三方 cubin/PTX。

- **显式数值语义**：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`，单步 RNE 常量和显式类型转换，不做隐式提升或广播。
- **自有编译与融合**：常量折叠、DCE、多输出逐元素融合；Host 生成 `.o`，CUDA 生成 NVPTX LLVM IR，由调用方运行 `llc` / `ptxas` 生成 cubin。
- **通用原生执行计划**：C++ 一次绑定并提交整段计划，管理中间缓冲区、完成事件和资源寿命；已完成工作区、显存分配与事件可复用，支持 CUDA Graph replay。
- **外部产物边界**：外部 kernel 保持独立，通过设备缓冲区连接；支持指针/null、按值标量、多输入与多输出，不改写第三方代码。

ASK 只交付 C++ SDK 与 `ask-opt`，不交付 Python 包或绑定。FugaTensor 保留 Python 用户 API、图语义、薄适配器及编译编排，自己的原生扩展直接链接 ASK。第三方 worker 调用、磁盘缓存和编译包校验由调用方实现，协议与 JSON Schema 由 ASK 维护。

接口从 [用户 API](docs/用户API.md) 开始；职责见 [核心思想](docs/核心思想.md)，FugaTensor 接入见 [上层执行说明](../fugatensor/docs/architecture/04-ASK桥接与多后端执行.md)。

## 构建、安装与验证

锁定 SDK 面向 Linux x86-64。构建需要 C++17、CMake 3.28+、Ninja 和匹配 [llvm.lock.json](cmake/llvm.lock.json) 的 LLVM/MLIR；依赖下载脚本使用 Python 标准库，库本身不依赖 Python。

```bash
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
cmake --install build/core --prefix /path/to/ask-sdk
```

`ASK::Compiler` 提供不暴露 LLVM 类型的编译接口，`ASK::Runtime` 提供原生运行时；只消费这两个组件无需 LLVM 开发 SDK。需要直接操作 MLIR/LLVM 时使用 `ASK::Core`，并提供匹配的 LLVM/MLIR SDK。完整步骤见 [安装说明](docs/安装.md)。

```cmake
find_package(ASK CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

## 示例与测试

```bash
build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --partition \
  -o build/partitioned.mlir --report-json build/plan-report.json

# 物理 GPU 上启用原生运行时、资源复用与 Graph replay 验证。
cmake --preset core -DASK_TEST_CUDA=ON
cmake --build --preset core
ctest --preset core --output-on-failure
```

默认 CTest 包含 `ask-core` 与 `ask-compiler`；显式启用 GPU 后增加 `ask-runtime`。独立消费示例见 [编译门面](examples/cpp/compiler.cpp)、[底层 MLIR 接口](examples/cpp/frontend.cpp)与[原生 CUDA 计划](examples/cpp-runtime/main.cpp)。

Python 集成、数值矩阵、外部 worker 与 FugaTensor JIT 测试位于调用方的 [tests/backend](../fugatensor/tests/backend)。缺少工具或驱动而跳过的测试不能作为对应功能通过的证据。

[文档目录](docs/README.md) · [外部 Worker 协议](docs/09-外部编译器适配接入规范.md) · [执行计划](docs/08-图切分与多Kernel执行计划.md)
