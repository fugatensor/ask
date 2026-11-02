# ASK

ASK 是面向张量框架的 C++ 编译与运行时引擎。它接收 MLIR `ask` 静态图，检查类型与形状，优化自有 CPU/CUDA 计算，并用原生执行计划调度自有及第三方 cubin/PTX。

- **显式数值语义**：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`，单步 RNE 常量和显式类型转换，不做隐式提升或广播。
- **自有编译与融合**：常量折叠、DCE、多输出逐元素融合；Host 生成 `.o`，CUDA 生成 NVPTX LLVM IR，由调用方运行 `llc` / `ptxas` 生成 cubin。
- **通用原生执行计划**：C++ 一次绑定并提交整段计划，管理中间缓冲区、完成事件和资源寿命；已完成工作区、显存分配与事件可复用，支持 CUDA Graph replay。
- **外部产物边界**：外部 kernel 保持独立，通过设备缓冲区连接；支持指针/null、按值标量、多输入与多输出，不改写第三方代码。

ASK 只交付 C++ SDK 与 `ask-opt`，不交付 Python 包或绑定。FugaTensor 保留 Python 用户 API、图语义、薄适配器及编译编排，自己的原生扩展直接链接 ASK。第三方 worker 调用、磁盘缓存和编译包校验由调用方实现，协议与 JSON Schema 由 ASK 维护。

接口从 [用户 API](docs/用户API.md) 开始；职责见 [核心思想](docs/核心思想.md)。推荐以 [FugaTensor](../fugatensor/README.md) 作为完整接入示例：从 [Tensor API](../fugatensor/docs/API/README.md) 了解使用方式，再看[图降级、原生绑定与执行](../fugatensor/docs/architecture/04-ASK桥接与多后端执行.md)。

## 构建、安装与验证

默认 SDK 面向 Linux x86-64，使用 LLVM/MLIR 23.1.4 稳定发行版。构建需要 C++17、CMake 3.28+、Ninja；LLVM 与 MLIR 必须来自同一套受支持的 23.1.x SDK。[依赖记录](cmake/llvm.lock.json)固定默认下载的版本与摘要以便复现，不要求跟随 Triton 的 LLVM 提交。依赖获取脚本使用 Python 标准库，库本身不依赖 Python。

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

默认 CTest 包含 `ask-core` 与 `ask-compiler`；显式启用 GPU 后增加 `ask-runtime`。断言与测试输入位于 `tests/`，SDK 独立消费验证也由该目录提供；示例代码写在[用户 API 文档](docs/用户API.md)中。测试入口与覆盖范围见[测试体系](docs/06-工程结构与测试体系.md)。

FugaTensor 的[端到端测试](../fugatensor/docs/测试.md)覆盖图降级、原生调用、数值矩阵和外部 kernel 调度。缺少工具或驱动而跳过的测试不能作为对应功能通过的证据。

[文档目录](docs/README.md) · [外部 Worker 协议](docs/09-外部编译器适配接入规范.md) · [执行计划](docs/08-图切分与多Kernel执行计划.md)
