# ASK

ASK 是面向张量框架的底层编译与运行时引擎。上层提交 MLIR `ask` 静态图，ASK 检查类型与形状，优化并编译自有 CPU/CUDA 算子，将外部 cubin/PTX 与自有 kernel 纳入统一 CUDA 执行计划。

- **显式数值语义**：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`，单步 RNE 常量和 6×6 显式 `ask.cast`，不做隐式类型提升或广播。
- **自有融合与代码生成**：常量折叠、DCE、多输出逐元素融合；Host 生成 `.o`，CUDA 生成 cubin，支持异构多端口、0-D 张量和空张量。
- **外部可执行接口**：调用方 worker 返回 cubin 或 PTX，ASK 推导端口并校验启动契约。外部 kernel 保持独立，与自有区域通过显存缓冲区连接，在单 stream 上顺序执行。
- **常驻会话与缓存**：进程内 nanobind 编译入口、`ExecutionSession`、CUDA Array Interface 零拷贝、异步事件、自有 IR/机器码缓存和外部产物缓存。

ASK 不依赖 Triton、NumPy、PyTorch 或 FugaTensor。第三方编译器适配器由调用方维护；FugaTensor 的 Triton 接入见 [上层执行说明](../fugatensor/docs/architecture/04-ASK桥接与多后端执行.md)。架构导航见 [文档目录](docs/README.md) 与 [核心思想](docs/核心思想.md)。

## 构建与验证

锁定 SDK 支持 Linux x86-64，需要 C++17、CMake 3.28+、Ninja 和 Python 3.10+ 开发环境。LLVM/MLIR 与 nanobind 锁分别见 [llvm.lock.json](cmake/llvm.lock.json)、[nanobind.lock.json](cmake/nanobind.lock.json)。

```bash
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
```

ASK 只有 `core` 构建预设，包含自有 CPU/CUDA 与通用外部执行能力。安装到 Python 环境使用 `python3 -m pip install .`；无需为第三方编译器重建 ASK。任意路径构建和独立安装见 [安装说明](docs/安装.md)，版本与分发边界见 [版本契约](docs/02-前端接入与构建配置.md#version-contract)。

## 示例与接入

```bash
# 准备含外部节点的图，只检查语义和形状并导出请求。
PYTHONPATH=build/core/python:python python3 examples/framework_frontend.py

# 自有图常量折叠、DCE 和逐元素融合分组。
PYTHONPATH=build/core/python:python python3 examples/optimize_frontend.py

build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --partition \
  -o build/partitioned.mlir --report-json build/plan-report.json
```

Python `Compiler()` 默认在进程内准备、优化和切分图。`Compiler.compile_cuda(...)` 编译自有区域，并通过显式 `compilers={id: CompilerEnvironment(...)}` 调用外部 worker。纯外部图不需要 ASK 的 `llc` / `ptxas`；包含自有 CUDA 步骤时仍须提供这两个工具。接口见 [外部产物与执行接口](docs/04-外部产物与执行接口.md)。

编译后的单 kernel 或 `CUDAProgram` 均可绑定常驻会话：

```python
from ask import ExecutionSession

with ExecutionSession(device=0) as session:
    executable = program.prepare(session)
    x = session.from_host(values, shape=executable.shape, dtype=executable.dtype)
    y = session.tensor(executable.shape, executable.dtype)
    done = executable.submit(x, y)
    done.wait()
    result = y.to_host()
```

C++ 宿主通过 `find_package(ASK CONFIG REQUIRED COMPONENTS Core)` 使用 `ASK::Core`，无需初始化 Python；示例见 [frontend.cpp](examples/cpp/frontend.cpp)。

## 测试

CTest 包含两个目标：

- `ask-unit`：`tests/test_*.py`，CPU 单测、自有编译、通用协议、缓存与执行计划验证。
- `ask-gpu`：`tests/gpu/test_gpu_*.py`，物理 GPU 数值、常驻会话、外部产物和多 kernel 调度验证。

```bash
PYTHONPATH=build/core/python:python:tests ASK_OPT=build/core/bin/ask-opt \
  python3 -m unittest discover -s tests -p "test_*.py" -v
```

缺少工具或驱动而跳过的用例不能作为对应功能通过的证据。CI 与硬件验证边界见 [测试体系](docs/06-工程结构与测试体系.md#continuous-integration)。

## 文档

[文档目录](docs/README.md)列出全部专题。接入时可依次阅读 [IR 与数值语义](docs/01-ASK-IR与数值语义.md)、[前端与构建](docs/02-前端接入与构建配置.md)、[外部 Worker 协议](docs/09-外部编译器适配接入规范.md)和[执行计划](docs/08-图切分与多Kernel执行计划.md)。
