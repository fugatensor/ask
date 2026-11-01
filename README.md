# ASK

ASK 是一个面向张量框架的底层编译与常驻运行时引擎：上层框架通过 MLIR `ask` dialect 提交静态计算图和可选的外部 DSL（如 Triton），ASK 负责语义与形状检查、基础图优化、原生 LLVM IR 降级、跨编译器值边界内联融合，并生成 CPU 目标代码或 CUDA `cubin` 交由常驻会话执行。

当前已落地的核心能力：

- **显式 6 精度与确定性数值语义**：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16` 单基本块静态图、单步最近偶数舍入（RNE）常量、6×6 显式 `ask.cast` 与 `f32` 中间累加混合精度，拒绝隐式类型提升与隐式广播。
- **自有 CPU 与自有 CUDA 代码生成**：默认 `core` 构建（`ASK_THIRD_PARTY_COMPILERS=none`）不依赖 Triton、NumPy 或 PyTorch，即可执行常量/类型折叠、死代码消除、多输出逐元素融合，并直接生成 Host 目标代码或自有 CUDA kernel（支持异构多输入输出、0-D 标量与空张量）。
- **跨编译器 LLVM 值边界融合**：当图中包含可证明的逐元素 Triton 算子（`P → F → Q`）时，通过插件化的 [Triton 适配器](adapters/triton/README.md) 导出原生 LLVM IR，利用通用值边界内联器把前后 ASK 计算合入同一条 GPU kernel，并精确隔离两边的浮点收缩（FMA）与转换规则。
- **进程内 SDK、常驻会话与双层缓存**：`nanobind` 进程内编译器入口、`ExecutionSession` 常驻设备内存与 `__cuda_array_interface__` 零拷贝接入、异步事件同步，以及磁盘级 LLVM IR / `cubin` 双层持久缓存。

完整的架构脉络与工程细节请从 [文档目录](docs/README.md) 和 [核心思想](docs/核心思想.md) 开始阅读。

## 快速构建与验证

环境要求：C++17 编译器、CMake 3.28+、Ninja、Python 3.10+（含开发头文件）。LLVM/MLIR（`23.0.0git`）与 `nanobind` 的版本和 SHA256 锁定在 [cmake/llvm.lock.json](cmake/llvm.lock.json) 与 [cmake/nanobind.lock.json](cmake/nanobind.lock.json) 中，使用统一脚本拉取到 `.deps/` 后，CMake 会自动识别，无需手填长路径：

```bash
# 1. 拉取锁定版 LLVM/MLIR SDK 与 nanobind 源码
python3 tools/fetch_deps.py

# 2. 构建并测试核心编译器与自有 CPU/CUDA 后端（无需 Triton）
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
```

启用全部已注册的第三方编译器适配器（当前为 Triton 3.7.1，依赖锁定见 [adapters/triton/dependencies.lock.json](adapters/triton/dependencies.lock.json)）：

```bash
cmake --preset all -DPython_EXECUTABLE=/path/to/triton-env/bin/python
cmake --build --preset all
ctest --preset all --output-on-failure
```

将包含原生扩展的 `ask-compiler` 安装进当前 Python 环境：

```bash
python3 -m pip install .
```

> 若使用外部自备的同修订 LLVM/MLIR，可通过 `-DLLVM_DIR=... -DMLIR_DIR=...`（或 `pip install` 时的 `-Ccmake.define.LLVM_DIR=...`）显式覆盖，详见 [前端接入与构建配置](docs/02-前端接入与构建配置.md#build-workflow)。

## 示例与接入方式

### 1. Python 进程内构建、准备与优化计算图

`Compiler()` 默认直接走进程内 `nanobind` 原生扩展，也可传入 `--ask-opt` 切换 CLI 子进程模式：

```bash
# 构造 P -> F(Triton) -> Q 图并执行语义/形状检查与外部请求导出
PYTHONPATH=build/core/python:python python3 examples/framework_frontend.py

# 演示常量折叠、冗余 cast 消除、多输出逐元素融合及 LLVM IR 导出
PYTHONPATH=build/core/python:python python3 examples/optimize_frontend.py
```

直接使用原生命令行工具 `ask-opt`：

```bash
build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --prepare -o build/prepared.mlir --report-json build/report.json
```

### 2. 常驻 GPU 会话与 `ASK → Triton → ASK` 硬件融合

在配有 NVIDIA GPU、`ptxas` 及 Triton 3.7.1 环境的机器上运行端到端硬件融合与多精度数值校验（脚本会自动探测 `.deps/` 中的 `llc` 与系统/环境中的 `ptxas`）：

```bash
PYTHONPATH=build/all/python:python /path/to/triton-env/bin/python examples/triton_fused_gpu.py
```

上层框架在循环中执行 GPU kernel 时，应复用常驻 `ExecutionSession` 与设备张量，避免反复初始化上下文或来回拷贝主机列表：

```python
from ask import ExecutionSession

with ExecutionSession(device=0) as session:
    executable = kernel.prepare(session)  # kernel 来自 Compiler().compile_cuda(...)
    x = session.from_host(values, shape=executable.shape, dtype=executable.dtype)
    y = session.tensor(executable.shape, executable.dtype)
    done = executable.submit(x, y)        # 异步入队，不隐式同步主机
    done.wait()
    result = y.to_host()
```

### 3. 独立 C++ 接入（`ASK::Core`）

纯 C++ 宿主无需初始化 Python 即可通过 `find_package(ASK CONFIG REQUIRED)` 构造与准备 `ask` IR，完整独立工程示例见 [examples/cpp/frontend.cpp](examples/cpp/frontend.cpp) 与 [examples/cpp/CMakeLists.txt](examples/cpp/CMakeLists.txt)。

## 测试分层

`ctest --preset all` 统一调度 5 个测试集（`core` 运行前 4 个）：

- `ask-compiler-adapter` / `ask-boundary-inline`：C++ 适配器注册隔离与通用 LLVM 值边界内联单测。
- `ask-unit`：`tests/test_*.py` 共 7 个纯 CPU 快速单测模块（SDK、IR 正反例与优化、Host 执行、自有 CUDA 离线契约、缓存/运行时校验、6 精度极限参考值、工程工具测试）。
- `ask-gpu`：`tests/gpu/test_gpu_{native,triton}.py` 真实 NVIDIA GPU 硬件执行、零拷贝会话与双层持久缓存验证。
- `ask-triton`：`adapters/triton/tests/test_*.py` Triton 导出协议、AST 静态融合判定与 LLVM 边界融合测试。

日常单独快速运行 CPU Python 单测：

```bash
PYTHONPATH=build/core/python:python:tests ASK_OPT=build/core/bin/ask-opt \
  ASK_BOUNDARY_INLINE_BIN=build/core/ask-boundary-inline-test \
  python3 -m unittest discover -s tests -p "test_*.py" -v
```

## 文档导航

| 文档 | 说明 |
| --- | --- |
| [文档目录](docs/README.md) / [核心思想](docs/核心思想.md) | 全局设计脉络与双向互链总入口 |
| [01. ASK IR 与数值语义](docs/01-ASK-IR与数值语义.md) | 6 种类型、单步 RNE、饱和 `ask.cast`、形状推导与外部算子声明 |
| [02. 前端接入与构建配置](docs/02-前端接入与构建配置.md) | 依赖锁、CMake Preset、可选适配器注册、Python / C++ / CLI 三套入口 |
| [03. 图优化与自有代码生成](docs/03-图优化与自有代码生成.md) | 折叠/DCE/融合组规划、Host 共享循环降级与自有 CUDA 降级 |
| [04. 外部编译器适配与边界内联](docs/04-外部编译器适配与边界内联.md) | `adapters/<id>/` 隔离、导出协议与 ABI v2、通用内联核心与 FMA 隔离 |
| [05. 常驻运行时与多级缓存](docs/05-常驻运行时与多级缓存.md) | `ExecutionSession`、零拷贝设备张量、异步事件与 IR / `cubin` 双层缓存 |
| [06. 工程结构与测试体系](docs/06-工程结构与测试体系.md) | 目录职责、5 个 CTest 测试集与独立任意精度数值参考矩阵 |
| [07. FugaTensor 协同与演进路线](docs/07-FugaTensor协同与演进路线.md) | 与上层框架 FugaTensor 的职责边界、当前能力清单与后续演进方向 |
