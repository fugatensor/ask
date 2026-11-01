# ASK

ASK 是面向张量框架的底层编译与运行时引擎。上层通过 MLIR `ask` dialect 提交静态计算图与可选的外部 DSL（如 Triton），ASK 完成类型与形状检查、基础图优化、原生 LLVM IR 降级、跨编译器值边界内联融合，并生成 CPU 目标代码或 CUDA `cubin` 交由常驻会话执行。

- **6 种显式类型与确定性数值**：支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16` 单基本块静态图、单步最近偶数舍入（RNE）常量与 6×6 显式 `ask.cast`，不做隐式类型提升或隐式广播。
- **自有 CPU 与自有 CUDA 代码生成**：`core` 构建（`ASK_THIRD_PARTY_COMPILERS=none`）不依赖 Triton、NumPy 或 PyTorch，直接支持常量/类型折叠、死代码消除、多输出逐元素融合，并生成 Host 目标文件或自有 CUDA kernel（支持异构多输入输出、0-D 标量与空张量）。
- **跨编译器 LLVM 值边界融合**：图中包含可证明的逐元素 Triton 算子（`P → F → Q`）时，通过 [Triton 适配器](adapters/triton/README.md) 导出原生 LLVM IR，利用通用值边界内联器将前后 ASK 计算合入同一条 GPU kernel，并按指令级标志隔离各自的浮点收缩（FMA）与转换规则。
- **进程内 SDK、常驻会话与双层缓存**：提供 `nanobind` 进程内编译器入口、`ExecutionSession` 常驻设备内存、`__cuda_array_interface__` 零拷贝接入、异步事件同步，以及磁盘级 LLVM IR / `cubin` 双层持久缓存。

架构与实现说明见 [文档目录](docs/README.md) 和 [核心思想](docs/核心思想.md)。

## 构建与验证

环境要求：C++17 编译器、CMake 3.28+、Ninja、Python 3.10+（含开发头文件）。LLVM/MLIR（`23.0.0git`）与 `nanobind` 的版本和校验值锁定在 [cmake/llvm.lock.json](cmake/llvm.lock.json) 与 [cmake/nanobind.lock.json](cmake/nanobind.lock.json) 中，拉取到 `.deps/` 后 CMake 会自动识别：

```bash
python3 tools/fetch_deps.py
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

使用外部同修订 LLVM/MLIR 安装时，可通过 `-DLLVM_DIR=... -DMLIR_DIR=...`（或 `pip install` 时的 `-Ccmake.define.LLVM_DIR=...`）覆盖，见 [前端接入与构建配置](docs/02-前端接入与构建配置.md#build-workflow)。

## 示例与接入方式

### 1. Python 构建、准备与优化计算图

`Compiler()` 默认调用进程内 `nanobind` 扩展，传入 `--ask-opt` 可切换 CLI 子进程模式：

```bash
# 构造 P -> F(Triton) -> Q 图并执行语义/形状检查与外部请求导出
PYTHONPATH=build/core/python:python python3 examples/framework_frontend.py

# 常量折叠、冗余 cast 消除、多输出逐元素融合及 LLVM IR 导出
PYTHONPATH=build/core/python:python python3 examples/optimize_frontend.py
```

直接调用命令行工具 `ask-opt`：

```bash
build/core/bin/ask-opt --capabilities
build/core/bin/ask-opt tests/ir/pipeline.mlir --prepare -o build/prepared.mlir --report-json build/report.json
```

### 2. 常驻 GPU 会话与 `ASK → Triton → ASK` 融合

在配有 NVIDIA GPU、`ptxas` 及 Triton 3.7.1 的环境中运行硬件融合与多精度数值验证：

```bash
PYTHONPATH=build/all/python:python /path/to/triton-env/bin/python examples/triton_fused_gpu.py
```

重复执行 GPU kernel 时复用 `ExecutionSession` 与设备张量：

```python
from ask import ExecutionSession

with ExecutionSession(device=0) as session:
    executable = kernel.prepare(session)  # kernel 来自 Compiler().compile_cuda(...)
    x = session.from_host(values, shape=executable.shape, dtype=executable.dtype)
    y = session.tensor(executable.shape, executable.dtype)
    done = executable.submit(x, y)        # 异步入队
    done.wait()
    result = y.to_host()
```

### 3. 独立 C++ 接入（`ASK::Core`）

C++ 宿主可通过 `find_package(ASK CONFIG REQUIRED)` 直接构造与准备 `ask` IR，无需初始化 Python，示例见 [examples/cpp/frontend.cpp](examples/cpp/frontend.cpp) 与 [examples/cpp/CMakeLists.txt](examples/cpp/CMakeLists.txt)。

## 测试

`ctest --preset all` 运行 5 个测试集（`core` 运行前 4 个）：

- `ask-compiler-adapter` / `ask-boundary-inline`：C++ 适配器注册隔离与通用 LLVM 值边界内联单测。
- `ask-unit`：`tests/test_*.py` 共 7 个纯 CPU 单测模块。
- `ask-gpu`：`tests/gpu/test_gpu_{native,triton}.py` 物理 GPU 执行、零拷贝会话与双层缓存验证。
- `ask-triton`：`adapters/triton/tests/test_*.py` Triton 导出协议、AST 静态融合判定与 LLVM 边界融合测试。

单独运行纯 CPU Python 单测：

```bash
PYTHONPATH=build/core/python:python:tests ASK_OPT=build/core/bin/ask-opt \
  ASK_BOUNDARY_INLINE_BIN=build/core/ask-boundary-inline-test \
  python3 -m unittest discover -s tests -p "test_*.py" -v
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [文档目录](docs/README.md) / [核心思想](docs/核心思想.md) | 架构总览与各模块导航 |
| [01. ASK IR 与数值语义](docs/01-ASK-IR与数值语义.md) | 6 种类型、单步 RNE、饱和 `ask.cast`、形状推导与外部算子声明 |
| [02. 前端接入与构建配置](docs/02-前端接入与构建配置.md) | 依赖锁、CMake Preset、可选适配器注册、Python / C++ / CLI 入口 |
| [03. 图优化与自有代码生成](docs/03-图优化与自有代码生成.md) | 折叠/DCE/融合分组、Host 共享循环降级与自有 CUDA 降级 |
| [04. 外部编译器适配与边界内联](docs/04-外部编译器适配与边界内联.md) | `adapters/<id>/` 隔离、导出协议与 ABI v2、通用内联核心与 FMA 隔离 |
| [05. 常驻运行时与多级缓存](docs/05-常驻运行时与多级缓存.md) | `ExecutionSession`、零拷贝设备张量、异步事件与 IR / `cubin` 缓存 |
| [06. 工程结构与测试体系](docs/06-工程结构与测试体系.md) | 目录职责、5 个 CTest 测试集与共享多精度数值参考 |
| [07. FugaTensor 协同与演进路线](docs/07-FugaTensor协同与演进路线.md) | 与 FugaTensor 的职责边界、已实现范围与后续规划 |
