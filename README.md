# ASK

ASK 用 MLIR 的 `ask` dialect 接收上层框架生成的计算图，以原生 LLVM IR 承接低层产物。这个仓库已开始实现编译器初稿；当前入口是 Python SDK 和原生 `ask-opt`。

目前可以构建、检查和准备单基本块的 bool / i32 / i64 / f32 / f16 / bf16 图，推导具体形状，并在导出追踪记录后删除 `ask.mark`。显式 `ask.cast` 支持六种类型互转及混合精度表达式。基础优化包括常量折叠、死代码消除和逐元素计算合并。

无 external 的基础图直接降到 CPU 代码或自有 CUDA kernel。`none` 构建也能编译并执行 GPU 图，不需要 Triton；支持多输入、多输出、不同输出类型、零维和空张量，详见[自有 GPU 降级](docs/19-自有GPU降级.md)。Python 通过 nanobind 调用 C++ 前端，设备张量、常驻 Session 和异步事件用于重复执行。

图中明确提交 Triton DSL 时，还可以使用六类型静态逐元素 ASK→Triton→ASK 融合：链接并内联原生 LLVM IR，再生成 cubin。自有图和融合图都提供持久 IR / 可执行产物缓存及独立命中统计。外部融合范围见[GPU 融合与 IR 缓存](docs/12-GPU融合与IR缓存.md)，显式 f32 中间计算见[类型转换与混合精度](docs/15-类型转换与混合精度.md)。完整混合图调度仍待实现。

## 构建

需要 C++17、CMake 3.28 或更新版本、Ninja、Python 3.10+ 及其扩展模块开发文件、锁定 nanobind，以及配套的 LLVM/MLIR 开发安装。当前使用 LLVM/MLIR `23.0.0git`，修订和 SDK 校验值锁定在 [cmake/llvm.lock.json](cmake/llvm.lock.json)；不要只按版本号混用别的构建。Triton 导出与 reader 的兼容性记录见 [dependencies.lock.json](adapters/triton/dependencies.lock.json)。

```bash
python3 tools/fetch_llvm.py
python3 tools/fetch_nanobind.py
cmake --preset core \
  -DLLVM_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/llvm" \
  -DMLIR_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/mlir"
cmake --build --preset core
ctest --preset core
build/core/bin/ask-opt --capabilities
```

下载脚本取锁定的 Ubuntu x64 SDK。其他平台需要准备同一修订的 LLVM/MLIR，并将两个路径指向对应安装。

`core` 使用 `ASK_THIRD_PARTY_COMPILERS=none`，不需要 Triton、NumPy 或 PyTorch。Python API 和 nanobind 绑定是基础组成，三种模式全部构建，不提供独立关闭选项。它仍能检查包含 `ask.external` 的图；真正请求编译外部 DSL 时才检查该编译器是否启用。第三方编译器可选择 `none`、`all` 或明确的名称。历史选项 `ASK_TARGET_BACKENDS=none|host` 只控制 host 目标文件生成；自有 CUDA LLVM 降级和运行时在三种编译器模式中都属于基础核心。见[构建说明](docs/09-构建与可选编译器.md)。

第三方适配按 `adapters/<id>/` 集中存放：构建规则、worker、Python 策略、原生边界检查和专属测试都在该目录。公共核心通过显式注册接口调用它们，CUDA 运行时只读取标准化的参数与启动契约。新增编译器的文件和接口见[适配与注册](docs/17-第三方编译器适配与注册.md)；当前真实适配器仍只有 Triton。

## 从上层框架提交图

构建完成后，使用匹配构建解释器的原生扩展。`Compiler()` 默认进程内调用 C++；显式传入 `ask-opt` 路径则使用 CLI 兼容入口：

```bash
PYTHONPATH=build/core/python:python python3 examples/framework_frontend.py --emit-only
PYTHONPATH=build/core/python:python python3 examples/framework_frontend.py \
  --ask-opt build/core/bin/ask-opt
```

[完整示例](examples/framework_frontend.py) 生成 `P: t=x+1 → F: u=t*2 → Q: y=u+3`。F 的 Triton 源码作为字符串写入实现声明；示例进程不会导入或执行 Triton。第二条命令调用真实 MLIR 编译器，输出准备后的 IR、形状快照、追踪记录和外部编译请求。

也可以直接提交 MLIR 文件：

```bash
build/core/bin/ask-opt tests/ir/pipeline.mlir \
  --prepare -o /tmp/ask-prepared.mlir --report-json /tmp/ask-report.json
```

SDK 用法、当前支持范围和诊断处理见[上层框架接入](docs/10-上层框架接入.md)。架构设计从[核心思想](docs/核心思想.md)开始，分阶段说明见[文档目录](docs/README.md)。设计文档中的完整接口不等于当前版本已经实现的 API。

低层还提供[通用 LLVM 值边界内联接口](docs/13-通用LLVM值边界内联.md)：接收已确认的边界和 typed helper，支持多参数、多站点、整数、浮点和定长向量，失败时保留输入模块。Triton 的源码、布局和启动检查放在适配器中。六种类型由 ASK IR 决定，设备张量通过 `kernel.prepare(session).submit_many(inputs, outputs)` 提交。`kernel.run(values)` 保留为单输入单输出、相同 shape/dtype 的同步列表接口，见[类型传递](docs/10-上层框架接入.md#dtype-path)。bf16 会检查实际硬件支持，包括只在中间值使用 bf16 的图，不支持就报错。类型转换、CPU/GPU 检查与显式 f32 计算见[类型转换与混合精度](docs/15-类型转换与混合精度.md)。

GPU 示例改为共用测试矩阵：`--dtypes bool i32 i64 f32 f16 bf16` 用同一组图验证六种类型和 6×6 转换方向，`--cases` 选图，`--extended` 增加形状与启动配置。host 和原生融合测试也复用该图构造器，规则和命令见[多精度测试](docs/14-多精度测试.md)。

C++ 框架也可以通过安装后的 `ASK::Core` 直接构造和准备 MLIR，运行时无需初始化 Python。[C++ 示例](examples/cpp/frontend.cpp)及其 [CMake 项目](examples/cpp/CMakeLists.txt)演示独立消费已安装的 SDK，构建命令见接入文档第 8 节。

## 设备张量与重复执行

`kernel.run(values)` 保留为同步验证入口。框架循环中使用常驻资源：

```python
from ask import ExecutionSession

with ExecutionSession(device=0) as session:
    executable = kernel.prepare(session)  # kernel 来自 compile_cuda
    x = session.from_host(values, shape=executable.shape, dtype=executable.dtype)
    y = session.tensor(executable.shape, executable.dtype)
    done = executable.submit(x, y)        # 入队，不下载结果
    done.wait()
    result = y.to_host()
```

重复 `submit` 复用上下文、模块和缓冲区。已有连续 CUDA 张量可通过 `session.from_cuda_array(tensor)` 零拷贝接入；外部 bf16 的 DLPack 接入尚未实现。可执行缓存命中后跳过 `llc/ptxas`，查询 `kernel.codegen["executable_cache"]`；IR 统计仍在 `kernel.cache`。生命周期、异步顺序和支持边界见[常驻运行时](docs/16-Python绑定与常驻运行时.md)。

Python 安装包使用 scikit-build-core 构建，并包含原生扩展；不再生成只有 Python 客户端的包。准备好上述依赖后，在使用同一 LLVM SDK 的环境中安装：

```bash
python3 -m pip install . \
  -Ccmake.define.LLVM_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/llvm" \
  -Ccmake.define.MLIR_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/mlir"
```

## 基础优化

框架调用 `compiler.optimize(module)` 可取得优化后的 ASK IR 和报告；默认的 `prepare` 保持原有行为。也可以通过 CLI 运行：

```bash
PYTHONPATH=python python3 examples/optimize_frontend.py --emit-only > build/optimization-input.mlir
build/core/bin/ask-opt build/optimization-input.mlir --optimize --emit-llvm \
  -o build/optimized.ll --report-json build/optimization.json
build/core/bin/ask-opt build/optimized.ll --optimize-llir -o build/cleaned.ll
```

此处的融合计划由 host 后端落实为共享循环。`Compiler.compile_cuda` 根据图是否含 external 选择自有 GPU 降级或外部融合；纯图不会加载适配器。跨 Triton 边界的 LLVM 融合也可以通过 `--fuse-external` 入口启用。开关、形状和标记更新规则见[基础优化与逐元素融合](docs/11-基础优化与逐元素融合.md)，GPU 数值验证见[可运行示例](examples/triton_fused_gpu.py)。

## 测试

`ctest --preset core` 会运行原生绑定、Python API 和真实 `ask-opt` 集成测试。只检查 Python 部分时：

```bash
PYTHONPATH=build/core/python:python python3 -m unittest discover -s tests/python -v
```

未设置 `ASK_OPT` 时，真实编译器测试会显示 skipped。要单独运行它们，设置 `ASK_OPT` 为编译出的 `ask-opt` 路径；不能把跳过当成 native 验证通过。
