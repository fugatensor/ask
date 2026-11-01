# ASK

ASK 用 MLIR 的 `ask` dialect 接收上层框架生成的计算图，以原生 LLVM IR 承接低层产物。这个仓库已开始实现编译器初稿；当前入口是 Python SDK 和原生 `ask-opt`。

目前可以构建、检查和准备单基本块的 i32 图，注册外部算子的语义与 Triton DSL，推导具体形状，并在导出追踪记录后删除 `ask.mark`。可选优化支持常量折叠、死代码消除、自有逐元素多输出融合，以及原生 LLVM 基础清理。另已跑通受限的静态 i32 ASK→Triton→ASK 融合：链接并内联原生 LLVM IR，重新生成 cubin，在 GPU 上执行；提供持久 IR 缓存和命中统计。支持范围及复现命令见[GPU 融合与 IR 缓存](docs/12-GPU融合与IR缓存.md)。完整混合图调度仍待实现。

## 构建

需要 C++17、CMake 3.28 或更新版本、Ninja，以及配套的 LLVM/MLIR 开发安装。当前使用 LLVM/MLIR `23.0.0git`，修订和 SDK 校验值锁定在 [cmake/llvm.lock.json](cmake/llvm.lock.json)；不要只按版本号混用别的构建。Triton 导出与 reader 的兼容性记录见 [dependencies.lock.json](adapters/triton/dependencies.lock.json)。

```bash
python3 tools/fetch_llvm.py
cmake --preset core \
  -DLLVM_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/llvm" \
  -DMLIR_DIR="$PWD/.deps/llvm-1f126a6d-ubuntu-x64-1/lib/cmake/mlir"
cmake --build --preset core
ctest --preset core
build/core/bin/ask-opt --capabilities
```

下载脚本取锁定的 Ubuntu x64 SDK。其他平台需要准备同一修订的 LLVM/MLIR，并将两个路径指向对应安装。

`core` 使用 `ASK_THIRD_PARTY_COMPILERS=none`，不需要 Triton、NumPy 或 PyTorch。它仍能检查包含 `ask.external` 的图；真正请求编译外部 DSL 时才检查该编译器是否启用。第三方编译器可选择 `none`、`all` 或明确的名称，目标后端另外选择；当前后端选项只实现 `none` 和 `host`。见[构建说明](docs/09-构建与可选编译器.md)。

## 从上层框架提交图

Python SDK 支持 Python 3.10 或更新版本，运行时只依赖标准库。可以安装 `python3 -m pip install -e .`，也可以直接从源码使用：

```bash
PYTHONPATH=python python3 examples/framework_frontend.py --emit-only
PYTHONPATH=python python3 examples/framework_frontend.py \
  --ask-opt build/core/bin/ask-opt
```

[完整示例](examples/framework_frontend.py) 生成 `P: t=x+1 → F: u=t*2 → Q: y=u+3`。F 的 Triton 源码作为字符串写入实现声明；示例进程不会导入或执行 Triton。第二条命令调用真实 MLIR 编译器，输出准备后的 IR、形状快照、追踪记录和外部编译请求。

也可以直接提交 MLIR 文件：

```bash
build/core/bin/ask-opt tests/ir/pipeline.mlir \
  --prepare -o /tmp/ask-prepared.mlir --report-json /tmp/ask-report.json
```

SDK 用法、当前支持范围和诊断处理见[上层框架接入](docs/10-上层框架接入.md)。架构设计从[核心思想](docs/核心思想.md)开始，分阶段说明见[文档目录](docs/README.md)。设计文档中的完整接口不等于当前版本已经实现的 API。

低层还提供[通用 LLVM 值边界内联接口](docs/13-通用LLVM值边界内联.md)：接收已确认的边界和 typed helper，支持多参数、多站点、整数、浮点和定长向量，失败时保留输入模块。Triton 的源码、布局和启动检查放在适配器中。GPU 示例的 `--extended` 覆盖多维形状、不同 BLOCK/warp 配置及每线程多个元素。

C++ 框架也可以通过安装后的 `ASK::Core` 直接构造和准备 MLIR，无需 Python。[C++ 示例](examples/cpp/frontend.cpp)及其 [CMake 项目](examples/cpp/CMakeLists.txt)演示独立消费已安装的 SDK，构建命令见接入文档第 8 节。

## 基础优化

框架调用 `compiler.optimize(module)` 可取得优化后的 ASK IR 和报告；默认的 `prepare` 保持原有行为。也可以通过 CLI 运行：

```bash
PYTHONPATH=python python3 examples/optimize_frontend.py --emit-only > build/optimization-input.mlir
build/core/bin/ask-opt build/optimization-input.mlir --optimize --emit-llvm \
  -o build/optimized.ll --report-json build/optimization.json
build/core/bin/ask-opt build/optimized.ll --optimize-llir -o build/cleaned.ll
```

此处的融合计划由 host 后端落实为共享循环。跨 Triton 边界的 LLVM 融合通过独立的 `Compiler.compile_cuda` / `--fuse-external` 入口启用。开关、形状和标记更新规则见[基础优化与逐元素融合](docs/11-基础优化与逐元素融合.md)，GPU 数值验证见[可运行示例](examples/triton_fused_gpu.py)。

## 测试

`ctest --preset core` 会运行 Python 客户端测试和真实 `ask-opt` 集成测试。只检查 Python 部分时：

```bash
python3 -m unittest discover -s tests/python -v
```

未设置 `ASK_OPT` 时，真实编译器测试会显示 skipped。要单独运行它们，设置 `ASK_OPT` 为编译出的 `ask-opt` 路径；不能把跳过当成 native 验证通过。
