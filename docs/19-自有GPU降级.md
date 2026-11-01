# 基础 ASK 图怎样直接在 GPU 上执行

[返回顶层设计：B 层](核心思想.md#ask-b) · [构建选择](09-构建与可选编译器.md) · [形状与降级](ask-ir/03-降级与信息保留.md) · [常驻运行时](16-Python绑定与常驻运行时.md) · [FugaTensor](18-FugaTensor接入示例.md)

没有 `ask.external` 的图由 ASK 自己生成 GPU kernel。`none` 构建包含这条路径，执行 `x + y` 不需要注册或启动 Triton。上层提交高层 ASK IR，ASK 生成线程索引、边界判断、访存和计算，然后把原生 LLVM IR 交给代码生成工具。

## 1. 当前能编译什么

支持静态形状、单基本块中的 `constant`、`splat`、`add`、`mul`、`cast` 和返回已有值。准备阶段仍负责形状推导、收集和删除 `ask.mark`；优化可以做常量折叠和死代码消除。计算类型为 bool、i32、i64、f32、f16、bf16；bool 先显式转为数值类型再做加乘。

一个 kernel 可以有多个输入和输出。端口可以有不同形状和类型；加乘的两个操作数仍须同形同类型。支持零维张量、空张量、多个结果和没有输入的常量图。每个输入或输出对应一个指针参数，总数为 1–64。标量参数也通过一个元素的缓冲区传递，不另外增加按值参数 ABI。

CPU 和 CUDA 现在都有当前基础算子的完整编译执行链路。矩阵乘法、归约、广播、任意 stride 和图级动态控制流仍未实现。遇到不支持的图会给出诊断；不能把“已有 GPU 后端”当成“任意算子都已实现”。

## 2. 一次降级具体做什么

入口是 [CUDA.h](../include/ask/Lowering/CUDA.h) 中的自有降级函数，实现位于 [CUDA.cpp](../lib/Lowering/CUDA.cpp)。它接收准备后的 MLIR Module、显式 CUDA 目标和 launch 配置，返回原生 LLVM Module 及 CUDA 产物说明。整个过程不查第三方适配器注册表。

先检查每个参数和结果的静态形状与元素类型。随后为所有输入、输出创建 global 地址空间的指针参数，输入在前、输出在后；逻辑类型保存到 `port_map`。最终 LLVM 参数已经没有张量形状，运行时从这张端口表获取 shape、dtype 和对应的物理参数序号。

输出按完全相同的 shape 分组。每组用一条线性 grid-stride 循环计算，组内共享相同索引上的 SSA 计算和输入读取；不同形状的组各有自己的循环边界。索引使用 64 位整数：

```text
i = blockIdx.x * blockDim.x + threadIdx.x
stride = gridDim.x * blockDim.x
while i < 输出元素数:
    读取这组计算需要的输入元素
    按 ASK SSA 依赖计算结果
    写入各个输出端口的第 i 个元素
    i += stride
```

默认每个 block 有 128 个线程，grid 的 x 维最多 65535；超过 grid 可一次覆盖的元素数时，由循环继续处理。多维张量按连续行优先布局展开，逻辑 shape 仍留在快照与端口表。空输出的循环不会进入，不发生占位读取或写入。零维 shape `[]` 的元素数为 1。

中间逐元素值保留在线程的 SSA 值中，不分配中间张量缓冲区。一个输出用到另一个输出对应的表达式时，读取的是计算值，不从尚未写完的输出 buffer 取数。因此没有跨线程同步需求，也没有为了套用某个外部 kernel 而改变图形状。

这是一种基础执行映射。以后可以增加向量化、分块或其他目标后端，但这些实现选择不改变 ASK IR 的算子含义。

## 3. 数值规则和 CPU 一致

自有 CPU、GPU 和 ASK LLVM helper 使用同一套数值构造函数。i32/i64 加乘按对应位宽回绕，不附加未经证明的无溢出假设。bool 在存储中使用一个字节，计算时转为 i1；cast 的饱和、舍入和 NaN 规则见[类型转换与混合精度](15-类型转换与混合精度.md)。

每个 SSA 值的类型决定这一步的精度。f16/bf16 输入转为 f32、在 f32 做加乘、再转回原类型，需要图中显式的 `ask.cast`。代码生成使用 `fp_contract=off`，不把 ASK 分开的乘法和加法暗中改为一次 FMA。

产物的 `required_element_types` 包含中间值使用的类型。图只在内部转换到 bf16，也会要求 bf16 硬件支持。编译目标低于 sm80 时拒绝 bf16；加载到实际 GPU 时再次检查设备能力。支持 bf16 的 GPU 正常执行，不因为 CPU 缺少 BF16 指令而拒绝 GPU 图。

## 4. Python 和 CLI 入口

`Compiler.compile_cuda` 先准备图。有 external 时使用已有的适配和融合路径；没有 external 时调用自有降级，并要求传入目标。纯图不需要 `compilers` 环境：

```python
from ask import Compiler, ExecutionSession, Module, TensorType

module = Module()
type_ = TensorType((3,), "f32")
graph = module.graph("add", inputs={"x": type_, "y": type_}, results=[type_])
graph.return_(graph.add(graph.inputs["x"], graph.inputs["y"]))

with ExecutionSession(device=0) as session:
    major, minor = session.capability
    kernel = Compiler("build/core/bin/ask-opt").compile_cuda(
        module, target={"backend": "cuda", "arch": 10 * major + minor},
        llc="/path/to/llvm/bin/llc", ptxas="/path/to/cuda/bin/ptxas",
        cache_dir="build/add-cache", output_dir="build/add-output",
    )
    executable = kernel.prepare(session)
    x = session.from_host([1, 2, 3], shape=(3,), dtype="f32")
    y = session.from_host([10, 20, 30], shape=(3,), dtype="f32")
    out = session.tensor((3,), "f32")
    executable.submit_many([x, y], [out]).wait()
    print(out.to_host())  # [11.0, 22.0, 33.0]
```

`executable.inputs` 和 `outputs` 按逻辑端口顺序返回类型与形状说明，框架可据此分配结果。`submit_many` 检查端口数量、shape、dtype、可写性和缓冲区重叠，再按物理参数表提交。输入可以重复使用同一只读数据；输出不得覆盖任何输入或其他输出。原来的单输入、单输出 `submit(x, y)` 仍可使用，也允许输入输出类型不同。同步主机列表入口 `kernel.run(values)` 保留原先的一输入一输出、相同 shape/dtype 约定；通用多端口执行使用 `prepare` 与 `submit_many`。

只生成 LLVM 时使用 CLI：

```bash
build/core/bin/ask-opt input.mlir --optimize --emit-cuda-llvm \
  --cuda-arch=86 --cuda-block-size=128 --cuda-ptx-version=80 \
  -o build/graph.ll --report-json build/graph.json
```

`--cuda-arch` 必须明确指定。报告的 `phase` 为 `cuda_llvm`；`cuda_artifact` 保存完整 LLVM 字节、目标、数值选项、端口和 invocation v2；`cuda_lowering` 记录循环分组、生成的输入 load 条数及中间缓冲区数量；这些是降级统计，不是执行时的访存计数。形状快照仍在准备报告中，降成指针后也能核对端口与原始高层值。

`ASK_TARGET_BACKENDS` 这个历史选项目前仍只控制 host 目标文件生成，接受 `none` 或 `host`；CUDA LLVM 生成和运行时属于基础核心，不由第三方编译器选项关闭。`ASK_THIRD_PARTY_COMPILERS=none` 的含义是没有 Triton 这类 DSL 编译器适配器，LLVM/MLIR 仍是 ASK 的基础依赖。CUDA 代码生成还需要 `llc` 和 NVIDIA `ptxas`，执行需要驱动与 GPU；`ptxas` 可来自独立 CUDA 工具安装，不必安装 Triton。

## 5. 缓存、执行和验证

纯图和外部融合共用两级缓存与常驻运行时。自有 IR 缓存键包含高层图、形状、目标、block 配置、数值规则、ASK 工具与流水线版本；张量数值不进入键。IR 命中跳过降级，可执行缓存命中跳过 `llc/ptxas`。分别查看 `kernel.cache` 和 `kernel.codegen["executable_cache"]`。输出目录必须为空；重复编译时使用新的 `output_dir`，保留同一个 `cache_dir` 才能核对缓存复用。

编译结果经过现有的 CUDA v2 契约检查：物理参数与逻辑端口一致、所需类型明确、LLVM 文件与产物 payload 一致。`prepare` 验证产物和 cubin 摘要并加载模块；`submit_many` 只检查设备张量元数据并入队。中间没有 Python 逐元素计算，也不把 GPU 请求转到 CPU。

[数值验证脚本](../tests/test_native_cuda_execution.py) 用 host 与外部融合已有的 `pointwise` 构图器和独立有理数参考，覆盖六类型、类型转换、显式 f32 中间计算、尾部、多 block、零维、空张量及多输入输出。每个唯一图分别验证冷、热编译和真实 GPU 执行；命令与记录字段见[多精度测试](14-多精度测试.md#自有-gpu-图验证)。

FugaTensor 根据图是否包含 external 选择同一 ASK 接口。普通 `@ft.jit` 函数接收 CUDA 张量时即可走本章路径，基础计算不要求注册 Triton；用户编写的 Triton kernel 仍按[第 18 章](18-FugaTensor接入示例.md)提交。

本次在 RTX 3080 上使用 `none` 构建完成六类型矩阵及 grid-stride、标量输入专项：228 次真实 GPU 执行、206244 个结果比较全部通过。两级缓存分别是 114 次 miss、114 次 hit；多端口的重叠缓冲区、缺失输入和错误输出类型在 launch 前拒绝。CPU 及编译器回归检查为 core 12/12、triton 15/15、all 15/15；FugaTensor 使用 none 构建通过 56 项，7 项 Triton 专属测试因未启用适配器而跳过，启用 Triton 的构建通过 63/63。测试覆盖当前基础算子链路，不据此宣称已支持未实现的矩阵乘法或归约。
