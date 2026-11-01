# Triton / ASK 原生 LLVM IR 融合、GPU 执行与缓存

[返回顶层设计](核心思想.md#ask-c) · [B 层导入](03-B层多源降级与低层归一化.md) · [运行时设计](05-代码生成与运行时.md) · [框架 API](10-上层框架接入.md)

本章描述已经实现的窄路径。输入仍是框架生成的 ASK IR，外部节点仍携带 Triton DSL。例子是 `P: t=x+1 → F: u=t*2 → Q: y=u+3`，三个计算最终在一个 GPU kernel 中执行。

## 1. 一次编译到底做了什么

1. `prepare` 检查图，推导具体形状，提取并删除高层 `ask.mark`，保存 ShapeSnapshot。
2. Triton 适配器检查 F 的整个 DSL 函数体。只有输入、输出访问能被完整说明时，才加入两个 identity 追踪点。追踪点不包含 P/Q 的算法。
3. 通过公开 `triton.compile` 取得原生 LLVM IR。普通 load/store 可能已经变成内联 PTX，按原样保留。同时检查插入追踪点前后的 TTGIR 布局和启动资源一致。
4. ASK 将 P 和 Q 分别降成原生 LLVM 标量函数。本例生成 `__ask_pre(i32 x) = x+1` 和 `__ask_post(i32 u) = u+3`。整数运算不带 `nsw/nuw`。
5. LLVM Linker 合并这两个函数与 F 的模块。融合器在确认的输入、输出位置插入调用，再用 LLVM `InlineFunction` 内联，删除追踪点和 helper 定义，运行基础 LLVM 清理并再次验证模块。
6. 将**融合后的 LLVM 文件**交给锁定 SDK 的 `llc`，生成 PTX，再用 `ptxas` 生成新 cubin。CUDA Driver API 加载这份 cubin，绑定参数并启动。

这条路径没有使用 Triton 原来的 cubin，也没有在 CPU 上先执行 P/Q。导出目录中的 `ask.helpers.ll` 和 `external.ll` 是合并的两个输入，`fused.ll` 是实际代码生成的输入。

## 2. 怎样找到边界，为什么允许内联 PTX

首版适配器接受以下形式，变量名可以变化；`mask` 和 `other` 使用关键字参数：

```python
i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
x = tl.load(input_ptr + i, mask=i < N, other=0)
# 可有若干仅由 x、此前结果、i32 常量组成的加、减、乘。
tl.store(output_ptr + i, x * 2, mask=i < N)
```

AST 检查要求只有一次输入读、一次输出写，二者索引和掩码相同。禁止重赋值、指针逃逸、额外调用、原子操作、分支、循环和归约。无法说明整个函数体，就返回 `UnsupportedFusion`。

适配器用公开 `tl.inline_asm_elementwise` 在读取结果、写入值上各插入一个已知 identity：`mov.b32 $0, $1`。它显式携带值、地址、索引和掩码。编译时暂用 `is_pure=False` 保留这两个位置；这只是低层追踪手段，不是给高层 `ask.mark` 增加副作用。标记既不读写内存，也不做同步，融合结束后必须全部删除。

融合器识别的是 ASK 自己定义的精确标记协议，并检查 LLVM 调用类型、端口地址、两端索引和掩码，以及绑定该模块的摘要。它不根据未知 PTX 字符串猜测 load/store 的语义。普通外部 LLVM 导入也不要求有这些标记；只有这个可选融合 Pass 要求边界证明。

输入侧实际替换为 `select(mask, P(loaded), loaded)`。无效 lane 继续使用 F 原来的 `other=0`。输出侧把值替换为 `Q(value)`，保留 F 原来的 store 和 mask。x、y 使用独立分配，t、u 不再物化。

`fusion_contract` 记录 DSL 检查结论、原源码摘要、插入标记后的源码摘要和导出 LLVM 摘要。它来自受信任的适配器，并不是对任意外部程序的形式化证明。模块被改写后，这份契约移到 `source_fusion_contract`，只表示来源；不能再用它定位新 IR 的指令。高层 ShapeSnapshot 则写入融合产物及 `!ask.shape_snapshot`，不会从 LLVM 指针反推形状。

## 3. 当前支持范围

| 项目 | 当前要求 |
|---|---|
| ASK 图 | 一个函数、一个一维静态 i32 输入、一个同形状输出、一个 external |
| P / Q | `constant/splat/add/mul`；P 只依赖图输入，Q 只依赖 F 输出及常量 |
| F | 上述 AST 子集，Triton 3.7.1；边界 identity 可保留为一对标量 LLVM 调用 |
| 形状 | `1 <= N <= 2147483520`，N 必须与 ASK 类型一致 |
| 固定策略 | `BLOCK=128`、4 warps、`grid=[ceil(N/128),1,1]`、无共享或 scratch 存储 |
| 目标 | CUDA sm80，warp32；运行器核对设备二进制兼容性 |
| ABI | 两个 i32 数据指针、两个已证明未使用的 scratch 指针，后两者传 null |
| 运行输入 | Python i32 列表，复制到连续且互不重叠的设备缓冲区 |

这些限制属于首个适配器和融合 Pass，不是 ASK IR 的通用规则。F 的 DSL 算法、grid、warps 和布局由请求确定；ASK 不为融合改选这些参数。没有覆盖的图会报不支持，当前没有完整混合图调度器替它自动选择三次独立调用。零长度融合也明确拒绝。

## 4. 框架怎样调用

```python
compiler = Compiler("build/triton/bin/ask-opt", timeout=120)
kernel = compiler.compile_cuda(
    module,
    cache_dir="build/ir-cache",
    output_dir="build/my-run",             # 使用空目录
    triton_python="/path/to/triton/python",
    triton_worker="adapters/triton/worker.py",
    llc=".deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc",
    ptxas="/path/to/triton/backends/nvidia/bin/ptxas",
)
execution = kernel.run(values)
print(execution["output"])
print(kernel.cache["status"], kernel.cache["counters"])
```

Python SDK 本身不导入 Triton 或 CUDA。显式调用编译入口时先检查构建允许清单，`none` 构建在读缓存、启动 worker 前就拒绝请求。`run` 才加载 CUDA driver；无需 PyTorch、NumPy 或 CUDA Python 包。

`--capabilities` 中的 `gpu_execution` / `cuda_execution_via_python` 描述本构建是否启用了这条 SDK 路径，不是 GPU 探测结果。`ask-opt` 本身没有 launch 命令，实际工具和设备是否可用由编译、运行时检查。

代码生成记录 LLVM / PTX / cubin 摘要。运行器核对输入 LLVM、调用产物和 cubin 的摘要，检查启动描述、设备能力，并查询加载后 kernel 的参数 ABI 和资源。每次调用分配输入、输出，在独立 CUDA context 中启动并同步后读取结果，最后释放资源。提交失败会报告错误并清理，不自动重放。它还不是框架共享 stream、异步事件或零拷贝张量运行时。

只想查看 LLVM 融合结果时，可用原生入口：

```bash
build/triton/bin/ask-opt tests/ir/pipeline.mlir --fuse-external \
  --export-dir build/fusion-export \
  --triton-python /path/to/triton/python \
  --triton-worker adapters/triton/worker.py \
  -o build/fused.ll --report-json build/fusion-report.json
```

CLI 导出并融合，不启动 GPU，也不访问持久缓存。`--export-external` 仍只导出未经追踪改写的 F。

## 5. 缓存命中究竟省了什么

`Compiler.compile_cuda` 使用 ASK 自己的磁盘 IR 缓存。缓存保存融合 LLVM、ASK helper LLVM、带边界标记的 Triton LLVM、各自调用说明及形状/融合报告。它不保存运行结果、设备地址或 context。

键包含原始 ASK 图和 DSL 的内容摘要、完整固定配置、输入形状及形状规则、目标、i32 数值规则、流水线版本、LLVM revision、ASK 可执行文件、worker/边界适配器、选中的 Python、llc、ptxas 的内容摘要。Triton 安装包中的源码、libtriton、libdevice 等文件也按内容计算摘要；只看版本号不够。环境变量整体计算摘要，报告不保存变量明文。

每次调用先重新检查图和当前依赖，再查缓存。命中后跳过 Triton 导出和 LLVM 融合，继续用缓存中的 `fused.ll` 重新生成 PTX/cubin。因此 `hit` 指 **ASK 的融合 IR 命中**，不表示 Triton 自身缓存命中，也不表示复用了旧 GPU 二进制。

首版融合禁止 `TRITON_KERNEL_OVERRIDE`、自定义 libdevice/ptxas 路径、编译插件等环境覆盖，因为仅记录路径不能说明文件内容是否变化，也不能保证被替换的代码仍对应 DSL 证明。检查发生在缓存读取前，worker 同样检查。允许的 Triton 环境项仅为 `TRITON_CACHE_DIR`、`TRITON_HOME`、`TRITON_ALWAYS_COMPILE`、`TRITON_KERNEL_DUMP`、`TRITON_DUMP_DIR`、`TRITON_DISABLE_LINE_INFO`。

读取时逐项核对文件大小和 SHA-256；缺文件或内容损坏视为 `invalid`，重建后才能返回。相同 key 使用进程锁，只有拿锁的请求负责构建；完整文件写入临时目录后原子发布。构建抛错时不产生成功条目。

每次成功请求返回一次统计，可由上层累加：

| status | requests | hits | misses | invalid | builds |
|---|---:|---:|---:|---:|---:|
| `miss` | 1 | 0 | 1 | 0 | 1 |
| `hit` | 1 | 1 | 0 | 0 | 0 |
| `invalid`，重建成功 | 1 | 0 | 1 | 1 | 1 |

修改 P/Q 常量、F DSL、形状、目标或工具链会形成另一个 key。改变输入张量的**数值**不改变 key：这些值在运行时复制给 kernel。

## 6. 如何复现数值验证

[triton_fused_gpu.py](../examples/triton_fused_gpu.py) 自行构造 ASK 图，执行冷缓存和热缓存两次编译，并与逐步 i32 回绕的参考结果逐位比较：

```bash
PYTHONPATH=python python3 examples/triton_fused_gpu.py \
  --ask-opt build/triton/bin/ask-opt \
  --triton-python /path/to/triton/python \
  --triton-worker adapters/triton/worker.py \
  --llc .deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc \
  --ptxas /path/to/triton/backends/nvidia/bin/ptxas \
  --cache-dir build/new-validation-cache \
  --output-dir build/new-validation-results
```

脚本要求新缓存，验证 N 为 1、10、127、128、129、257、4097 的七组输入，覆盖整 block、尾部 mask、多 block、正负边界和固定种子随机数；再分别修改 ASK 常量、Triton DSL 和形状检查失效。输出 `execution.json`、代码生成命令和文件摘要，以及总表 `validation.json`。GPU 测试需要实际设备，普通 CTest 不会把“无 GPU 时跳过”记作数值验证成功。

无 GPU 的测试另行检查真实 Triton 导出后的 LLVM link/inline、失配的形状/摘要/边界拒绝，以及缓存损坏、失败后重试和多进程竞争。相关实现：[Fuse.cpp](../lib/LLIR/Fuse.cpp)、[Triton 边界导出](../adapters/triton/fusion.py)、[GPU 编译入口](../python/ask/gpu.py)、[CUDA 运行器](../python/ask/cuda.py)、[IR 缓存](../python/ask/cache.py)。

本次在 RTX 3080（sm86）上使用 Triton 3.7.1、锁定 LLVM 23.0.0git 和 ptxas 12.8.93 实测：17 次 GPU 执行全部逐位一致，IR 缓存 7 次命中、10 次未命中。另一次故意破坏 `fused.ll` 缓存文件，返回 `invalid` 并重建，GPU 结果仍正确。核心构建的 8 组 CTest 和 Triton 构建的 10 组 CTest 均通过。这里只证明表中子集，未宣称任意 Triton kernel 可以融合。
