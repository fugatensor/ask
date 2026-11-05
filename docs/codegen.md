返回核心思想：[图优化与自有代码生成](concepts.md#ask-owned-codegen) · 相关细节：[ASK IR 与数值语义](ir.md) · [外部产物与执行接口](artifacts.md) · [常驻运行时与多级缓存](runtime.md)

# 图优化与自有代码生成

ASK 对自有计算做图优化，再生成 CPU 目标文件或 CUDA LLVM IR。

<a id="graph-passes"></a>

## 1. 高层图优化（`lib/Transforms/Optimize.cpp`）

`ask::compiler::prepare` 在 `PrepareOptions.optimize = true` 时，完成图校验后按顺序运行三项变换。统计保存在结果 `diagnosticsJSON` 的 `optimization` 字段；CLI 对应 `ask-opt --optimize`：

1. **常量与类型折叠（`fold_constants`）**：
   - 折叠标量 `ask.constant` 上的 `ask.add`、`ask.mul` 与 `ask.cast`，以及 `div`、`compare`、`select` 和输入为 `ask.splat(c)` 的逐元素常量运算（归约为 `ask.splat(folded_c)`）；均匀常量也可穿过 reshape、broadcast 和 transpose 继续参与折叠。
   - 整数遵循 32/64 位二补数回绕；浮点运算在 `llvm::APFloat` 目标格式（`IEEEsingle` / `IEEEhalf` / `BFloat`）下按单步 RNE 执行；浮点转整数执行饱和截断。
   - 含运行时张量的浮点表达式保留运算顺序，例如 `(x + c1) + c2` 和 `x * 0.0`。这样保留各步舍入及 `NaN`、`-0.0` 语义；NaN 的具体 payload 随目标实现变化。
2. **死代码消除（`eliminate_dead_code`）**：
   - 逆序遍历单基本块，删除结果无使用者的自有纯操作（包括形状变换、归约、逐元素操作和 matmul），连带清理变得无用的前驱。`ask.external` 保守保留，即使声明为 `effects = "none"`；其它副作用声明目前会在校验时被拒绝。
   - `ask.mark` 在此前的 `prepare` 阶段提取并擦除。
3. **多输出逐元素融合分组（`fuse_elementwise`）**：
   - 按基本块顺序，将相邻且结果形状一致的自有张量计算划组，至少两个操作才标记 `ask.fusion_group`；纯标量计算可穿插其间，结果形状变化、matmul 或 external 会截断分组。reshape 与同目标形状的下游运算可以同组。操作属性及优化统计中的 `fusion_groups` 记录结果，同时作为函数返回值和下游操作数的中间值（如 `return t, t * 2`）可在共享循环内写出。

matmul 前连续的末两轴转置在融合开启时被吸收到 `transpose_a` / `transpose_b`，仅被该节点消费的转置由 DCE 删除。

<a id="host-codegen"></a>

## 2. Host CPU 共享循环代码生成（`lib/Lowering/Host.cpp`）

`ask::compiler::codegen(source, options)` 将自有图降级为本机 CPU 代码，默认先优化图。`options.target = CodegenTarget::HostLLVM` 返回 LLVM 文本；`HostObject` 将 `.o` 写入 `options.outputPath`。两种模式均通过结果的 `abiJSON` 返回端口契约，调用方负责链接和加载。[C++ 编译接口](api.md#compiler-api)与 `ask-opt --emit-llvm` / `--emit-object` 共用 `ask::codegenSource`：

- **共享循环生成**：按 `fusion_groups` 将同形状的逐元素链合并到同一个 `i64` 线性下标循环中，逐元素中间值保留在 LLVM SSA 寄存器；0-D 张量有一个元素，空输出直接结束对应计算。
- **索引和归约**：reshape 复用线性下标，broadcast 与 transpose 映射到输入下标；归约生成内层循环。共享的 `ScalarEmitter` 为 Host/CUDA 生成这些表达式；跨 kernel 边界或公开输出写入缓冲区。
- **C ABI**：生成函数与 MLIR `func.func` 同名。参数依次为全部输入、全部输出：标量输入按值传递，张量输入和所有输出使用指针。输出指针带 `noalias`，须与所有输入及其它输出缓冲区不重叠；输入之间可共享存储。`abiJSON` 的 `functions` 中各函数的 `inputs` / `outputs` 给出 `kind`、`shape`、`dtype`、`element_bytes` 与物理参数序号，上层（如 [FugaTensor 的 `HostBackend`](../../fugatensor/docs/architecture/backend.md#host-backend)）据此分配缓冲区，按 `abiJSON.link_libraries` 从 capabilities 的 `host_link_libraries` 取得库路径，交给系统链接器并设置加载路径，再调用导出的 C ABI 函数。
- **Host 类型**：CPU 代码生成支持 `bool/i32/i64/f32/f16`，在图优化前静态校验类型。`--capabilities` 的 `host_supported_element_types` 给出可用类型。
- **f32 矩阵乘法**：每个批次调用单线程 OpenBLAS SGEMM。M/N/K、转置、leading dimension、alpha=1 和 beta=0 在编译期固定，矩阵尺寸按 LP64 范围校验。连续输入直接传指针，末两轴转置转换为库参数，输入侧逐元素计算融合后物化；结果写入公开输出或现有临时槽位，后续逐元素链继续融合。K=0 生成零填充，空输出跳过计算。库算法采用 f32 运算并决定累加顺序。
- **f16 矩阵乘法**：在每个批次内按 4 行 × 16 列生成 f32 向量累加，四行共享 B 的读取，每个输出沿 K 次序执行独立乘法与加法，最终舍入为 f16。尾行和尾列由静态形状决定。
- **机器码优化**：`HostObject` 在发出本机目标代码前运行目标相关的 LLVM O3 pipeline，执行循环与向量优化；浮点乘加收缩保持关闭。`HostLLVM` 返回降级阶段的 LLVM IR，ABI 中 `statistics_stage = "host_lowering"` 的统计同样描述该阶段。[通用 LLVM IR 清理](#llir-cleanup) 使用独立的清理规则。

`lowering.loop_count` 记录非空输出批次数，`reduction_loop_count` 记录降级阶段发出的 K 循环及归约循环个数。`blas_matmul_count` 记录生成的 SGEMM 调用位置。实际机器循环由后续目标优化确定；性能分析使用最终机器码和独立计时。

标量归约在输出循环前计算。部分归约结果被扩展或供矩阵乘法复用时，先物化到函数级临时空间。代码生成器记录各物化阶段及输出批次实际发出的临时区读取，按最后读取位置复用容量和对齐足够的槽位；同一阶段的读写占用不同槽位。布局在编译期校验后写为固定偏移，执行时统一 `malloc`、结束时 `free`。单元素结果通常保留为 SSA 值；SGEMM 需要地址的输入和中间结果使用同一物化临时区。函数的 `lowering.temporary_workspace_bytes` 报告复用后的字节数，`materialized_tensor_count` 报告物化值数量；容量为零时生成代码保留纯计算路径。沿用的 void C ABI 在临时空间分配失败时调用 `abort`。

<a id="owned-cuda"></a>

## 3. 自有 CUDA 降级（`lib/Lowering/CUDA.cpp`）

自有 GPU 图通过 `ask::compiler::codegen` 降级，选项指定 `CodegenTarget::CUDALLVM` 和目标 `architecture`，例如 `86`。结果的 `ir` 返回 LLVM 文本，`cudaArtifactJSON` 返回 CUDA 产物契约。调用方用 `llc` / `ptxas` 生成 PTX 和 cubin，并在编译缓存中记录 Compiler 共享库或所选 CLI 的文件摘要。exp/log 产生 `__nv_expf` / `__nv_logf` 引用；调用方先用匹配的 `llvm-link` 链接 CUDA Toolkit 的 `nvvm/libdevice/libdevice.10.bc`。FugaTensor 将链接工具及 libdevice 摘要纳入产物缓存。

- **NVPTX 原生 LLVM IR**：生成 `target triple = "nvptx64-nvidia-cuda"` 模块与 `define ptx_kernel void @<entry>(...)`，使用 `addrspace(1)` 全局显存指针。
- **64 位 grid-stride 循环**：读取 `@llvm.nvvm.read.ptx.sreg.{ctaid,tid}.x` 后扩展为 `i64`，按固定 block size 计算起始下标。grid 的 x 维最多为 65535；线程调度的步长为 `grid.x * block_size`，warp 调度的步长再除以 32。混合输出组共用满足各组需求的 grid。
- **Warp 协作归约**：在 `sm_70+`、block size 为 32 倍数时，独立且归约域至少为 32 的 `reduce_max` 与 `reduce_sum` 可由一个 warp 协作计算一个输出。选择还要求归约结果与该输出组元素数一致；各 lane 按 32 步进读取（大尺寸启用 4 路 ILP 展开与独立累加流），五次 shuffle-XOR 合并，lane 0 写出结果。尾部读取只为不足 32 的部分生成谓词，之后全部 lane 汇合。浮点求和使用 `reassoc` fast-math 标志以充分利用硬件并行流水线，浮点最大值使用传播 NaN、区分正负零的 `llvm.maximum`，i64 通过两个 32 位 shuffle 保留完整位模式。前后逐元素表达式继续在同一 kernel 中融合。
- **多端口与边界处理**：支持多张量/标量输入、异构形状与 dtype 的多输出端口、非整块尾元素掩码保护（`idx < N`）以及空张量。
- **浮点与架构检查**：
  - 浮点指令不附加 `contract` / `fast` 标志，配合 `llc -fp-contract=off` 保证 `a * b + c` 按独立乘法和加法两次 RNE 执行。
  - 图中输入、输出或中间值包含 `bf16` 时，编译期要求 `arch >= 80`，并在产物 `required_element_types` 中记录 `"bf16"` 供 [CUDA 运行时](runtime.md#execution-session)加载时核对物理显卡能力。

连续逐元素表达式在一个 kernel 内计算，中间值留在寄存器。归约后再次扩展结果时，分区器先写出归约值，后续 kernel 复用它。`ask.matmul` 经分区后交给 cuBLASLt，见[库步骤](partitioning.md#partition-boundaries)。报告中的 `reduction_loop_count` 记录生成的归约循环数；`warp_reduction_groups` 记录 warp 协作输出组数，各组的 `threads_per_output` 为 1 或 32。

<a id="llir-cleanup"></a>

## 4. 原生 LLVM IR 清理（`lib/LLIR/Optimize.cpp`）

`ask-opt --optimize-llir` 单独清理读入的 LLVM IR。自有 Host/CUDA 降级后的优化由对应的 LLVM 后端处理。

- 按顺序执行 InstCombine、SimplifyCFG、ADCE，跳过带 `optnone` 的函数；不启用通用函数内联、循环变换、向量化或模块级未引用函数删除。
- 移除已失效的 ASK 追踪元数据和指令级 ASK 附件，保留 `ask.abi`、`ask.shape_snapshot` 等模块接口元数据，并在报告中列出未验证的其它 ASK 命名元数据。
- 优化前后验证 LLVM 模块，并检查目标信息与对外函数 ABI 不变。内联 PTX 按其 LLVM `sideeffect`、操作数和约束语义参与优化；没有副作用且结果未使用的汇编可以被删除。
