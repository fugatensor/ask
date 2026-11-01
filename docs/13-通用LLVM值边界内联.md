# 通用 LLVM 值边界内联

[返回顶层设计](核心思想.md#ask-c) · [C 层优化](04-C层统一处理与可选优化.md) · [Triton GPU 路径](12-GPU融合与IR缓存.md) · [适配器注册](17-第三方编译器适配与注册.md)

一个融合实现需要回答两个不同的问题：在哪里替换值才是合法的，以及确认位置后怎样改写 LLVM。前者需要图、访存和编译器信息；后者可以共用。现在两部分通过 `BoundaryInlineSite` 交接。

## 1. 核心接收什么

[inlineValueBoundaries](../include/ask/LLIR/Inline.h) 接收两个原生 `llvm::Module` 和一组站点：

```cpp
struct BoundaryInlineSite {
  llvm::CallInst *marker;
  llvm::Function *helper;
  llvm::SmallVector<llvm::Value *> arguments;
  llvm::Value *predicate;
  llvm::Value *passthrough;
};
```

`marker` 位于消费者模块。调用方已经确认它是可移除的 identity 追踪点；它的返回值就是要替换的值。`helper` 位于另一个模块，描述要接入的计算；`arguments` 来自消费者当前已经可用的 SSA 值。

例如，图分析确认某次读取取得 x[i]，而 P 定义 t[i]=x[i]+1。调用方把这个读取结果传给 `P_value`，把对应边界标记作为 `marker`。如果 P 还需要另一个已经取得的标量值，也可以作为第二个参数传入；内联核心不限制参数数量。

有掩码时，`predicate` 表示本次是否应用 helper，`passthrough` 表示无效位置保留什么值。标量掩码用条件分支保护 helper 的执行；向量掩码按 lane 选择结果，只接受能够安全逐 lane 求值的直线 helper。这样不会让无效 lane 的值进入一个本来不该执行的条件分支。

向量掩码只规定选择哪些输出。若 helper 使用 shuffle 等跨 lane 运算，提供方仍须证明活跃输出依赖的输入 lane 有效；掩码本身不提供这个证明。

这些字段使用当前 LLVM 模块的对象引用，属于进程内 C++ API。它们不是持久化格式。文件重载、内联或克隆后，需要重新绑定站点；缓存仍保存 LLVM 文件及来源说明，不保存指针。

## 2. 核心负责什么

内联核心按顺序做以下检查和改写：

1. 两个模块通过 LLVM verifier，使用同一个 context，目标和 data layout 兼容。
2. helper 的参数、返回值和站点类型相符；参数、掩码和 passthrough 支配替换位置。
3. 检查 helper 的实际指令。支持整数、浮点和定长向量的纯计算、受限 intrinsic、无环控制流及无递归的纯 helper 调用。不根据一个自称 `memory(none)` 的属性就信任未知函数。
4. 克隆输入模块，在副本上链接、内联和删除 identity 标记，再验证结果。
5. 成功时返回新模块及报告；失败时不修改输入模块，也不发布成功报告。调用方可以继续使用原产物。

helper 不能引入内存访问、指针计算、未知 asm/call、整数除零风险、递归或无限循环。影响消费者整体含义的 module flags 和未经归一的目标属性也不能借链接偷偷带入。完整约束以 [Inline.h](../include/ask/LLIR/Inline.h) 为准。

浮点 helper 还要检查消费者函数的数值属性。若消费者要求忽略 NaN、丢掉零的符号、冲掉非正规数或使用不兼容的异常规则，helper 内联进去后会继承这些条件；普通类型检查发现不了这种变化。核心检查实际 helper 调用链中的浮点计算，遇到冲突则拒绝，并保留原模块和报告。指令各自携带的合法数值标志保留，不把一个区域的许可传播给另一个区域。

这里没有 Triton 版本、CUDA 型号、block 大小、warp 数、张量 rank 或固定 i32 条件。一个 helper 可以用于多个站点；消费者可以包含原有的分支、循环、目标 intrinsic 和内联汇编。核心保留这些代码，不尝试解释其中未知的访存行为。

## 3. 合法性依据由谁提供

`BoundaryInlineSite` 不是“这个位置可以融合”的自证。构造它的分析或适配器必须先确认：逻辑值对应正确、所需参数已经产生、无效位置语义明确、别名和副作用允许替换、需要的执行顺序不会丢失。核心只能检查 LLVM 中可直接检查的类型、支配和 helper 行为，无法从一个 CallInst 指针知道完整图语义。

目前有两种使用方式：

| 来源 | 如何确定站点 | 调用的改写实现 |
|---|---|---|
| 已掌握含义的普通 LLVM 模块 | 提供方直接绑定 identity 调用及参数 | `inlineValueBoundaries` |
| 当前 Triton 适配器 | DSL 数据流分析、公开 identity 追踪、产物摘要和 LLVM 地址/掩码检查 | 同一个 `inlineValueBoundaries` |

Triton 的检查放在 [fusion.py](../adapters/triton/fusion.py) 和 [TritonBoundary.cpp](../adapters/triton/lib/TritonBoundary.cpp)。它们处理版本、PTX 标记格式和启动说明，再把已确认的位置交给核心。[Fuse.cpp](../lib/LLIR/Fuse.cpp) 接收调用方传入的 `CompilerAdapter`，调用它的 `prepareFusion`，取得输入/输出值边界以及已规范化的代码生成选项，再降级 ASK helper。核心不根据编译器名字选择分支。

适配器的 `validateArtifact` 检查私有证据，`finalizeArtifact` 在模块改写后处理已失效的来源信息。通用导出入口负责请求与响应的一致性、LLVM 模块验证和实际入口 ABI。生产注册表只装配构建选中的适配器；核心测试可以直接注入测试描述符，不把假编译器放进生产注册表。接口和接入步骤见[第 17 章](17-第三方编译器适配与注册.md)。

接入另一种编译器时，需要提供该编译器的导出和边界绑定方式，无需复制链接、内联、类型检查和事务处理。暂时拿不到边界依据的模块，仍可以导入、验证和运行其他适用的 Pass；不能因此擅自删除其访存。

## 4. 当前已经放宽了哪些限制

Triton 路径使用请求里的 `BLOCK` 和 `num_warps`，保留原来的执行策略。BLOCK 可以是 1–1024 的二次幂，warp 数目前支持 4 或 8。适配器比较原始与插入追踪后的实际布局，不再要求所有输入都采用同一个布局字符串。

当一个线程处理多个元素时，编译器可能把一个逻辑边界展开成多条 LLVM 标记。契约记录实际数量，绑定层逐个核对，核心在每个站点内联同一个 helper。没有“必须刚好两个标记”的限制。

DSL 检查沿赋值追踪整数、索引、掩码、指针和已加载值。它接受位置/关键字 mask、局部别名、等价的交换运算，以及能按赋值顺序解释的局部重赋值。索引仍需满足已支持的逻辑关系，但不再要求源码逐句长成示例的样子。

ASK 图接受相同形状的任意 rank 静态 bool / i32 / i64 / f32 / f16 / bf16 张量。当前 Triton 高层融合入口仍使用一个输入、一个同形状同 dtype 的输出和连续缓冲区。Triton 适配器中的 N 是元素总数，ShapeSnapshot 和规范 `port_map.shape` 保留原始多维形状。通用运行器只读取端口形状和三轴启动尺寸，不解释 BLOCK、N 或 warp 数。

## 5. 哪些能力仍然没有实现

ASK 图、helper 降级、Triton 边界标记和 GPU 缓冲区支持六种声明类型；运行入口按产物 dtype 选择数据编码。`ask.cast` 支持六类互转，可以显式写出低精度存储、f32 中间计算再窄化的图，见[类型转换与混合精度](15-类型转换与混合精度.md)。通用 LLVM 接口还支持其他整数、浮点和定长向量类型，但高层图仍不支持 f64 或向量元素类型。

第三方 LLVM 融合路径目前仍是一个 external 的单输入、单输出、同存储 dtype 边界的逐元素组合。无 external 的图已经有[自有 GPU 降级](19-自有GPU降级.md)，支持多个输入输出，不需要借用 Triton kernel。归约、任意 gather/scatter、跨线程供值、多输出别名以及任意未知 PTX 的访问分析尚未实现；失败时也还没有完整混合图运行时自动调度多个 kernel。这些限制在各自层继续明确报告，不写成通用内联接口的限制。

## 6. 如何验证通用性

`ask-boundary-inline` 检查普通 LLVM 的多站点、多参数、类型、掩码和失败回滚。`ask-boundary-execution` 将同一结果通过 host `llc` 生成机器码，执行 i32、i64、f32、f64 和向量计算，包含无效分支、整数溢出和浮点特殊值。两个测试都不需要 Triton 或 GPU。

GPU 验证使用同一组图覆盖不同 dtype，包含二维、三维形状、不同 BLOCK/warp 组合、多个 LLVM 边界站点、DSL 别名以及 P/Q 链式算术：

```bash
# 工具路径与第 12 章相同，使用新的缓存和输出目录。
PYTHONPATH=build/triton/python:python python3 examples/triton_fused_gpu.py \
  --ask-opt build/triton/bin/ask-opt \
  --triton-python /path/to/triton/python \
  --triton-worker adapters/triton/worker.py \
  --llc .deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc \
  --ptxas /path/to/triton/backends/nvidia/bin/ptxas \
  --cache-dir build/generalized-cache \
  --output-dir build/generalized-results --dtypes bool i32 i64 f32 f16 bf16 --extended
```

该验证同时检查 GPU 输出、冷/热缓存状态和多维 ShapeSnapshot。LLVM 结构合法、融合位置正确和设备数值一致，分别验收。图与数值参考的复用方式见[多精度测试](14-多精度测试.md)。

此前类型与混合精度实现阶段，六类型基础矩阵加专项共通过 266 次融合 GPU 执行，比较 28842 个结果，缓存命中与未命中各 133 次。覆盖 36 个 cast 方向、显式 f32 中间计算、二维 / 三维形状、4 / 8 warps、多个边界站点和源码别名；i64 舍入、混合精度下溢和 F 的 FMA 开关也通过。完整数量和硬件验证范围见[GPU 数值验证](12-GPU融合与IR缓存.md#6-如何复现数值验证)。独立 host 测试还执行了其他浮点、整数和向量接口，结果不依赖 Triton 导出。

适配器拆分后，另用手写 LLVM kernel 在 `none` 构建的运行时上执行 2/5 参数、三维启动和共享内存，576 个数值结果通过。此检查覆盖通用 CUDA 执行契约；测试描述符覆盖适配接口，并不构成第二个真实编译器的接入记录。
