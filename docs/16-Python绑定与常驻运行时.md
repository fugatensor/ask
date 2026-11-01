# 从 Python 提交设备张量，重复执行已编译代码

[返回顶层设计：执行流程](核心思想.md#ask-pipeline) · [构建选择](09-构建与可选编译器.md#core) · [框架接入](10-上层框架接入.md) · [GPU 编译](12-GPU融合与IR缓存.md)

Python 负责构图和调用。`Compiler.prepare/optimize/capabilities` 以及 CUDA 资源管理通过 nanobind 调用 C++。冷编译的 GPU 融合流程仍由 Python 装配 `ask-opt`、外部 worker 和代码生成工具；这部分没有冒充已经全部迁入进程内。Python 绑定属于 ASK 基础组成，`none/all/triton` 都包含它；Triton 仍在独立 worker 中编译 DSL。这两条路径不能混在一起。

## 1. 编译一次，加载一次，再重复提交

原来的 `kernel.run(values)` 接收 Python 列表，创建上下文和缓冲区，执行后读回列表。它保留为同步验证入口。训练循环应把输入输出保留在设备上：

```python
from ask import ExecutionSession

# kernel 是 Compiler.compile_cuda(...) 返回的 CUDAKernel。
with ExecutionSession(device=0) as session:
    executable = kernel.prepare(session)
    x = session.from_host(values, shape=executable.shape, dtype=executable.dtype)
    y = session.tensor(executable.shape, executable.dtype)

    for _ in range(100):
        done = executable.submit(x, y)

    done.wait()
    result = y.to_host()
    print(session.stats)
```

`prepare` 校验产物、LLVM 和 cubin 摘要，固定参数说明，再加载模块。之后 `submit` 检查张量的形状、类型、所属 Session 和读写条件，绑定设备地址并入队；不读取产物文件，不上传输入，不下载结果，也不创建上下文或重新分配 x/y。

当前高级执行入口沿用已有逐元素图 ABI：一个输入、一个输出、两个空 scratch 指针。运行时没有扩展编译器的算子集合，也没有实现任意混合图调度。`DeviceTensor` 的形状描述本身支持多维；同样有 15 个元素的 `(15,)` 与 `(3, 5)` 不被当成同一张量类型。

## 2. 设备张量保存什么

`DeviceTensor` 保存设备地址、shape、dtype、字节数、只读状态和资源所有权。支持 bool、i32、i64、f32、f16、bf16；存储宽度沿用 ASK IR。它不实现自动微分，也不替代上层框架的 Tensor 类。

`session.tensor(shape, dtype)` 分配未初始化的连续存储。`from_host`、`copy_from_host` 和 `to_host` 是明确同步的便捷接口，Python 列表的逐元素编码只发生在这些边界。反复提交设备张量不经过这段编码。

上层已有 CUDA 张量时，可用 `session.from_cuda_array(owner)` 接入 `__cuda_array_interface__`。接口检查设备、连续布局、类型、可查询的分配范围和只读状态，保留 owner，并按来源 stream 建立依赖。每次提交外部张量时重新读取来源 stream，核对指针、形状、类型和只读声明没有改变，再插入必要的事件等待；上层在两次调用间更新数据后，ASK 不会沿用过期的生产依赖。自有张量不需要读取这份外部协议。旧版 CAI v2 没有定义同步。缺少 stream 声明时，调用方必须显式传 `producer_stream=...`，否则拒绝导入；`1/2` 分别代表 legacy / per-thread 默认流，其他正整数代表真实 stream，显式 `None` 表示调用方保证每次访问前数据已经完成。不能根据一个缺失字段猜“已经就绪”。

导入对象在异步执行期间属于借用状态。上层再次读写原对象前，必须等待返回的 Event，或在自己的 stream 建立完成依赖；ASK 不会自动给外部框架的后续调用加锁。它也不能把任意整数地址当成可信张量，或把不连续视图偷偷当成连续数组。输出不允许只读，当前图还要求输入输出存储不重叠。

CUDA Array Interface 没有通用的 bf16 类型编码，不能把 `uint16` 冒充 bf16。ASK 自有 bf16 张量可以执行；外部 bf16 张量互操作还需要单独接通支持该类型的协议，例如 DLPack。当前没有声明完整 DLPack 支持。[CUDA Array Interface 规范](https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html)

## 3. Session 和事件怎样管理资源

Session 保留选定设备的 primary context、自己的 stream 和已加载模块。每次进入驱动调用时设置当前线程的 context，离开时恢复调用者的 context。使用 primary context 是为了与框架管理的设备内存互操作，不是每次执行创建一份新的私有上下文。

同一个 Session 的提交按同一 stream 排序。`submit` 返回 Event，表示任务已提交；`query()` 查询完成状态，`wait()` 等待该事件。读取结果、把结果交给其他 stream 或释放外部所有者之前，必须建立相应完成依赖。异步错误在后续查询或等待时抛出，不自动重放已经提交的计算。

未完成任务保留模块、参数缓冲区和外部 owner。用户删除返回的 Event，不能提前释放 GPU 仍在访问的对象。Session 在回收已完成事件时释放这些保留引用，`close()` 等待自己的 stream 后关闭会话。仍被调用方持有的张量或内核对象可以继续保有底层资源，最后一个引用释放时再回收；关闭后的会话不能继续提交或复制。它不是全设备同步接口，也不能销毁其他框架持有的 primary context。

bf16 检查针对实际选定的设备，且包括只在内核中间值使用 bf16 的情况。缓存命中不绕过这项检查。cubin 也按真实设备计算能力检查兼容性，不因为文件能读到就加载。

## 4. 三种缓存分别省什么

| 缓存 | 保存内容 | 命中后省掉什么 |
|---|---|---|
| IR 磁盘缓存 | 融合后的 LLVM、产物说明与报告 | 外部 DSL 导出及 ASK 融合 |
| 可执行产物磁盘缓存 | PTX、cubin 与代码生成记录 | `llc` 和 `ptxas` |
| Session 模块缓存 | 当前 context 中的 module/function 句柄 | 重复模块加载与符号查询 |

可执行缓存键包含实际 LLVM 字节、完整产物契约、目标架构、启动 ABI、数值选项、工具二进制摘要及代码生成版本。输出目录不是编译语义；同一个结果可以写到新的目录，新的 receipt 使用新路径。张量数据不参与编译缓存键。

缓存按条目加锁，完整写入并检查后才发布。损坏条目记为 invalid 和 miss，重新构建；工具失败不发布半成品。命中仍检查文件摘要。磁盘上不保存设备指针、CUDA 句柄或 Python 对象，进程重启后必须重新加载模块。

`kernel.cache` 保留 IR 缓存统计；`kernel.codegen["executable_cache"]` 与 `kernel.report["executable_cache"]` 返回可执行缓存统计。两者分别统计 requests、hits、misses、invalid、builds，不能拿 IR hit 冒充机器码 hit。Session 的 `stats` 另外报告模块加载、模块命中、分配、提交与同步次数。

## 5. 怎样检查这条路径确实工作

数值测试复用[第 14 章](14-多精度测试.md)的图构造器与独立参考值，同一套图遍历六种类型，另外检查 f16/bf16 输入、f32 计算再转回。真实 GPU 验证同时检查：冷编译 miss，重复编译两级缓存 hit；重复 prepare 命中模块；多次 submit 期间分配、模块加载和同步计数不增加；最终等待后输出与参考值一致。

参数错误、缓存损坏、类型或形状不符、只读输出和重叠缓冲区在各自入口拒绝。没有 GPU 的测试只能验证契约和缓存行为；真实设备运行单独记录，不能用模拟驱动通过代替。

在 RTX 3080（sm86）上，本轮 15 个图/dtype 组合全部通过，覆盖六种存储类型、多维形状和显式 f32 中间计算。120 次重复提交期间没有增加分配、模块加载或同步；12 个组合还验证了跨 Session 的 stream 依赖与零拷贝传递。IR 和可执行缓存分别为 15 次 miss、15 次 hit、0 次 invalid。另测 120 次跨线程提交、事件及外部 owner 生命周期；`none` 构建独立执行六类型产物也通过，未导入 Triton。这里记录功能和计数验证，不据此推算性能加速倍数。

实现入口：[原生运行时](../include/ask/Runtime/CUDA.h)、[设备张量 API](../python/ask/runtime.py)、[产物检查与提交](../python/ask/executable.py)、[共用图验证](../tests/support/runtime_validation.py)。

相关设计：[代码生成与调用契约](05-代码生成与运行时.md)、[形状保留](ask-ir/03-降级与信息保留.md)、[混合精度](15-类型转换与混合精度.md)。
