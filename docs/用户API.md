# ASK 用户 API

返回核心思想：[构建与进程内绑定](核心思想.md#ask-frontend-build) · 相关细节：[安装](安装.md) · [外部编译器接入](09-外部编译器适配接入规范.md) · [常驻运行时](05-常驻运行时与多级缓存.md)

本文面向直接调用 ASK 的框架和应用开发者，介绍公开 Python 接口及其调用顺序。ASK 的 `Value` 表示图中的值，`DeviceTensor` 表示设备存储；构图本身不执行计算。需要 Tensor 算术、自动组图或自动微分时，使用上层 [FugaTensor](../../fugatensor/README.md)。

<a id="api-entrypoints"></a>

## 1. 接口入口

安装包名为 `ask-compiler`，Python 导入名为 `ask`。

| 用途 | 入口 |
| --- | --- |
| 描述类型、构造图 | `ScalarType`、`TensorType`、`Module`；由 Module 创建 `Graph`、`Semantics`、`Implementation` |
| 校验、优化、切分、CUDA 编译 | `Compiler`；准备结果为 `PreparedModule` |
| 配置调用方编译器 | `CompilerEnvironment` |
| 查看编译产物 | `CUDAKernel` 或 `CUDAProgram` |
| 分配设备存储、执行 | `ExecutionSession`、`DeviceTensor`、`Event` |
| 编译错误 | `CompilerError`；CUDA 错误从 `ask.cuda` 导入 `CUDAError` |

这些对象的源码入口见 [`ask/__init__.py`](../python/ask/__init__.py)。通常通过工厂方法取得 Graph、Value、设备张量和可执行对象，无需手动构造它们。

<a id="graph-api"></a>

## 2. 类型与构图

`ScalarType(dtype)` 和 `TensorType(shape, dtype)` 支持 `bool`、`i32`、`i64`、`f32`、`f16`、`bf16`；`fp16` 是 `f16` 的别名。默认 dtype 为 `i32`。`TensorType((), "f32")` 是 0-D 张量，仍通过指针传递；它与 `ScalarType("f32")` 不同。形状声明可含 `None`，实际编译执行前须推导为具体维度。

下面构造 `y = x * 2 + 1`，并检查图：

```python
from ask import Compiler, Module, TensorType

module = Module()
tensor = TensorType((4,), "f32")
graph = module.graph("affine", inputs={"x": tensor}, results=[tensor])
x = graph.inputs["x"]
two = graph.splat(graph.constant(2.0, dtype="f32"), (4,))
one = graph.splat(graph.constant(1.0, dtype="f32"), (4,))
graph.return_(graph.add(graph.mul(x, two), one))

compiler = Compiler()
prepared = compiler.prepare(module)
assert isinstance(prepared.ir, str)
assert isinstance(prepared.report, dict)
```

| 调用 | 行为 |
| --- | --- |
| `module.graph(name, inputs={...}, results=[...])` | 创建单基本块图；输入和输出的顺序确定公开参数顺序 |
| `graph.constant(value, dtype="i32")` | 构造标量常量 |
| `graph.splat(value, shape)` | 将标量填充为静态形状的张量 |
| `graph.cast(value, dtype=...)` | 显式转换元素类型，保留形状 |
| `graph.add(lhs, rhs)` / `graph.mul(lhs, rhs)` | 构造同形状、同类型的加法或乘法；不做隐式广播，bool 不支持这两种运算 |
| `graph.external(*operands, semantics=..., implementation=...)` | 调用外部算子，见[外部 kernel](#external-api) |
| `graph.mark(value, mark_id=..., role="value", port_id=None)` | 记录溯源标记；`role="input"/"output"` 时需提供 `port_id` |
| `graph.return_(*values)` | 结束图，返回值类型须与图声明一致 |
| `module.to_mlir()` | 导出 MLIR 文本；每个图必须已有 `return_` |

构图方法接收同一 Graph 的 Value；不使用 `x + y` 这样的 Tensor 运算符。操作可通过 `id=` 指定身份，单结果操作还可通过 `value_id=` 指定结果身份。精确舍入、转换和溯源规则见 [IR 与数值语义](01-ASK-IR与数值语义.md)。

<a id="compiler-api"></a>

## 3. 校验与编译

`Compiler(executable=None, timeout=60.0)` 默认使用进程内原生扩展。传入 `executable="/path/to/ask-opt"` 时使用 CLI；`timeout` 约束工具子进程，不中断进程内准备调用。`prepare`、`optimize`、`partition` 和 `compile_cuda` 均接受 `Module` 或 MLIR 字符串。

| 调用 | 返回值与用途 |
| --- | --- |
| `compiler.capabilities()` | 当前构建的能力字典 |
| `compiler.prepare(module)` | `PreparedModule(ir, report)`：校验、形状推导和溯源导出 |
| `compiler.optimize(module, fold_constants=True, eliminate_dead_code=True, fuse_elementwise=True)` | `PreparedModule`：准备并优化 ASK 自有计算 |
| `compiler.partition(module, optimize=True)` | `PreparedModule`：报告中提供 `execution_plan`、`step_sources` 和外部请求 |
| `compiler.compile_cuda(module, ...)` | 生成 `CUDAKernel` 或 `CUDAProgram`，随后绑定会话执行 |

`PreparedModule` 不包含可执行 kernel。CUDA 编译参数如下：

```python
compiled = compiler.compile_cuda(
    module,
    cache_dir="/path/to/reusable-cache",
    output_dir="/path/to/new-output",
    target={"backend": "cuda", "arch": 86},
    llc="/path/to/llc",
    ptxas="/path/to/ptxas",
    compilers=None,
    block_size=128,
    ptx_version=80,
)
```

- `output_dir` 必须为空或不存在；`cache_dir` 可跨调用复用。
- 纯自有图须显式提供 `target`、`llc` 和 `ptxas`。`arch=86` 表示编译到 `sm_86`，应选择实际设备兼容的目标；下一节示例从设备查询目标。
- 纯外部图可省略 ASK 的 `llc` / `ptxas`，其目标可从 `implementation` 推导；混合图的自有步骤仍需要这两个工具。外部声明的目标必须与本次编译目标一致。
- `compilers` 将编译器 ID 映射到 `CompilerEnvironment`；`block_size` 和 `ptx_version` 控制 ASK 自有 CUDA 代码生成，外部 kernel 的编译选项放在 `implementation.options` 中。

CPU 对象文件通过 `ask-opt --emit-object` 或 C++ 接口生成，当前没有 Python `Compiler.compile_host()` 方法。CPU ABI 与链接方式见 [Host 代码生成](03-图优化与自有代码生成.md#host-codegen)，CLI 和 C++ 入口见 [前端接入](02-前端接入与构建配置.md#cpp-sdk)。

<a id="execution-api"></a>

## 4. CUDA 会话与执行

下面接着使用第 2 节的 `module` 和 `compiler`。将 `llc`、`ptxas` 替换为本机路径；需要 CUDA 驱动和设备。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from ask import ExecutionSession

with TemporaryDirectory() as directory, ExecutionSession(device=0) as session:
    root = Path(directory)
    major, minor = session.capability
    compiled = compiler.compile_cuda(
        module,
        cache_dir=root / "cache",
        output_dir=root / "compiled",
        target={"backend": "cuda", "arch": major * 10 + minor},
        llc="/path/to/llc",
        ptxas="/path/to/ptxas",
    )
    executable = compiled.prepare(session)
    x = session.from_host([1.0, 2.0, 3.0, 4.0], dtype="f32")
    outputs = [session.tensor(port["shape"], port["element_type"])
               for port in executable.outputs]
    done = executable.submit_many([x], outputs)
    done.wait()
    assert outputs[0].to_host() == [3.0, 5.0, 7.0, 9.0]
```

`CUDAKernel` 和 `CUDAProgram` 都提供 `prepare(session)`。返回对象绑定当前会话，具有以下接口：

| 接口 | 说明 |
| --- | --- |
| `inputs` / `outputs` | 按逻辑顺序返回端口描述，含 `logical_port`、`shape`、`element_type` 等字段 |
| `submit_many(inputs, outputs)` | 接收按端口顺序排列的 DeviceTensor 序列，异步提交并返回带 `wait/query` 的事件 |
| `submit(input, output)` | 单输入、单输出的便捷调用 |
| `shape` / `dtype` | 仅在恰好一个输出时可用；多输出读取 `outputs` 中各端口的属性 |

单 kernel 的输出存储必须与输入及其它输出不重叠。多步骤程序自行分配中间缓冲区；若公开输出是输入别名或同一值的重复返回，调用方须按执行计划提供同一设备存储。别名规则见 [编译包与常驻调度](08-图切分与多Kernel执行计划.md#program-runtime)。

| 会话或张量接口 | 说明 |
| --- | --- |
| `session.tensor(shape, dtype="f32")` | 分配未初始化的设备张量 |
| `session.from_host(values, shape=None, dtype="f32")` | 接收扁平 list/tuple；省略 shape 时取 `(len(values),)` |
| `session.from_cuda_array(owner, producer_stream=...)` | 零拷贝借用提供 CUDA Array Interface 的连续设备数组 |
| `tensor.copy_from_host(values)` / `tensor.to_host()` | 拷入或读回扁平数据；会等待相关会话工作完成 |
| `tensor.shape/dtype/nbytes/data_ptr/readonly` | 查询存储元数据 |
| `event.query()` / `event.wait()` | 查询是否完成，或等待该事件；`wait()` 返回事件自身 |
| `session.synchronize()` / `session.reap()` | 等待整个 stream，或回收已完成任务的保活引用 |
| `session.device/stream/capability/stats/closed` | 查询设备、stream、计算能力、累计计数和关闭状态 |
| `session.close()` | 同步关闭；推荐使用 `with ExecutionSession(...)` |

CAI 导入支持 `bool/i32/i64/f32/f16`，不把 `uint16` 存储解释为 bf16。没有 stream 信息的 CAI v2 数组须显式提供 `producer_stream`，其中 `None` 表示生产者已完成。完整同步和生命周期契约见 [CAI 互操作](05-常驻运行时与多级缓存.md#device-tensor-and-cai)。

单输入、单输出且 shape/dtype 相同的 `CUDAKernel` 还提供 `run(values, device_ordinal=0)` 同步便捷入口，返回包含 `"output"` 的报告字典。反复运行或处理多端口时使用常驻会话。

<a id="external-api"></a>

## 5. 外部 kernel：标量与多输出

一个外部调用由三个接口组成：`module.semantics` 声明图上的输入输出；`module.implementation` 保存源码、编译配置和参数绑定；`graph.external` 将它接入图。每个外部节点独立调度，自有区域仍可融合。

下面展示双输出契约的构造。`source_text` 应是调用方 worker 能编译的源码，其入口参数与 `signature` 一致；本段只准备图，不调用 worker。

```python
from ask import Compiler, Module, TensorType

module = Module()
tensor = TensorType((4,), "f32")
semantics = module.semantics(
    "pair", inputs=[tensor], results=[tensor, tensor], shape_rule="fixed")
implementation = module.implementation(
    "pair_impl", compiler="my_compiler", source=source_text, entry="pair_kernel",
    signature={"x": "*fp32", "y": "*fp32", "z": "*fp32",
               "n": "i32", "scale": "fp32"},
    source_bindings={"x": "input:0", "y": "output:0", "z": "output:1",
                     "n": 4, "scale": 0.5},
    constants={"BLOCK": 128}, options={},
    target={"backend": "cuda", "arch": 86}, grid=(1, 1, 1),
)
graph = module.graph("main", inputs={"x": tensor}, results=[tensor, tensor])
y, z = graph.external(graph.inputs["x"], semantics=semantics,
                      implementation=implementation)
graph.return_(y, z)
prepared = Compiler().prepare(module)
assert len(prepared.report["external_requests"]) == 1
```

`shape_rule="same_as"` 使用 `shape_operand` 指定的输入形状；固定或不同输出形状使用 `"fixed"`。`graph.external` 可以显式传 `result_types=[...]`；单结果返回一个 Value，并兼容 `result_type=`，多结果返回 Value 元组。

`signature` 与 `source_bindings` 必须使用相同的参数名。张量参数绑定 `input:N/output:N`，普通标量绑定具体数值；它们按声明 ABI 类型传入 kernel。`constants` 是编译期配置，两者不混用。启动标量在构图时冻结并参与缓存身份，`submit_many` 不覆盖它们。类型范围和递归 JSON 配置见 [请求协议](09-外部编译器适配接入规范.md#worker-protocol)。

编译时显式选择 worker：

```python
from ask import CompilerEnvironment

compilers = {"my_compiler": CompilerEnvironment(
    python="/path/to/compiler-env/bin/python",
    worker="/path/to/worker.py",
)}
# 将 compilers=compilers 传给 compiler.compile_cuda(...)。
```

映射键对应 `implementation.compiler`。原生可执行 worker 可以省略 `python`；ASK 不扫描或自动注册第三方编译器。完整可运行实例见 [PTX 接入示例](../examples/external_ptx.py)。Triton 的 `from_jit`、`scalars`、`outputs` 属于 [FugaTensor 接口](../../fugatensor/docs/architecture/04-ASK桥接与多后端执行.md#cuda-and-triton)。

已持有 cubin/PTX 且自行管理参数契约的调用方，也可直接使用 `session.load(image_bytes, entry, required_types=[...], arch=..., parameter_types=[...])`，随后 `session.submit(kernel, bindings, grid=..., block=..., shared_bytes=0, writable_indices=())`。`parameter_types` 使用 `"pointer"` 或协议中的标量类型；bindings 中传 DeviceTensor、`None` 或已编码的标量 bytes。`writable_indices` 用来检查输出参数可写性；此底层入口不替代 `prepare().submit_many()` 的张量形状与别名检查。

<a id="artifact-api"></a>

## 6. 编译产物与持久化

`compile_cuda` 根据图返回不同产物，统一执行入口是 `compiled.prepare(session)`。

| 产物 | 可读取内容 | 保存与加载 |
| --- | --- | --- |
| `CUDAKernel` | `artifact`、`codegen`、`report`、`cache` | 编译目录保留产物；对象没有 `save/load` 方法 |
| `CUDAProgram` | `plan`、`kernels`、`report`、`cache` | `save(directory)` 返回 `program.json` 的路径；`CUDAProgram.load(directory)` 读取程序目录 |

纯自有图通常返回 CUDAKernel；单个外部步骤与公开端口完全对应时也可返回 CUDAKernel。多个步骤或需要表达公开端口别名的计划返回 CUDAProgram，因此不能对编译结果无条件调用 `save`。

多步骤编译会自动在 `output_dir` 写入程序包，可直接 `CUDAProgram.load(output_dir)`。另存使用新目录，已有 `program.json` 时不会覆盖。保存/加载校验摘要与端口契约；包不携带 worker、编译器或 CUDA 驱动，执行设备仍须与产物目标兼容。文件布局见 [编译包与常驻调度](08-图切分与多Kernel执行计划.md#program-runtime)。

<a id="errors-and-limits"></a>

## 7. 错误与支持边界

Python 构图参数错误通常抛出 `TypeError` 或 `ValueError`；编译器拒绝、worker 或工具失败抛出 `CompilerError`，可读取 `diagnostics` 和 `returncode`；设备、ABI 或执行包错误抛出 `ask.cuda.CUDAError`。失败不会切换到另一种计算实现。

执行计划使用静态形状、连续行主序张量和单 stream 调度。外部算子仅接受 `effects="none"`，没有原地写入、非空 scratch、描述符或集群/协作启动契约；张量类型仍为六种 ASK 基础类型，按值标量 ABI 的类型集合不改变这一点。完整范围见 [当前限制](07-FugaTensor协同与演进路线.md#roadmap)。
