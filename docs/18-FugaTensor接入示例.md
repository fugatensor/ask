# FugaTensor 怎样使用 ASK

[返回顶层设计](核心思想.md#10-fugatensor-放在哪里) · [框架接入接口](10-上层框架接入.md) · [编译器适配](17-第三方编译器适配与注册.md) · [文档目录](README.md)

FugaTensor 位于同级 `fugatensor/` 仓库。用户写张量计算，FugaTensor 记录独立的 Fuga IR 和实际执行历史，在 backward 时生成反向计算，再把可执行静态部分降成高层 ASK IR。Python 动态条件由 Fuga 分片调度，ASK 继续接收静态图。ASK 负责检查、目标优化和生成实际代码。Fuga IR 是跨设备的逻辑表示，基本类型与操作不带 CPU/CUDA 放置；对象模型、region 和转换边界见 [Fuga IR 文档](../../fugatensor/docs/ir/README.md)。

## 日常代码

配置好 ASK 后，CPU 计算只需：

```python
import fugatensor as ft

@ft.jit
def calculate(x, y):
    return (x + y) * 2

x = ft.tensor([1, 2, 3])
y = ft.tensor([10, 20, 30])
print(calculate(x, y).tolist())  # [22.0, 44.0, 66.0]
```

默认 dtype 为 f32、设备为 CPU。首次调用追踪 Python 函数，生成 `fuga.add`、`fuga.mul` 等结构化操作，再经显式转换生成 ASK 操作，最后由 ASK 生成本机目标文件并加载。相同形状、dtype 和静态参数的下一次调用直接执行已加载函数；换输入值不会重新追踪。CPU 和自有 CUDA 图都支持多个张量输入与输出。

## 同一张图在 GPU 上执行

配置好 `llc`、`ptxas` 和 ASK 后，只需把输入放到 CUDA：

```python
x = ft.tensor([1, 2, 3], device="cuda")
y = ft.tensor([10, 20, 30], device="cuda")
print(calculate(x, y).tolist())  # [22.0, 44.0, 66.0]
```

这里仍是上面的 `calculate`，没有 Triton 代码或 external 节点。同一基础函数、shape 和 dtype 在 CPU/CUDA 上生成相同 Fuga IR，再提交相同语义的基础 ASK 图。执行阶段选择目标；ASK 为 GPU 生成索引、边界判断、访存和逐元素计算，再用自己的常驻 Session 执行。`ASK_THIRD_PARTY_COMPILERS=none` 即可使用。过程和依赖见[自有 GPU 降级](19-自有GPU降级.md)。

## 图中的 Triton 算子

用户先在 `kernels.py` 写实际 Triton 代码，例如加载一个值、乘二再写回的 `scale_kernel`。FugaTensor 原样读取这个文件，不生成内核函数体。然后声明源码入口、参数绑定、特化常量和 grid：

```python
scale = ft.triton.kernel(
    "kernels.py", entry="scale_kernel", input="inp", output="out",
    constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
    grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
    options={"num_warps": 4},
)
```

`spec` 只包含静态 shape、dtype、元素个数和设备，不读取张量数据。也可以用 `ft.triton.from_source` 提交用户手写的完整源码文本。注册时保存源码快照，之后编辑文件需要重新注册并清理对应 JIT 特化。当前入口接收文件或文本，尚不直接接收第三方进程内的 JIT 对象。

CUDA 工具和 worker 只在初始化时配置。注册完成后，计算部分写成：

```python
@ft.jit
def mixed(x):
    return scale(x + 1) + 3

x = ft.tensor([1, 2, 3], device="cuda")
print(mixed(x).tolist())  # [7.0, 9.0, 11.0]
```

前后加法先成为 Fuga 操作，`scale` 成为保存用户源码的 `fuga.external`；转换后才对应 ASK 加法与 `ask.external`。具体 GPU 架构在这一边界绑定，不写进通用 Fuga 值的类型。换成用户写的其他合法算法，仍通过同一注册接口接入。FugaTensor 提交 DSL、语义、端口绑定和固定编译选项；ASK 的 Triton 适配器取得 LLVM IR 并证明值边界，通用内联 Pass 插入 ASK helper，然后重新生成 cubin。FugaTensor 复用 ASK 的常驻 Session 提交设备张量，`tolist()` 等待并下载结果。独立 Fuga IR 不改变这条原生 LLVM 链接与融合路径。

目前 external 融合入口仍要求一个实际使用的输入、一个同形状同存储类型输出和一个 external。Fuga 可删除无用的物理参数，并保存逻辑输入到实际端口的映射。这个限制只属于外部融合路径；自有 GPU 图可有多个输入和输出，各端口保留自己的形状与类型。fp16/bf16 存储可以在图中显式转为 f32 计算再转回；实际硬件检查沿用 ASK。

external 后的计算目前只使用其结果与常量，不支持绕过 external 再读取原始输入。用户手写 backward 也遵守这一限制；只需要 dy 的反向可以删除未使用的原始 x，正常接收运行时梯度。确实需要多个实际输入的 external 反向仍会拒绝，基础 ASK 多输入反向没有这项限制。

## 配置、测试和检查产物

FugaTensor 的 `python3 tools/bootstrap.py` 默认以 `all` 构建同级 ASK，安装到自己的 `.venv` 并保存运行配置。之后 `.venv/bin/python examples/jit_demo.py --device both` 运行基础 CPU/CUDA 示例，`examples/mixed_triton.py` 运行用户源码与 ASK 运算的混合示例。这是 FugaTensor 的初始化默认值，不修改 ASK 全局默认 `none`。已有环境仍可用 `ft.configure(device=..., ...)` 或环境变量覆盖工具路径。

`function.last_fuga_ir` 显示追踪后的 Fuga 图，`last_ask_ir` 显示提交给 ASK 的图，`last_ir` 保留 ASK 文本的兼容含义。编译目录同时保存 `fuga.ir`、`frontend.mlir` 及逐值映射 `fuga-to-ask.json`。`last_report` 保留编译报告；`last_artifact_dir` 定位目标文件、LLVM 或 cubin。`cache_info()` 统计 FugaTensor 内存特化缓存，ASK 的 IR 与可执行磁盘缓存仍分别记在 GPU 报告里。CPU 当前没有跨进程持久代码缓存。

Fuga 的 region 能表达 if/for/while，直接桥接仍只处理静态条件与有界 for。普通 JIT 的 Python 零维 Tensor 条件通过图中断执行前缀、读取条件并选择下一片段；反向连接本次实际执行路径。这些框架层工作不要求 ASK A 层接收动态控制流。

公开自动微分使用 `requires_grad=True`、`output.backward()` 和叶子 `.grad`。create_graph 可保留由已知 Fuga 基础运算组成的梯度图并继续求导；不透明第三方正向或反向的高阶请求会提前拒绝。fun 的 forward/backward 都需 `@ft.jit`；Fuga 先检查全部反向规则，再生成和执行反向图，缺少规则时不会先执行部分反向。外部 DSL 继续进入原生 LLVM 链接与合法内联。接口、保存值和图释放见 [Fuga backward](../../fugatensor/docs/ir/04-JIT与自定义梯度提案.md)。

Fuga 开始把 SSA 依赖、反向路径和活跃性分析放入自己的 nanobind C++ 扩展，Python 暂时保留追踪和求导规则生成。这个拆分不改变 ASK 的职责：ASK 继续接收静态计算图，负责目标编译与设备执行。

默认运行时由 `ft.shutdown()` 关闭；也可用 `with ft.Engine(...)` 明确管理生命周期。关闭会等待设备工作并清除所属的 JIT 特化。追踪期间不能读取真实张量数据来生成 Python 常量，所有计算输入必须作为参数进入图。
