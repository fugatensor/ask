# FugaTensor 怎样使用 ASK

[返回顶层设计](核心思想.md#10-fugatensor-放在哪里) · [框架接入接口](10-上层框架接入.md) · [编译器适配](17-第三方编译器适配与注册.md) · [文档目录](README.md)

首个 FugaTensor 原型位于同级 `fugatensor/` 仓库。用户写张量计算，FugaTensor 记录高层 ASK IR；ASK 负责检查、优化和生成实际执行的代码。

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

默认 dtype 为 f32、设备为 CPU。首次调用追踪 Python 函数，生成 `ask.add`、`ask.mul` 等操作，再用 ASK 生成本机目标文件并加载。相同形状、dtype 和静态参数的下一次调用直接执行已加载函数；换输入值不会重新追踪。CPU 支持多个张量输入与输出。

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

前后加法是 ASK 操作，`scale` 对应用户内核的 `ask.external`；换成用户写的其他合法算法，仍通过同一注册接口接入。FugaTensor 提交 DSL、语义、端口绑定和固定编译选项；ASK 的 Triton 适配器取得 LLVM IR 并证明值边界，通用内联 Pass 插入 ASK helper，然后重新生成 cubin。FugaTensor 复用 ASK 的常驻 Session 提交设备张量，`tolist()` 等待并下载结果。

目前这条 CUDA 高层入口要求一个输入、一个同形状同存储类型输出和一个 external。纯 ASK CUDA 图的 lowering 尚未实现，超出范围会明确报错。fp16/bf16 存储可以在图中显式转为 f32 计算再转回；实际硬件检查沿用 ASK。

## 配置、测试和检查产物

`ft.configure(device=..., ...)` 或环境变量负责设置 ASK CLI、worker、`llc` 和 `ptxas`；普通张量调用不重复传工具路径。无显式配置时，CPU 采用已安装 ASK 的配套工具。详细命令在 FugaTensor 的 `README.md`，直接可读的创建与计算用例在 `tests/test_quickstart.py`，完整演示为 `examples/jit_demo.py`。

`function.last_ir` 显示提交给 ASK 的高层图；`last_report` 保留编译报告；`last_artifact_dir` 定位目标文件、LLVM 或 cubin。`cache_info()` 统计 FugaTensor 内存特化缓存，ASK 的 IR 与可执行磁盘缓存仍分别记在 GPU 报告里。CPU 当前没有跨进程持久代码缓存。

默认运行时由 `ft.shutdown()` 关闭；也可用 `with ft.Engine(...)` 明确管理生命周期。关闭会等待设备工作并清除所属的 JIT 特化。追踪期间不能读取真实张量数据来生成 Python 常量，所有计算输入必须作为参数进入图。
