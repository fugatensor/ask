# Triton 适配器

这个目录负责把 Triton DSL 编译结果接到 ASK。它固定使用 Triton 3.7.1 的公开 API：
`ASTSource`、`triton.compile`、`GPUTarget` 和 `compiled.asm`。它不修改 Triton 源码，
也不替换 Triton 的编译阶段。worker 导出 LLVM IR；ASK 再验证、融合和生成自己的 CUDA 二进制。

## 文件各自负责什么

| 文件 | 职责 |
| --- | --- |
| `registration.json` | 声明 compiler ID、协议版本、C++ 工厂、Python 工厂、worker 和兼容参数名。 |
| `CMakeLists.txt` | 只有选中本适配器时才检查 Triton 环境、编译专属 C++、注册测试并安装资源。 |
| `worker.py` | 检查请求，通过公开 API 编译 DSL，读取真实 ABI、类型和执行配置，发布产物。 |
| `fusion.py` | 检查可追踪的数据流，插入恒等标记，对比加标记前后的布局和资源。 |
| `python/__init__.py` | Python 编译环境检查、Triton 安装指纹、请求数值契约、旧产物迁移。安装后为 `ask._adapters.triton`。 |
| `lib/`、`include/` | 实现 [C++ 适配器接口](../../include/ask/Adapters/Compiler.h)，验证 Triton 标记并保留原 F 的浮点策略。 |
| `tests/` | worker 和策略单测；`integration/` 包含 ASK 导出、融合及拒绝错误产物的测试。 |

中央 [registry.json](../registry.json) 只列出允许构建的 ID 和目录。
构建选中 `triton` 后，CMake 将本目录的注册信息写入原生注册表和
`ask/_adapter_registry.json`。`none` 构建不执行这里的依赖检查，也不注册本适配器。
公共 Python 代码先检查原生构建允许的集合，再加载明确注册的工厂；不会扫描 Python entrypoints。

公共层只处理编译器 ID、请求/响应协议、LLVM IR 和标准执行契约。
ASK 请求签名到 Triton API 的转换，以及 `N/BLOCK`、`num_warps`、安装指纹和环境开关，都在这个目录处理。
其他 MLIR 编译器可提供自己的目录和工厂，不需要在公共编译流程里增加品牌分支。

## 当前支持范围

- Triton 3.7.1；锁定环境记录见 `dependencies.lock.json` 和 `requirements.lock`。
- CUDA 普通整数架构 `70 <= arch <= 999`、`warp_size=32`、`num_warps=4` 或 `8`。
  保留请求中的架构；这个范围不保证每个整数都对应有效 GPU，仍须 Triton、LLVM 和 ptxas 支持。
  `90a` 等带后缀目标暂不接受。
- 一个 `@triton.jit` 入口，两个存储类型相同的数据指针，一个逻辑输入和一个逻辑输出。
  支持 `*i1`、`*i32`、`*i64`、`*fp32`、`*fp16`、`*bf16`，分别对应 ASK 的
  `bool/i32/i64/f32/f16/bf16`。constexpr 单独传入，源码参数通过 `source_bindings` 对应逻辑端口。
- 当前逐元素导出需要明确的 `N`；原始产物的端口形状为 `[N]`。
  融合后使用 ASK 图的实际形状，包括多维形状。
- shared、global scratch、profile scratch 均为零，不支持 tensor descriptor、协作启动等资源。

`bool` 在 LLVM 中使用 i1 值，在内存中每元素占一字节。`i64` 保持完整的 64 位整数值。
`f16` 和 `bf16` 都占两字节，但不能互相冒充。

任何 BF16 端口都要求目标至少 sm80。即使端口是 f32，内部 BF16 转换也会从真实
`compiled.asm["ttir"]` 的类型中识别；字符串、注释和变量名不会被当成类型。
不满足条件时，不发布成功产物。原始导出与融合产物都携带 `required_element_types`，
运行时还会检查实际 GPU 的 BF16 能力；不能靠改目标标签在旧设备上执行。

超出支持范围会报错。新增 Triton 版本需要检查实际 ABI 并补测试，不能只移除版本检查。
DSL 只允许规定的 imports 和单个入口定义；这项限制不是不可信 Python 的安全沙箱。

## 从 ASK 调用

SDK 使用构建 ASK 时的 Python，Triton worker 可以使用另一个 Python 环境。
源码树运行时，将选中的构建包加入路径，例如 `PYTHONPATH=python:build/triton/python`。

```python
from pathlib import Path
from ask import Compiler, CompilerEnvironment

kernel = Compiler().compile_cuda(
    Path("graph.mlir").read_text(),  # 上层框架已经生成的 ASK IR
    compilers={
        "triton": CompilerEnvironment(
            python="/path/to/triton-env/bin/python",
            worker="adapters/triton/worker.py",
        ),
    },
    llc="/path/to/llvm/bin/llc",
    ptxas="/path/to/triton/backends/nvidia/bin/ptxas",
    cache_dir="build/example-cache",
    output_dir="build/example-result",  # 每次调用使用空输出目录
)
print(kernel.report["ir_cache"])
print(kernel.report["executable_cache"])
```

选择哪个适配器由 ASK IR 中的 `compiler` ID 决定。环境映射只提供这个 ID 所需的工具路径。
其他适配器可以使用原生可执行 worker，此时 `CompilerEnvironment(worker=...)` 不需要 Python；
Triton 自己要求提供 Python。旧的 `triton_python` / `triton_worker` Python 参数仍由
`registration.json` 的 aliases 兼容，公共函数不硬编码这些名字。

CLI 使用同一套按 ID 配置的参数：

```bash
build/triton/bin/ask-opt graph.mlir --fuse-external \
  --compiler-python triton=/path/to/triton-env/bin/python \
  --compiler-worker triton=adapters/triton/worker.py \
  --export-dir build/example-export -o build/example-fused.ll
```

也可以单独运行 worker 检查环境或导出：

```bash
/path/to/triton-env/bin/python adapters/triton/worker.py --probe
/path/to/triton-env/bin/python adapters/triton/worker.py \
  --request adapters/triton/tests/fixtures/scale2.request.json \
  --output /tmp/ask-scale2-artifact.json
```

请求仍使用 ASK 导出协议 1，包含 DSL、入口、签名、绑定、constexpr、固定选项、target 和 grid。
worker 对照源码与公开 JIT 签名检查这些字段；JSON 对象的键顺序不决定参数 ABI。
成功时原子发布响应文件；失败时返回非零状态，并向 stderr 写 JSON 诊断。
调用者必须检查退出状态，不能把上次留下的文件当成这次成功的结果。

## 产物交给公共层的字段

导出协议版本和执行契约版本是两件事：外层 `protocol_version` 为 1，
其中 `invocation.version` 为 2。

| 字段 | 含义 |
| --- | --- |
| `payload`、`entry`、`entry_signature` | LLVM 原文、实际入口和物理参数 ABI。 |
| `port_map` | 逻辑端口、对应的参数序号、`element_type`、`shape`、`storage_bytes` 和 `layout="contiguous_row_major"`。 |
| `invocation` | `grid`、`block`、`shared_bytes`、实际参数顺序以及 buffer/null 绑定。公共 CUDA 层直接使用这些值。 |
| `codegen` | 明确的 `fp_contract` 和 `ptx_version`。原 F 使用 `fast/off`；融合后使用 `on`，由 F 指令自己的 contract 标志决定可收缩范围。 |
| `required_element_types` | 外部计算及融合图需要的类型，包含内部 BF16 等硬件需求。 |
| `producer_metadata` | Triton 私有 `compiler_metadata`、`num_warps`、`num_ctas` 和 resources；公共执行层不读取它们。 |

Triton 3.7.1 即使不使用 scratch，也会增加两个物理指针参数。
本适配器确认它们未使用，并在通用 `arguments` 中绑定为 `null`。
公共 CUDA 层按列表处理参数，不依赖“四个参数”或某个 scratch 名字。

worker 不序列化 `CompiledKernel`、Python launcher、函数指针或 GPU handle。
它的 LLVM 头部检查也不等于完整验证，因此 `validation.llvm_module_verified` 为 false；
ASK 的 LLVM parser/verifier 随后负责检查模块。外部库摘要记录编译输入，不代表未解析符号已经解决。

旧 `invocation.version=1` 产物不会被公共运行时猜测转换。需要迁移时，显式调用：

```python
from ask._adapters.triton import normalize_legacy_artifact

artifact = normalize_legacy_artifact(old_artifact, shape=(2, 5))
```

形状必须与旧 `N` 的元素数相符。迁移检查旧 ABI、launch 和数值收据，保留 LLVM 内容，
并把私有字段放入适配器范围。产物摘要会改变，必须重新生成 codegen 收据，不能继续使用旧收据。

## 合法融合的检查顺序

请求加上 `"fusion": {"enabled": true}` 后，先证明源码只有一次受保护的输入 load、
逐元素纯计算和一次使用相同索引/掩码的输出 store，再通过公开
`tl.inline_asm_elementwise` 插入恒等标记。局部别名、等价仿射表达式、重赋值、位置参数和关键字参数均可使用。
显式 `.to(tl.float32)`、`.to(value.dtype)` 等数值转换保留 Triton 的原始语义；
不会把 F 的转换改成 ASK 的饱和转换规则。未知调用、指针转换和不能证明的内存访问会被拒绝。

随后同时编译原始和加标记的 DSL，比较 blocked layout、启动配置和资源。
`BLOCK` 可以是 1 到 1024 的二次幂，物理线程数仍是原 F 的 `num_warps * 32`。
一个逻辑边界可对应多处 LLVM 标记，每处都必须绑定和验证。

标记只复制位：bool 用 `mov.pred`，16/32/64 位值分别用 `mov.b16/b32/b64`。
C++ binder 用真实字节宽度检查地址，并证明掩码等价于合法逻辑索引范围。
[F 的数值策略与标记校验](lib/TritonBoundary.cpp) 通过后，公共
[LLVM inliner](../../include/ask/LLIR/Inline.h) 才能插入 ASK helper。
原 F 的内存 inline assembly 保留，不通过解码 PTX 猜测内存语义。最终产物移除追踪标记。

## 验证

```bash
python3 -m unittest discover -s adapters/triton/tests -v
ASK_TEST_TRITON_PYTHON=/path/to/triton-env/bin/python \
  python3 -m unittest discover -s adapters/triton/tests -v
ctest --test-dir build/triton -R '^ask-triton-' --output-on-failure
```

第二条命令会实际导出 LLVM IR，并检查各类型标记、原始/加标记布局、混合精度、数值策略和 BF16 目标拒绝。
它不启动 GPU。CMake 集成测试还会运行 ASK 导出和原生 LLVM 融合，检查错误产物是否被拒绝。
GPU 数值、IR/可执行缓存及常驻运行时验证使用共享图和独立数值参考，见
[GPU 融合与缓存](../../docs/12-GPU融合与IR缓存.md)、[多精度测试](../../docs/14-多精度测试.md)
和[类型转换](../../docs/15-类型转换与混合精度.md)。

当前 LLVM SDK 固定在 `1f126a6dea50d185c0781743a667390037ae88bd`，版本为 LLVM 23.0.0git，
与 Triton 3.7.1 的上游 LLVM pin 对齐。获取方式为 `python3 tools/fetch_llvm.py`，
来源、归档摘要和版本记录在 `cmake/llvm.lock.json`。
LLVM 18.1.8 会拒绝真实导出的 `captures(none)` 属性；这里没有通过删属性来迁就旧 reader。
