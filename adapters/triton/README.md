# Triton 适配器

本目录负责将 Triton DSL 编译结果接入 ASK，固定调用 Triton 3.7.1 的公开 API（`ASTSource`、`triton.compile`、`GPUTarget` 与 `compiled.asm`），不修改 Triton 源码，也不替换其内部编译流程。`worker.py` 在独立子进程中导出原生 LLVM IR 与标准执行契约，ASK 公共层随后完成校验、值边界内联并生成 CUDA `cubin`。

注册机制与通用 LLVM 值边界内联设计见 [04. 外部编译器适配与边界内联](../../docs/04-外部编译器适配与边界内联.md)。

## 目录分工

| 文件 / 目录 | 职责 |
| --- | --- |
| `registration.json` | 声明编译器 ID（`"triton"`）、协议版本、C++ 工厂、Python 策略工厂、`worker.py` 与兼容参数别名。 |
| `dependencies.lock.json` / `requirements.lock` | 锁定 `triton==3.7.1`、配套 LLVM 提交哈希（`1f126a6d...`）与 Python 依赖。 |
| `CMakeLists.txt` | 在 `ASK_THIRD_PARTY_COMPILERS` 选中 `triton` 时检查环境、编译专属 C++ 库、注册 `ask-triton` 测试并安装资源。 |
| `worker.py` | 校验请求，调用 Triton 公开 API 编译 DSL，提取物理参数 ABI、类型需求与启动配置，原子写出 JSON 产物。 |
| `fusion.py` | 对 `@triton.jit` 源码做 AST 数据流检查并插入恒等位复制内联汇编标记；`worker.py` 负责双编译及标记前后布局、资源的对比。 |
| `python/__init__.py` | Python 环境探测、Triton 安装指纹计算、请求数值契约构建及 `invocation` v1 旧产物显式迁移（安装为 `ask._adapters.triton`）。 |
| `include/`、`lib/TritonBoundary.cpp` | 实现 [C++ `CompilerAdapter` 接口](../../include/ask/Adapters/Compiler.h)，校验 LLVM 标记站点的地址与掩码等价性，按原 `F` 策略设置浮点 `contract` 标志。 |
| `tests/` | `test_worker.py`、`test_python_policy.py`、`test_fusion.py` 与 `test_integration.py`，覆盖导出协议、AST 判定、策略迁移与 `ask-opt --fuse-external` LLVM 融合。 |

## 当前支持范围

- **版本与目标架构**：Triton 3.7.1；普通整数 CUDA 架构 `70 <= arch <= 999`（暂不支持 `90a` 等带后缀目标）、`warp_size = 32`、`num_warps = 4` 或 `8`。
- **签名与端口类型**：单个 `@triton.jit` 入口，使用显式逻辑输入/输出指针（`*i1`、`*i32`、`*i64`、`*fp32`、`*fp16`、`*bf16`，对应 ASK 六种基础元素类型，见 [01. ASK IR 与数值语义](../../docs/01-ASK-IR与数值语义.md#types-and-precision)）；特化常量通过 `constants` 传入，源码参数通过 `source_bindings` 映射到逻辑端口。
- **独立外部 kernel**：请求通过 `port_shapes` 分别声明每个 `input:i` / `output:i` 的静态形状，支持不同形状与存储类型。当前 ASK `ask.external` 仍只返回一个结果；worker 产物协议可以描述多个输出，并不表示前端已支持多结果外部操作。允许非零动态共享内存，`num_stages` 可显式指定为 1–5；`global_scratch`、`profile_scratch`、`tmem_size` 仍须为 0，不支持 tensor descriptor、集群或协作启动。无 `port_shapes` 的旧请求只支持两个同类型端口，并用 `N` 构成 `[N]`。
- **BF16 检查**：任何 `bf16` 端口或从 `compiled.asm["ttir"]` 类型中识别出的内部 `bf16` 运算，均要求目标至少 `sm_80`，并写入 `required_element_types` 供运行时核对物理 GPU 能力。
- **值边界融合限制**：融合片段须为一个非空静态输入与同形状同存储类型输出、恰好一个 `ask.external`，且外部源码通过逐元素 AST/LLVM 证明。该路径仍要求 `shared=0`，不把矩阵乘或注意力当作逐元素算子。图执行计划将不适合值边界融合的复杂算子独立编译，并在前后阶段传递实际张量；无效请求或损坏产物直接报错。

## 从 ASK 调用

SDK 进程与 Triton `worker.py` 可使用不同的 Python 环境：

```python
from pathlib import Path
from ask import Compiler, CompilerEnvironment

kernel = Compiler().compile_cuda(
    Path("graph.mlir").read_text(),
    compilers={
        "triton": CompilerEnvironment(
            python="/path/to/triton-env/bin/python",
            worker="adapters/triton/worker.py",
        ),
    },
    llc="/path/to/llvm/bin/llc",
    ptxas="/path/to/cuda/bin/ptxas",
    cache_dir="build/example-cache",
    output_dir="build/example-result",
)
```

命令行调用：

```bash
build/all/bin/ask-opt graph.mlir --fuse-external \
  --compiler-python triton=/path/to/triton-env/bin/python \
  --compiler-worker triton=adapters/triton/worker.py \
  --export-dir build/example-export -o build/example-fused.ll
```

单独探测 worker 环境或执行单次导出：

```bash
/path/to/triton-env/bin/python adapters/triton/worker.py --probe
/path/to/triton-env/bin/python adapters/triton/worker.py \
  --request /path/to/request.json --output /tmp/ask-triton-artifact.json
```

## 产物契约与融合检查

1. **协议与执行契约（`protocol_version = 1` / `invocation.version = 2`）**：
   - 适配器修订由 `dependencies.lock.json` 的 `adapter_version` 定义（当前 `0.7.0`），worker 与 C++ 边界校验使用同一份锁。它与 ASK 软件包版本、Triton 版本及两项协议版本分别管理；旧适配器修订导出的产物须重新导出。
   - 产物包含文本格式 LLVM IR（`payload`）、`entry_signature`、`port_map`、`invocation`、`codegen`（含 `fp_contract` 与 `ptx_version`）、`required_element_types` 及放在 `producer_metadata` 中的 Triton 私有元数据，不序列化 Python 对象或 GPU 句柄。
   - Triton 3.7.1 自动追加的两个未使用物理 scratch 指针参数经校验后在 `invocation.arguments` 中绑定为 `"null"`，公共 CUDA 层按参数列表传递。
   - 历史 `invocation.version = 1` 产物可通过 `ask._adapters.triton.normalize_legacy_artifact(old_artifact, shape=...)` 显式迁移。
2. **融合检查顺序**：
   - 开启 `"fusion": {"enabled": true}` 后，依次执行：AST 静态单次受保护 load/store 数据流检查 → `tl.inline_asm_elementwise` 恒等位标记插入与原版/加标记版双编译布局对比 → `TritonBoundary.cpp` 指针字节偏移与 `0 <= idx < N` 掩码等价检查 → [通用 LLVM 值边界内联器](../../include/ask/LLIR/Inline.h) 内联 `P`/`Q` helper 并移除标记。
   - 用户在 Triton kernel 内编写的 `.to(...)` 类型转换与 `enable_fp_fusion` 浮点收缩策略保留原行为；ASK 的 `P`、`Q` 指令不带 `contract` 标志，最终 `llc` 使用 `-fp-contract=on` 区分两侧策略（见 [跨编译器 FMA 隔离](../../docs/04-外部编译器适配与边界内联.md#fma-and-numerics)）。

## 验证命令

```bash
ctest --preset all -R ask-triton --output-on-failure

ASK_TEST_TRITON_PYTHON=/path/to/triton-env/bin/python \
  PYTHONPATH=build/all/python:python \
  python3 -m unittest discover -s adapters/triton/tests -v
```

物理 GPU 执行、缓存命中与常驻会话验证由顶层 `ask-gpu` 测试集覆盖，见 [05. 常驻运行时与多级缓存](../../docs/05-常驻运行时与多级缓存.md) 与 [06. 工程结构与测试体系](../../docs/06-工程结构与测试体系.md)。配套 LLVM SDK（`23.0.0git`）通过根目录 `python3 tools/fetch_deps.py --only llvm` 获取，与 Triton 3.7.1 导出的 `captures(none)` 等 LLVM IR 属性对齐。

## FlashAttention-2 风格前向示例

[`examples/flash_attention2.py`](../../examples/flash_attention2.py) 构造三个输入与一个输出的独立外部操作：`Q[B,H,M,D]`、`K[B,H,N,D]`、`V[B,H,N,DV]` → `O[B,H,M,DV]`。其[实际 Triton 源码](../../examples/kernels/flash_attention2.py)按 query 块并行，循环读取 key/value 块，以 FP32 在线最大值、归一化和输出累加器计算 softmax，两次 `tl.dot` 使用 FP16/BF16 输入；不分配完整注意力矩阵。

示例固定 `BLOCK_M=BLOCK_N=32`、`num_warps=4`、`num_stages=2`，支持 `D` / `DV` 为 32、64、128 与正长度连续张量。`causal` 按左上对齐定义为 `key_index <= query_index`，矩形输入也采用该规则。没有 dropout、bias、变长批次、任意 stride、反向实现或性能承诺。GPU 测试用下载后的量化输入计算独立 Python 稳定 softmax 参考；FP16 的 `atol=rtol=0.002`，BF16 为 `0.02`。

算法依据 [FlashAttention-2 论文 v1](https://arxiv.org/abs/2307.08691v1)和 [Triton 官方 fused attention 教程的固定提交](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/tutorials/06-fused-attention.py)。该提交对应 Triton 3.7.1；示例保留了 OpenAI kernel team 来源说明及 [MIT 许可](../../examples/kernels/LICENSE.triton)。为 Ampere 使用普通指针和掩码加载，没有使用较新 GPU 的 TensorDescriptor 路径。
