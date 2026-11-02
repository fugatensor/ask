# ASK 仓库工程规范

## 工作约定

1. 完成一组代码或文档修改并通过相关验证后，创建本地 Git 提交，便于回退。若有未完成的检查点，在提交说明中标明状态。
2. 第三方 SDK（`.deps/`）、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存不纳入 Git；保留依赖锁文件（`cmake/*.lock.json`）及获取脚本（`tools/fetch_deps.py`）。
3. 文档使用自然直接的中文说明设计原因、模块配合与支持边界，避免空泛套话、AI 式过渡语和跨篇重复。
4. 代码与文档保持同步：只把已落地的能力写成当前行为；修改接口、构建参数、语义规则或目录结构时，同步更新文档、示例及双向链接锚点。

---

## 架构与模块边界

- **双层 IR 分工**：
  - 高层 MLIR `ask` dialect（`include/ask/IR/`、`lib/IR/`）负责显式类型、静态形状推导、图级优化（折叠、DCE、逐元素融合分组）、`ask.mark` 溯源导出与 `ask.external` 外部算子契约检查。
  - 低层原生 LLVM IR（`lib/Lowering/`、`lib/LLIR/`）负责自有 CPU/CUDA 算子降级、LLVM 校验与优化及目标机器码（`.o` / `cubin`）生成。
- **核心与第三方适配器隔离**：
  - `core` 构建不依赖 Triton、NumPy 或 PyTorch；自有 CPU 后端与自有 CUDA 后端属于基础核心。
  - 第三方编译器专属 worker、版本策略和测试由调用方维护；Triton 适配器位于 FugaTensor。ASK 只提供协议 v2 的 worker 调用、cubin/PTX 接入、缓存和调度，不设置编译器构建白名单，不硬编码特定 DSL，也不扫描 Python `entrypoints`。
- **图切分与执行计划**：
  - 高层 MLIR 分区函数与 JSON v1 记录缓冲区、步骤依赖和单 stream 调度；自有区域内部保留融合，每个外部节点始终独立执行。非法 IR、损坏产物和工具失败直接报错。
  - 外部 kernel 通过显存缓冲区连接；支持静态连续多输出张量、指针/null 与按值标量参数，使用普通单 stream 启动，不实现第三方代码融合。编译期配置使用递归 JSON，编译器专属值由调用方编码解释。
- **数值语义**：
  - 6 种基础类型（`bool`、`i32`、`i64`、`f32`、`f16`、`bf16`）不做隐式类型提升或隐式广播。
  - `ask.constant` 按单步最近偶数舍入（RNE）编码浮点字面量，64 位整数保持完整精度。
  - `ask.cast` 浮点转整数采用确定性饱和截断（`NaN` 置 0，溢出钳制到目标整数极值）。
  - 自有 CUDA 计算通过 `llc -fp-contract=off` 保持乘加分别舍入；外部 cubin/PTX 保留生产者数值策略，ASK 不改写第三方代码。

---

## 构建与测试

- **依赖与构建指令**：
  - 拉取依赖使用 `python3 tools/fetch_deps.py`（支持 `--only llvm` / `--only nanobind`）。`CMakeLists.txt` 会根据 `cmake/llvm.lock.json` 自动识别 `.deps/` 下的 LLVM/MLIR：
    - 核心构建与测试：`cmake --preset core && cmake --build --preset core && ctest --preset core --output-on-failure`
- **测试组织**：
  - 2 个 CTest 目标：`ask-unit`（`tests/test_*.py`）和 `ask-gpu`（`tests/gpu/test_gpu_*.py`）。第三方编译器专属测试放在调用方仓库，ASK 使用通用协议与产物用例。
  - `tests/gpu/` 不放 `__init__.py`，使顶层 `unittest discover -s tests` 只运行纯 CPU 单测（`ask-unit`），物理 GPU 测试统一由 `ask-gpu` 调度。
  - 新增数值与多精度测试优先复用 `tests/support/pointwise.py` 的参考实现与 `plan_matrix` 矩阵，避免拆出零散微测试文件。
  - 因缺少 `ASK_OPT`、GPU 驱动或编译工具而 `skipped` 的测试不能视作功能验证通过。

---

## 文档与双向链接

- 文档采用 `docs/核心思想.md` ↔ `docs/01..09-*.md` ↔ `../fugatensor/docs/` 双向链接结构：
  - 总览各节使用 `<a id="..."></a>` 定义锚点，节末以 `工程细节：...` 链向专题章节；
  - 专题首行以 `返回核心思想：... · 相关细节：...` 链回总览与关联专题，内部小节同样提供 `<a id="..."></a>` 锚点。
- 修改章节标题、文件名或锚点时，同步检查仓内及同级 `../fugatensor/docs/` 的相对路径与 `#anchor` 引用。
