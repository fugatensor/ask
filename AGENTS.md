# ASK 仓库工程规范

## 基本工作铁律

1. **验证后创建本地提交**：完成一组代码或文档修改并跑通相关验证后，必须创建本地 Git 提交，方便出问题时随时回退。若存在未完成的检查点，必须在提交说明中明确标明状态。
2. **保持版本库干净可复现**：第三方 SDK（`.deps/`）、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和各类编译/运行缓存绝不纳入 Git；只保留依赖锁文件（`cmake/*.lock.json`、`adapters/*/dependencies.lock.json`）及可复现的获取脚本（`tools/fetch_deps.py`）。
3. **文档要写得像人话**：用自然、通顺、直奔重点的工程中文解释“为什么这么设计、代码怎么配合、边界在哪里”，拒绝生硬的公文腔、防御性套话堆砌和跨篇重复罗列。
4. **代码与文档严格同步**：文档只把真实落地的能力写成当前行为；修改接口、构建参数、语义规则或目录结构时，必须同步更新相关文档与示例，并保持双链接锚点有效。

---

## 架构与模块边界规范

- **双层 IR 职责清晰**：
  - 高层 MLIR `ask` dialect（`include/ask/IR/`、`lib/IR/`）负责显式类型、静态形状推导、图级优化（折叠、DCE、逐元素融合组规划）、`ask.mark` 溯源导出与 `ask.external` 外部算子契约检查。
  - 低层原生 LLVM IR（`lib/Lowering/`、`lib/LLIR/`）负责自有 CPU/CUDA 算子降级、跨编译器值边界内联（`Inline.cpp` / `Fuse.cpp`）及目标机器码（`.o` / `cubin`）生成。
- **核心不依赖可选第三方包**：
  - `core` 构建（`ASK_THIRD_PARTY_COMPILERS=none`）不得依赖 Triton、NumPy 或 PyTorch；自有 CPU 后端与自有 CUDA 后端始终是基础核心能力。
  - 所有第三方编译器专属逻辑（导出 `worker`、AST/标记融合、Python 环境策略、C++ 边界验证与专属测试）必须严格收敛在 `adapters/<id>/` 下，经 `adapters/registry.json` 与 `registration.json` 显式注册。公共核心（`lib/Adapters/`、`lib/LLIR/Inline.cpp`、`lib/Runtime/CUDA.cpp`）严禁硬编码特定外部编译器品牌分支或扫描 Python `entrypoints`。
- **确定性数值语义不可妥协**：
  - 6 种基础类型（`bool`、`i32`、`i64`、`f32`、`f16`、`bf16`）严禁隐式类型提升与隐式广播。
  - `ask.constant` 必须遵循一步到位最近偶数舍入（RNE，杜绝 `double -> float -> half` 双重舍入）及满精度 64 位整数解析。
  - `ask.cast` 浮点转整数必须保持 NaN 置 0、正负溢出饱和到目标整数极值的跨平台确定性行为。
  - 跨编译器融合时，必须通过指令级 `contract` fast-math 标志 + `llc -fp-contract=on` 隔离外部 kernel 与 ASK 自有计算的 FMA 策略；边界内联失败时必须原子保持原 `llvm::Module` 未被污染。

---

## 构建、测试与验证规范

- **统一依赖与构建指令**：
  - 拉取依赖统一使用 `python3 tools/fetch_deps.py`（支持 `--only llvm` / `--only nanobind`）。`CMakeLists.txt` 会自动从 `cmake/llvm.lock.json` 识别 `.deps/` 下的锁定版 LLVM/MLIR，日常构建直接使用零参数预设：
    - 核心构建与测试：`cmake --preset core && cmake --build --preset core && ctest --preset core --output-on-failure`
    - 全适配器构建与测试：`cmake --preset all -DPython_EXECUTABLE=/path/to/triton-env/bin/python && cmake --build --preset all && ctest --preset all --output-on-failure`
- **测试体系组织约定**：
  - 测试按职责收敛在 5 个 CTest 目标中：`ask-compiler-adapter`、`ask-boundary-inline`（`tests/cpp/`）、`ask-unit`（`tests/test_*.py`）、`ask-gpu`（`tests/gpu/test_gpu_*.py`）与 `ask-triton`（`adapters/triton/tests/test_*.py`）。
  - `tests/gpu/` 目录故意不放 `__init__.py`，确保顶层 `unittest discover -s tests` 只运行秒级完成的纯 CPU 单元测试（`ask-unit`），物理 GPU 测试统一由 `ask-gpu` 调度。
  - 新增数值与多精度测试优先复用 `tests/support/pointwise.py` 的任意精度整数 IEEE-754 参考实现与 `plan_matrix` 正交矩阵，避免新增碎片化微测试文件或重复样板代码。
  - 不要把因缺少 `ASK_OPT`、GPU 驱动或 Triton 环境而 `skipped` 的测试当作真实功能验证通过。

---

## 文档与双链接维护规范

- 仓库文档采用“**顶层总览（`docs/核心思想.md`） ↔ 工程专题（`docs/01..07-*.md`） ↔ 上层框架（`../fugatensor/docs/`）**”双向互链结构：
  - 总览文档每节用 `<a id="..."></a>` 暴露稳定锚点，并在节末以 `工程细节：...` 指向专题文档对应章节；
  - 专题文档首行以 `返回核心思想：... · 相关细节：...` 回链总览及上下游专题，内部关键小节同样提供显式 `<a id="..."></a>` 锚点。
- 增删或重命名文档章节时，务必检查仓内及同级 `../fugatensor/docs/` 的所有相对路径与 `#anchor` 引用，确保零断链。
