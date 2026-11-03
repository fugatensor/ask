# ASK 仓库工程规范

## 工作约定

1. 完成一组代码或文档修改并通过相关验证后，创建本地 Git 提交，便于回退。若有未完成的检查点，在提交说明中标明状态。
2. 第三方 SDK（`.deps/`）、构建产物（`build/`、`dist/`、`*.egg-info/`）、Python 虚拟环境（`.venv/`）和缓存不纳入 Git；保留依赖锁文件（`cmake/*.lock.json`）及获取脚本（`tools/fetch_deps.py`）。
3. 文档用直接的中文说明用途、做法和必要条件。删去重复铺垫、历史对比和无关的“无需／不需要”说明；限制放在相关接口处。
4. 代码与文档保持同步：只把已落地的能力写成当前行为；修改接口、构建参数、语义规则或目录结构时，同步更新文档、示例及双向链接锚点。
5. 用户示例写在文档中，展示合法调用；测试代码位于 `tests/`，保留行为与数值断言，不把测试入口称作示例。算子实现属于生产模块，测试直接调用公开接口。

---

## 架构与模块边界

- **双层 IR 分工**：
  - 高层 MLIR `ask` dialect（`include/ask/IR/`、`lib/IR/`）负责显式类型、静态形状推导、图级优化（折叠、DCE、逐元素融合分组）、`ask.mark` 溯源导出与 `ask.external` 外部算子契约检查。
  - 低层原生 LLVM IR（`lib/Lowering/`、`lib/LLIR/`）负责自有 CPU/CUDA 算子降级、LLVM 校验与优化及目标机器码（`.o` / `cubin`）生成。
  - 前端自动微分与停止梯度由 FugaTensor 的 Fuga IR 处理；`fuga.stop_gradient` 降级时映射为输入值，ASK 接收处理后的数值计算图。
- **核心与第三方适配器隔离**：
  - ASK 只交付 C++ SDK 与 CLI，不包含 Python 包或 nanobind 绑定；`core` 构建不依赖 Python、Triton、NumPy 或 PyTorch，自有 CPU/CUDA 降级属于基础核心。Python 仅用于可选的依赖获取和工程脚本。
  - `ASK::Compiler` 提供不暴露 LLVM 类型的编译门面，`ASK::Runtime` 提供原生计划与设备资源管理；消费两者不要求 LLVM 开发 SDK。直接操作 MLIR/LLVM 的调用方使用 `ASK::Core`。
  - 第三方 worker、版本策略、编译编排、磁盘缓存和 Python 产物校验由调用方维护；FugaTensor 通过自己的原生绑定直接链接 ASK。ASK 维护协议 v2/schema、标准化请求、cubin/PTX 加载与原生调度，不硬编码 DSL 或扫描 Python entrypoints。
- **图切分与执行计划**：
  - 高层 MLIR 分区函数与 JSON v1 记录缓冲区、步骤依赖和单 stream 调度；自有区域内部保留融合，每个外部节点始终独立执行。非法 IR、损坏产物和工具失败直接报错。
  - `Session::prepareProgram` 使用 SUKA PASS 的有限步骤规则证明读端口与公开输出的可用性；ASK 继续校验索引、单一生产者、kernel ABI、完成事件和在途资源。未证成的计划直接拒绝。
  - 外部 kernel 通过显存缓冲区连接；支持静态连续多输出张量、指针/null 与按值标量参数，使用普通单 stream 启动，不实现第三方代码融合。编译期配置使用递归 JSON，编译器专属值由调用方编码解释。
  - `PreparedProgram` 在 C++ 中一次绑定并提交整段计划，完成事件保留在途资源。工作区、分配和事件按完成状态复用；CUDA Graph replay 允许公开端口地址变化，不修改在途 graph 实例，不覆盖仍被调用方持有的输出。
- **数值语义**：
  - 6 种基础类型（`bool`、`i32`、`i64`、`f32`、`f16`、`bf16`）不做隐式类型提升或隐式广播。
  - `ask.constant` 按单步最近偶数舍入（RNE）编码浮点字面量，64 位整数保持完整精度。
  - `ask.cast` 浮点转整数采用确定性饱和截断（`NaN` 置 0，溢出钳制到目标整数极值）。
  - 自有 CUDA 计算通过 `llc -fp-contract=off` 保持乘加分别舍入；外部 cubin/PTX 保留生产者数值策略，ASK 不改写第三方代码。

---

## 构建与测试

- **依赖与构建指令**：
  - 拉取依赖使用 `python3 tools/fetch_deps.py`（支持 `--only llvm`）。`CMakeLists.txt` 根据 `cmake/llvm.lock.json` 自动识别 `.deps/` 下的 LLVM/MLIR：
    - 核心构建与测试：`cmake --preset core && cmake --build --preset core && ctest --preset core --output-on-failure`
- **测试组织**：
  - 原生测试源码位于 `tests/cpp/`，默认 CTest 为 `ask-core`、`ask-compiler`；显式配置 `-DASK_TEST_CUDA=ON` 增加物理 GPU 测试 `ask-runtime`。
  - 推荐以 FugaTensor 的生产接口和实现作为完整接入示例，文档不使用测试目录充当示例，也不叙述测试迁移历史。端到端数值、图组合和第三方编译器验证由 FugaTensor 维护；新增覆盖优先复用独立参考与 `plan_matrix`，避免重复实现。
  - SDK 使用 `cmake --install build/core --prefix /path/to/ask-sdk` 安装；验证独立消费 Compiler/Runtime 时不依赖 LLVM 开发 SDK，消费 Core 时匹配 ASK 实际构建的 LLVM/MLIR 版本。默认依赖记录用于复现稳定发行版；自有 LLVM/MLIR 与第三方编译器内部 LLVM 独立选择。
  - 因缺少 `ASK_OPT`、GPU 驱动或编译工具而 `skipped` 的测试不能视作功能验证通过。

---

## 文档与双向链接

- 文档采用 `docs/核心思想.md` ↔ `docs/` 下的专题文档 ↔ `../fugatensor/docs/` 双向链接结构：
  - 文件名按主题命名，不带数字序号前缀；
  - 总览各节使用 `<a id="..."></a>` 定义锚点，节末以 `工程细节：...` 链向专题章节；
  - 专题首行以 `返回核心思想：... · 相关细节：...` 链回总览与关联专题，内部小节同样提供 `<a id="..."></a>` 锚点。
- 修改章节标题、文件名或锚点时，同步检查仓内及同级 `../fugatensor/docs/` 的相对路径与 `#anchor` 引用。
