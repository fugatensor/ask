返回核心思想：[常驻运行时与缓存](concepts.md#ask-runtime-cache) · 相关细节：[计划静态校验](verification.md#plan-proof) · [Host 代码生成](codegen.md#host-codegen) · [Fuga 接入](../../fugatensor/docs/architecture/backend.md#native-plan-replay)

# CUDA Graph 依赖调度与 CPU 临时区复用验证报告

CPU 八层归约扩展链的临时区减少 75%，整体延迟基本持平。CUDA 分支并行已通过设备时间戳验证；本机短图的整体延迟没有稳定改善，四分支小负载出现退化，部分计划的内部显存增加。以下分别报告这些收益和成本。

<a id="implementation"></a>

## 实现与接口

CUDA Graph 从各步骤的读写端口建立数据依赖，独立分支保留并行机会。CPU 物化张量按实际读取阶段复用临时区。两项优化均由专用 C++ 静态校验验收：GPU 在 `prepareProgram` 时校验，CPU 在生成代码时校验。

`Session::prepareProgram`、Fuga 用户 API 和 Host C ABI 保持原有用法。新增 Session 诊断计数报告已准备的内部容量和实际实例化的节点、边；它们在准备或实例化时更新。Graph 热重放沿用现有端口绑定和启动路径，生成的 CPU 循环也沿用原有分支。

ASK 仍独立交付 Compiler/Runtime SDK，Fuga 消费安装包。CPU 类型为 bool、i32、i64、f32、f16。

<a id="method"></a>

## 测量方法

基线为 ASK `98a8bc6` 的已安装 SDK。环境为 Xeon E5-2696 v4、RTX 3080、WSL2、GCC 14.2、LLVM/MLIR 23.1.4。两版库和 Fuga 安装分别快照，独立进程运行同一负载；正式测量时停止构建和测试。

共进行 7 轮，轮间交替版本先后次序。每格先取进程内样本中位数，再汇总 7 轮中位数；原始表保留每轮结果，汇总表同时记录范围与成对变化。CPU 计时固定 CPU 2，输入生成、模块加载和独立参考计算置于原生执行计时外。

- 原生 GPU：链/分支汇合 × 1,024/32,768 个元素 × 2/4 分支 × stream/Graph × 固定/交替公开地址。每个 work kernel 每线程执行 512 轮整数递推，结果逐元素核对。分别计 prepare、首次提交到完成、热 submit 和热提交到完成。
- 设备执行窗口：独立插桩 kernel 的每个 block 在首尾记录 `%globaltimer`，聚合最早开始和最晚结束，并检查不同分支的时间区间重叠。该交叉核验为一组 32 格配对，每格 21 次中位数；正常整体延迟来自七轮无插桩测量。
- 原生 CPU：32/8,192 行 × 4/16 列 × 3/8 层归约扩展 × 是否额外返回首层结果。分别测 HostObject 编译、首次调用和热函数调用，核对 ABI 工作区及数值。
- Fuga：CPU 复用上述 16 列负载；GPU 测独立输入分支的归约扩展及单 kernel 融合控制。冷路径包括构图、编译和首次完成；热路径每次新建表达式，分别计构图、evaluate 和同步完成。每格断言 25 次实际执行、1 次编译。

ASK 显存计数记录自有分配请求的字节数，包含分配池；Graph 和驱动内部存储由驱动管理。CPU 精确临时区另用独立 malloc/free 采样核对，采样进程不参与正常计时。进程 RSS 高水位包含 LLVM、Python、输入与参考数据，单独报告。

设备计时最初尝试了 host callback gate 和 GPU gate。两种方法对并行 Graph 的事件区间产生显著扰动，原始值保存在诊断表；本文采用无 gate 的设备时间戳交叉证据。数据汇总中的中位数各自计算，构图、evaluate 和完整时间的中位数相加可能不同于完整调用中位数。

<a id="cuda-graph"></a>

## CUDA Graph

槽位复用要求旧值的生产者及全部读取节点都通过数据依赖先于新生产者完成。未被消费的输出和 cuBLASLt scratch 也包含写访问。独立分支保留各自活跃存储；库步骤仍作为子图参与调度和变址更新。

专用校验器独立验证拓扑、读取可用性、容量和共享槽位的全部访问顺序。成功缓存键包含调度模式与完整依赖边。祖先搜索以可复用候选的产生位置限制范围，排除最后访问没有后继的候选。

下表为固定公开地址、Graph 模式的七轮中位数，延迟单位为 μs；显存为内部槽位的 KiB：

| 分支数 × 元素数 | 热 prepare：前 → 后 | 主机 submit：前 → 后 | 提交到完成：前 → 后 | 内部显存：前 → 后 |
| --- | --- | --- | --- | --- |
| 2 × 32,768 | 6.507 → 9.565 | 12.112 → 15.624 | 62.818 → 59.485 | 384 → 512 |
| 4 × 1,024 | 12.336 → 18.453 | 12.017 → 27.037 | 66.643 → 76.910 | 20 → 32 |
| 4 × 32,768 | 12.059 → 17.969 | 12.354 → 27.108 | 79.989 → 79.342 | 640 → 1,024 |

四分支大负载包含 11 个步骤，数据依赖为 10 条边。单组插桩配对的设备窗口从 32.768 降到 27.648 μs，并检测到独立分支执行重叠。正常主机 submit 成本上升抵消了这项设备收益。含预分配公开端口的 ASK 自有存储从 896 增到 1,280 KiB。

同例首次提交到完成为 236.992 → 407.350 μs，包含工作区分配及 Graph 实例化；独立新进程的冷 prepare 为 60.388 → 65.050 μs。这两项成本发生在热重放之前。

多根图的启动开销与 NVIDIA 的[多根 CUDA Graph 测量](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)一致。另行测试了共同 empty root 原型：7 轮、448 条数值正确记录中，四分支大负载的 submit 为 28.335 → 28.301 μs，完整时间为 84.810 → 85.881 μs。因此生产实现保留直接数据 DAG。

规模审计覆盖深链后接分支，以及前置未消费节点。8,000 步深分叉的 prepare 从初版搜索的 138.594 ms 降到 12.525 ms；最终普通链为 13.025 ms。该组是三次中位数的工程诊断，未与正式七轮静默测量混合；永久测试只断言结构和容量。

<a id="host-workspace"></a>

## CPU 物化临时区

ScalarEmitter 在编译时记录实际发出的临时区 load。物化阶段、提前计算的标量与公开输出批次共同确定最后读取位置；uniform SSA 和同批次缓存命中保留其原有行为。同阶段读写使用不同槽位，后续阶段选择容量增量较小的空闲槽，并满足所有成员的对齐。

独立校验完成后，生成器把固定容量和偏移写入 LLVM 指令。函数沿用一次 malloc/free；空工作区保留纯计算路径。650 组编译矩阵检查容量、物化数量、分配次数及精确条件分支数。

以下为 8,192 × 16、八层归约扩展链：

| 指标 | 基线 | 修改后 |
| --- | --- | --- |
| 临时区，最终输出 | 256 KiB | 64 KiB（减少 75%） |
| 临时区，保留首层输出 | 256 KiB | 96 KiB（减少 62.5%） |
| 单次 malloc/free | 1 / 1 | 1 / 1 |
| 原生热调用，最终输出 | 2,042.088 μs | 2,054.584 μs |
| 原生热调用，保留首层输出 | 2,117.026 μs | 2,120.743 μs |

独立 malloc 采样的 32 条记录均满足实际请求字节和峰值等于 ABI 工作区，函数返回后存活字节为 0。物化次数和计算循环保留原状；本次收益来自空间复用。

<a id="fuga-profile"></a>

## Fuga 整体路径

用户接口的构图、缓存命中后的 evaluate 与同步完成分别计时。单位为 μs，冷路径包含首次编译：

| 路径 | 冷路径：前 → 后 | 热构图：前 → 后 | 热 evaluate：前 → 后 | 构图到同步完成：前 → 后 |
| --- | --- | --- | --- | --- |
| CPU 8,192 × 16，八层 | 245,687 → 249,647 | 490.786 → 490.241 | 2,238.752 → 2,227.109 | 2,735.123 → 2,746.924 |
| GPU 512 × 256，四分支 | 378,811 → 386,317 | 358.771 → 367.379 | 197.029 → 197.811 | 731.383 → 762.170 |
| GPU 同尺寸逐元素融合 | 44,468 → 46,834 | 71.329 → 72.482 | 107.446 → 113.055 | 200.337 → 209.353 |

四分支计划保持 5 个计算步骤，ASK 自有显存从 4,100 增至 4,610 KiB；融合控制保持 1 个 kernel、1,536 KiB。整体耗时包含前端与驱动成本，实际改进应按这一路径判断。本次 CPU 主要获得空间收益，GPU 数据依赖调度提供了真实并行机会，同时承担额外活跃存储和启动成本。

七轮整进程 RSS 高水位记录中，原生 CPU 为 21,288 → 31,328 KiB，Fuga CPU 为 57,288 → 67,188 KiB。随后用相同原始二进制、库摘要与完整 wrapper 复测，原生 CPU 为 21,284 → 21,276 KiB；连续 16 次编译后的匿名内存为 3,432 → 3,412 KiB。原记录中的约 10 MiB 差额未复现，原表与逐映射复核结果均已归档。工作区节省由独立 malloc 计数和 ABI 字节数共同确认。

<a id="validation"></a>

## 构建与测试

| 验证项 | 结果 |
| --- | --- |
| ASK 核心与物理 GPU | 6/6 CTest 套件通过，零跳过 |
| 静态计划 | 原有 2,439 组主矩阵，加 4,752 组 Graph 分配/依赖变异；独立闭包及成对访问参考，ASan/UBSan 通过 |
| CUDA Runtime | 640 组拓扑、尺寸、地址、在途状态组合，加 16 组深链分叉结构 |
| cuBLASLt | 原有 360 组，加 48 组 dtype × 尺寸 × 分支 × scratch 上限 × 调度模式；4 MiB 实际 scratch 路径通过 |
| Host 编译 | 650 组 dtype、形状、归约、布局、保活、融合与混合类型组合 |
| Fuga 完整接入 | 60/60 方法通过，零跳过；新增 192 组 CPU 复用、50 组混合类型及 8 组原生分支组合 |
| 最终 Runtime 接入复核 | 重装后 9/9 GPU 方法通过，零跳过 |
| 安装 SDK 独立消费 | Compiler 2/2、Runtime 2/2；关闭 LLVM/MLIR 包查找 |
| 工程脚本与包依赖 | 5/5；Fuga `pip check` 通过 |

测试组合由生成器交叉各轴，数值参考独立计算；代码位于 ASK `tests/cpp/` 与 Fuga `tests/`。公开输出保活、变址子图、在途工作区、空张量、单元素 SSA 和逐元素融合均纳入验证。

ASK 已构建并安装到 `build/sdk`，Fuga 已以该 SDK 重新构建。构建目录、SDK 和 Fuga 安装目录中的共享库 SHA-256 一致：

| 库 | SHA-256 |
| --- | --- |
| Compiler | `5b30943de8845711e9cdb84cf03a97ab5ea2c010e0bba6fbb73687825a8ea208` |
| Runtime | `bd09c738c2553675d2358383b3665bec72f746deec6753d200b9f3cf332ebcac` |

<a id="evidence"></a>

## 测量记录与清理

- [完整汇总](assets/graph-workspace/summary.csv)、[详细方法](assets/graph-workspace/method.md)、[环境与哈希](assets/graph-workspace/profile-metadata.json)。汇总保留各版中位数、最小/最大值及逐轮成对变化。
- [CPU 原始数据](assets/graph-workspace/host-raw.csv)、[基线 malloc](assets/graph-workspace/baseline-host-memory.csv)、[修改后 malloc](assets/graph-workspace/candidate-host-memory.csv)。
- [GPU 原始数据](assets/graph-workspace/gpu-raw.csv)、[冷准备](assets/graph-workspace/cold-raw.csv)、[设备时间戳](assets/graph-workspace/trace-raw.csv)、[Fuga CPU](assets/graph-workspace/fuga-cpu-raw.csv)、[Fuga GPU](assets/graph-workspace/fuga-gpu-raw.csv)。
- [规模审计](assets/graph-workspace/scaling.csv)、[empty-root 对照](assets/graph-workspace/empty-root-summary.csv)、[RSS 原记录](assets/graph-workspace/process-rss.csv)、[RSS 归因复核](assets/graph-workspace/rss-attribution.csv)、[安装库一致性](assets/graph-workspace/installed-libraries.json)。

临时测量驱动、独立原型、插桩 kernel、内存探针、生成产物及基线快照在归档后统一删除。仓库保留生产代码、生成式测试矩阵和报告数据。
