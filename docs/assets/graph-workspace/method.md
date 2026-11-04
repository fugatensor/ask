返回核心思想：[常驻运行时](../../核心思想.md#ask-runtime-cache) · 相关细节：[验证报告](../../依赖调度与临时区复用验证报告.md#method)

# CUDA Graph 与 Host 临时区量测方法

基线为 ASK `98a8bc689a69c1db66e6602aa6cd5533a546e7e5` 已安装 SDK；候选为本次验证构建。库、驱动、硬件、临时驱动的 SHA-256 见 metadata。相同原生量测二进制分别在独立进程加载两套 SDK，避免同名动态库符号混用。Fuga 分别使用独立 venv 的完整 site-packages 快照，以 `python -I` 启动；Python 包、已安装 native extension 和其 `_lib` 同时固定。

正式量测分 7 轮，每轮交替版本先后顺序。CPU 两种量测进程固定 CPU 2；GPU 保持默认调度。输入、PTX 模块加载和独立数值参考在计时外。`*-raw.csv` 每行是一个进程内的样本中位数；`summary.csv` 再汇总这 7 个进程中位数，并保存最小值、最大值和成对变化中位数。smoke 结果不进入正式统计。

原生 GPU 矩阵为 chain/fork-join × 1024/32768 个 i32 × 2/4 个分支 × stream/Graph × 固定/交替公开地址，共 32 格。输入为从 1 开始的连续 u32。每个 work kernel 的每线程执行 512 轮 `x = x * 1664525 + 1013904223 (mod 2^32)`；chain 执行 2B 个 work，fork-join 的每分支执行两个 work，再通过 B-1 个相加 kernel 合并。每个 kernel block 为 128 线程。所有输出逐元素比对独立主机整数参考。

每格测首次 prepare、7 次重复 prepare 中位数、首次提交至完成，以及 4 次预热后的 31 次热 submit 和 submit-to-complete 中位数。首次提交至完成包含工作区分配、Graph 创建/实例化和实际执行。矩阵中的“首次 prepare”指该 PreparedProgram 首次调用，进程的完整计划缓存可能已含同一槽位计划；另用 `cold-raw.csv` 在独立进程首次准备 fork-join/32768/4分支/Graph/固定地址，明确测冷准备开销。

设备交叉核验另外运行一组配对进程：在相同 kernel 每个 block 的首尾记录 `%globaltimer`，末尾同步该 block 的线程；时间戳存放于各 step 独立公开输出，保持原数值图的 DAG 和内部槽复用。聚合所有 block 的最早开始、最晚结束，21 次取执行窗口中位数；同时记录不同分支首阶段 kernel 时间区间的重叠对数。该插桩运行与 7 轮正常整体计时分列，数据在 `trace-raw.csv`。本机计时呈约1.024us量化。

先前尝试同 stream host callback gate，以及有界 mapped-memory GPU gate 等命令排队后释放，再使用 CUDA events。二者都使多根 Graph 的事件跨度大于普通提交至完成时间，因此保存为诊断数据 `event-host-gate-raw.csv` 和 `event-device-gate-candidate-smoke.csv`，不混入最终 summary。具体调度原因尚未由驱动级跟踪确定。

`internal_bytes` 为首次执行前后 ASK `resident_device_bytes` 差，每格建立新 Session，避免历史分配池影响；`peak_bytes` 为首个已完成执行时 ASK 自有总分配字节，包含该格全部公开端口与内部槽。计数为分配请求 payload，不包含 CUDA 驱动/Graph 内部存储。baseline 未提供的 graph_nodes/graph_edges 留空。候选记录实际实例化节点/边；stream 的二者为 0。

Host 矩阵为 rows32/8192 × width4/16 × depth3/8 × 是否额外返回首阶段结果，共 16 格。每阶段 `reduce_max(axis=1)` 后 broadcast 回原形状。输入是 `(i*37 % 101)-50` 的 f32；逐元素比对独立行最大值。测完整 CompilerAPI HostObject 编译、首次调用与热调用。热调用先标定约 2ms 的批次，21 批取每调用中位数；输出保持外部分配。工作区与物化数量取实际 ABI。另开 LD_PRELOAD malloc/free 计数进程，只包围一次生成函数调用，核对实际次数、请求字节、峰值和返回后存活字节；这些运行的时间不并入正式计时。

Fuga CPU 矩阵复用上述 width16 的 8 格；CUDA 使用 rows32/512 × width256 × 2/4 个独立输入分支，分支先 amax，再 broadcast 和相加，共 4 格，另有两个尺寸的 `square(x)+x` 融合控制，共 6 格。冷路径计时包含新表达式构造、编译、链接/内核产物、准备及首次完成；热路径每次重新构造 lazy 输出，并分别计构图、evaluate、构图至同步完成。每格执行 25 次（首次 + 3 次预热 + 21 次测量），断言 executions=25、compilations=1，并在首尾校验全部数值，避免对已计算 Tensor 再 evaluate 造成假加速。

`process-rss.csv` 通过独立子进程的 `getrusage(RUSAGE_CHILDREN).ru_maxrss` 记录每个完整量测进程的主机 RSS 高水位，单位 KiB；包含 Python/LLVM、量测输入与参考数据，用于系统级观察。精确工作区改善以 ABI 和独立 malloc 计数为准。
