# ASK 用户 API

返回核心思想：[构建与进程内接入](核心思想.md#ask-frontend-build) · 相关细节：[安装](安装.md) · [外部编译器接入](外部编译器适配接入规范.md) · [常驻运行时](常驻运行时与多级缓存.md)

ASK 的 C++ 接口分为编译器和运行时。接入示例见 [FugaTensor](../../fugatensor/README.md)，其原生扩展链接 Compiler 与 Runtime；Python 用法见 [FugaTensor API](../../fugatensor/docs/API/README.md)。

<a id="api-entrypoints"></a>

## 1. 接入与职责

```cmake
find_package(ASK 0.1.0 EXACT CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

| 用途 | 头文件与组件 |
| --- | --- |
| 图校验、优化、分区、自有代码生成 | `ask/Frontend/CompilerAPI.h`，链接 `ASK::Compiler` |
| 设备缓冲区、kernel、计划、CUDA Graph、事件 | `ask/Runtime/CUDA.h`，链接 `ASK::Runtime` |
| 直接构造或变换 MLIR | `ask/Frontend/Compiler.h` 等，链接 `ASK::Core` 并准备与 ASK 实际构建版本匹配的 LLVM/MLIR SDK |

Compiler 与 Runtime 使用标准 C++ 类型，编译和 GPU 提交在进程内完成。代码生成库内部调用 LLVM，CUDA runtime 按需加载驱动。

<a id="graph-api"></a>

## 2. 提交 ASK 图

Compiler 接收 ASK MLIR 文本，调用方可直接生成文本，或通过 Core 的 MLIR API 构图。FugaTensor 先将 Fuga IR 降级为 ASK IR，再提交编译。

下面是 `y = x + 1` 的静态图：

```mlir
module attributes {ask.schema_version = 1 : i64} {
  func.func @increment(%x: tensor<4xi32>) -> tensor<4xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 4>} : (i32) -> tensor<4xi32>
    %y = "ask.add"(%x, %ones) {overflow = "wrap"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %y : tensor<4xi32>
  }
}
```

类型包括 `bool/i32/i64/f32/f16/bf16`，MLIR 的 bool 写作 `i1`。类型转换用 `ask.cast`，标量填充用 `ask.splat`，执行前所有形状须确定。`tensor<f32>` 是通过指针传递的单元素 0-D 张量；Host 标量 `f32` 输入按值传递。舍入和溯源规则见 [ASK IR](ASK-IR与数值语义.md)。

形状变化使用 `ask.reshape`，例如 `"ask.reshape"(%x) {shape = array<i64: 2, 2>} : (tensor<4xi32>) -> tensor<2x2xi32>`。输入输出保持相同元素类型、总数和连续次序；目标形状中的 `-1` 须先由调用方解析。外部 kernel 会接收变换后的 `2×2` 端口。

<a id="compiler-api"></a>

## 3. 编译接口

编译入口位于 `namespace ask::compiler`，使用具名选项和按阶段区分的结果类型。所有公开类型仅依赖 C++17 标准库。

| 调用 | 结果 |
| --- | --- |
| `prepare(source, options={})` | `PrepareResult`：校验后的 `ir`、`shapeSnapshotJSON`、`externalRequestsJSON` 与 `diagnosticsJSON` |
| `partition(source, options={})` | `PartitionResult`：上述字段，以及 `executionPlanJSON` 和按步骤 ID 索引 MLIR 的 `stepSources` |
| `codegen(source, options)` | `CodegenResult`：`target`、`outputPath`、`ir`、形状快照、外部请求、诊断，以及按目标提供的 `abiJSON` 或 `cudaArtifactJSON` |

`PrepareOptions.optimize` 默认 `false`，`PartitionOptions.optimize` 和 `CodegenOptions.optimize` 默认 `true`。三个选项类型都通过 `optimization` 指定 `OptimizationOptions`，其中 `foldConstants`、`eliminateDeadCode`、`fuseElementwise` 默认 `true`。

`CodegenOptions.target` 使用 `CodegenTarget` 枚举，默认 `HostLLVM`：

- `HostLLVM`：`ir` 返回本机 LLVM 文本，`abiJSON` 有值。
- `HostObject`：将本机 `.o` 写入必填 `outputPath`，`ir` 为空，`abiJSON` 有值。链接和加载由调用方负责。
- `CUDALLVM`：必须指定 `architecture`，例如 `86`；`ir` 返回 CUDA LLVM 文本，`cudaArtifactJSON` 有值。`blockSize` 默认 `128`，`ptxVersion` 默认 `80`。调用方继续用 `llc` / `ptxas` 生成 PTX/cubin，自有计算保持 `-fp-contract=off`。

`outputPath` 只用于 `HostObject`，LLVM 文本目标须保持为空。`abiJSON` 与 `cudaArtifactJSON` 为 `std::optional<std::string>`，仅对应目标的字段有值。其它 JSON 字段为 `std::string`，保留现有版本化契约；`stepSources` 为 `std::map<std::string, std::string>`。`shapeSnapshotJSON` 供形状校验与编译缓存使用，`externalRequestsJSON` 是外部请求数组。`diagnosticsJSON` 保存阶段、溯源和优化统计，编译编排通过上述具名字段获取产物。

```cpp
#include <ask/Frontend/CompilerAPI.h>

// source 为上面的 ASK MLIR 字符串。
ask::compiler::PrepareOptions preparation;
preparation.optimize = true;
auto prepared = ask::compiler::prepare(source, preparation);
auto partitioned = ask::compiler::partition(source);

ask::compiler::CodegenOptions cudaOptions;
cudaOptions.target = ask::compiler::CodegenTarget::CUDALLVM;
cudaOptions.architecture = 86;
auto cuda = ask::compiler::codegen(source, cudaOptions);

ask::compiler::CodegenOptions hostOptions;
hostOptions.target = ask::compiler::CodegenTarget::HostObject;
hostOptions.outputPath = "increment.o";
auto host = ask::compiler::codegen(source, hostOptions);
```

`ask::compilerCapabilitiesJSON()` 返回构建版本、类型、目标后端与运行时能力；`ask::compilerLibraryPath()` 返回当前实际载入的 Compiler 共享库路径，可用于内容摘要与编译缓存身份。

`namespace ask` 下的 `prepareJSON`、`partitionJSON`、`codegenJSON` 保留兼容接口，返回 `{ir, reportJSON}` 字符串对。`ask-opt --report-json` 同样输出含形状快照、外部请求、执行计划和目标产物的完整报告。具名结果与完整报告使用相同的序列化规则，保留浮点配置的词法类型、负零和 64 位整数精度。

| 兼容接口 | 默认参数 |
| --- | --- |
| `prepareJSON(source, optimize, foldConstants, eliminateDeadCode, fuseElementwise)` | `false, true, true, true` |
| `partitionJSON(source, optimize, foldConstants, eliminateDeadCode, fuseElementwise)` | `true, true, true, true` |
| `codegenJSON(source, target, output, architecture, blockSize, ptxVersion, optimize)` | `output="", architecture=0, blockSize=128, ptxVersion=80, optimize=true`；`target` 为 `"host-llvm"`、`"host-object"` 或 `"cuda-llvm"` |

各请求使用独立 MLIR context，支持并发调用。失败抛出带原生诊断的 `std::runtime_error`。接口验证见[编译器测试](../tests/cpp/compiler.cpp)。

<a id="execution-api"></a>

## 4. 原生会话与计划执行

以下对象位于 `namespace ask::runtime`，由 `std::shared_ptr` 管理。会话持有一条非阻塞 CUDA stream，所有步骤按计划顺序执行。

| 接口 | 行为 |
| --- | --- |
| `Session::create(device=0)` | 创建目标设备的常驻会话，保留 primary context |
| `session->allocate(bytes)` | 分配原生设备缓冲区，可复用同字节数的空闲分配 |
| `session->importBuffer(pointer, bytes, readonly=false)` | 借用外部设备存储，不取得其实际分配的所有权 |
| `session->load(binary, entry, requiredTypes, arch, parameterWidths)` | 加载 cubin 字节或 PTX 文本，校验设备与 kernel ABI，复用驻留模块 |
| `session->prepareProgram(bufferSizes, inputs, outputs, steps, graph=false)` | 校验执行计划和内部缓冲区槽位映射，创建不可变 `PreparedProgram` |
| `program->submit(inputs, outputs)` | 绑定公开缓冲区，一次提交完整计划，返回完成事件 |
| `program->run(inputs)` | 自动分配公开输出，返回 `{outputs, event}`；计划中的输入别名与重复输出保持原存储 |
| `event->query()` / `event->wait()` | 查询完成或等待完成 |
| `session->reap()` / `synchronize()` / `close()` | 回收完成事件、同步 stream、同步关闭会话 |
| `session->stats()` | 返回模块、分配、工作区、事件和 graph 等累计计数 |

`parameterWidths` 按 kernel 的物理参数顺序填写：指针（含 null）用 `0`，按值标量用其 ABI 字节数 `1/2/4/8`。驱动报告的参数数量、宽度和偏移必须匹配。`requiredTypes` 声明产物实际使用的基础类型，`arch` 来自编译目标。

每个 `PlanStep` 保存已加载的 `kernel`、`arguments`、三维 `grid/block`、`sharedBytes` 和逻辑输入输出索引。`PlanArgument.buffer` 指向整个计划的缓冲区表，负数表示 null；非空 `scalar` 保存冻结的标量 ABI 字节。启动标量不随 `submit` 改变，变更配置须准备新程序。

下面将一个双指针 kernel 接入单步骤计划；`kernel` 已由 `load` 获取：

```cpp
using namespace ask::runtime;
PlanStep step;
step.kernel = kernel;
step.arguments = {{0, {}}, {1, {}}};
step.grid = {1, 1, 1};
step.block = {128, 1, 1};
step.inputs = {0};
step.outputs = {1};
auto program = session->prepareProgram({16, 16}, {0}, {1}, {step}, true);
auto x = session->allocate(16);
std::int32_t values[4] = {1, 2, 3, 4};
x->copyFromHost(values, sizeof(values));
auto [outputs, done] = program->run({x});
done->wait();
std::int32_t result[4];
outputs.front()->copyToHost(result, sizeof(result));
```

缓冲区字节数按执行计划和 kernel ABI 中的形状、元素类型计算。输入允许共享存储；不同逻辑输出须与其它公开端口分离。计划中的输入别名和重复输出须绑定同一存储。ASK PASS 在准备时校验所有内部缓冲区的物理槽位容量与生存期；公开端口不参与槽位共享。全部端口通过检查后才启动 kernel。

传 `graph=true` 使用 CUDA Graph replay：首次为工作区实例化 graph，后续更新 kernel 的输入输出地址后复用。拓扑、kernel、形状和启动配置固定，地址可变化。在途调用各用独立工作区和 graph 实例，完成后复用中间缓冲区和事件。顺序提交和 graph 共用参数绑定、校验及资源管理；驱动能力不足或调用失败会报错。验证见[运行时测试](../tests/cpp/runtime.cpp)。

<a id="external-api"></a>

## 5. 第三方 kernel 与外部存储

适配器提供 cubin/PTX、入口、端口和启动配置，通过 `load` 与 `prepareProgram` 接入。编译器检查 `ask.external` 的声明，`ask::compiler::partition` 将每个外部节点分为独立步骤，自有区域内部继续融合。

worker 请求、产物格式和标量类型见[外部适配规范](外部编译器适配接入规范.md)。调用方负责 worker、工具环境、磁盘缓存和原生程序准备，FugaTensor 已提供相应实现。

借用外部设备指针时，调用方须保留实际存储所有者直到完成事件结束；`importBuffer` 仅持有 ASK 包装对象。使用 `session->waitForExternalStream(producerStream)` 等待生产者，外部框架再次访问缓冲区前也须等待 ASK 完成或建立等价依赖。

<a id="artifact-api"></a>

## 6. 原始字节、资源寿命与缓存

`DeviceBuffer` 提供 `pointer()`、`size()`、`readonly()`；`copyFromHost(pointer, bytes)` 与 `copyToHost(pointer, bytes)` 批量复制原始字节，大小须完全一致，主机导出地址须可写。无参 `copyToHost()` 返回 `std::string` 字节。复制返回前等待会话工作。

会话保留待完成事件，事件保留程序、参数缓冲区和工作区，直到设备工作结束。完成后的工作区和事件句柄可复用，旧事件对象保持已完成状态，调用方持有的旧结果继续使用原存储。

调用方保存编译产物及其目标、端口和参数 ABI、启动配置、内容摘要，并在加载时校验。执行计划 JSON v1 见[图切分与计划](图切分与多Kernel执行计划.md#plan-v1)。

<a id="errors-and-limits"></a>

## 7. 错误与边界

编译接口抛出 `std::runtime_error`，设备层抛出 `ask::runtime::CUDAError`。启动失败时，前序步骤可能已提交；运行时等待这些工作结束，再释放引用并传播错误，设备写入不回滚。

当前计划支持静态连续缓冲区和单 stream 的普通 kernel 调度；动态形状、多 stream 和数据依赖控制流不在支持范围内。FugaTensor 负责 Python API 和框架级控制流，CUDA Graph 用于减少计划提交开销。
