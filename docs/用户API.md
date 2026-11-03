返回核心思想：[构建与进程内接入](核心思想.md#ask-frontend-build) · 相关细节：[安装](安装.md) · [外部编译器接入](外部编译器适配接入规范.md) · [常驻运行时](常驻运行时与多级缓存.md)

# ASK 用户 API

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

先把上面的图保存为 `std::string source`，生成一个 Host 对象文件：

```cpp
#include <ask/Frontend/CompilerAPI.h>

ask::compiler::CodegenOptions options;
options.target = ask::compiler::CodegenTarget::HostObject;
options.outputPath = "increment.o";
auto result = ask::compiler::codegen(source, options);
```

`result.outputPath` 是对象文件路径，`result.abiJSON` 描述入口和参数。调用方链接、加载该文件。编译失败抛出带诊断的 `std::runtime_error`。

按需要选择编译阶段：

| 目的 | 调用 | 读取的结果 |
| --- | --- | --- |
| 检查图和推导形状 | `prepare(source)` | `ir`、`shapeSnapshotJSON` |
| 获取 CUDA 执行计划 | `partition(source)` | `executionPlanJSON`、按步骤 ID 索引的 `stepSources` |
| 生成自有代码 | `codegen(source, options)` | `ir` 或 `outputPath`，以及目标 ABI |

三个结果都提供 `externalRequestsJSON`，供调用方交给第三方 worker；`diagnosticsJSON` 保存溯源和优化统计。各请求使用独立 MLIR context，支持并发调用。

<details>
<summary>CUDA 代码生成与编译选项</summary>

```cpp
ask::compiler::CodegenOptions options;
options.target = ask::compiler::CodegenTarget::CUDALLVM;
options.architecture = 86;
auto result = ask::compiler::codegen(source, options);
```

`result.ir` 是 CUDA LLVM 文本，`result.cudaArtifactJSON` 描述启动 ABI。调用方用 `llc -fp-contract=off` 和 `ptxas` 继续生成 PTX/cubin。

| 选项 | 用法 |
| --- | --- |
| `target` | `HostLLVM`（默认）返回 Host LLVM；`HostObject` 写入对象文件；`CUDALLVM` 返回 CUDA LLVM |
| `outputPath` | `HostObject` 的必填输出路径；LLVM 文本目标使用默认空值 |
| `architecture` | CUDA 架构编号，如 `86` |
| `blockSize` / `ptxVersion` | CUDA 默认值分别为 `128`、`80` |
| `optimize` | `prepare` 默认 `false`；`partition` 和 `codegen` 默认 `true` |
| `optimization` | `foldConstants`、`eliminateDeadCode`、`fuseElementwise` 默认均为 `true` |

`abiJSON` 和 `cudaArtifactJSON` 是可选字符串，按目标返回对应字段。完整类型定义见 [CompilerAPI.h](../include/ask/Frontend/CompilerAPI.h)。

</details>

构建信息、CLI 和 JSON 报告接口见[前端接入](前端接入与构建配置.md#native-compiler)。

<a id="execution-api"></a>

## 4. 原生会话与计划执行

运行时接入分为三步：加载 kernel、准备计划、绑定缓冲区执行。下面假设 `binary` 是一个面向 `sm_86` 的 cubin/PTX 字符串，入口 `increment` 读取四个 `i32` 并写入四个 `i32`：

```cpp
#include <ask/Runtime/CUDA.h>

using namespace ask::runtime;
auto session = Session::create(0);
auto kernel = session->load(binary, "increment", {"i32"}, 86, {0, 0});

PlanStep step;
step.kernel = kernel;
step.arguments = {{0, {}}, {1, {}}};
step.grid = {1, 1, 1};
step.block = {128, 1, 1};
step.inputs = {0};
step.outputs = {1};
auto program = session->prepareProgram({16, 16}, {0}, {1}, {step});

std::int32_t values[4] = {1, 2, 3, 4};
auto x = session->allocate(sizeof(values));
x->copyFromHost(values, sizeof(values));
auto [outputs, done] = program->run({x});
done->wait();
std::int32_t result[4];
outputs.front()->copyToHost(result, sizeof(result));
```

`prepareProgram(bufferSizes, inputs, outputs, steps, graph=false)` 接收逻辑缓冲区字节数、公开输入输出索引和步骤列表。它校验计划并生成内部槽位映射，返回可反复执行的 `PreparedProgram`。`run(inputs)` 自动分配公开输出；已有输出缓冲区时使用 `submit(inputs, outputs)`，得到完成事件。

传 `graph=true` 使用 CUDA Graph replay。计划固定 kernel、形状和启动配置，每次可以传入新的公开缓冲区地址。工作区与事件的复用方式见[常驻运行时](常驻运行时与多级缓存.md#cuda-graph)。

<details>
<summary>kernel 参数和缓冲区契约</summary>

`load(binary, entry, requiredTypes, arch, parameterWidths)` 按产物内容和入口复用模块。`requiredTypes` 声明使用的基础类型，`arch` 是编译目标。`parameterWidths` 按物理参数顺序填写：指针（含 null）为 `0`，按值标量为 ABI 字节数 `1/2/4/8`。加载时核对驱动报告的参数数量、宽度和偏移。

`PlanStep` 保存 `kernel`、`arguments`、三维 `grid/block`、`sharedBytes` 和逻辑输入输出索引。`PlanArgument.buffer` 指向计划的缓冲区表，`-1` 表示 null；`scalar` 保存冻结的标量 ABI 字节。修改标量或启动配置时准备新程序。

公开输入允许共享存储。不同逻辑输出须与其它公开端口分离；重复输出或输入别名须绑定同一存储。每次提交核对实际缓冲区的设备、字节数、可写性和别名。内部槽位的容量和生存期在准备阶段由 [ASK PASS](规则证明与计划校验.md) 验证。

</details>

<a id="external-api"></a>

## 5. 第三方 kernel 与外部存储

适配器提供 cubin/PTX、入口、端口和启动配置，通过 `load` 与 `prepareProgram` 接入。编译器检查 `ask.external` 的声明，`ask::compiler::partition` 将每个外部节点分为独立步骤，自有区域内部继续融合。

worker 请求、产物格式和标量类型见[外部适配规范](外部编译器适配接入规范.md)。调用方负责 worker、工具环境、磁盘缓存和原生程序准备，FugaTensor 已提供相应实现。

借用外部设备指针时，调用方须保留实际存储所有者直到完成事件结束；`importBuffer` 仅持有 ASK 包装对象。使用 `session->waitForExternalStream(producerStream)` 等待生产者，外部框架再次访问缓冲区前也须等待 ASK 完成或建立等价依赖。

<a id="artifact-api"></a>

## 6. 原始字节、资源寿命与缓存

会话与资源对象由 `std::shared_ptr` 管理。会话持有一条非阻塞 CUDA stream，步骤按计划顺序执行。

| 接口 | 用途 |
| --- | --- |
| `DeviceBuffer::pointer()` / `size()` / `readonly()` | 读取设备地址、字节数和只读属性 |
| `copyFromHost(pointer, bytes)` / `copyToHost(pointer, bytes)` | 复制完整缓冲区；返回前等待会话工作 |
| `copyToHost()` | 将缓冲区复制为 `std::string` 字节 |
| `Event::query()` / `wait()` | 查询完成状态或等待完成 |
| `Session::reap()` / `synchronize()` / `close()` | 回收完成事件、同步 stream、同步关闭会话 |
| `Session::stats()` | 查看模块、分配、工作区、事件和 graph 的累计计数 |

会话保留待完成事件，事件保留程序、参数缓冲区和工作区，直到设备工作结束。完成后的工作区和事件句柄可复用，旧事件对象保持已完成状态，调用方持有的旧结果继续使用原存储。

调用方保存编译产物及其目标、端口和参数 ABI、启动配置、内容摘要，并在加载时校验。执行计划 JSON v1 见[图切分与计划](图切分与多Kernel执行计划.md#plan-v1)。

<a id="errors-and-limits"></a>

## 7. 错误处理

编译接口抛出 `std::runtime_error`，设备层抛出 `ask::runtime::CUDAError`。启动失败时，前序步骤可能已提交；运行时等待这些工作结束，再释放引用并传播错误，设备写入不回滚。

计划使用静态连续缓冲区，在单 stream 上执行普通 kernel。FugaTensor 负责 Python API 和框架级控制流，CUDA Graph 用于减少计划提交开销。
