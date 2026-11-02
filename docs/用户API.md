# ASK 用户 API

返回核心思想：[构建与进程内接入](核心思想.md#ask-frontend-build) · 相关细节：[安装](安装.md) · [外部编译器接入](外部编译器适配接入规范.md) · [常驻运行时](常驻运行时与多级缓存.md)

本文描述 ASK 的公开 C++ 接口。推荐接入示例是 [FugaTensor](../../fugatensor/README.md)：它提供 Tensor API、自动组图和自动微分，原生扩展直接链接这里的 Compiler 与 Runtime 共享库。Python 用法见 [FugaTensor API](../../fugatensor/docs/API/README.md)。

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

Compiler 与 Runtime 的公开接口不包含 LLVM、MLIR 或 Python 类型，消费这两个组件无需安装 LLVM 开发头文件。代码生成库内部使用 LLVM；CUDA runtime 按需加载驱动。编译请求和 GPU 执行均为同进程调用。

<a id="graph-api"></a>

## 2. 提交 ASK 图

Compiler 接收高层 ASK MLIR 文本。调用方可以生成文本，也可以通过 Core 的 MLIR API 构造模块。Fuga IR 保持独立，在 FugaTensor 的边界层降级后提交。

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

支持 `bool/i32/i64/f32/f16/bf16`，MLIR 的 bool 拼写为 `i1`。类型转换与广播必须显式表达；设备执行前所有形状须为具体值。`tensor<f32>` 是单元素 0-D 张量，仍通过指针传递；Host 标量 `f32` 输入按值传递。图语义、舍入与溯源规则见 [ASK IR](ASK-IR与数值语义.md)。

形状变化使用 `ask.reshape`，例如 `"ask.reshape"(%x) {shape = array<i64: 2, 2>} : (tensor<4xi32>) -> tensor<2x2xi32>`。输入输出元素类型和总数必须一致，保持连续元素次序；目标形状中的 `-1` 须先由调用方解析。该操作属于自有图计算，外部 kernel 接收到的端口形状保持为 `2×2`。

<a id="compiler-api"></a>

## 3. 编译接口

以下函数位于 `namespace ask`。除 capabilities 和库路径外，返回类型均为 `std::pair<std::string, std::string>`：第一项是 IR，第二项是 JSON 报告。

| 调用 | 结果 |
| --- | --- |
| `compilerCapabilitiesJSON()` | 构建版本、类型、目标后端与运行时能力的 JSON 文本 |
| `compilerLibraryPath()` | 当前实际载入的 Compiler 共享库路径，可用于内容摘要与编译缓存身份 |
| `prepareJSON(source, optimize=false, foldConstants=true, eliminateDeadCode=true, fuseElementwise=true)` | 校验、静态形状与溯源报告；可选自有图优化 |
| `partitionJSON(source, optimize=true, foldConstants=true, eliminateDeadCode=true, fuseElementwise=true)` | 分区后 MLIR；报告提供 `execution_plan`、`step_sources` 和外部请求 |
| `codegenJSON(source, target, output="", architecture=0, blockSize=128, ptxVersion=80, optimize=true)` | 生成自有 Host/CUDA 代码与 ABI 报告 |

`codegenJSON` 的 target 有三种：

- `"host-llvm"`：第一项返回本机 LLVM 文本，JSON 含 `abi`。
- `"host-object"`：将本机 `.o` 写入必填 `output`，第一项为空，JSON 含 `abi`。链接和加载由调用方负责。
- `"cuda-llvm"`：必须指定 architecture，例如 `86`；返回 CUDA LLVM 文本与含 `cuda_artifact` 的报告。调用方继续用 `llc` / `ptxas` 生成 PTX/cubin，自有计算保持 `-fp-contract=off`。

```cpp
#include <ask/Frontend/CompilerAPI.h>

// source 为上面的 ASK MLIR 字符串。
auto [preparedIR, preparationReport] = ask::prepareJSON(source, true);
auto [cudaIR, cudaReport] = ask::codegenJSON(source, "cuda-llvm", "", 86);
auto [unused, hostReport] = ask::codegenJSON(source, "host-object", "increment.o");
```

各次请求使用独立 MLIR context，可以并发调用。失败抛出 `std::runtime_error`，包含原生诊断；不会启动另一条编译路径掩盖错误。接口回归由[编译器测试](../tests/cpp/compiler.cpp)验证。

<a id="execution-api"></a>

## 4. 原生会话与计划执行

以下对象位于 `namespace ask::runtime`，由 `std::shared_ptr` 管理。会话持有一条非阻塞 CUDA stream，所有步骤按计划顺序执行。

| 接口 | 行为 |
| --- | --- |
| `Session::create(device=0)` | 创建目标设备的常驻会话，保留 primary context |
| `session->allocate(bytes)` | 分配原生设备缓冲区，可复用同字节数的空闲分配 |
| `session->importBuffer(pointer, bytes, readonly=false)` | 借用外部设备存储，不取得其实际分配的所有权 |
| `session->load(binary, entry, requiredTypes, arch, parameterWidths)` | 加载 cubin 字节或 PTX 文本，校验设备与 kernel ABI，复用驻留模块 |
| `session->prepareProgram(bufferSizes, inputs, outputs, steps, graph=false)` | 创建不可变参数布局和逻辑端口表，返回原生 `PreparedProgram` |
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

缓冲区字节数由编译报告中的具体形状与元素类型计算，不能凭裸指针猜测。输入允许共享存储；不同逻辑输出不得覆盖其它公开端口。计划明确表达的输入别名或重复输出必须绑定同一存储。所有端口检查完成后才启动 kernel。

传 `graph=true` 使用 CUDA Graph replay：首次为工作区实例化 graph，后续复用时更新 kernel 的输入输出地址。拓扑、kernel、形状与启动配置固定，地址可变化。每个在途调用使用独立工作区和 graph 实例，完成后复用中间缓冲区和事件。普通顺序提交与 graph 使用同一套绑定、校验和资源保活；驱动能力不足或调用失败直接报错。多步骤数值、资源寿命与 graph 复用由[运行时测试](../tests/cpp/runtime.cpp)验证。

<a id="external-api"></a>

## 5. 第三方 kernel 与外部存储

ASK 不导入第三方 DSL，也不融合第三方 LLVM。调用方的薄适配器提供 cubin/PTX、入口、端口契约和启动描述，再通过 `load` 与 `prepareProgram` 接入。高层 `ask.external` 的语义声明与实现请求由编译器验证，`partitionJSON` 将每个外部节点分成独立步骤；自有区域仍可融合。

worker 请求、产物 schema 与冻结标量类型见[外部适配规范](外部编译器适配接入规范.md)。worker 启动、工具链环境、磁盘缓存和图到原生程序的安装由调用方管理；FugaTensor 已提供这层编排，不要求 ASK 持有 Python 解释器。

借用外部设备指针时，调用方必须让实际存储所有者活到完成事件之后。`importBuffer` 的引用只保活 ASK 包装对象，不能替代外部框架的所有权。使用 `session->waitForExternalStream(producerStream)` 建立生产者依赖；外部框架再次访问该缓冲区前，也须等待 ASK 完成或建立等价依赖。

<a id="artifact-api"></a>

## 6. 原始字节、资源寿命与缓存

`DeviceBuffer` 提供 `pointer()`、`size()`、`readonly()`；`copyFromHost(pointer, bytes)` 与 `copyToHost(pointer, bytes)` 按原始存储字节批量复制，大小必须完全一致，主机导出地址须可写。`copyToHost()` 的无参形式返回 `std::string` 字节。复制返回前等待会话工作，不做 dtype 或字节序转换。

会话保留待完成事件，事件保留程序、参数缓冲区和工作区。丢弃调用方持有的程序或事件不会提前释放在途资源；已完成工作区和事件句柄可以复用，旧事件对象仍维持已完成状态。`run` 不覆盖调用方仍持有的旧结果。

编译缓存与包序列化由调用方维护，原生共享库不要求某个 Python 缓存实现。保存编译产物时应保留目标、端口与参数 ABI、启动配置及内容摘要；加载之前仍须验证完整契约。执行计划 JSON v1 见[图切分与计划](图切分与多Kernel执行计划.md#plan-v1)。

<a id="errors-and-limits"></a>

## 7. 错误与边界

编译 facade 抛出 `std::runtime_error`，设备层抛出 `ask::runtime::CUDAError`。失败不会静默切换计算路径。设备实际启动失败可能发生在前序步骤已经提交之后，此时运行时等待已提交工作结束再释放引用并传播错误，不提供 GPU 事务回滚。

当前计划为静态连续缓冲区、单 stream 的普通 kernel 调度，不表达动态形状、多 stream 或数据依赖控制流。CUDA Graph 是提交优化，不改变张量运算语义或第三方融合边界。Python 用户 API 与框架级控制流属于 FugaTensor。
