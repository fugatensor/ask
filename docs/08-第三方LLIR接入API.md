# 第三方怎样把 LLVM IR 交给 ASK

[返回顶层设计：B 层](核心思想.md#ask-b) · [统一层的范围](核心思想.md#ask-architecture) · [文档索引](README.md)

本篇说明产物契约。`LLIRArtifact`、`ImportResult` 和 `ask.llir.import_artifact` 是完整接口的设计示例，尚未成为 Python 类或方法；当前实现使用 JSON 产物、CLI worker 协议和 C++ 回调，代码接口及目录见[适配与注册](17-第三方编译器适配与注册.md)。已有 SDK 见[框架接入](10-上层框架接入.md)，含 external 的 GPU 入口见[融合与缓存](12-GPU融合与IR缓存.md)，纯 ASK 图见[自有 GPU 降级](19-自有GPU降级.md)。LLIR 指原生 LLVM IR，允许包含目标 intrinsic 和 inline asm。第三方编译器使用 MLIR 时，由其导出路径生成原生 LLVM IR，再交给 ASK。

首期只实现 Triton 导出器。其他编译器先遵循同一个产物格式，不要求现在就解决所有工具链的导出问题。

## 0. 图编译从 ASK IR 中的 external 节点开始

上层框架负责生成基础高层 ASK IR，用 `ask.external` 表示第三方编译区域，同时提交该编译器的 DSL。节点的 `semantics` 和 `implementation` 是符号引用，分别指向算法语义和具体实现记录；它是编译边界，与只记录追踪位置的 `ask.mark` 不同。定义见[ASK IR 的 external 输入约定](ask-ir/00-结构类型与操作.md#external)。ASK 不负责从任意框架 IR 猜出这些区域，也不从原生 LLVM IR 反向生成 DSL。

`ImplementationReference` 关联 `compiler_id`、`DSLSource`、`source_bindings`、`constants` 和 `fixed_options`。`DSLSource` 保存 `source_ref`、内容 `digest`、`language`、源入口 `entry` 和源码依赖 `deps`。框架提供这些输入，B 层验证引用与特化后，按 `compiler_id` 找到注册的导出器，把 DSL 放入 `ExportRequest.source`。导出器负责通过公开编译接口获得实际 LLVM 入口、参数与调用说明，再返回 `LLIRArtifact`。

`source_ref` 定位 worker 可以读取的文件或内容存储。客户端可以接收适配器支持的 callable，但首期独立进程路径要求提交前将其解析成源码、入口、捕获的编译期值和辅助函数依赖；无法取得这些内容就明确报不支持，不能直接传 Python 对象。对象地址、对象名、`repr` 或任意 pickle 都不能代替稳定源码身份。源码依赖 `DSLSource.deps` 与产物链接依赖 `LLIRArtifact.dependencies` 分别记录，缓存同时考虑两者。工作环境与版本锁定见[构建设计](09-构建与可选编译器.md#workers)。

下面的离线 LLVM 导入保留为低层工具入口，便于检查或接入现成产物；它不是上述图编译流程的主入口。

## 1. 已经有文件，就直接导入

最低限度只需要内容和格式：

```python
artifact = LLIRArtifact(
    format="llvm-ir",             # 或 "llvm-bitcode"
    payload=llvm_text,             # 文本用 str，bitcode 用 bytes
)
result = ask.llir.import_artifact(artifact)
```

ASK 选择相应 LLVM reader，运行 verifier，再把模块存入当前编译会话。失败时返回带位置的诊断；成功时返回模块句柄和检查状态。内部使用 `llvm::Module`，Python 等上层接口拿到的是由会话管理的句柄，不是裸指针。

格式由 `format` 声明，不靠文件后缀猜测。Triton 名为 `llir` 的文本属于 `llvm-ir`。MLIR LLVM Dialect 文本不属于这两种格式，导入器应直接说明输入类型不对。

这次调用足以检查和保存模块。没有入口绑定、launch 或追踪信息时，结果列出缺项，而不是拒绝合法 LLVM IR。之后可以带齐信息再次导入；内容摘要允许复用已解析模块。

## 2. 要调用哪个入口，就补它的调用说明

同一模块可以包含多个函数。需要执行时，为选定入口提供目标、参数来源和调用方式；一个 `LLIRArtifact` 描述一个选定入口，其他函数作为模块内定义或依赖保留。需要公开多个入口时，可以复用同一模块内容建立多条入口记录。

```python
artifact = LLIRArtifact(
    format="llvm-ir",
    payload=llvm_text,
    producer=Producer("compiler-name", compiler_version),
    entry=entry_name,
    target=exported_target,
    port_map=exported_port_map,
    invocation=exported_invocation,
    specializations=exported_constants,
    dependencies=exported_dependencies,
)
result = ask.llir.import_artifact(artifact, binding=operator_binding)
```

这里的 `exported_*` 来自本次产物及其公开调用约定，下面逐项说明。ASK 不要求用户再手写一份物理函数签名；参数和返回值类型直接从 LLVM IR 读取，然后检查这些说明是否匹配。

| 字段 | 表达什么 | 何时需要 |
|---|---|---|
| `schema_version` | ASK 产物格式版本，构造器默认当前版本 | 序列化时始终写入 |
| `format`, `payload` | 原生 LLVM 文本或 bitcode | 导入必需 |
| `producer` | 生成编译器、版本和导出器版本 | 在线导出必需；离线未知时明确记为未知 |
| `entry` | 本次使用的 LLVM 函数符号 | 为入口生成可调用包时 |
| `target` | triple、数据布局、目标特性和工具链要求 | 目标代码生成前补齐，并与模块核对 |
| `port_map` | 逻辑输入输出怎样对应本次产物的参数、返回值和常量 | 绑定高层算子时 |
| `invocation` | 参数封送、调用或启动配置、临时资源和完成方式 | 构造执行计划前 |
| `specializations` | 已固化的形状、常量与编译配置 | 存在特化时，用于核对请求和缓存 |
| 数值代码生成设置 | LLVM 文本之外仍影响运算结果的舍入、收缩、非正规数等设置 | 重新生成代码或把其他区域内联进来之前 |
| `dependencies` | 外部符号、设备库或其他链接输入及版本/内容摘要 | 有外部依赖时，链接或加载前解析 |
| `trace` | 可选的元素、访存和执行映射 | 某项分析或优化请求时 |
| `fallback` | 可选的独立合法调用产物 | 明确选择独立调用路径时 |

缺省与空值要区分。例如未知依赖不能写成“已确认无依赖”，未知资源大小不能写成零。模块已含目标信息时可以据此补充；缺少的信息也可由调用方显式提供，但不能从当前宿主机或同名文件猜测。

LLVM 文本不一定包含原编译器的所有数值设置。已实现的 Triton 路径会把实际 `enable_fp_fusion` / `enable_reflect_ftz` 保存在 `producer_metadata.compiler_metadata`，由 Triton 适配器核对并转换为标准 `codegen` 字段。Triton 可以在 LLVM 到 PTX 时才允许 FMA，即使 `.ll` 中的乘加没有 `contract`。ASK 在插入自己的操作前，将这项许可落实到 F 的指令上，随后用标准 LLVM 规则生成代码。它不是为了融合放宽精度，而是把外部隐含的配置显式化。具体代码和验证见[GPU 融合](12-GPU融合与IR缓存.md)。其他导出器也要说明这类设置；有冲突时拒绝该次融合，不能用全模块 fast-math 覆盖所有区域。

`binding` 引用 A 层已经注册的 `OperatorSemantics`，并带本次类型、形状和常量。它不重复定义 F 的算法。导入器能检查签名、端口和声明是否一致，不能自动证明任意 LLVM 程序实现了该算法；实现注册方负责对应关系，实际测试验证结果。

只生成一个低层可调用入口而不放进 A 层图时，可以使用显式的物理 ABI 调用说明，不强制创建高层算子。此时 ASK 不宣称知道它的张量语义，也不能执行依赖该语义的变换。

## 3. 参数对应必须取自导出结果

编译之前使用 `source_bindings`，编译之后使用 `port_map`，两者不能混为一份参数表。考虑 DSL 源接口 `F(t_ptr, u_ptr, N: constexpr)`，图中表达为 `F(t) -> u`。框架提交的源绑定可以是：

```text
source_bindings:
  t_ptr -> input t
  u_ptr -> output u
  N     -> constant N
constants:
  N     -> 10
```

输出 u 对应的是 DSL 的输出指针参数，不必因此成为图的额外输入。导出器用源绑定、已验证的类型与常量发起编译。若 N 被 constexpr 消除，实际导出的入口只剩两个指针参数，则根据产物回填：

```text
port_map:
  input t   -> argument 0
  output u  -> argument 1, written through pointer
specializations:
  N         -> 本次编译的确定值
```

这只是该签名下的例子，不能规定所有编译器都按这个顺序输出。参数可能被消除、拆分或聚合，也可能出现 scratch 指针等隐藏参数，最终符号也可能不同于 DSL 的源入口 `entry`。导出器结合源绑定和真实编译结果说明每个参数从哪里来；ASK 检查物理签名、输出位置与常量特化是否匹配。只有 `source_bindings` 而没有实际 `port_map`，还不能构造调用。

完整设计允许 `port_map` 表达直接参数、写入指针的输出、返回值和特化常量。当前 CUDA 执行入口仅实现一输入一输出的连续指针端口，形状、dtype 和实际参数位置必须明确；更多参数形式在实现之前报不支持。具体字段与限制见[CUDA v2 契约](17-第三方编译器适配与注册.md#3-先交付能调用的-llvm再考虑融合)。

`invocation` 按调用种类区分。例如普通主机调用需要 calling convention 和返回规则，GPU kernel 还需要 grid、block 或等价配置、共享/临时资源及完成事件。通用层只保存类型与版本，目标后端验证和执行对应描述。PTX 的启动字段不进入所有编译器都必须实现的公共字段。

## 4. 用注册的 worker 代为编译

当前导出流程按 `ask.external` 实现记录的 compiler ID 选 worker。worker 的计算职责仍是 `export(request) -> artifact`；进程协议使用 `--request FILE --output FILE`，传 JSON 文件。Python worker 用指定解释器执行，原生 worker 可以直接执行。尚未提供动态 `ask.llir.register_exporter` API，也不扫描系统插件。

一个新适配器需要在自己的目录交付三部分：worker 负责公开 DSL 编译与产物导出；Python 策略负责环境与缓存身份；原生 `CompilerAdapter` 负责产物的专属验证，按需提供融合边界。它们通过同一份 `registration.json` 注册，当前构建允许的 ID 才能加载。新增步骤见[适配器目录与注册](17-第三方编译器适配与注册.md#6-新增一个编译器的具体步骤)。

请求包含源码字符串、源入口、签名、源参数绑定、常量、目标、固定选项和 grid。worker 核对自己的源码与配置约定，通过第三方公开编译 API 取得 LLVM，再依据实际输出建立端口映射，不能把源参数表直接当成降低后的 ABI。首期传递源码字符串；完整 `DSLSource` 内容存储与依赖包接口仍待实现。

公共驱动核对响应协议、源码摘要和固定配置，解析 LLVM 并对照真实入口签名；适配器再检查本编译器的来源证据。普通 `--export-external` 不要求融合标记。`--fuse-external` 才请求已证明的边界；没有融合能力时明确拒绝这次请求，不影响导出能力。

F 的策略一旦选定，源码、常量、目标和固定选项就是基线。可以为取得缺失 IR 按相同配置重新导出，不能为了融合改 F 的 tiling、布局或执行映射。

只有二进制、没有 LLVM 的 `CallableArtifact` 及自动独立调用回退仍属后续设计。当前 worker 不返回第三方 launcher、进程内句柄或空 payload 冒充 LLVM。导出不支持时返回错误；现有图融合入口不会悄悄建立多 kernel 调度。

## 5. 返回的是几项检查结果，不是一个“可融合”标记

`ImportResult` 持有模块句柄、内容摘要、IR revision、诊断和缺失项。各阶段分别记录以下状态；导入函数不自动编译或执行代码，后两项在后续请求完成相应工作时更新。

| 维度 | 结果示例 | 说明 |
|---|---|---|
| 原生 IR | `accepted` / `error` | LLVM 解析和结构验证是否通过 |
| 算子绑定 | `not_requested` / `missing` / `checked` / `error` | 请求的端口和特化是否与实际入口一致 |
| 目标代码生成 | `unchecked` / `supported` / `unsupported` | 所选工具链是否支持这份模块 |
| 执行准备 | `unchecked` / `missing` / `prepared` / `error` | 调用包、参数和资源是否准备好 |
| 某项分析 | `not_requested` / `available` / `unknown` | 本次版本及作用范围内能否给出所需事实 |

`codegen.supported` 只能在所选后端完成相应验证或试生成后记录，不能因为 triple 相同就写入。`execution.prepared` 不代表已经运行并得到正确答案；运行结果另行返回。刚导入时，两者通常都是 `unchecked`。

例如 LLVM reader 能读入含 PTX 的模块，但当前会话未配置 NVIDIA 后端：模块仍为 `accepted`，目标检查可返回 `unsupported`。换到支持该模块的目标环境后，可以继续代码生成，不必重写 IR。目标后端已支持但 grid 缺失时，可以生成代码，执行准备仍为 `missing`。

绑定字段彼此矛盾、追踪摘要不匹配或 LLVM IR 无效，应给出明确错误。可选追踪无效时丢弃该追踪，不用它授权优化；这与模块本身无效分开处理。缺失可选分析则是 `unknown`。请求某个 Pass 时，驱动检查它所依赖的分析和作用范围，只跳过无法建立正确性前提的尝试；具体行为见 [C 层](04-C层统一处理与可选优化.md)。

## 6. Triton 导出器具体做什么

首期选定一个 Triton 版本及对应 LLVM 工具链。适配器通过公开编译入口获得 `CompiledKernel`，读取 `asm["llir"]`；这已经是原生 LLVM IR 文本。调用方另外提供本次确定的输入特化、固定选项与 grid，适配器结合实际入口和公开产物信息生成上述字段。[Triton 生成 LLVM IR](https://github.com/triton-lang/triton/blob/b89bce47e2d9e1c7fb4bd568c1e55e19c476a31e/third_party/nvidia/backend/compiler.py#L492-L544)、[读取编译阶段产物](https://github.com/triton-lang/triton/blob/b89bce47e2d9e1c7fb4bd568c1e55e19c476a31e/python/triton/compiler/compiler.py#L407-L431)

具体分三步：先从实际 LLVM 入口取得物理签名；再将未被特化消除的逻辑参数对应到它；最后检查 launcher 要求的启动、共享存储和临时资源是否都有来源。grid 由调用方给出的启动规则确定，不能从函数参数猜测；Triton 自己也是在启动时另外计算并传递它。[grid 处理](https://github.com/triton-lang/triton/blob/b89bce47e2d9e1c7fb4bd568c1e55e19c476a31e/python/triton/runtime/jit.py#L787-L798)

适配器不需要先把 PTX 访存还原成 LLVM load/store。即使完全没有 `trace`，也应能够交付 F 的代码和调用说明。随后由目标后端重新生成代码并执行；这才验证原生 IR 接入路径，不能只回调原二进制作为成功依据。

缓存命中可能跳过编译阶段，某些产物保留配置也可能使 llir 不存在，因此不能把“阶段回调一定执行”当作保证。先检查公开返回的产物；缺失时在公开支持的配置下按同一 F 基线重新导出，仍拿不到就报告该配置不支持 IR 导出。不要扫描私有缓存结构、复制内部函数或修改源码来补洞。[缓存命中与阶段执行](https://github.com/triton-lang/triton/blob/b89bce47e2d9e1c7fb4bd568c1e55e19c476a31e/python/triton/compiler/compiler.py#L268-L339)

源码链接固定在一个快照，仅用于说明已核对的行为，不构成跨版本兼容承诺。每新增支持版本，都要重新验证导出字段、参数绑定和目标代码生成。初期代码生成与公开 override 的适用范围见[代码生成与运行时](05-代码生成与运行时.md)。

## 7. 追踪是扩展数据，不能成为另一套隐藏 IR

`trace` 可以携带逻辑元素与低层访存、SSA 值及执行实例的对应关系。它需要声明格式版本、生产者、适用范围和产物摘要，导入后绑定当前 IR revision。序列化引用在读入时解析，不能直接保存内存地址；失效规则见[跨层身份与追踪](02-跨层身份与追踪契约.md)。

没有 trace 的模块仍然完整。trace 提供的信息必须对应原生 LLVM 模块，不能绕开 LLVM 语义另藏一份真正执行的程序。修改、内联或链接后，分析必须更新或失效。

若希望分析某类 inline asm，可以另外提供版本明确的语义模型，说明地址、mask、访问宽度、效果和适用条件。通用驱动只在能核验模型与当前产物对应时使用它；一个 `trusted=true` 或 `can_fuse=true` 不足以授权变换。LLVM 对 inline asm 的约束及效果声明仍须遵守。[LLVM inline assembly](https://llvm.org/docs/LangRef.html#inline-assembler-expressions)

## 8. 把编译器差异、目标差异和通用处理分开

导出器负责取得某种编译器的产物和真实调用说明；目标后端负责代码生成、链接、参数封送与启动；ASK 核心负责 LLVM 解析、结构检查、版本、分析和 Pass 驱动。接入另一种同目标编译器时，应主要新增导出器；遇到新目标，再实现相应后端。

跨进程传递文本、bitcode 和版本化说明，不直接共享来自不同 LLVM 构建的 C++ 对象。bitcode 也不承诺任意版本互读。工具链版本、DSL 内容与源码依赖摘要、`source_bindings`、常量、固定选项、目标、产物依赖内容、调用说明和导出器版本都进入缓存键；可选分析另外绑定 IR revision。

最初只需验证三组结果：真实 Triton 导出后重新生成并正确执行；缺少追踪或出现未知 asm 时仍能导入，相关优化按需跳过；错误 IR、错误绑定和缺失执行信息各自得到正确诊断。之后才增加第二个真实编译器和更强的优化。具体顺序见[实施路线](07-实施路线与端到端示例.md)。
