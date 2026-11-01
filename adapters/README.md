# 第三方编译器适配器

每个第三方编译器的导出 `worker`、Python 策略、C++ 边界校验、CMake 构建规则、依赖锁文件和专属测试均放在 `adapters/<id>/` 目录下，不在公共编译或运行时代码中编写特定编译器的分支。

- [registry.json](registry.json) 记录允许构建的编译器 ID 到子目录的映射。
- 构建时通过 `ASK_THIRD_PARTY_COMPILERS` 选择 `none`、`all` 或具体 ID 列表，CMake 仅加载选中目录的构建规则并生成静态注册表；未选中的适配器不会因本机装有同名包而自动启用。
- CUDA 设备会话与内核启动位于 `lib/Runtime/`，只读取标准 `invocation` v2 执行契约，不归属具体编译器适配器。

当前已实现适配器：[Triton 3.7.1](triton/README.md)。新增 Triton 同类编译器的目录、注册字段、worker 请求与 artifact 契约、缓存身份和验收清单见[适配器接入规范](../docs/09-外部编译器适配接入规范.md#adapter-contract)；值边界内联细节见 [04. 外部编译器适配与边界内联](../docs/04-外部编译器适配与边界内联.md)，常驻执行与缓存见 [05. 常驻运行时与多级缓存](../docs/05-常驻运行时与多级缓存.md)。
