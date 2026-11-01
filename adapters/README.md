# 第三方编译器适配器

每个编译器的 worker、Python 策略、原生边界检查、构建规则、锁定记录和专属测试都放在 `adapters/<id>/`。

[registry.json](registry.json) 是显式支持名录，只记录 ID 到目录。构建选择 `none`、`all` 或 ID 集合后，仅执行选中目录的 CMake；未选中适配器不会因本机安装了同名包而自动启用。

当前真实适配器：[Triton](triton/README.md)。新增编译器所需的文件、接口和验证顺序见[适配与注册](../docs/17-第三方编译器适配与注册.md)。通用 CUDA 运行时属于目标后端，不放在任何编译器目录下。
