返回文档目录：[ASK 文档](README.md) · 返回核心思想：[构建与绑定](concepts.md#ask-frontend-build) · 相关细节：[构建参数与版本契约](build.md#version-contract)

# ASK 独立安装

通过 CMake 构建并安装 ASK C++ SDK 和 `ask-opt`，调用方按组件链接所需库。

<a id="requirements"></a>

## 1. 构建与消费依赖

构建面向 Linux x86-64，需要 C++17、CMake 3.28+、Ninja 和同一安装中的 LLVM/MLIR 23.1.x SDK。依赖脚本使用 Python 3 标准库，下载 LLVM/MLIR 23.1.4，并通过 C 编译器与 Make 构建 OpenBLAS 0.3.33。执行计划校验随 ASK 的 C++ 源码构建。

| 用途 | 需要的组件 |
| --- | --- |
| 从源码构建 ASK | C++ 工具链、CMake/Ninja、zlib/zstd/libxml2 开发库，以及同一安装中的 LLVM/MLIR 23.1.x 稳定版。 |
| 消费 `ASK::Compiler` | 已安装 ASK SDK。接口使用 C++17 具名选项、结果结构和版本化 JSON 字段。 |
| 消费 `ASK::Runtime` | 已安装 ASK SDK；GPU 执行需要 NVIDIA 驱动与设备。 |
| 消费 `ASK::Core` | 已安装 ASK SDK、与 ASK 实际构建版本匹配的 LLVM/MLIR 开发文件，以及该 LLVM SDK 使用的 zlib/zstd/libxml2 开发库。 |
| 编译自有 CUDA 图 | 调用方将 ASK 生成的 LLVM IR 交给匹配的 `llc` 和 CUDA `ptxas`。 |
| 执行外部 cubin/PTX | 调用方提供二进制、入口及 ABI；PTX 在驱动加载时 JIT。 |
| 将 Host `.o` 变成动态库 | 调用方提供系统链接器，并按 Host ABI 的 `link_libraries` 链接 SDK 提供的库。 |

Debian/Ubuntu 系统依赖如下。另请确认 `cmake --version` 为 3.28 或更新版本：

```bash
sudo apt-get install build-essential ninja-build zlib1g-dev libzstd-dev libxml2-dev
```

<a id="source-install"></a>

## 2. 构建并安装 SDK

```bash
cd /path/to/ask-source
python3 tools/fetch_deps.py
cmake --preset core
cmake --build --preset core
ctest --preset core --output-on-failure
cmake --install build/core --prefix /path/to/ask-sdk
```

脚本校验归档 SHA-256，将 LLVM/MLIR SDK 和原生 OpenBLAS SDK 放入 `.deps/`，供 CMake 自动发现。`--only llvm` / `--only openblas` 单独准备一项；`--jobs 4` 控制 OpenBLAS 构建并行度。OpenBLAS 使用 LP64、单线程、并发调用锁和动态 CPU 分派，导出 `ask_` 前缀符号。指定其他依赖和构建目录时：

```bash
python3 /path/to/ask-source/tools/fetch_deps.py --destination /path/to/ask-deps
llvm_sdk=/path/to/ask-deps/LLVM-23.1.4-Linux-X64
cmake -S /path/to/ask-source -B /path/to/ask-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$llvm_sdk/lib/cmake/llvm" \
  -DMLIR_DIR="$llvm_sdk/lib/cmake/mlir" \
  -DASK_OPENBLAS_ROOT=/path/to/ask-deps/openblas-0.3.33
cmake --build /path/to/ask-build
cmake --install /path/to/ask-build --prefix /path/to/ask-sdk
```

发布编译与运行时组件时，可以分别安装各自的共享库、公开头文件和 CMake 配置：

```bash
cmake --install build/core --prefix /path/to/ask-sdk --component Compiler
cmake --install build/core --prefix /path/to/ask-sdk --component Runtime
```

完整安装目录包含 `bin/ask-opt`、`include/ask/`、共享库、Core 静态库和 `lib/cmake/ASK/`。LLVM/MLIR 与 OpenBLAS 的版本、来源和 SHA-256 分别保存在 `cmake/llvm.lock.json`、`cmake/openblas.lock.json`。Host 编译组件携带 `libask_openblas.so.0`、OpenBLAS 锁文件及 `share/licenses/ask/OpenBLAS/LICENSE`。选择其他 23.1.x 稳定版时，LLVM 和 MLIR 须来自同一安装；构建报告记录实际版本。

<a id="wheel-install"></a>

## 3. 消费安装后的 C++ 接口

```cmake
find_package(ASK CONFIG REQUIRED COMPONENTS Compiler Runtime)
target_link_libraries(my_framework PRIVATE ASK::Compiler ASK::Runtime)
```

配置调用方项目时传入 `-DCMAKE_PREFIX_PATH=/path/to/ask-sdk`。Host f32 对象引用 SGEMM 时，CMake 调用方链接 `ASK::HostBLAS`；动态 JIT 调用方按 capabilities 的 `host_link_libraries` 获取实际库路径。执行已有产物只需选择 `Runtime`。直接操作 MLIR 时选择 `Core`，并将匹配的 LLVM SDK 加入搜索路径。LLVM 的系统依赖安装在自定义目录时，将该前缀一并加入 `CMAKE_PREFIX_PATH`。

可以单独构建 Runtime 的 SDK 消费测试：

```bash
cmake -S /path/to/ask-source/tests/cpp \
  -B /path/to/runtime-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH=/path/to/ask-sdk -DASK_TEST_COMPONENT=Runtime
cmake --build /path/to/runtime-consumer
ctest --test-dir /path/to/runtime-consumer --output-on-failure
```

该测试检查 PTX 执行和资源复用，运行时需要兼容的 NVIDIA GPU。将 `ASK_TEST_COMPONENT` 改为 `Compiler` 或 `Core` 可测试另外两类组件。接口见[用户 API](api.md)，测试范围见[测试体系](testing.md#ctest-targets)。

<a id="cuda-and-adapters"></a>

## 4. CUDA 工具与第三方编译器

自有 CUDA 计算由 ASK 生成 LLVM IR，调用方运行 `llc` / `ptxas` 并缓存产物。ASK Runtime 通过 `libcuda.so.1` 加载驱动。exp/log 还使用匹配的 `llvm-link` 和 CUDA Toolkit 的 `nvvm/libdevice/libdevice.10.bc`。

CUDA matmul 使用 cuBLASLt。`ASK_ENABLE_CUBLASLT=ON`（默认）在找到 `cublasLt.h` 时启用它；CMake 搜索 `CUDA_PATH`、`CUDA_HOME` 和 `/usr/local/cuda/include`，也可指定 `ASK_CUBLASLT_INCLUDE_DIR`。运行时动态加载 `libcublasLt.so.13`、`.12` 或无版本名，库目录须在动态加载器搜索路径中。`ASK_ENABLE_CUBLASLT=OFF` 构建基础 Runtime，准备非空 matmul 时会报错。Compiler/Runtime 的公开头文件使用标准 C++ 类型。

第三方 kernel 由调用方的 worker 编译，ASK 按[协议](../../schemas/)接收 cubin/PTX 和启动信息。FugaTensor 提供 Triton worker，环境配置见 [FugaTensor 安装](../../../fugatensor/docs/zh/install.md)。

<a id="framework-install"></a>

## 5. 与 FugaTensor 连接

先安装 ASK SDK，再通过 `CMAKE_PREFIX_PATH` 将其传给 FugaTensor 构建。FugaTensor 的 nanobind 扩展链接 `ASK::Compiler` 和 `ASK::Runtime`，wheel 携带这两个共享库、Host OpenBLAS 库及其许可证。

FugaTensor 的 `pure` / `all` 选项决定是否由 pip 安装 Triton。安装命令及 CUDA 工具配置见 [FugaTensor 安装](../../../fugatensor/docs/zh/install.md#installation-variants)。
