# ASK Documentation Index

Browse documentation by installation, API, compilation, and runtime.

| Document | Description | Corresponding Implementation |
| --- | --- | --- |
| [Standalone Installation](install.md) | CMake SDK installation, component consumption, and CUDA tooling | `CMakeLists.txt`, `tools/fetch_deps.py` |
| [User API](api.md) | C++ compilation interfaces, CLI, execution plans, and buffers | `include/ask/Frontend/`, `include/ask/Runtime/` |
| [Core Concepts](concepts.md) | Framework division of labor, IR, and execution flow | Overall Architecture |
| [ASK IR and Numerical Semantics](ir.md) | Explicit types, RNE, cast, shapes, and external declarations | `lib/IR/`, `lib/Frontend/Prepare.cpp` |
| [Frontend Integration and Build Configuration](build.md) | Compiler/Runtime/Core components, build system, and version contracts | `CMakeLists.txt`, `cmake/` |
| [Graph Optimization and Native Code Generation](codegen.md) | Constant folding, DCE, native fusion, Host/CUDA lowering | `lib/Transforms/`, `lib/Lowering/` |
| [External Artifacts and Execution Interface](artifacts.md) | cubin/PTX, physical ABI, and worker responsibilities | `schemas/`, `lib/Runtime/` |
| [Resident Runtime and Multi-Level Caching](runtime.md) | Session, native plan, resource reuse, CUDA Graph replay, and caller caching | `lib/Runtime/` |
| [Project Structure and Testing Suite](testing.md) | C++ project layout, CTest, and upstream integration test boundaries | `tests/cpp/`, `tests/ir/` |
| [Responsibilities of FugaTensor and ASK](fugatensor.md) | Division of labor and dependencies between framework and compiler runtime | FugaTensor, ASK |
| [Graph Partitioning and Multi-Kernel Execution Plans](partitioning.md) | owned/external/library partitioning, JSON v1, and native scheduling | `lib/Frontend/Partition.cpp`, `lib/Runtime/` |
| [Static Verification of ASK PASS Execution Plans](verification.md) | Plan validity, full slot mapping, and build dependencies | `lib/Pass/` |
| [External Compiler Adapter Integration Specification](compilers.md) | Universal worker request/response protocol v2 and cache identity | `schemas/` |
| [Primitives and Matrix Multiplication Verification Report](matmul_report.md) | Primitives, fusion, cuBLASLt, numerical matrix, and layered performance benchmarks | ASK, FugaTensor |

## Related Repository Documentation

- [FugaTensor Documentation Index](../../../fugatensor/docs/en/README.md)
- [FugaTensor Architecture Overview](../../../fugatensor/docs/en/architecture/README.md)
- [Fuga IR Lowering to ASK](../../../fugatensor/docs/en/ir/lowering.md)
