Back to Core Concepts: [External Artifacts and Execution Interface](concepts.md#external-execution) · Related details: [Executable Artifact Boundaries](artifacts.md#binary-artifact) · [Multi-Kernel Execution Plans](partitioning.md#complex-external)

# External Compiler Adapter Integration Specification

An adapter is a worker: it receives requests exported by ASK, invokes a third-party compiler, and returns binary artifacts along with launch information. Callers are responsible for spawning workers, validating and caching artifacts, and handing them over to the ASK Runtime. FugaTensor provides this invocation pipeline along with a Triton worker.

<a id="adapter-contract"></a>

## 1. Process Interface

The worker provides two commands:

```text
worker --probe --request /path/to/request.json
worker --request /path/to/request.json --output /path/to/artifact.json
```

`--probe` computes a dependency fingerprint based on the source code and compilation options in the request, outputting JSON to stdout. The result contains at least `producer: {name, version}` and a non-empty string `adapter_version`, recording compiler installations, referenced modules, dependency locks, and anything else affecting compilation results. Callers incorporate the entire result into the cache key and verify the `producer` of the actual artifact; FugaTensor already implements this verification.

Upon successful compilation, it writes the response JSON and exits with 0; on failure, it exits with a non-zero status and writes diagnostics to stderr. Output is atomically replaced via temporary files. Python workers run using the designated interpreter.

<a id="worker-protocol"></a>

## 2. Request Protocol v2

Requests originate from `externalRequestsJSON` in Compiler results; the CLI and full JSON interface provide the same content via `external_requests` in the full report, which callers pass to the worker. Compiler IDs, graph node IDs, and function names are used for routing and reporting and are not passed to the worker. Protocol fields are as follows:

| Field | Meaning |
| --- | --- |
| `protocol_version` | Integer `2`. |
| `source` / `entry` | Source code snapshot and entry point name, interpreted by the worker. |
| `signature` | Mapping from parameters to ABI types. Tensors use `*i1`, `*i32`, `*i64`, `*fp32`, `*fp16`, or `*bf16`; see below for pass-by-value scalar types. |
| `source_bindings` | Shares keys with `signature`. Pointers bind to `input:N` / `output:N` with contiguous, 1-to-1 mapped port numbers; scalars bind directly to booleans, integers, or finite floating-point values. |
| `constants` / `fixed_options` | Recursive JSON compile-time constants and fixed options supporting null, booleans, integers, finite floats, strings, arrays, and objects; specific semantics defined by the worker. |
| `target` | CUDA target containing `backend`, `arch`, and optionally `warp_size`. |
| `grid` | Three positive integers representing a fixed launch grid. |
| `port_shapes` | Concrete shape of each logical input and output, obtained from ASK graph validation. |

Scalars support `i1` (boolean), `i8/i16/i32/i64`, `u8/u16/u32/u64`, `fp16/bf16/fp32/fp64`. Setting `source_bindings = {"x": "input:0", "n": 1024, "scale": 0.5, "y": "output:0"}` along with the corresponding signature passes standard kernel scalar parameters; these are passed by value via the CUDA launch ABI, distinct from compile-time `constants`. Integers must fall within their bit-width range; floating-point numbers are encoded using target-format RNE.

Launch values are frozen during graph construction, participating in request caching and compilation package identity; modifying values requires reconstructing the invocation. `PreparedProgram` binds tensor buffers at submission time while preserving scalar values from preparation time. Compile-time configured integers span `[-2^63, 2^64-1]`. Requests carry only tensor metadata; see [external-compiler-request-v2.schema.json](../../schemas/external-compiler-request-v2.schema.json) for the schema.

<a id="artifact-contract"></a>

## 3. Minimal Response

The response must contain exactly the following fields:

```json
{
  "protocol_version": 2,
  "format": "cubin",
  "payload": "<base64-encoded cubin>",
  "entry": "kernel_entry",
  "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
  "producer": {"name": "example", "version": "1.0"},
  "launch": {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_bytes": 0},
  "arguments": ["input:0", {"type": "i32", "value": 1024}, "output:0"],
  "required_element_types": ["f32"]
}
```

When `format = "ptx"`, `payload` is PTX text. `target` must match the request, and `launch.grid` must match the requested grid; `block` and `shared_bytes` come from actual compilation results. `arguments` are ordered by physical parameter position: each tensor port appears exactly once; scalars are written as objects containing only `type` and `value`, with type, encoded value, and occurrence count matching the request; unused pointers that are explicitly nullable may be written as `"null"`. The example above requires an `i32` launch parameter bound to `1024` in the request.

`required_element_types` declares tensor element types that must be checked at runtime; callers merge them with port types before passing them to `Session::load`. The producer compiles according to `target` and is responsible for satisfying architectural requirements of internal instructions.

FugaTensor's [`normalize_artifact`](../../../fugatensor/src/fugatensor/_ask/adapters.py) constructs `port_map` and invocation v2 from requests, validates artifacts, and translates them into native plans. Other callers may adopt the same convention. For format specifications, see [external-kernel-artifact-v2.schema.json](../../schemas/external-kernel-artifact-v2.schema.json).

<a id="cache-identity"></a>

## 4. Caching and Supported Scope

Before querying the external compilation cache, FugaTensor executes `--probe` with the request, and probes again after compilation to confirm dependencies remained consistent throughout. The cache key includes the request, probe results, worker/interpreter file digests, environment variables, and implementation digests of the invocation pipeline; cache hits are still validated against the artifact. The worker's fingerprint should cover imported helper modules and third-party dependencies.

Artifacts use standard CUDA kernels, static contiguous multi-input multi-output layouts, mixed pointer and scalar arguments, fixed launch configurations, and dynamic shared memory. Scratchpad memory is empty; `null` is used for pointers that are actually nullable. For launch requirements, see [Executable Artifacts and Ports](artifacts.md#binary-artifact).

<a id="conformance"></a>

## 5. Integration Conformance

New workers should cover multi-port scenarios across varied shapes, mixed scalar bit-widths and argument orders, cubin/PTX encodings, dynamic shared memory, cache hits, and dependency shifts. Negative testing encompasses worker failures, corrupted artifacts, and target or port mismatches. Numerical results must be validated on physical GPUs and recorded separately from CPU protocol tests.

Implementations can reference FugaTensor's [Triton worker](../../../fugatensor/src/fugatensor/_triton_worker.py). Compiler versions and resource constraints are maintained by individual adapters.
