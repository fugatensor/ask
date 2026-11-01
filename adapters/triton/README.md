# Triton exporter prototype

This worker implements ASK request protocol 1 using `triton.compiler.ASTSource`,
`triton.compile`, `GPUTarget`, and the returned compilation artifacts. It does not
patch Triton, replace compiler stages, invoke private hooks, or launch a kernel.
The core communicates with this Python process; importing the worker module alone
does not import Triton.

The current adapter contract is:

- Triton 3.7.1; observed Python 3.14.7.
- CUDA ordinary `arch >= 70`, `warp_size=32`, requested `num_warps=4` or `8`; BF16 requires `arch >= 80`. The requested integer architecture is retained; Triton, LLVM and ptxas must also support it. Suffixed targets such as `90a` are not accepted.
- One `@triton.jit` entry, two matching `*i1`, `*i32`, `*i64`, `*fp32`, `*fp16`, or `*bf16` data pointers, integer `tl.constexpr` values.
- One logical input and one logical output, with explicit source parameter bindings.
- No shared, global scratch, profile scratch, tensor descriptors, or cooperative resources.

`*i1` corresponds to ASK `bool`: LLVM computation uses i1, while each buffer
element occupies one byte containing 0 or 1. `i64` uses signed interpretation and
ASK arithmetic wraps at 64 bits. `fp16` uses the ASK spelling `f16`; `bf16` remains
distinct despite the same two-byte storage size. BF16 pointer types reject targets
below sm80 before compilation. Internal computation types are also collected from
the public `compiled.asm["ttir"]` output, excluding comments, strings and names;
an internal BF16 conversion rejects an older target before artifact publication.
The resulting top-level `required_element_types` applies to ordinary exports as
well as fusion. The CUDA runner checks the actual selected device
before context creation, requiring compute capability major >= 8 whenever any
ASK value or the external computation requires BF16. Intermediate BF16 casts
count too. Backend target support, BF16 device capability, and cubin compatibility
are separate checks.

Configurations outside this contract fail explicitly. Supporting another compiler version requires
checking its physical ABI and updating the tests, not merely removing the version
check. The source format allows `import triton`, `import triton.language as tl`,
and the single entry definition. It is a source-format restriction, not a security
sandbox for untrusted Python.

## Run

Use an explicitly prepared Python environment containing the locked compiler:

```bash
/path/to/triton-env/bin/python adapters/triton/worker.py --probe
/path/to/triton-env/bin/python adapters/triton/worker.py \
  --request adapters/triton/tests/fixtures/scale2.request.json \
  --output /tmp/ask-scale2-artifact.json
```

The request contains DSL text, the source entry, data-pointer `signature` and
`source_bindings`, `constants`, `fixed_options`, `target`, and the explicit `grid`.
In this protocol version, `source_bindings` covers data parameters; constexpr
parameters are provided separately in `constants`. The worker checks both against
the source and public JIT signature. Signature JSON ordering has no ABI meaning.

`--probe` emits one JSON object to stdout. An export publishes the response file
atomically after all checks succeed. Python and native compiler stdout are routed
to stderr. Failure returns a nonzero exit status with a JSON diagnostic on stderr;
callers must check the exit status rather than use a previously existing output.

The response includes unchanged LLVM text, actual entry and pointer signature,
`port_map`, source/constant provenance, target and invocation descriptions. Triton
3.7.1 adds two physical scratch pointers even when both scratch sizes are zero.
The worker verifies this pinned ABI and records explicit null bindings for them.
It never serializes `CompiledKernel`, function pointers, Python launchers, or GPU
handles. Data parameters follow the source order prescribed by this compiler ABI;
the restricted header parser checks pointer types, count, and known attributes.
It does not infer memory semantics from PTX and does not prove the source algorithm.

The response's `validation.llvm_module_verified` is false: header checking is not
a replacement for the ASK core's LLVM parser and verifier. External library digests
record producer compilation inputs; unresolved link dependencies remain explicitly
unanalyzed until the core processes the module. No execution readiness is claimed.

## Test

The fixture and protocol tests use only the Python standard library:

```bash
python3 -m unittest discover -s adapters/triton/tests -v
ASK_TEST_TRITON_PYTHON=/path/to/triton-env/bin/python \
  python3 -m unittest discover -s adapters/triton/tests -v
```

The second command adds a real public-API export of the `scale2` fixture. It does
not query a GPU, submit a kernel, or count the producer's original binary as ASK
code generation. Test fixtures are not substitutes for this export test.

`dependencies.lock.json` records the actual observed environment and compatibility
result. `requirements.lock` pins Triton, but is not a portable, hash-locked complete
environment. Triton v3.7.1's [upstream LLVM pin](https://github.com/triton-lang/triton/blob/v3.7.1/cmake/llvm-hash.txt)
is `1f126a6dea50d185c0781743a667390037ae88bd`, whose
[version file](https://github.com/llvm/llvm-project/blob/1f126a6dea50d185c0781743a667390037ae88bd/cmake/Modules/LLVMVersion.cmake)
reports LLVM 23.0.0git. This is upstream release provenance, not an assertion that
the installed wheel exposes or attests its own LLVM build revision.

The pinned LLVM 23.0.0git SDK has parsed and verified the original real export.
Fetch it with `python3 tools/fetch_llvm.py`; its URL, archive SHA256, revision and
version are recorded in `cmake/llvm.lock.json`. Export and LLVM verification have
been tested. The optional fusion path has also been recompiled through that SDK's
`llc` and packaged `ptxas`, then executed on an RTX 3080 using the CUDA Driver API.
See [GPU fusion and IR caching](../../docs/12-GPU融合与IR缓存.md) for the tested subset,
numerical checks, and reproducible commands.

LLVM 18.1.8 rejected the same artifact's `captures(none)` parameter attribute.
The worker did not delete that attribute or rewrite the export to accommodate
the older reader.

## Optional boundary tracing

An additional request field `"fusion": {"enabled": true}` asks the adapter to
validate a one-load/one-store pointwise typed dataflow and insert two logical
identity traces through public `tl.inline_asm_elementwise`. Local aliases,
reassignments, equivalent affine arithmetic and positional/keyword arguments
are accepted without prescribing a single source template. Explicit numeric
`.to(tl.float32)` and `.to(value.dtype)` conversions are checked as dataflow;
the adapter retains their Triton semantics and does not replace them with ASK
cast rules. Pointer conversions and unknown calls remain unsupported. It compares the
unmodified and traced kernels' blocked layouts and execution metadata. Ordinary
export requests keep their existing behavior.

`BLOCK` is a requested power of two from 1 through 1024. The physical thread
count is `num_warps * 32`; it need not equal BLOCK. Each logical identity may
lower to multiple LLVM calls, so the contract records actual `marker_count`
for each boundary. All sites are bound before using the compiler-independent
[LLVM value inliner](../../include/ask/LLIR/Inline.h).

Markers preserve the declared logical value type. The 16-bit formats use
`mov.b16`, 32-bit formats use `mov.b32`, and i64 uses `mov.b64`; bool uses
`mov.pred` to preserve its i1 value. These are bit-preserving
traces, not numeric casts. Address validation uses the actual element byte width.
The producer's optional `enable_fp_fusion` setting is recorded and normalized
onto F's own instructions before inserting separately rounded ASK helpers.

Native and GPU tests share one graph builder across bool/i32/i64/f32/f16/bf16,
with an independent exact arithmetic oracle. Its `compute_dtype`,
`roundtrip_dtype`, and `arch` parameters produce explicit ASK casts and matching
F source without copying the graph per type. The current fusion path still has
one external region and matching storage dtypes on all boundaries; P/Q may use
other explicit intermediate types. See [multi-precision tests](../../docs/14-多精度测试.md)
and [cast and mixed-precision rules](../../docs/15-类型转换与混合精度.md).

The response includes `instrumented_source` and a versioned `fusion_contract`
bound to original source, instrumented source and exact LLVM content hashes.
The producer-specific binder checks addresses with ScalarEvolution and proves
mask expressions equivalent using LLVM normalization on a disposable clone.
The shared inliner checks helper types, dominance and effects before rewriting
a clone transactionally. Original memory assembly is preserved; it is never decoded to guess
access semantics. Traces are removed before final code generation. Unsupported
DSL operations or compiler environment overrides fail explicitly.
