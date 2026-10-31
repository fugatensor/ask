# Triton exporter prototype

This worker implements ASK request protocol 1 using `triton.compiler.ASTSource`,
`triton.compile`, `GPUTarget`, and the returned compilation artifacts. It does not
patch Triton, replace compiler stages, invoke private hooks, or launch a kernel.
The core communicates with this Python process; importing the worker module alone
does not import Triton.

The first tested configuration is deliberately narrow:

- Triton 3.7.1; observed Python 3.14.7.
- CUDA `arch=80`, `warp_size=32`, `num_warps=4`.
- One `@triton.jit` entry, two `*i32` data pointers, integer `tl.constexpr` values.
- One logical input and one logical output, with explicit source parameter bindings.
- No shared, global scratch, profile scratch, tensor descriptors, or cooperative resources.

Other configurations fail explicitly. Supporting another compiler version requires
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

LLVM 18.1.8 rejects the real artifact's `captures(none)` parameter attribute.
The worker does not delete that attribute or otherwise rewrite the export to
make an older reader accept it; use a compatible LLVM/MLIR SDK for the core.
