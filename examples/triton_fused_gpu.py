"""Compile P -> Triton F -> Q, execute the fused cubin, and verify persistent cache reuse.

From the repository, supply explicit tool paths and a fresh validation cache:

  PYTHONPATH=python python3 examples/triton_fused_gpu.py \
    --ask-opt build/triton/bin/ask-opt \
    --triton-python /path/to/triton/python \
    --triton-worker adapters/triton/worker.py \
    --llc .deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc \
    --ptxas /path/to/triton/backends/nvidia/bin/ptxas \
    --cache-dir /tmp/ask-fused-cache --output-dir /tmp/ask-fused-results

This program needs a CUDA driver and a compatible compute capability 8.x GPU.
All graph and DSL sources are defined here; no test fixture is required.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import random

from ask import Compiler, Module, TensorType


def build_module(size: int, *, bias: int = 1, factor: int = 2) -> Module:
    """y = ((x + bias) * factor) + 3, with i32 wrapping at every operation."""
    source = f'''import triton
import triton.language as tl

@triton.jit
def scale(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(t + i, mask=i < N, other=0)
    tl.store(u + i, value * {factor}, mask=i < N)
'''
    module = Module()
    tensor = TensorType((size,), "i32")
    dynamic = TensorType((None,), "i32")
    semantics = module.semantics("scale_i32_v1", inputs=[dynamic], results=[dynamic])
    implementation = module.implementation(
        "triton_scale", compiler="triton", source=source, entry="scale",
        signature={"t": "*i32", "u": "*i32"},
        source_bindings={"t": "input:0", "u": "output:0"},
        constants={"N": size, "BLOCK": 128}, options={"num_warps": 4},
        target={"backend": "cuda", "arch": 80, "warp_size": 32},
        grid=(max(1, (size + 127) // 128), 1, 1))
    graph = module.graph("pipeline", inputs={"x": tensor}, results=[tensor])
    increment = graph.splat(graph.constant(bias, id="bias"), (size,), id="bias_tensor")
    before = graph.add(graph.inputs["x"], increment, id="P", value_id="v_t")
    before = graph.mark(before, mark_id="before_F", role="input", port_id="F.in")
    external = graph.external(before, semantics=semantics, implementation=implementation,
                              id="F", value_id="v_u")
    external = graph.mark(external, mark_id="after_F", role="output", port_id="F.out")
    offset = graph.splat(graph.constant(3, id="offset"), (size,), id="offset_tensor")
    result = graph.add(external, offset, id="Q", value_id="v_y")
    graph.return_(result)
    return module


def input_values(size: int) -> list[int]:
    edges = [2147483647, -2147483648, -1, 0, 1, 1073741824, -1073741824]
    randomizer = random.Random(7357 + size)
    return (edges[:size] + [randomizer.randint(-(2**31), 2**31 - 1)
                           for _ in range(max(0, size - len(edges)))])


def expected_values(values: list[int], bias: int, factor: int) -> list[int]:
    wrap = lambda value: ctypes.c_int32(value).value
    return [wrap(wrap(wrap(value + bias) * factor) + 3) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ask-opt", required=True)
    parser.add_argument("--triton-python", required=True)
    parser.add_argument("--triton-worker", required=True)
    parser.add_argument("--llc", required=True)
    parser.add_argument("--ptxas", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="use a fresh directory to demonstrate first-call misses")
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(arguments.ask_opt, timeout=120)
    records = []

    def check(label, size, *, bias=1, factor=2, expected_hit=False):
        module = build_module(size, bias=bias, factor=factor)
        kernel = compiler.compile_cuda(
            module, cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir / label,
            triton_python=arguments.triton_python, triton_worker=arguments.triton_worker,
            llc=arguments.llc, ptxas=arguments.ptxas)
        values = input_values(size)
        execution = kernel.run(values)
        actual = execution["output"]
        expected = expected_values(values, bias, factor)
        if actual != expected:
            raise AssertionError(f"{label}: fused GPU output differs from i32 reference")
        if (kernel.cache["status"] == "hit") != expected_hit:
            raise AssertionError(f"{label}: expected cache hit={expected_hit}, got {kernel.cache}; "
                                 "use a fresh --cache-dir when repeating the validation")
        if not expected_hit and kernel.cache["status"] != "miss":
            raise AssertionError(f"{label}: expected a cold miss, got {kernel.cache}")
        if not execution["device_execution_performed"] or execution["original_triton_binary_used"]:
            raise AssertionError(f"{label}: expected execution of the newly generated cubin")
        (arguments.output_dir / label / "execution.json").write_text(
            json.dumps(execution, indent=2) + "\n")
        record = {"case": label, "N": size, "bias": bias, "factor": factor,
                  "cache": kernel.cache, "llvm_sha256": kernel.codegen["llvm_sha256"],
                  "cubin_sha256": kernel.codegen["cubin_sha256"],
                  "device": execution["device"],
                  "verified_values": len(values), "output_sample": actual[:7]}
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        return kernel

    # Each source shape is a distinct static compilation. Repeating it must reuse
    # cached ASK IR, then run llc/ptxas again and execute the newly generated cubin.
    for size in (1, 10, 127, 128, 129, 257, 4097):
        first = check(f"n{size}-cold", size)
        second = check(f"n{size}-warm", size, expected_hit=True)
        if first.codegen["cubin_sha256"] != second.codegen["cubin_sha256"]:
            raise AssertionError("cache hit changed the compiled cubin")
    check("changed-ask-add", 10, bias=2)
    check("changed-triton-dsl", 10, factor=3)
    check("changed-shape", 11)
    counters = {name: sum(record["cache"]["counters"][name] for record in records)
                for name in ("requests", "hits", "misses", "invalid", "builds")}
    expected_counters = {"requests": 17, "hits": 7, "misses": 10, "invalid": 0, "builds": 10}
    if counters != expected_counters:
        raise AssertionError(f"unexpected cache counters: {counters}")
    summary = {"gpu_executions": len(records), "cache_counters": counters, "cases": records}
    (arguments.output_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"cache_counters": counters}, sort_keys=True))
    print(f"Verified {len(records)} GPU executions; results: {arguments.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
