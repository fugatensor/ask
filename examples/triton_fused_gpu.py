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
Add --extended to cover multidimensional tensors, independent logical tiles and
thread-block sizes, equivalent DSL spellings, and longer ASK expression chains.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import random

from ask import Compiler, Module, TensorType


def build_module(size: int | None = None, *, shape: tuple[int, ...] | None = None,
                 block: int = 128, num_warps: int = 4, bias: int = 1, factor: int = 2,
                 pre_factor: int = 1, post_factor: int = 1, aliased_dsl: bool = False) -> Module:
    """Build pointwise arithmetic over a concrete shape and declared F strategy.

    Runtime values are flat contiguous lists; shape remains in ASK's logical
    types and ShapeSnapshot. BLOCK counts elements, not CUDA threads.
    """
    if shape is None:
        if type(size) is not int or size < 1:
            raise ValueError("provide a positive size or a concrete nonempty shape")
        shape = (size,)
    if any(type(dimension) is not int or dimension < 1 for dimension in shape):
        raise ValueError("shape dimensions must be positive integers")
    elements = math.prod(shape)
    if size is not None and size != elements:
        raise ValueError("size must equal the product of shape dimensions")
    size = elements
    source = f'''import triton
import triton.language as tl

@triton.jit
def scale(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(t + i, mask=i < N, other=0)
    tl.store(u + i, value * {factor}, mask=i < N)
'''
    if aliased_dsl:
        source = f'''import triton
import triton.language as tl

@triton.jit
def scale(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = BLOCK * tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offsets = lane + tile
    active = offsets < N
    input_address = t + offsets
    output_address = offsets + u
    value = tl.load(input_address, active, other=0)
    scaled = value * {factor}
    tl.store(output_address, scaled, active)
'''
    module = Module()
    tensor = TensorType(shape, "i32")
    dynamic = TensorType((None,) * len(shape), "i32")
    semantics = module.semantics("scale_i32_v1", inputs=[dynamic], results=[dynamic])
    implementation = module.implementation(
        "triton_scale", compiler="triton", source=source, entry="scale",
        signature={"t": "*i32", "u": "*i32"},
        source_bindings={"t": "input:0", "u": "output:0"},
        constants={"N": size, "BLOCK": block}, options={"num_warps": num_warps},
        target={"backend": "cuda", "arch": 80, "warp_size": 32},
        grid=(max(1, (size + block - 1) // block), 1, 1))
    graph = module.graph("pipeline", inputs={"x": tensor}, results=[tensor])
    increment = graph.splat(graph.constant(bias, id="bias"), shape, id="bias_tensor")
    before = graph.add(graph.inputs["x"], increment, id="P" if pre_factor == 1 else "P_add",
                       value_id="v_t" if pre_factor == 1 else "v_pre_add")
    if pre_factor != 1:
        multiplier = graph.splat(graph.constant(pre_factor, id="pre_factor"), shape,
                                  id="pre_factor_tensor")
        before = graph.mul(before, multiplier, id="P", value_id="v_t")
    before = graph.mark(before, mark_id="before_F", role="input", port_id="F.in")
    external = graph.external(before, semantics=semantics, implementation=implementation,
                              id="F", value_id="v_u")
    external = graph.mark(external, mark_id="after_F", role="output", port_id="F.out")
    offset = graph.splat(graph.constant(3, id="offset"), shape, id="offset_tensor")
    result = graph.add(external, offset, id="Q" if post_factor == 1 else "Q_add",
                       value_id="v_y" if post_factor == 1 else "v_post_add")
    if post_factor != 1:
        multiplier = graph.splat(graph.constant(post_factor, id="post_factor"), shape,
                                  id="post_factor_tensor")
        result = graph.mul(result, multiplier, id="Q", value_id="v_y")
    graph.return_(result)
    return module


def input_values(size: int) -> list[int]:
    edges = [2147483647, -2147483648, -1, 0, 1, 1073741824, -1073741824]
    randomizer = random.Random(7357 + size)
    return (edges[:size] + [randomizer.randint(-(2**31), 2**31 - 1)
                           for _ in range(max(0, size - len(edges)))])


def expected_values(values: list[int], bias: int, factor: int, *,
                    pre_factor: int = 1, post_factor: int = 1) -> list[int]:
    wrap = lambda value: ctypes.c_int32(value).value
    result = []
    for value in values:
        before = wrap(value + bias)
        before = wrap(before * pre_factor)
        external = wrap(before * factor)
        after = wrap(external + 3)
        result.append(wrap(after * post_factor))
    return result


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
    parser.add_argument("--extended", action="store_true",
                        help="also verify multidimensional shapes, producer launch strategies and DSL aliases")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(arguments.ask_opt, timeout=120)
    records = []

    def check(label, size=None, *, shape=None, block=128, num_warps=4, bias=1, factor=2,
              pre_factor=1, post_factor=1, aliased_dsl=False, expected_hit=False):
        shape = (size,) if shape is None else shape
        size = math.prod(shape)
        module = build_module(size, shape=shape, block=block, num_warps=num_warps,
                              bias=bias, factor=factor, pre_factor=pre_factor,
                              post_factor=post_factor, aliased_dsl=aliased_dsl)
        kernel = compiler.compile_cuda(
            module, cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir / label,
            triton_python=arguments.triton_python, triton_worker=arguments.triton_worker,
            llc=arguments.llc, ptxas=arguments.ptxas)
        tensor_bindings = [binding for binding in kernel.report["shape_snapshot"]["shape_bindings"]
                           if binding["type"].startswith("tensor<")]
        if not tensor_bindings or any(binding["shape"] != list(shape) for binding in tensor_bindings):
            raise AssertionError(f"{label}: LLVM fusion lost the logical tensor shape")
        if kernel.report["fusion"]["entry"] != kernel.artifact["entry"]:
            raise AssertionError(f"{label}: fusion report no longer identifies the compiled entry")
        values = input_values(size)
        execution = kernel.run(values)
        actual = execution["output"]
        expected = expected_values(values, bias, factor,
                                   pre_factor=pre_factor, post_factor=post_factor)
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
        record = {"case": label, "N": size, "shape": list(shape), "bias": bias, "factor": factor,
                  "pre_factor": pre_factor, "post_factor": post_factor,
                  "logical_block": block, "num_warps": num_warps,
                  "launch_block": execution["block"],
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
    extended_cases = [
        ("shape-3x5", {"shape": (3, 5), "block": 64, "num_warps": 4}),
        ("shape-2x3x17", {"shape": (2, 3, 17), "block": 128, "num_warps": 8}),
        ("multi-lane-256", {"shape": (257,), "block": 256, "num_warps": 4}),
        ("multi-lane-512", {"shape": (513,), "block": 512, "num_warps": 4}),
        ("wide-block-512", {"shape": (513,), "block": 512, "num_warps": 8}),
        ("source-aliases", {"shape": (19,), "block": 256, "aliased_dsl": True}),
        ("ask-expression-chains", {"shape": (3, 7), "pre_factor": -3, "post_factor": 5}),
    ] if arguments.extended else []
    for label, options in extended_cases:
        first = check(label + "-cold", **options)
        second = check(label + "-warm", **options, expected_hit=True)
        if first.codegen["cubin_sha256"] != second.codegen["cubin_sha256"]:
            raise AssertionError("extended cache hit changed the compiled cubin")
    counters = {name: sum(record["cache"]["counters"][name] for record in records)
                for name in ("requests", "hits", "misses", "invalid", "builds")}
    extra = len(extended_cases)
    expected_counters = {"requests": 17 + 2 * extra, "hits": 7 + extra,
                         "misses": 10 + extra, "invalid": 0, "builds": 10 + extra}
    if counters != expected_counters:
        raise AssertionError(f"unexpected cache counters: {counters}")
    summary = {"gpu_executions": len(records), "cache_counters": counters, "cases": records}
    (arguments.output_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"cache_counters": counters}, sort_keys=True))
    print(f"Verified {len(records)} GPU executions; results: {arguments.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
