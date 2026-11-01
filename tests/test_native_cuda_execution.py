"""Compile basic ASK graphs and execute them on CUDA without a DSL compiler.

Uses the same typed graph and independent numeric oracle as host/fusion tests.
This is an explicit GPU test, not a GPU requirement for ordinary CPU CTest.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from ask import Compiler, ExecutionSession, Module, ScalarType, TensorType
from ask.cuda import CUDAError
from support.pointwise import (DTYPES, GRAPH_CASES, GraphCase, build_module,
                               cast_reference, expected_values, input_values,
                               json_values, numerically_equal, round_reference,
                               arithmetic_reference)


def compare(actual, expected, dtype, label):
    if not numerically_equal(actual, expected, dtype):
        differences = [(index, left, right)
                       for index, (left, right) in enumerate(zip(actual, expected))
                       if not numerically_equal([left], [right], dtype)]
        raise AssertionError(f"{label}: {dtype} mismatch {differences[:8]}; "
                             f"lengths {len(actual)}/{len(expected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ask-opt", "llc", "ptxas"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtypes", choices=DTYPES, nargs="+", default=DTYPES)
    parser.add_argument("--cases", nargs="+", help="restrict the shared graph matrix")
    parser.add_argument("--require-no-adapters", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = Compiler(args.ask_opt, timeout=120)
    caps = compiler.capabilities()
    if args.require_no_adapters:
        assert caps["third_party_compilers"] == [], caps
        assert caps["compiler_adapters"] == [], caps
    records, rejected, seen = [], [], set()
    graph_names = {"single", "tail", "shape-3x5", "multi-block", "expression-chain",
                   "overflow", "fractions", "special", "ask-rounding",
                   "mixed-accumulate", "mixed-special", "mixed-underflow",
                   *("cast-" + dtype for dtype in DTYPES)}
    selected = [case for case in GRAPH_CASES if case.name in graph_names]
    selected += [GraphCase("rank-zero", {"shape": ()}),
                 GraphCase("empty", {"shape": (2, 0, 3)}),
                 GraphCase("grid-stride", {"size": 65539, "block": 1}, storage_types=("i32",))]
    if args.cases:
        unknown = set(args.cases) - {case.name for case in selected}
        if unknown:
            parser.error("unknown native CUDA cases: " + ", ".join(sorted(unknown)))
        selected = [case for case in selected if case.name in args.cases]

    with ExecutionSession() as session:
        major, minor = session.capability
        arch = major * 10 + minor
        supports_bf16 = major >= 8

        def compile_pair(label, module, *, block_size=128):
            assert '"ask.external"' not in module.to_mlir()
            kernels = []
            for suffix in ("cold", "warm"):
                kernel = compiler.compile_cuda(
                    module, target={"backend": "cuda", "arch": arch},
                    block_size=block_size, ptx_version=80,
                    cache_dir=args.output / "cache", output_dir=args.output / (label + "-" + suffix),
                    llc=args.llc, ptxas=args.ptxas)
                expected_status = "miss" if suffix == "cold" else "hit"
                assert kernel.cache["status"] == expected_status, (label, kernel.cache)
                assert kernel.codegen["executable_cache"]["status"] == expected_status, (label, kernel.codegen)
                assert kernel.report.get("external_requests") == [], kernel.report
                assert kernel.artifact["target"]["arch"] == arch
                kernels.append(kernel)
            assert kernels[0].codegen["cubin_sha256"] == kernels[1].codegen["cubin_sha256"]
            return kernels

        def execute_pair(label, kernels, values, expected):
            for index, kernel in enumerate(kernels):
                prepared = kernel.prepare(session)
                inputs = [session.from_host(data, shape=tuple(port["shape"]), dtype=port["element_type"])
                          for port, data in zip(prepared.inputs, values)]
                outputs = [session.tensor(tuple(port["shape"]), port["element_type"])
                           for port in prepared.outputs]
                assert len(inputs) == len(prepared.inputs)
                assert len(expected) == len(outputs)
                before = session.stats
                completion = prepared.submit_many(inputs, outputs)
                completion.wait()
                after = session.stats
                assert after["launches"] == before["launches"] + 1
                for key in ("allocations", "module_loads", "host_to_device_copies", "device_to_host_copies"):
                    assert after[key] == before[key], (label, key, before, after)
                for port, output, reference in zip(prepared.outputs, outputs, expected):
                    compare(output.to_host(), reference, port["element_type"], label)
                record = {"case": label, "cache": kernel.cache,
                          "executable_cache": kernel.codegen["executable_cache"],
                          "phase": "cold" if index == 0 else "warm",
                          "outputs": [{"shape": port["shape"], "dtype": port["element_type"],
                                       "verified_values": len(reference),
                                       "sample": json_values(reference[:8])}
                                      for port, reference in zip(prepared.outputs, expected)]}
                records.append(record)
            print(json.dumps({"case": label, "passed": True}), flush=True)
            return prepared, inputs, outputs

        for case in selected:
            for dtype in dict.fromkeys(args.dtypes):
                options = case.for_dtype(dtype)
                if options is None:
                    continue
                module = build_module(dtype=dtype, external=False, **options)
                source = module.to_mlir()
                if source in seen:
                    continue
                seen.add(source)
                label = dtype + "-" + case.name
                if not supports_bf16 and "bf16" in source:
                    try:
                        compile_pair(label, module)
                    except (CUDAError, RuntimeError) as exc:
                        assert "bf16" in str(exc).lower(), str(exc)
                        rejected.append({"case": label, "reason": str(exc)})
                    else:
                        raise AssertionError("unsupported bf16 graph compiled successfully")
                    continue
                shape = options.get("shape", (options.get("size"),))
                values = input_values(math.prod(shape), dtype)
                numeric_options = {key: value for key, value in options.items()
                                   if key not in {"shape", "size", "block", "num_warps", "aliased_dsl"}}
                numeric_options.setdefault("bias", 1)
                numeric_options.setdefault("factor", 2)
                expected = expected_values(values, dtype=dtype, external=False, **numeric_options)
                kernels = compile_pair(label, module, block_size=options.get("block", 128))
                if case.name == "grid-stride":
                    assert kernels[0].artifact["invocation"]["grid"] == [65535, 1, 1]
                    assert kernels[0].artifact["invocation"]["block"] == [1, 1, 1]
                execute_pair(label, kernels, [values], [expected])

        # Direct cast outputs prevent an incorrect conversion being hidden by a
        # second conversion back to the original type. One graph covers every
        # destination type for a given source; no per-direction graph copies.
        destinations = [dtype for dtype in DTYPES if supports_bf16 or dtype != "bf16"]
        for dtype in dict.fromkeys(args.dtypes):
            if dtype == "bf16" and not supports_bf16:
                continue
            shape = (129,)
            module = Module()
            graph = module.graph("direct_casts", inputs={"x": TensorType(shape, dtype)},
                                 results=[TensorType(shape, target) for target in destinations])
            graph.return_(*(graph.cast(graph.inputs["x"], dtype=target) for target in destinations))
            values = input_values(129, dtype)
            expected = [[cast_reference(round_reference(value, dtype), dtype, target) for value in values]
                        for target in destinations]
            execute_pair("direct-casts-" + dtype, compile_pair("direct-casts-" + dtype, module),
                         [values], expected)

        # Independent input shapes and output types have distinct metadata and
        # bounds. The shorter result must not inherit the large input's extent.
        module = Module()
        graph = module.graph("several_ports", inputs={"x": TensorType((129,), "i64"),
                                                     "y": TensorType((129,), "i64"),
                                                     "z": TensorType((3,), "f32")},
                             results=[TensorType((129,), "i64"), TensorType((129,), "i64"),
                                      TensorType((129,), "f32"), TensorType((3,), "f32"),
                                      TensorType((), "f16")])
        total = graph.add(graph.inputs["x"], graph.inputs["y"])
        product = graph.mul(graph.inputs["x"], graph.inputs["y"])
        small = graph.add(graph.inputs["z"], graph.splat(graph.constant(5, dtype="f32"), (3,)))
        scalar = graph.splat(graph.constant(1.5, dtype="f16"), ())
        graph.return_(total, product, graph.cast(total, dtype="f32"), small, scalar)
        x, y, z = [2**60 + index for index in range(129)], [index - 64 for index in range(129)], [0.5, -1.0, 2.0]
        sums = [arithmetic_reference("add", a, b, "i64") for a, b in zip(x, y)]
        products = [arithmetic_reference("mul", a, b, "i64") for a, b in zip(x, y)]
        expected = [sums, products, [cast_reference(value, "i64", "f32") for value in sums],
                    [5.5, 4.0, 7.0], [1.5]]
        prepared, inputs, outputs = execute_pair("several-ports", compile_pair("several-ports", module),
                                                 [x, y, z], expected)
        for label, bad_inputs, bad_outputs in (
            ("input-output-overlap", inputs, [inputs[0], *outputs[1:]]),
            ("output-output-overlap", inputs, [outputs[0], outputs[0], *outputs[2:]]),
            ("missing-input", inputs[:-1], outputs),
            ("wrong-output-dtype", inputs, [outputs[2], *outputs[1:]]),
        ):
            before = session.stats["launches"]
            try:
                prepared.submit_many(bad_inputs, bad_outputs)
            except (CUDAError, ValueError) as exc:
                rejected.append({"case": label, "reason": str(exc)})
            else:
                raise AssertionError(label + " unexpectedly launched")
            assert session.stats["launches"] == before

        # A graph with no inputs is still a real GPU compilation, not a host
        # materialization of tensor values.
        module = Module()
        graph = module.graph("constant_result", inputs={}, results=[TensorType((17,), "i32")])
        graph.return_(graph.splat(graph.constant(7, dtype="i32"), (17,)))
        execute_pair("constant-result", compile_pair("constant-result", module), [], [[7] * 17])

        # ScalarType arguments are pointer-backed single values. A splat must
        # reload offset zero, not use the surrounding tensor's thread index.
        module = Module()
        tensor = TensorType((129,), "f32")
        graph = module.graph("scalar_argument", inputs={"x": tensor, "s": ScalarType("f32")},
                             results=[tensor, ScalarType("i64")])
        broadcast = graph.splat(graph.inputs["s"], (129,))
        graph.return_(graph.add(graph.inputs["x"], broadcast), graph.cast(graph.inputs["s"], dtype="i64"))
        values = [index * 0.25 for index in range(-64, 65)]
        execute_pair("scalar-argument", compile_pair("scalar-argument", module),
                     [values, [2.5]], [[value + 2.5 for value in values], [2]])

    assert records, "no CUDA execution was performed"
    assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
    counters = {tier: {name: sum(record[tier]["counters"][name] for record in records)
                       for name in ("requests", "hits", "misses", "invalid", "builds")}
                for tier in ("cache", "executable_cache")}
    summary = {"passed": True, "graph_family": "typed-pointwise-v1",
               "third_party_compilers": caps["third_party_compilers"],
               "arch": arch, "dtypes": list(args.dtypes),
               "gpu_executions": len(records),
               "values_checked": sum(port["verified_values"] for record in records for port in record["outputs"]),
               "cache_counters": counters, "rejections": rejected, "cases": records}
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("cases", "rejections")}))


if __name__ == "__main__":
    main()
