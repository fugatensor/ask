"""Validate native resident execution using the shared dtype/graph matrix."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ask import Compiler, CompilerEnvironment
from ask.runtime import ExecutionSession
from .gpu_validation import plan_matrix
from .pointwise import (DTYPES, build_module, expected_values, input_values,
                        numerically_equal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ask-opt", "triton-python", "triton-worker", "llc", "ptxas"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arch", type=int, default=86)
    parser.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=DTYPES)
    parser.add_argument("--cases", nargs="+", default=["same-shape", "shape-3x5", "mixed-accumulate"])
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least two")
    plan = plan_matrix(args.dtypes, args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(args.ask_opt, timeout=120)
    records = []
    with ExecutionSession(device=0) as session:
        for combination in plan.eligible:
            dtype, options = combination.dtype, combination.options
            module = build_module(dtype=dtype, arch=args.arch, **options)
            shape = tuple(options.get("shape", (options.get("size"),)))
            size = math.prod(shape)
            values = input_values(size, dtype)
            reference_options = {key: options[key] for key in
                                 ("bias", "factor", "pre_factor", "post_factor", "offset",
                                  "tail_offset", "external_offset", "enable_fp_fusion",
                                  "compute_dtype", "roundtrip_dtype") if key in options}
            reference_options.setdefault("bias", 1)
            reference_options.setdefault("factor", 2)
            expected = expected_values(values, dtype=dtype, **reference_options)
            label = dtype + "-" + combination.graph.name
            kernels = []
            for temperature in ("cold", "warm"):
                kernel = compiler.compile_cuda(module, cache_dir=args.cache_dir,
                    output_dir=args.output_dir / (label + "-" + temperature),
                    compilers={"triton": CompilerEnvironment(python=args.triton_python,
                                                          worker=args.triton_worker)},
                    llc=args.llc, ptxas=args.ptxas)
                wanted = "miss" if temperature == "cold" else "hit"
                assert kernel.cache["status"] == wanted, (label, kernel.cache)
                assert kernel.codegen["executable_cache"]["status"] == wanted, (
                    label, kernel.codegen["executable_cache"])
                kernels.append(kernel)

            executable = kernels[0].prepare(session)
            before_reload = session.stats
            kernels[1].prepare(session)
            after_reload = session.stats
            assert after_reload["module_loads"] == before_reload["module_loads"]
            assert after_reload["module_hits"] == before_reload["module_hits"] + 1
            x = session.from_host(values, shape=shape, dtype=dtype)
            y = session.tensor(shape, dtype)
            before = session.stats
            for _ in range(args.repeats):
                done = executable.submit(x, y)
            after = session.stats
            for field in ("allocations", "module_loads", "primary_context_retains", "synchronizations"):
                assert after[field] == before[field], (label, field, before, after)
            assert after["launches"] - before["launches"] == args.repeats
            assert type(done.query()) is bool
            done.wait()
            actual = y.to_host()
            assert numerically_equal(actual, expected, dtype), (label, actual[:8], expected[:8])

            # Use the result on a second non-default stream without a host wait
            # between its new producer launch and the import. CAI carries that
            # dependency; the consumer must retain the external tensor owner.
            cross_stream = False
            if dtype != "bf16":
                with ExecutionSession(device=0) as consumer:
                    next_executable = kernels[0].prepare(consumer)
                    executable.submit(x, y)
                    imported = consumer.from_cuda_array(y)
                    assert imported.data_ptr == y.data_ptr
                    output = consumer.tensor(shape, dtype)
                    pending = next_executable.submit(imported, output)
                    pending.wait()
                    expected_twice = expected_values(expected, dtype=dtype, **reference_options)
                    assert numerically_equal(output.to_host(), expected_twice, dtype), label
                    cross_stream = True
            record = {"case": label, "dtype": dtype, "shape": shape,
                      "repeated_launches": args.repeats, "verified_values": size,
                      "cross_stream_zero_copy": cross_stream,
                      "ir_cache": [kernel.cache for kernel in kernels],
                      "executable_cache": [kernel.codegen["executable_cache"] for kernel in kernels],
                      "hot_path_before": before, "hot_path_after": after}
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        stats = session.stats
    summary = {"combinations": len(records), "cases": records,
               "repeated_launches": sum(record["repeated_launches"] for record in records),
               "verified_values": sum(record["verified_values"] for record in records),
               "session_stats": stats}
    destination = args.output_dir / "validation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(records)} resident graph/dtype combinations: {destination}")


if __name__ == "__main__":
    main()
