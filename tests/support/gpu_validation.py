"""Execute one graph-case matrix over selected ASK element types on CUDA."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from ask import Compiler
from .pointwise import (DTYPES, GRAPH_CASES, GraphCase, build_module, expected_values,
                        input_values, json_values, numerically_equal)


@dataclass(frozen=True)
class MatrixCase:
    graph: GraphCase
    dtype: str
    options: dict


@dataclass(frozen=True)
class MatrixPlan:
    dtypes: tuple[str, ...]
    cases: tuple[GraphCase, ...]
    eligible: tuple[MatrixCase, ...]
    skipped: tuple[dict, ...]

    @property
    def gpu_executions(self):
        return 2 * len(self.eligible)

    @property
    def cache_counters(self):
        pairs = len(self.eligible)
        return {"requests": 2 * pairs, "hits": pairs, "misses": pairs,
                "invalid": 0, "builds": pairs}


def plan_matrix(dtypes, case_names=None, *, extended=False):
    """Select every combination before tools, caches or device work can begin."""
    dtypes = tuple(dict.fromkeys("f16" if dtype == "fp16" else dtype for dtype in dtypes))
    if not dtypes or any(dtype not in DTYPES for dtype in dtypes):
        raise ValueError("select at least one supported dtype")
    if case_names is not None:
        unknown = set(case_names) - {case.name for case in GRAPH_CASES}
        if unknown:
            raise ValueError("unknown graph cases: " + ", ".join(sorted(unknown)))
    cases = tuple(case for case in GRAPH_CASES if
                  (case_names is not None and case.name in case_names) or
                  (case_names is None and (extended or not case.extended)))
    eligible, skipped = [], []
    for case in cases:
        for dtype in dtypes:
            options = case.for_dtype(dtype)
            if options is None:
                skipped.append({"case": case.name, "dtype": dtype,
                                "reason": "case requires a floating-point dtype"})
            else:
                eligible.append(MatrixCase(case, dtype, options))
    if not eligible:
        raise ValueError("no eligible dtype/graph combinations; floating-point cases cannot run as i32")
    return MatrixPlan(dtypes, cases, tuple(eligible), tuple(skipped))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("ask-opt", "triton-python", "triton-worker", "llc", "ptxas"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="use a fresh directory to validate cold and warm requests")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtypes", nargs="+", choices=(*DTYPES, "fp16"),
                        help="run the same graph cases for each selected element type")
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in GRAPH_CASES],
                        help="select graph cases; no per-dtype graph copies are needed")
    parser.add_argument("--extended", action="store_true", help="include extra shapes and launch strategies")
    parser.add_argument("--float", action="store_true", help="include f32 (compatibility spelling)")
    parser.add_argument("--low-precision", action="store_true", help="include f16 and bf16")
    arguments = parser.parse_args(argv)
    selected = arguments.dtypes or (["i32"] + (["f32"] if arguments.float else [])
                                    + (["f16", "bf16"] if arguments.low_precision else []))
    try:
        plan = plan_matrix(selected, arguments.cases, extended=arguments.extended)
    except ValueError as error:
        parser.error(str(error))
    dtypes = plan.dtypes
    print(json.dumps({"planned_gpu_executions": plan.gpu_executions,
                      "eligible_combinations": len(plan.eligible),
                      "skipped_combinations": plan.skipped,
                      "skipped_combination_count": len(plan.skipped)}, sort_keys=True), flush=True)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(arguments.ask_opt, timeout=120)
    records = []

    def check(label, size=None, *, shape=None, block=128, num_warps=4, bias=1, factor=2,
              pre_factor=1, post_factor=1, aliased_dsl=False, expected_hit=False,
              dtype="i32", offset=3, tail_offset=None, external_offset=None,
              enable_fp_fusion=None):
        shape = (size,) if shape is None else shape
        size = math.prod(shape)
        module = build_module(size, shape=shape, block=block, num_warps=num_warps,
                              bias=bias, factor=factor, pre_factor=pre_factor,
                              post_factor=post_factor, aliased_dsl=aliased_dsl,
                              dtype=dtype, offset=offset, tail_offset=tail_offset,
                              external_offset=external_offset, enable_fp_fusion=enable_fp_fusion)
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
        values = input_values(size, dtype)
        execution = kernel.run(values)
        actual = execution["output"]
        expected = expected_values(values, bias, factor,
                                   pre_factor=pre_factor, post_factor=post_factor,
                                   dtype=dtype, offset=offset, tail_offset=tail_offset,
                                   external_offset=external_offset, enable_fp_fusion=enable_fp_fusion)
        if not numerically_equal(actual, expected, dtype):
            mismatches = [(i, left, right) for i, (left, right) in enumerate(zip(actual, expected))
                          if not numerically_equal([left], [right], dtype)]
            raise AssertionError(f"{label}: fused GPU output differs from {dtype} reference: {mismatches[:8]}")
        if execution["element_type"] != dtype:
            raise AssertionError(f"{label}: execution lost its dtype")
        if (kernel.cache["status"] == "hit") != expected_hit:
            raise AssertionError(f"{label}: expected cache hit={expected_hit}, got {kernel.cache}; "
                                 "use a fresh --cache-dir when repeating the validation")
        if not expected_hit and kernel.cache["status"] != "miss":
            raise AssertionError(f"{label}: expected a cold miss, got {kernel.cache}")
        if not execution["device_execution_performed"] or execution["original_triton_binary_used"]:
            raise AssertionError(f"{label}: expected execution of the newly generated cubin")
        (arguments.output_dir / label / "execution.json").write_text(
            json.dumps({**execution, "output": json_values(actual)}, indent=2, allow_nan=False) + "\n")
        record = {"case": label, "N": size, "shape": list(shape), "dtype": dtype,
                  "bias": bias, "factor": factor, "offset": offset, "tail_offset": tail_offset,
                  "external_offset": external_offset, "enable_fp_fusion": enable_fp_fusion,
                  "pre_factor": pre_factor, "post_factor": post_factor,
                  "logical_block": block, "num_warps": num_warps,
                  "launch_block": execution["block"],
                  "cache": kernel.cache, "llvm_sha256": kernel.codegen["llvm_sha256"],
                  "cubin_sha256": kernel.codegen["cubin_sha256"],
                  "device": execution["device"],
                  "verified_values": len(values), "output_sample": json_values(actual[:12])}
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        return kernel

    for combination in plan.eligible:
        dtype, options = combination.dtype, combination.options
        label = dtype + "-" + combination.graph.name
        first = check(label + "-cold", dtype=dtype, **options)
        second = check(label + "-warm", dtype=dtype, **options, expected_hit=True)
        if first.codegen["cubin_sha256"] != second.codegen["cubin_sha256"]:
            raise AssertionError(label + ": cache hit changed the compiled cubin")
    if len(records) != plan.gpu_executions:
        raise AssertionError(f"planned {plan.gpu_executions} GPU executions but completed {len(records)}")
    counters = {name: sum(record["cache"]["counters"][name] for record in records)
                for name in ("requests", "hits", "misses", "invalid", "builds")}
    if counters != plan.cache_counters:
        raise AssertionError(f"unexpected cache counters: {counters}")
    comparisons = {dtype: sum(record["verified_values"] for record in records if record["dtype"] == dtype)
                   for dtype in dtypes}
    summary = {"graph_family": "typed-pointwise-v1", "dtypes": dtypes,
               "selected_cases": [case.name for case in plan.cases], "gpu_executions": len(records),
               "planned_gpu_executions": plan.gpu_executions,
               "eligible_combinations": len(plan.eligible),
               "skipped_combinations": plan.skipped, "skipped_combination_count": len(plan.skipped),
               "comparisons_by_dtype": comparisons, "cache_counters": counters, "cases": records}
    (arguments.output_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"cache_counters": counters, "comparisons_by_dtype": comparisons}, sort_keys=True))
    print(f"Verified {len(records)} GPU executions; results: {arguments.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
