"""Construct a graph with foldable constants, dead work, and shared outputs."""
import argparse
import json
from pathlib import Path

from ask import Compiler, Module, TensorType


def build_module():
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor, tensor])
    one = graph.constant(1, id="one")
    two = graph.add(one, one, id="two")
    dead = graph.mul(graph.splat(graph.constant(99), (10,)),
                     graph.splat(graph.constant(99), (10,)), id="unused")
    graph.mark(dead, mark_id="unused_trace")
    t = graph.add(graph.inputs["x"], graph.splat(one, (10,)), id="P", value_id="v_t")
    y = graph.mul(t, graph.splat(two, (10,)), id="Q", value_id="v_y")
    graph.return_(t, y)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask-opt", default="ask-opt")
    parser.add_argument("--output", type=Path, help="write optimized ASK IR to this file")
    parser.add_argument("--emit-only", action="store_true", help="print input ASK IR without optimizing")
    args = parser.parse_args()
    module = build_module()
    if args.emit_only:
        print(module.to_mlir(), end="")
        return
    result = Compiler(args.ask_opt).optimize(module)
    if args.output:
        args.output.write_text(result.ir, encoding="utf-8")
    else:
        print(result.ir, end="")
    print(json.dumps(result.report["optimization"], indent=2))


if __name__ == "__main__":
    main()
