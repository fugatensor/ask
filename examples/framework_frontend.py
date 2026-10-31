"""A framework constructs P -> external Triton F -> Q through the ASK SDK.

Run from an installed package, or use PYTHONPATH=python from the repository:
  python3 examples/framework_frontend.py --emit-only
  python3 examples/framework_frontend.py --ask-opt build/bin/ask-opt

Preparing this graph verifies the declaration of F; it does not execute its DSL.
"""

from __future__ import annotations

import argparse
import json
import os

from ask import Compiler, Module, TensorType


TRITON_SCALE2 = '''import triton
import triton.language as tl

@triton.jit
def scale2(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(t + i, i < N, other=0)
    tl.store(u + i, value * 2, i < N)
'''


def build_module() -> Module:
    """Describe y = (x + 1) * 2 + 3 with i32 wrapping arithmetic."""
    module = Module()
    tensor = TensorType((10,), "i32")
    dynamic_tensor = TensorType((None,), "i32")
    scale2 = module.semantics(
        "scale2_i32_v1", inputs=[dynamic_tensor], results=[dynamic_tensor],
        shape_rule="same_as",
    )
    triton_scale2 = module.implementation(
        "triton_scale2", compiler="triton", source=TRITON_SCALE2, entry="scale2",
        signature={"t": "*i32", "u": "*i32"},
        source_bindings={"t": "input:0", "u": "output:0"},
        constants={"N": 10, "BLOCK": 128}, options={"num_warps": 4},
        target={"backend": "cuda", "arch": 80, "warp_size": 32}, grid=(1, 1, 1),
    )
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
    one = graph.splat(graph.constant(1, id="one"), (10,), id="ones")
    t = graph.add(graph.inputs["x"], one, id="P", value_id="v_t")
    marked_t = graph.mark(t, mark_id="before_F", role="input", port_id="F.in")
    u = graph.external(marked_t, semantics=scale2, implementation=triton_scale2,
                       id="F", value_id="v_u")
    marked_u = graph.mark(u, mark_id="after_F", role="output", port_id="F.out")
    three = graph.splat(graph.constant(3, id="three"), (10,), id="threes")
    y = graph.add(marked_u, three, id="Q", value_id="v_y")
    graph.return_(y)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask-opt", default=os.environ.get("ASK_OPT", "ask-opt"))
    parser.add_argument("--emit-only", action="store_true", help="print input MLIR without preparing it")
    arguments = parser.parse_args()
    module = build_module()
    if arguments.emit_only:
        print(module.to_mlir(), end="")
        return
    prepared = Compiler(arguments.ask_opt).prepare(module)
    print("Prepared ASK IR (external DSL has not been compiled or executed):")
    print(prepared.ir, end="")
    print("Preparation report:")
    print(json.dumps(prepared.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
