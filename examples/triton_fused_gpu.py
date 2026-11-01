"""Compile and run a fused ASK -> Triton -> ASK graph on CUDA.

Usage:
  python3 examples/triton_fused_gpu.py --emit-only
  PYTHONPATH=python:build/all/python python3 examples/triton_fused_gpu.py \
    --triton-python /path/to/triton-venv/bin/python
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import tempfile

from ask import Compiler, CompilerEnvironment, ExecutionSession, Module, TensorType

ROOT = Path(__file__).resolve().parents[1]

TRITON_AFFINE = """import triton
import triton.language as tl

@triton.jit
def affine(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(t + i, mask=i < N, other=0.0)
    tl.store(u + i, x * 2.0 + 0.5, mask=i < N)
"""


def build_module(shape: tuple[int, int] = (2, 8), arch: int = 80) -> Module:
    """Describe y = ((x + 1.0) * 2.0 + 0.5) + 3.0 over a 2x8 f32 tensor."""
    count = shape[0] * shape[1]
    module = Module()
    tensor = TensorType(shape, "f32")
    dynamic = TensorType((None, None), "f32")
    semantics = module.semantics("affine_f32", inputs=[dynamic], results=[dynamic])
    impl = module.implementation(
        "triton_affine", compiler="triton", source=TRITON_AFFINE, entry="affine",
        signature={"t": "*fp32", "u": "*fp32"},
        source_bindings={"t": "input:0", "u": "output:0"},
        constants={"N": count, "BLOCK": 128}, options={"num_warps": 4},
        target={"backend": "cuda", "arch": arch, "warp_size": 32}, grid=(1, 1, 1),
    )
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
    p = graph.add(graph.inputs["x"], graph.splat(graph.constant(1.0, dtype="f32"), shape), id="P")
    u = graph.external(p, semantics=semantics, implementation=impl, id="F")
    y = graph.add(u, graph.splat(graph.constant(3.0, dtype="f32"), shape), id="Q")
    graph.return_(y)
    return module


def _default_llc() -> str:
    candidates = sorted((ROOT / ".deps").glob("llvm-*/bin/llc"))
    return str(candidates[0]) if candidates else (shutil.which("llc") or "llc")


def _default_ptxas() -> str:
    found = shutil.which("ptxas")
    if found:
        return found
    for candidate in sorted(Path("/usr/local").glob("cuda*/bin/ptxas"), reverse=True):
        if os.access(candidate, os.X_OK):
            return str(candidate)
    for dist in importlib.metadata.distributions():
        for entry in dist.files or ():
            if entry.name == "ptxas":
                located = Path(dist.locate_file(entry))
                if located.is_file() and os.access(located, os.X_OK):
                    return str(located)
    return "ptxas"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-only", action="store_true", help="print ASK IR without compiling")
    parser.add_argument("--ask-opt", default=os.environ.get("ASK_OPT"))
    parser.add_argument("--triton-python", default=os.environ.get("ASK_TRITON_PYTHON", sys.executable))
    parser.add_argument("--triton-worker", default=str(ROOT / "adapters/triton/worker.py"))
    parser.add_argument("--llc", default=os.environ.get("ASK_LLC", _default_llc()))
    parser.add_argument("--ptxas", default=os.environ.get("ASK_PTXAS", _default_ptxas()))
    args = parser.parse_args()

    if args.emit_only:
        print(build_module().to_mlir(), end="")
        return

    with ExecutionSession() as session, tempfile.TemporaryDirectory(prefix="ask-example-") as work:
        major, minor = session.capability
        module = build_module(arch=major * 10 + minor)
        compiler = Compiler(args.ask_opt) if args.ask_opt else Compiler()
        kernel = compiler.compile_cuda(
            module,
            cache_dir=Path(work) / "cache",
            output_dir=Path(work) / "out",
            compilers={"triton": CompilerEnvironment(python=args.triton_python, worker=args.triton_worker)},
            llc=args.llc,
            ptxas=args.ptxas,
        )
        executable = kernel.prepare(session)
        values = [float(i) * 0.5 for i in range(16)]
        x = session.from_host(values, shape=(2, 8), dtype="f32")
        y = session.tensor((2, 8), "f32")
        executable.submit(x, y).wait()
        result = y.to_host()
        expected = [((v + 1.0) * 2.0 + 0.5) + 3.0 for v in values]
        assert result == expected, (result, expected)
        print(f"Fused kernel executed on sm_{major * 10 + minor}: output[:4] = {result[:4]}")


if __name__ == "__main__":
    main()
