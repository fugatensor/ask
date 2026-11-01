#!/usr/bin/env python3
"""Validate an installed core wheel. Run with its Python interpreter and -I."""

from __future__ import annotations

import ctypes
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tempfile

import ask
from ask import Compiler, Module, TensorType


def main() -> None:
    if not sys.flags.isolated:
        raise RuntimeError("run this check with python -I to exclude the checkout and PYTHONPATH")
    package = Path(ask.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"ASK was not loaded from the installation environment: {package}")
    compiler = Compiler()
    caps = compiler.capabilities()
    if ask.__version__ != metadata.version("ask-compiler") or ask.__version__ != caps["ask_version"]:
        raise RuntimeError("Python, wheel metadata and native compiler versions differ")
    if caps["third_party_compilers"] or not caps["host_codegen"]:
        raise RuntimeError("the default wheel must include host codegen and no third-party compilers")
    cli = Compiler(compiler.executable)
    if cli.capabilities() != caps:
        raise RuntimeError("installed ask-opt and extension capabilities differ")
    if not Path(compiler.executable).resolve().is_relative_to(package):
        raise RuntimeError("the companion ask-opt must come from the installed wheel")

    tensor = TensorType((4,), "i32")
    module = Module()
    graph = module.graph("increment", inputs={"x": tensor}, results=[tensor])
    graph.return_(graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (4,))))
    if compiler.prepare(module) != cli.prepare(module):
        raise RuntimeError("installed native and CLI preparation results differ")
    with tempfile.TemporaryDirectory(prefix="ask-wheel-smoke-") as directory:
        root = Path(directory)
        source, obj, library = root / "input.mlir", root / "output.o", root / "output.so"
        source.write_text(module.to_mlir(), encoding="utf-8")
        subprocess.run([compiler.executable, str(source), "--emit-object", "-o", str(obj)], check=True)
        subprocess.run(["cc", "-shared", str(obj), "-o", str(library)], check=True)
        function = ctypes.CDLL(str(library)).increment
        function.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        function.restype = None
        values = (ctypes.c_int32 * 4)(-2, 0, 7, 2147483647)
        result = (ctypes.c_int32 * 4)()
        function(values, result)
        if list(result) != [-1, 1, 8, -2147483648]:
            raise RuntimeError(f"installed compiler produced incorrect host code: {list(result)}")
    print(f"Installed ask-compiler {ask.__version__}: native/CLI parity and CPU execution passed")


if __name__ == "__main__":
    main()
