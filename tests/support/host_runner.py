"""Shared test helpers for invoking ask-opt, linking host objects, and probing GPU tools."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from ask.cuda import CUDAError
from ask.numerics import decode_float, encode_float

STORAGE = {
    "bool": ctypes.c_uint8,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "f32": ctypes.c_float,
    "f16": ctypes.c_uint16,
    "bf16": ctypes.c_uint16,
}


def ask_opt_path() -> str:
    env = os.environ.get("ASK_OPT")
    if env:
        return str(Path(env).resolve())
    for preset in ("all", "triton", "core"):
        candidate = ROOT / "build" / preset / "bin" / "ask-opt"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("ask-opt")
    if found:
        return found
    raise unittest.SkipTest("ASK_OPT is not set and no build/<preset>/bin/ask-opt was found")


def boundary_inline_test_path() -> str:
    env = os.environ.get("ASK_BOUNDARY_INLINE_BIN")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    driver = Path(ask_opt_path())
    build_root = driver.parent.parent
    candidate = build_root / "ask-boundary-inline-test"
    if candidate.is_file():
        return str(candidate)
    for preset in ("all", "triton", "core"):
        fallback = ROOT / "build" / preset / "ask-boundary-inline-test"
        if fallback.is_file():
            return str(fallback)
    raise unittest.SkipTest("ask-boundary-inline-test binary not found")


def llvm_tools_dir() -> Path:
    env = os.environ.get("ASK_LLVM_TOOLS_DIR")
    if env:
        return Path(env).resolve()
    lock = json.loads((ROOT / "cmake" / "llvm.lock.json").read_text(encoding="utf-8"))
    return (ROOT / ".deps" / lock["directory"] / "bin").resolve()


def llc_path() -> str:
    env = os.environ.get("ASK_LLC")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    candidate = llvm_tools_dir() / "llc"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("llc")
    if found:
        return found
    raise unittest.SkipTest("pinned llc binary not found")


def ptxas_path() -> str:
    env = os.environ.get("ASK_PTXAS")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    found = shutil.which("ptxas")
    if found:
        return found
    for cuda_bin in sorted(Path("/usr/local").glob("cuda*/bin/ptxas"), reverse=True):
        if cuda_bin.is_file():
            return str(cuda_bin)
    triton_py = os.environ.get("ASK_TEST_TRITON_PYTHON") or os.environ.get("ASK_TRITON_PYTHON")
    if triton_py and Path(triton_py).is_file():
        probe = subprocess.run(
            [triton_py, "-c",
             "import pathlib, triton; p = pathlib.Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas'; "
             "print(p if p.is_file() else '')"],
            capture_output=True, text=True, timeout=15,
        )
        candidate = probe.stdout.strip()
        if candidate:
            return candidate
    if os.environ.get("ASK_REQUIRE_CUDA_DEVICE"):
        raise RuntimeError("ptxas is required when ASK_REQUIRE_CUDA_DEVICE is set")
    raise unittest.SkipTest("ptxas not found")


def run_ask_opt(*args, source: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ask_opt_path(), *map(str, args)],
        input=source,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def require_cc() -> str:
    cc = shutil.which("cc")
    if not cc:
        raise unittest.SkipTest("C compiler 'cc' not available")
    return cc


def link_object_to_cdll(obj_path: Path, so_path: Path) -> ctypes.CDLL:
    cc = require_cc()
    linked = subprocess.run([cc, "-shared", str(obj_path), "-o", str(so_path)],
                            capture_output=True, text=True, timeout=60)
    if linked.returncode != 0:
        raise AssertionError(f"cc -shared failed:\n{linked.stderr}")
    return ctypes.CDLL(str(so_path))


def compile_llvm_to_cdll(llvm_text_or_path: str | Path, work_dir: Path, stem: str = "module") -> ctypes.CDLL:
    llc = llc_path()
    if isinstance(llvm_text_or_path, Path):
        ll_path = llvm_text_or_path
    else:
        ll_path = work_dir / f"{stem}.ll"
        ll_path.write_text(llvm_text_or_path, encoding="utf-8")
    obj_path = work_dir / f"{stem}.o"
    so_path = work_dir / f"{stem}.so"
    compiled = subprocess.run(
        [llc, "-filetype=obj", "-relocation-model=pic", str(ll_path), "-o", str(obj_path)],
        capture_output=True, text=True, timeout=60,
    )
    if compiled.returncode != 0:
        raise AssertionError(f"llc failed:\n{compiled.stderr}")
    return link_object_to_cdll(obj_path, so_path)


def encode_scalar(value, dtype: str):
    if dtype in ("f16", "bf16"):
        return int.from_bytes(encode_float(value, dtype), "little")
    return value


def decode_scalar(value, dtype: str):
    if dtype in ("f16", "bf16"):
        return decode_float(int(value).to_bytes(2, "little"), dtype)
    return bool(value) if dtype == "bool" else value


def make_buffer(values, dtype: str):
    ctype = STORAGE[dtype]
    count = max(len(values), 1)
    encoded = [encode_scalar(v, dtype) for v in values]
    return (ctype * count)(*encoded)


def read_buffer(buf, count: int, dtype: str) -> list:
    raw = list(buf)[:count]
    if dtype in ("f16", "bf16"):
        return [decode_float(int(v).to_bytes(2, "little"), dtype) for v in raw]
    if dtype == "bool":
        return [bool(v) for v in raw]
    return raw


def open_cuda_session():
    from ask import ExecutionSession
    try:
        return ExecutionSession()
    except (CUDAError, OSError, RuntimeError) as exc:
        if os.environ.get("ASK_REQUIRE_CUDA_DEVICE"):
            raise
        raise unittest.SkipTest(f"CUDA device/driver unavailable: {exc}") from exc
