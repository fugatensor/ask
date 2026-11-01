"""CUDA Array Interface ownership, ExecutionSession lifecycle, IRCache, and executable cache tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import ask
from ask.cache import IRCache
from ask.cuda import CUDAError, compile_cuda
from ask.runtime import ExecutionSession, _cuda_array_description


def sample_cache_key() -> dict:
    return {
        "schema_version": 1, "pipeline_version": "gpu-llvm-v1",
        "graph": {"source": "P -> F -> Q"}, "dsl": "def F(t): return t * 2",
        "config": {"BLOCK": 128}, "llvm_revision": "llvm-commit",
        "producer": {"triton": "3.7.1"}, "adapter": {"revision": "adapter-commit"},
        "tools": {"ptxas": "tool-sha256"}, "target": {"arch": "sm_80"},
        "shape": [10], "numerics": {"overflow": "wrap"},
    }


def sample_artifact() -> dict:
    return {
        "format": "llvm-ir", "entry": "scale2", "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80},
        "port_map": [{"argument": i, "logical_port": b, "element_type": "i32",
                      "shape": [10], "layout": "contiguous_row_major"}
                     for i, b in enumerate(("input:0", "output:0"))],
        "invocation": {"kind": "cuda-kernel", "version": 2, "grid": [1, 1, 1], "block": [128, 1, 1],
                       "shared_bytes": 0,
                       "arguments": [{"argument": i, "binding": b, "llvm_type": "ptr addrspace(1)"}
                                     for i, b in enumerate(("input:0", "output:0"))]},
        "codegen": {"fp_contract": "fast", "ptx_version": 80},
        "required_element_types": ["i32"],
    }


def _worker_process(root: str, start, output) -> None:
    try:
        if not start.wait(10):
            raise RuntimeError("start timeout")
        def build():
            with (Path(root) / "builders.log").open("ab", buffering=0) as s:
                s.write(b"builder\n")
            time.sleep(0.05)
            return {"module.ll": b"define void @kernel() { ret void }"}
        artifacts, stats = IRCache(root).get_or_create(sample_cache_key(), build)
        output.put({"artifacts": artifacts, "stats": stats})
    except BaseException as exc:
        output.put({"error": repr(exc)})


def _crashing_builder(root: str, entered) -> None:
    def build():
        entered.set()
        os._exit(7)
    IRCache(root).get_or_create(sample_cache_key(), build)


class ArrayOwner:
    def __init__(self, **changes):
        self.description = {
            "version": 3, "shape": (2, 5), "typestr": "<f4",
            "data": (4096, False), "strides": None, "stream": 7,
        }
        self.description.update(changes)
        self.reads = 0

    @property
    def __cuda_array_interface__(self):
        self.reads += 1
        return copy.deepcopy(self.description)


class FakeEvent:
    def __init__(self):
        self.complete = False

    def query(self):
        return self.complete

    def wait(self):
        self.complete = True


class FakeNativeSession:
    def __init__(self):
        self.closed = False
        self.stream = 99
        self.device = 0
        self.capability = (8, 6)
        self.stats = {}
        self.log = []
        self.events = []
        self.next_pointer = 8192

    def check_types(self, types):
        self.log.append(("check_types", types))

    def allocate(self, nbytes):
        ptr = self.next_pointer
        self.next_pointer += max(nbytes, 64)
        return SimpleNamespace(data_ptr=ptr, nbytes=nbytes, readonly=False)

    def import_buffer(self, pointer, nbytes, readonly):
        self.log.append(("import", pointer, nbytes, readonly))
        return SimpleNamespace(data_ptr=pointer, nbytes=nbytes, readonly=readonly)

    def wait_for_external_stream(self, stream):
        self.log.append(("wait", stream))

    def submit(self, kernel, buffers, grid, block, shared_bytes):
        self.log.append(("submit", grid, block))
        ev = FakeEvent()
        self.events.append(ev)
        return ev

    def reap(self):
        pass

    def synchronize(self):
        for ev in self.events:
            ev.complete = True

    def close(self):
        self.synchronize()
        self.closed = True


class RuntimeOwnershipTests(unittest.TestCase):
    def test_cuda_array_interface_v2_v3_and_invalid_descriptors(self):
        for stream in (None, 1, 2, 7340032):
            self.assertEqual(_cuda_array_description(ArrayOwner(strides=(20, 4), stream=stream)),
                             ((2, 5), "f32", 4096, False, stream))
        v2 = ArrayOwner(version=2)
        del v2.description["stream"]
        with self.assertRaisesRegex(CUDAError, "version 2.*producer_stream"):
            _cuda_array_description(v2)
        self.assertEqual(_cuda_array_description(v2, producer_stream=11)[4], 11)

        for invalid in (
            {"version": 1}, {"typestr": "<u2"}, {"typestr": ">f4"},
            {"shape": (2, -1)}, {"data": (0, False)}, {"strides": (4, 8)},
            {"mask": object()}, {"stream": 0},
        ):
            with self.assertRaises((CUDAError, ValueError)):
                _cuda_array_description(ArrayOwner(**invalid))

    def test_session_stream_refresh_owner_retention_and_cross_session_guards(self):
        native = FakeNativeSession()
        ext = SimpleNamespace(CUDASession=SimpleNamespace(create=lambda device: native))
        with patch.object(ask, "_native", ext, create=True):
            with ExecutionSession() as session:
                empty = session.tensor((0, 5), "f32")
                self.assertEqual(empty.__cuda_array_interface__["data"], (0, False))
                self.assertEqual(session.from_cuda_array(empty).data_ptr, 0)

                owner = ArrayOwner()
                weak_owner = weakref.ref(owner)
                borrowed = session.from_cuda_array(owner)
                owner.description["stream"] = 19
                session.submit(object(), [borrowed], grid=(1, 1, 1), block=(128, 1, 1))
                self.assertIn(("wait", 19), native.log)

                owner.description["shape"] = (10,)
                with self.assertRaisesRegex(CUDAError, "changed after import"):
                    session.submit(object(), [borrowed], grid=(1, 1, 1), block=(128, 1, 1))
                owner.description["shape"] = (2, 5)

                del owner, borrowed
                gc.collect()
                self.assertIsNotNone(weak_owner())
                native.events[0].complete = True
                session.reap()
                gc.collect()
                self.assertIsNone(weak_owner())

                owned = session.tensor((2, 5))
                with ExecutionSession() as other:
                    with self.assertRaisesRegex(CUDAError, "different execution session"):
                        other.submit(object(), [owned], grid=(1, 1, 1), block=(128, 1, 1))


class IRCacheAndExecutableCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="ask-cache-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_ir_cache_persistence_canonical_keys_corruption_and_concurrency(self):
        cache_dir = self.root / "ir"
        cache = IRCache(cache_dir)
        key = sample_cache_key()
        calls = 0

        def builder():
            nonlocal calls
            calls += 1
            return {"llvm/module.ll": b"IR payload", "launch.json": b'{"grid":[1,1,1]}'}

        artifacts, miss = cache.get_or_create(key, builder)
        reordered = dict(reversed(list(key.items())), shape=(10,))
        loaded, hit = IRCache(cache_dir).get_or_create(reordered, builder)
        self.assertEqual((miss["status"], hit["status"], calls), ("miss", "hit", 1))
        self.assertEqual(loaded, artifacts)

        # Corrupt payload triggers rebuild and invalid counter.
        entry = cache_dir / "entries" / miss["key"]
        (entry / "artifacts/llvm/module.ll").write_bytes(b"corrupted")
        _, repaired = cache.get_or_create(key, builder)
        self.assertEqual((repaired["status"], calls), ("invalid", 2))

        # Tampered manifest path is rejected.
        manifest_path = entry / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["../escape.ll"] = manifest["artifacts"].pop("llvm/module.ll")
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest_path.write_bytes(raw)
        (entry / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
        self.assertEqual(cache.get_or_create(key, builder)[1]["status"], "invalid")

        for unsafe in ("../escape.ll", "/abs.ll", "a\\b"):
            with self.assertRaises(ValueError):
                cache.get_or_create({**key, "shape": [99]}, lambda: {unsafe: b"x"})

        with self.assertRaisesRegex(RuntimeError, "recursively"):
            cache.get_or_create({**key, "shape": [77]},
                                lambda: cache.get_or_create({**key, "shape": [77]}, builder)[0])

        # Multi-process single-builder lock & crash recovery.
        proc_dir = self.root / "mp"
        ctx = multiprocessing.get_context("spawn")
        start, queue = ctx.Event(), ctx.Queue()
        procs = [ctx.Process(target=_worker_process, args=(str(proc_dir), start, queue)) for _ in range(3)]
        for p in procs:
            p.start()
        start.set()
        results = [queue.get(timeout=20) for _ in procs]
        for p in procs:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(sorted(r["stats"]["status"] for r in results), ["hit", "hit", "miss"])
        self.assertEqual((proc_dir / "builders.log").read_bytes(), b"builder\n")
        queue.close()

        crash_dir = self.root / "crash"
        entered = ctx.Event()
        dying = ctx.Process(target=_crashing_builder, args=(str(crash_dir), entered))
        dying.start()
        self.assertTrue(entered.wait(10))
        dying.join(timeout=10)
        self.assertEqual(IRCache(crash_dir).get_or_create(key, builder)[1]["status"], "miss")

    def test_executable_cache_hits_and_tool_fingerprint_invalidation(self):
        llc, ptxas = self.root / "llc", self.root / "ptxas"
        llc.write_bytes(b"llc v1")
        ptxas.write_bytes(b"ptxas v1")
        tool_calls = []

        def fake_tool(cmd, **kwargs):
            tool_calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"PTX" if cmd[0] == str(llc) else b"CUBIN")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def run_compile(name: str):
            val = sample_artifact()
            ll = self.root / f"{name}.ll"
            ll.write_text(val["payload"], encoding="utf-8")
            return compile_cuda(ll, val, self.root / name, llc=llc, ptxas=ptxas, cache_dir=self.root / "exec_cache")

        with patch("ask.cuda.subprocess.run", side_effect=fake_tool):
            cold = run_compile("cold")
            warm = run_compile("warm")
            self.assertEqual((cold["executable_cache"]["status"], warm["executable_cache"]["status"]), ("miss", "hit"))
            self.assertEqual(len(tool_calls), 2)
            ptxas.write_bytes(b"ptxas v2")
            updated = run_compile("updated")
            self.assertEqual(updated["executable_cache"]["status"], "miss")


if __name__ == "__main__":
    unittest.main()
