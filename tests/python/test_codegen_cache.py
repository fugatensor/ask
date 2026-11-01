"""Executable cache integration: compilation counts and reusable local receipts."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ask.cuda import CUDAError, compile_cuda
from test_cuda import artifact


class ExecutableCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ask-codegen-cache-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.llc, self.ptxas = self.root / "llc", self.root / "ptxas"
        self.llc.write_bytes(b"synthetic llc version 1")
        self.ptxas.write_bytes(b"synthetic ptxas version 1")
        self.calls = []

    def tool(self, command, **kwargs):
        self.calls.append(command)
        Path(command[-1]).write_bytes(b"synthetic PTX" if command[0] == str(self.llc) else b"synthetic cubin")
        return SimpleNamespace(returncode=0, stdout="tool log\n", stderr="")

    def compile(self, name, value=None):
        value = artifact() if value is None else value
        source = self.root / (name + ".ll")
        source.write_text(value["payload"])
        return compile_cuda(source, value, self.root / name, llc=self.llc, ptxas=self.ptxas,
                            cache_dir=self.cache)

    def test_hit_skips_both_tools_and_rebases_every_local_receipt_path(self):
        with patch("ask.cuda.subprocess.run", side_effect=self.tool):
            cold = self.compile("cold")
            warm = self.compile("warm")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(cold["executable_cache"]["status"], "miss")
        self.assertEqual(warm["executable_cache"]["status"], "hit")
        self.assertEqual(warm["executable_cache"]["counters"],
                         {"requests": 1, "hits": 1, "misses": 0, "invalid": 0, "builds": 0})
        self.assertEqual((cold["compiler_invocations"], warm["compiler_invocations"]), (2, 0))
        self.assertEqual(cold["cubin_sha256"], warm["cubin_sha256"])
        self.assertEqual(warm["llvm_ir"], str(self.root / "warm.ll"))
        self.assertEqual(warm["commands"][0][-1], warm["ptx"])
        self.assertEqual(warm["commands"][0][-3], warm["llvm_ir"])
        self.assertEqual(warm["commands"][1][-3], warm["ptx"])
        self.assertEqual(warm["commands"][1][-1], warm["cubin"])
        for field in ("ptx", "cubin"):
            self.assertEqual(Path(warm[field]).parent, self.root / "warm")
            self.assertEqual(Path(cold[field]).read_bytes(), Path(warm[field]).read_bytes())
        self.assertEqual(json.loads((self.root / "warm/codegen.json").read_text()), warm)

    def test_key_covers_ir_target_abi_numerics_tool_bytes_and_pipeline_version(self):
        with patch("ask.cuda.subprocess.run", side_effect=self.tool):
            base = self.compile("base")
            changes = [
                lambda value: value.update(payload=value["payload"] + "; new LLVM bytes\n"),
                lambda value: value["target"].update(arch=86),
                lambda value: value["invocation"].update(block=[256, 1, 1], num_warps=8),
                lambda value: value["invocation"]["compiler_metadata"].update(enable_fp_fusion=False),
                lambda value: value["specializations"]["constants"].update(N=11),
                lambda value: value.update(producer={"version": "new producer receipt"}),
            ]
            keys = {base["executable_cache"]["key"]}
            for index, change in enumerate(changes):
                value = copy.deepcopy(artifact())
                change(value)
                result = self.compile(f"change-{index}", value)
                self.assertEqual(result["executable_cache"]["status"], "miss")
                keys.add(result["executable_cache"]["key"])
            self.assertEqual(len(keys), 1 + len(changes))
            for tool in (self.llc, self.ptxas):
                original = tool.read_bytes()
                tool.write_bytes(original + b" updated")
                self.assertEqual(self.compile("changed-" + tool.name)["executable_cache"]["status"], "miss")
                tool.write_bytes(original)
            with patch("ask.cuda._CODEGEN_CACHE_VERSION", "test-version-2"):
                self.assertEqual(self.compile("new-version")["executable_cache"]["status"], "miss")
            self.assertEqual(self.compile("original-again")["executable_cache"]["status"], "hit")
        self.assertEqual(len(self.calls), (1 + len(changes) + 3) * 2)

    def test_corrupt_cubin_rebuilds_and_reports_invalid_miss(self):
        with patch("ask.cuda.subprocess.run", side_effect=self.tool):
            cold = self.compile("cold")
            entry = self.cache / "entries" / cold["executable_cache"]["key"]
            (entry / "artifacts/kernel.cubin").write_bytes(b"broken")
            repaired = self.compile("repaired")
            warm = self.compile("warm")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(repaired["executable_cache"]["status"], "invalid")
        self.assertEqual(repaired["executable_cache"]["counters"]["invalid"], 1)
        self.assertEqual(repaired["executable_cache"]["counters"]["misses"], 1)
        self.assertEqual(warm["executable_cache"]["status"], "hit")
        self.assertEqual(Path(repaired["cubin"]).read_bytes(), b"synthetic cubin")

    def test_concurrent_requests_compile_once_and_materialize_separate_outputs(self):
        barrier = threading.Barrier(4)
        def slow_tool(command, **kwargs):
            time.sleep(0.03)
            return self.tool(command, **kwargs)
        def request(index):
            barrier.wait(timeout=5)
            return self.compile(f"parallel-{index}")
        with patch("ask.cuda.subprocess.run", side_effect=slow_tool), ThreadPoolExecutor(4) as pool:
            results = list(pool.map(request, range(4)))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(sorted(result["executable_cache"]["status"] for result in results),
                         ["hit", "hit", "hit", "miss"])
        self.assertEqual(len({result["cubin"] for result in results}), 4)
        self.assertTrue(all(Path(result["cubin"]).exists() for result in results))

    def test_failed_ptxas_never_publishes_or_leaves_partial_outputs(self):
        def failing(command, **kwargs):
            if command[0] == str(self.ptxas):
                return SimpleNamespace(returncode=1, stdout="", stderr="synthetic ptxas failure")
            return self.tool(command, **kwargs)
        with patch("ask.cuda.subprocess.run", side_effect=failing):
            with self.assertRaisesRegex(CUDAError, "synthetic ptxas failure"):
                self.compile("failed")
        self.assertEqual(list((self.root / "failed").iterdir()), [])
        self.assertEqual(list((self.cache / "entries").iterdir()), [])
        with patch("ask.cuda.subprocess.run", side_effect=self.tool):
            self.assertEqual(self.compile("retry")["executable_cache"]["status"], "miss")

    def test_tool_modified_during_build_does_not_publish_under_stale_fingerprint(self):
        def changing_tool(command, **kwargs):
            result = self.tool(command, **kwargs)
            if command[0] == str(self.ptxas):
                self.llc.write_bytes(b"replaced during build")
            return result
        with patch("ask.cuda.subprocess.run", side_effect=changing_tool):
            with self.assertRaisesRegex(CUDAError, "tool changed during compilation"):
                self.compile("changed")
        self.assertEqual(list((self.cache / "entries").iterdir()), [])
        self.assertEqual(list((self.root / "changed").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
