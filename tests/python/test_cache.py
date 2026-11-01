from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from ask.cache import IRCache


def cache_key():
    return {
        "schema_version": 1, "pipeline_version": "gpu-llvm-v1",
        "graph": {"source": "P(x + 1) -> F(t * 2) -> Q(u + 3)"},
        "dsl": "def F(t): return t * 2", "config": {"BLOCK": 128},
        "llvm_revision": "llvm-commit", "producer": {"triton": "3.7.1"},
        "adapter": {"revision": "adapter-commit"}, "tools": {"ptxas": "tool-sha256"},
        "target": {"arch": "sm_80"}, "shape": [10], "numerics": {"overflow": "wrap"},
    }


def _process_request(root, start, output):
    try:
        if not start.wait(10):
            raise RuntimeError("concurrent test did not start")
        def build():
            with (Path(root) / "builders.log").open("ab", buffering=0) as stream:
                stream.write(b"builder\n")
            time.sleep(0.1)
            return {"module.ll": b"define void @kernel() { ret void }"}
        artifacts, stats = IRCache(root).get_or_create(cache_key(), build)
        output.put({"artifacts": artifacts, "stats": stats})
    except BaseException as error:
        output.put({"error": repr(error)})


def _die_in_builder(root, entered):
    def build():
        entered.set()
        os._exit(7)
    IRCache(root).get_or_create(cache_key(), build)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ask-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = IRCache(self.root)
        self.key = cache_key()
        self.calls = 0

    def build(self):
        self.calls += 1
        return {"llvm/module.ll": b"IR payload", "launch.json": b'{"grid":[1,1,1]}' }

    def entry(self, stats):
        return self.root / "entries" / stats["key"]

    def rewrite_manifest(self, entry, transform):
        path = entry / "manifest.json"
        manifest = json.loads(path.read_bytes())
        transform(manifest)
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        path.write_bytes(payload)
        (entry / "manifest.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")

    def test_hit_persists_across_cache_instances(self):
        artifacts, miss = self.cache.get_or_create(self.key, self.build)
        loaded, hit = IRCache(self.root).get_or_create(self.key, self.build)
        self.assertEqual(loaded, artifacts)
        self.assertEqual(self.calls, 1)
        self.assertEqual(miss["status"], "miss")
        self.assertEqual(hit["status"], "hit")
        self.assertEqual(hit["counters"], {"requests": 1, "hits": 1, "misses": 0, "invalid": 0, "builds": 0})
        self.assertIsNone(hit["reason"])

    def test_canonical_keys_ignore_object_order_and_tuple_spelling(self):
        _, first = self.cache.get_or_create(self.key, self.build)
        reordered = dict(reversed(list(self.key.items())))
        reordered["shape"] = (10,)
        _, second = self.cache.get_or_create(reordered, self.build)
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(second["status"], "hit")

    def test_every_input_and_fingerprint_affects_the_key(self):
        _, original = self.cache.get_or_create(self.key, self.build)
        changes = {
            "schema_version": 2, "pipeline_version": "gpu-llvm-v2",
            "graph": {"source": "different graph"}, "dsl": "different source",
            "config": {"BLOCK": 256}, "llvm_revision": "other-llvm",
            "producer": {"triton": "different"}, "adapter": {"revision": "other-adapter"},
            "tools": {"ptxas": "other-tool"}, "target": {"arch": "sm_90"},
            "shape": [11], "numerics": {"overflow": "checked"},
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                key = deepcopy(self.key)
                key[field] = value
                _, result = self.cache.get_or_create(key, self.build)
                self.assertEqual(result["status"], "miss")
                self.assertNotEqual(result["key"], original["key"])

    def test_corrupt_payload_is_invalid_then_rebuilt(self):
        artifacts, initial = self.cache.get_or_create(self.key, self.build)
        (self.entry(initial) / "artifacts/llvm/module.ll").write_bytes(b"bad bytes!")
        rebuilt, stats = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(rebuilt, artifacts)
        self.assertEqual(stats["status"], "invalid")
        self.assertEqual(stats["counters"], {"requests": 1, "hits": 0, "misses": 1, "invalid": 1, "builds": 1})
        self.assertIn("mismatch", stats["reason"])
        self.assertEqual(self.calls, 2)
        self.assertEqual(self.cache.get_or_create(self.key, self.build)[1]["status"], "hit")

    def test_corrupt_manifest_and_unlisted_files_are_never_hits(self):
        _, initial = self.cache.get_or_create(self.key, self.build)
        (self.entry(initial) / "manifest.json").write_bytes(b"{}")
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("manifest checksum", result["reason"])
        (self.entry(initial) / "artifacts/extra.ll").write_bytes(b"unlisted")
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("inventory", result["reason"])

    def test_manifest_keys_and_sizes_are_validated(self):
        _, initial = self.cache.get_or_create(self.key, self.build)
        entry = self.entry(initial)
        self.rewrite_manifest(entry, lambda manifest: manifest["key"].update({"shape": [999]}))
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("key mismatch", result["reason"])
        self.rewrite_manifest(entry, lambda manifest: manifest["artifacts"]["llvm/module.ll"].update({"size_bytes": 999}))
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("size mismatch", result["reason"])

    def test_builder_failure_is_not_cached_and_can_retry(self):
        failure = RuntimeError("compiler failed")
        def fail():
            raise failure
        with self.assertRaises(RuntimeError) as caught:
            self.cache.get_or_create(self.key, fail)
        self.assertIs(caught.exception, failure)
        self.assertEqual(list((self.root / "entries").iterdir()), [])
        self.assertEqual(self.cache.get_or_create(self.key, self.build)[1]["status"], "miss")

    def test_publication_failure_leaves_no_partial_entry(self):
        with patch("ask.cache.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.cache.get_or_create(self.key, self.build)
        self.assertEqual(list((self.root / "entries").iterdir()), [])
        self.assertEqual(list((self.root / "tmp").iterdir()), [])
        self.assertEqual(self.cache.get_or_create(self.key, self.build)[1]["status"], "miss")

    def test_unsafe_artifact_names_never_escape_the_cache(self):
        for name in ("../escape.ll", "/escape.ll", "a/../escape.ll", "a//b", "a\\b", "C:/escape.ll",
                     "", ".", ".. /escape.ll", "a.", "CON.ll", "a/NUL", "bad\nname"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.cache.get_or_create(self.key, lambda: {name: b"bad"})
        with self.assertRaises(ValueError):
            self.cache.get_or_create(self.key, lambda: {"a": b"one", "a/b": b"two"})
        self.assertEqual(list((self.root / "entries").iterdir()), [])

    def test_malicious_manifest_path_is_invalid(self):
        _, initial = self.cache.get_or_create(self.key, self.build)
        def tamper(manifest):
            manifest["artifacts"]["../outside.ll"] = manifest["artifacts"].pop("llvm/module.ll")
        self.rewrite_manifest(self.entry(initial), tamper)
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("unsafe artifact path", result["reason"])

    @unittest.skipUnless(os.name == "posix", "symlink test uses POSIX permissions")
    def test_payload_symlink_is_not_followed(self):
        _, initial = self.cache.get_or_create(self.key, self.build)
        payload = self.entry(initial) / "artifacts/llvm/module.ll"
        outside = self.root / "outside.ll"
        outside.write_bytes(b"IR payload")
        payload.unlink()
        payload.symlink_to(outside)
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(outside.read_bytes(), b"IR payload")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO corruption requires POSIX")
    def test_fifo_payload_is_rejected_without_blocking(self):
        _, initial = self.cache.get_or_create(self.key, self.build)
        payload = self.entry(initial) / "artifacts/llvm/module.ll"
        payload.unlink()
        os.mkfifo(payload)
        _, result = self.cache.get_or_create(self.key, self.build)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("regular cache file", result["reason"])

    def test_invalid_keys_and_payload_types_fail_before_publication(self):
        key = deepcopy(self.key)
        del key["tools"]
        with self.assertRaises(ValueError):
            self.cache.get_or_create(key, self.build)
        for invalid in (float("nan"), float("inf"), {1: "non-string key"}, object()):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                self.cache.get_or_create({**self.key, "extra": invalid}, self.build)
        self.assertEqual(self.calls, 0)
        with self.assertRaises(TypeError):
            self.cache.get_or_create(self.key, lambda: {"module.ll": "not bytes"})
        self.assertEqual(list((self.root / "entries").iterdir()), [])

    def test_threads_share_one_builder_across_cache_instances(self):
        start = threading.Barrier(4)
        def request():
            start.wait(timeout=10)
            return IRCache(self.root).get_or_create(self.key, self.build)[1]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: request(), range(4)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(sorted(result["status"] for result in results), ["hit", "hit", "hit", "miss"])

    def test_processes_publish_one_entry_and_reuse_it(self):
        context = multiprocessing.get_context("spawn")
        start, output = context.Event(), context.Queue()
        processes = [context.Process(target=_process_request, args=(str(self.root), start, output)) for _ in range(4)]
        try:
            for process in processes:
                process.start()
            start.set()
            results = [output.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            for result in results:
                self.assertNotIn("error", result)
            self.assertEqual(sorted(result["stats"]["status"] for result in results), ["hit", "hit", "hit", "miss"])
            self.assertEqual((self.root / "builders.log").read_bytes(), b"builder\n")
            self.assertTrue(all(result["artifacts"] == results[0]["artifacts"] for result in results))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
            output.close()

    def test_process_death_releases_lock_without_a_cache_hit(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(target=_die_in_builder, args=(str(self.root), entered))
        process.start()
        try:
            self.assertTrue(entered.wait(10))
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 7)
            self.assertEqual(self.cache.get_or_create(self.key, self.build)[1]["status"], "miss")
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)

    def test_recursive_same_key_fails_instead_of_deadlocking(self):
        with self.assertRaisesRegex(RuntimeError, "recursively"):
            self.cache.get_or_create(self.key, lambda: self.cache.get_or_create(self.key, self.build)[0])
        self.assertEqual(list((self.root / "entries").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
