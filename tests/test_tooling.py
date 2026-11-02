"""Build metadata, dependency fetching and CI coverage verification."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ask_fetch_deps", ROOT / "tools/fetch_deps.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def make_llvm_archive(parent: Path, record: dict) -> Path:
    record["url"] = "https://example.invalid/test-sdk.tar.xz"
    archive = parent / "test-sdk.tar.xz"
    with tarfile.open(archive, "w:xz") as stream:
        for name in fetcher.LLVM_REQUIRED_FILES:
            data = (f"set(LLVM_PACKAGE_VERSION {record['version']})\n"
                    if name.endswith("LLVMConfig.cmake") else "fixture").encode()
            item = tarfile.TarInfo("test-sdk/" + name)
            item.size = len(data)
            stream.addfile(item, io.BytesIO(data))
    record["size_bytes"] = archive.stat().st_size
    record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive


class BuildMetadataTests(unittest.TestCase):
    def test_cmake_reads_native_version_and_rejects_invalid_release(self):
        version = (ROOT / "VERSION").read_text().strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cmake").mkdir()
            (root / "cmake/AskVersion.cmake").write_text((ROOT / "cmake/AskVersion.cmake").read_text())
            (root / "VERSION").write_text(version + "\n")
            script = root / "version.cmake"
            script.write_text(
                "cmake_minimum_required(VERSION 3.28)\n"
                f'include("{root}/cmake/AskVersion.cmake")\n'
                'message(STATUS "ASK_VERSION=${ASK_VERSION}")\n', encoding="utf-8")
            accepted = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("ASK_VERSION=" + version, accepted.stdout)
            (root / "VERSION").write_text("not-a-version\n")
            rejected = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("major.minor.patch", rejected.stderr)



class FetchDepsTests(unittest.TestCase):
    def llvm_lock(self) -> dict:
        value = fetcher.load_llvm_lock(ROOT / "cmake/llvm.lock.json")
        value["directory"] = "test-sdk"
        return value

    def test_llvm_fetch_receipt_and_mismatch_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), self.llvm_lock()
            make_llvm_archive(parent, record)
            with patch.object(fetcher.urllib.request, "urlopen", side_effect=AssertionError("network")):
                target = fetcher.fetch_llvm(record, parent)
                self.assertTrue((target / fetcher.LLVM_RECEIPT).exists())
                self.assertEqual(fetcher.fetch_llvm(record, parent), target)

            # Tampered receipt refuses overwrite.
            receipt = json.loads((target / fetcher.LLVM_RECEIPT).read_text(encoding="utf-8"))
            receipt["version"] = "bad"
            (target / fetcher.LLVM_RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                fetcher.fetch_llvm(record, parent)

        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), self.llvm_lock()
            make_llvm_archive(parent, record)
            target = fetcher.fetch_llvm(record, parent)
            (target / "lib/libMLIRIR.a").unlink()
            with self.assertRaisesRegex(ValueError, "SDK is incomplete"):
                fetcher.fetch_llvm(record, parent)

    def test_llvm_archive_digest_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = Path(directory), self.llvm_lock()
            archive = make_llvm_archive(root, lock)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetcher.fetch_llvm(dict(lock, sha256="0" * 64), root)
            self.assertTrue(archive.is_file())
            self.assertFalse((root / lock["directory"]).exists())

    def test_tar_filter_blocks_traversal_symlink_escape_and_special_files(self):
        record = self.llvm_lock()
        with tempfile.TemporaryDirectory() as directory:
            for bad_name in ("../escape", "/absolute", "test-sdk/../escape"):
                item = tarfile.TarInfo(bad_name)
                item.size = 1
                with self.assertRaises(ValueError):
                    fetcher.archive_filter(record)(item, directory)

            sym = tarfile.TarInfo("test-sdk/link")
            sym.type, sym.linkname = tarfile.SYMTYPE, "../../escape"
            with self.assertRaises(ValueError):
                fetcher.archive_filter(record)(sym, directory)

            fifo = tarfile.TarInfo("test-sdk/fifo")
            fifo.type = tarfile.FIFOTYPE
            with self.assertRaises(ValueError):
                fetcher.archive_filter(record)(fifo, directory)


class CITestReportTests(unittest.TestCase):
    def test_ci_rejects_skipped_missing_and_empty_suites(self):
        spec = importlib.util.spec_from_file_location("ask_ci_runner", ROOT / "tools/run_ci_tests.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.xml"
            valid = '<testsuite><testcase name="ask-compiler" status="run"><system-out>native compiler passed</system-out></testcase></testsuite>'
            report.write_text(valid, encoding="utf-8")
            runner.check_report(report, {"ask-compiler"})
            invalid = [
                valid.replace('status="run"', 'status="notrun"'),
                valid.replace("<system-out>", "<failure/><system-out>"),
                valid.replace("<system-out>", "<skipped/><system-out>"),
                '<testsuite/>',
            ]
            for content in invalid:
                with self.subTest(content=content):
                    report.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runner.check_report(report, {"ask-compiler"})
            report.write_text(valid, encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.check_report(report, {"ask-compiler", "ask-runtime"})


if __name__ == "__main__":
    unittest.main()
