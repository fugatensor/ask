"""CMake compiler selection allowlist and tools/fetch_deps.py verification tests."""
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
    archive = parent / "test-sdk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in fetcher.LLVM_REQUIRED_FILES:
            data = (f"set(LLVM_PACKAGE_VERSION {record['version']})\n"
                    if name.endswith("LLVMConfig.cmake") else "fixture").encode()
            item = tarfile.TarInfo("test-sdk/" + name)
            item.size = len(data)
            stream.addfile(item, io.BytesIO(data))
    record["size_bytes"] = archive.stat().st_size
    record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive


def make_nanobind_bytes(member_name: str = "nanobind-test/cmake/nanobind-config.cmake") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = 7
        archive.addfile(member, io.BytesIO(b"# test\n"))
    return buf.getvalue()


class CompilerSelectionTests(unittest.TestCase):
    def test_cmake_release_version_rejects_different_python_metadata(self):
        import runpy
        version = runpy.run_path(str(ROOT / "python/ask/_version.py"))["__version__"]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "version.cmake"
            script.write_text(
                "cmake_minimum_required(VERSION 3.28)\n"
                f'include("{ROOT}/cmake/AskVersion.cmake")\n'
                'message(STATUS "ASK_VERSION=${ASK_VERSION}")\n', encoding="utf-8")
            accepted = subprocess.run(["cmake", f"-DSKBUILD_PROJECT_VERSION={version}", "-P", str(script)],
                                      capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("ASK_VERSION=" + version, accepted.stdout)
            rejected = subprocess.run(["cmake", "-DSKBUILD_PROJECT_VERSION=999.0.0", "-P", str(script)],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("differs from ASK", rejected.stderr)

    def test_cmake_allowlist_resolution(self):
        cases = {
            "none": "", "all": "triton", "triton": "triton",
            "triton;triton": "triton", " triton ": "triton",
            "": None, "none;triton": None, "all;triton": None,
            "bogus": None, "triton;;triton": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "selection.cmake"
            for requested, expected in cases.items():
                with self.subTest(requested=requested):
                    script.write_text(
                        "cmake_minimum_required(VERSION 3.28)\n"
                        f'include("{ROOT}/cmake/AskCompilerSelection.cmake")\n'
                        f"ask_resolve_compilers([=[{requested}]=] triton selected)\n"
                        'message(STATUS "SELECTED=${selected}")\n',
                        encoding="utf-8",
                    )
                    res = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
                    if expected is None:
                        self.assertNotEqual(res.returncode, 0, res.stdout)
                    else:
                        self.assertEqual(res.returncode, 0, res.stderr)
                        self.assertIn("SELECTED=" + expected, res.stdout)


class FetchDepsTests(unittest.TestCase):
    def llvm_lock(self) -> dict:
        value = fetcher.load_llvm_lock(ROOT / "cmake/llvm.lock.json")
        value["directory"] = "test-sdk"
        return value

    def nanobind_lock(self, payload: bytes) -> dict:
        return {
            "schema_version": 1, "version": "test", "directory": "nanobind-test",
            "url": "https://example.test/nanobind.tar.gz",
            "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        }

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
            receipt["revision"] = "bad"
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

    def test_nanobind_fetch_and_digest_guard(self):
        payload = make_nanobind_bytes()
        lock = self.nanobind_lock(payload)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                target = fetcher.fetch_nanobind(lock, directory)
            self.assertEqual(json.loads((target / fetcher.NANOBIND_RECEIPT).read_text(encoding="utf-8")), lock)
            with patch.object(fetcher.urllib.request, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(fetcher.fetch_nanobind(lock, directory), target)

        bad_lock = dict(lock, sha256="0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    fetcher.fetch_nanobind(bad_lock, directory)
            self.assertFalse((Path(directory) / lock["directory"]).exists())

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
            valid = '<testsuite><testcase name="ask-unit" status="run"><system-out>Ran 4 tests in 0.1s\n\nOK</system-out></testcase></testsuite>'
            report.write_text(valid, encoding="utf-8")
            runner.check_report(report, {"ask-unit"})
            invalid = [
                valid.replace("OK", "OK (skipped=1)"),
                valid.replace("OK", "test_native ... skipped 'missing extension'"),
                valid.replace("Ran 4 tests", "Ran 0 tests"),
                valid.replace('status="run"', 'status="notrun"'),
                valid.replace("<system-out>", "<failure/><system-out>"),
                valid.replace("<system-out>", "<skipped/><system-out>"),
                '<testsuite/>',
            ]
            for content in invalid:
                with self.subTest(content=content):
                    report.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runner.check_report(report, {"ask-unit"})
            report.write_text(valid, encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.check_report(report, {"ask-unit", "ask-triton"})


if __name__ == "__main__":
    unittest.main()
