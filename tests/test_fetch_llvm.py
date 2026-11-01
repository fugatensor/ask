import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ask_fetch_llvm", ROOT / "tools/fetch_llvm.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def lock():
    value = fetcher.load_lock(ROOT / "cmake/llvm.lock.json")
    value["directory"] = "test-sdk"
    return value


def make_archive(parent, record):
    archive = parent / "test-sdk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in fetcher.REQUIRED_FILES:
            data = (f"set(LLVM_PACKAGE_VERSION {record['version']})\n"
                    if name.endswith("LLVMConfig.cmake") else "fixture").encode()
            item = tarfile.TarInfo("test-sdk/" + name)
            item.size = len(data)
            stream.addfile(item, io.BytesIO(data))
    record["size_bytes"] = archive.stat().st_size
    record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive


class FetchTests(unittest.TestCase):
    def test_cached_archive_extract_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            make_archive(parent, record)
            with mock.patch.object(fetcher.urllib.request, "urlopen", side_effect=AssertionError("network")):
                target = fetcher.fetch(record, parent)
                self.assertTrue((target / fetcher.RECEIPT).exists())
                self.assertEqual(fetcher.fetch(record, parent), target)

    def test_mismatched_sdk_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            make_archive(parent, record)
            target = fetcher.fetch(record, parent)
            receipt = json.loads((target / fetcher.RECEIPT).read_text())
            receipt["revision"] = "bad"
            (target / fetcher.RECEIPT).write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                fetcher.fetch(record, parent)
            self.assertEqual(json.loads((target / fetcher.RECEIPT).read_text())["revision"], "bad")

    def test_existing_directory_without_receipt_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            make_archive(parent, record)
            target = fetcher.fetch(record, parent)
            (target / fetcher.RECEIPT).unlink()
            with self.assertRaisesRegex(ValueError, "no valid receipt"):
                fetcher.fetch(record, parent)

    def test_matching_receipt_does_not_hide_incomplete_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            make_archive(parent, record)
            target = fetcher.fetch(record, parent)
            (target / "lib/libMLIRIR.a").unlink()
            with self.assertRaisesRegex(ValueError, "SDK is incomplete"):
                fetcher.fetch(record, parent)
            self.assertFalse((target / "lib/libMLIRIR.a").exists())

    def test_failed_download_writes_neither_sdk_nor_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            with mock.patch.object(fetcher.urllib.request, "urlopen", side_effect=OSError("network failure")):
                with self.assertRaisesRegex(OSError, "network failure"):
                    fetcher.fetch(record, parent)
            self.assertEqual(list(parent.iterdir()), [])

    def test_hash_mismatch_does_not_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, record = Path(directory), lock()
            archive = parent / "test-sdk.tar.gz"
            archive.write_bytes(b"bad")
            record["size_bytes"] = 3
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                fetcher.fetch(record, parent)
            self.assertEqual(archive.read_bytes(), b"bad")
            self.assertFalse((parent / "test-sdk").exists())

    def test_path_traversal_rejected(self):
        for name in ("../escape", "/absolute", "test-sdk/../escape"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                item = tarfile.TarInfo(name)
                item.size = 1
                with self.assertRaises(ValueError):
                    fetcher.archive_filter(lock())(item, directory)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            item = tarfile.TarInfo("test-sdk/link")
            item.type, item.linkname = tarfile.SYMTYPE, "../../escape"
            with self.assertRaises(ValueError):
                fetcher.archive_filter(lock())(item, directory)

    def test_special_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            item = tarfile.TarInfo("test-sdk/fifo")
            item.type = tarfile.FIFOTYPE
            with self.assertRaises(ValueError):
                fetcher.archive_filter(lock())(item, directory)


if __name__ == "__main__":
    unittest.main()
