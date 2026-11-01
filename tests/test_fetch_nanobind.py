"""The explicit source fetcher checks the pinned archive before extraction."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_nanobind", ROOT / "tools/fetch_nanobind.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def source_archive(name="nanobind-test/cmake/nanobind-config.cmake"):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 7
        archive.addfile(member, io.BytesIO(b"# test\n"))
    return data.getvalue()


def source_lock(contents):
    return {"schema_version": 1, "version": "test", "directory": "nanobind-test",
            "url": "https://example.test/archive.tar.gz", "size_bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest()}


class FetchTests(unittest.TestCase):
    def test_verified_source_and_receipt_are_reused_without_network(self):
        contents = source_archive()
        lock = source_lock(contents)
        with tempfile.TemporaryDirectory() as path:
            with patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(contents)):
                target = fetcher.fetch(lock, path)
            self.assertEqual(json.loads((target / ".ask-nanobind.json").read_text()), lock)
            with patch.object(fetcher.urllib.request, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(fetcher.fetch(lock, path), target)

    def test_bad_digest_does_not_publish_source(self):
        contents = source_archive()
        lock = source_lock(contents)
        lock["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as path:
            with patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(contents)):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    fetcher.fetch(lock, path)
            self.assertFalse((Path(path) / lock["directory"]).exists())

    def test_archive_cannot_escape_source_root(self):
        contents = source_archive("nanobind-test/../../escape")
        with tempfile.TemporaryDirectory() as path:
            with patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(contents)):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    fetcher.fetch(source_lock(contents), path)


if __name__ == "__main__":
    unittest.main()
