"""Compiler authorization and environment checks must happen before cache hits."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ask import CompilerError
from ask.gpu import compile_graph, _tool, _validate_environment


class GPUEntryTests(unittest.TestCase):
    def test_none_build_never_looks_up_cache_or_starts_worker(self):
        compiler = Mock()
        compiler.capabilities.return_value = {"third_party_compilers": []}
        with patch("ask.gpu.IRCache") as cache, patch("ask.gpu.subprocess.run") as run:
            with self.assertRaisesRegex(CompilerError, "does not enable"):
                compile_graph(compiler, "not parsed", cache_dir="unused", output_dir="unused",
                              triton_python="missing", triton_worker="missing", llc="missing", ptxas="missing")
            cache.assert_not_called()
            run.assert_not_called()
            compiler.prepare.assert_not_called()

    def test_override_cannot_bypass_gate_on_a_warm_cache(self):
        compiler = Mock()
        compiler.capabilities.return_value = {"third_party_compilers": ["triton"],
                                               "triton_llir_fusion": True}
        with patch.dict(os.environ, {"TRITON_LIBDEVICE_PATH": "/changed-in-place.bc"}, clear=True), \
             patch("ask.gpu.IRCache") as cache, patch("ask.gpu.subprocess.run") as run:
            with self.assertRaisesRegex(CompilerError, "TRITON_LIBDEVICE_PATH"):
                compile_graph(compiler, "not parsed", cache_dir="existing", output_dir="unused",
                              triton_python="missing", triton_worker="missing", llc="missing", ptxas="missing")
            cache.assert_not_called()
            run.assert_not_called()
            compiler.prepare.assert_not_called()

    def test_allowed_cache_location_and_false_override(self):
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/tmp/custom-cache",
                                     "TRITON_KERNEL_OVERRIDE": ""}, clear=True):
            _validate_environment()
        with patch.dict(os.environ, {"TRITON_KERNEL_OVERRIDE": "0"}, clear=True):
            with self.assertRaises(CompilerError):
                _validate_environment()

    @unittest.skipUnless(os.name == "posix", "executable symlinks need POSIX")
    def test_tool_fingerprint_preserves_virtualenv_executable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "python-real"
            link = Path(directory) / "venv-python"
            base.write_bytes(b"test executable")
            base.chmod(0o700)
            link.symlink_to(base)
            self.assertEqual(_tool(link)["path"], str(link))
            before = _tool(link)["sha256"]
            base.write_bytes(b"updated executable")
            self.assertNotEqual(_tool(link)["sha256"], before)


if __name__ == "__main__":
    unittest.main()
