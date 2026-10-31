"""Opt-in real export; no CUDA device is queried or kernel launched."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("ASK_TEST_TRITON_PYTHON")


@unittest.skipUnless(PYTHON, "set ASK_TEST_TRITON_PYTHON to run the real exporter")
class RealExportTests(unittest.TestCase):
    def test_public_api_export_without_gpu_execution(self):
        probe = subprocess.run([PYTHON, str(ROOT / "worker.py"), "--probe"],
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(json.loads(probe.stdout)["triton_version"], "3.7.1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            request = ROOT / "tests" / "fixtures" / "scale2.request.json"
            result = subprocess.run([PYTHON, str(ROOT / "worker.py"), "--request", str(request),
                                     "--output", str(output)],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["format"], "llvm-ir")
            self.assertIn("define ptx_kernel void @scale2", artifact["payload"])
            self.assertEqual(len(artifact["entry_signature"]["parameters"]), 4)
            self.assertEqual(artifact["invocation"]["arguments"][2]["binding"], "null")
            self.assertEqual(artifact["invocation"]["arguments"][3]["binding"], "null")
            self.assertEqual(artifact["invocation"]["grid"], [1, 1, 1])
            self.assertFalse(artifact["validation"]["device_execution_performed"])
            self.assertFalse(artifact["validation"]["llvm_module_verified"])


if __name__ == "__main__":
    unittest.main()
