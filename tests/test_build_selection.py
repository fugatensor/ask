"""Build allowlist checks without probing or importing any compiler package."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_selection_contract(self):
        cases = {"none": "", "all": "triton", "triton": "triton",
                 "triton;triton": "triton", " triton ": "triton",
                 "": None, "none;triton": None, "all;triton": None,
                 "none;all": None, "bogus": None, "triton;;triton": None}
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "selection.cmake"
            for requested, expected in cases.items():
                with self.subTest(requested=requested):
                    script.write_text(
                        'cmake_minimum_required(VERSION 3.28)\n'
                        f'include("{ROOT}/cmake/AskCompilerSelection.cmake")\n'
                        f'ask_resolve_compilers([=[{requested}]=] triton selected)\n'
                        'message(STATUS "SELECTED=${selected}")\n')
                    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
                    if expected is None:
                        self.assertNotEqual(result.returncode, 0, result.stdout)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("SELECTED=" + expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
