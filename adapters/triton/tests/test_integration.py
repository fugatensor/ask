"""End-to-end Triton export and ASK native LLVM value-boundary fusion integration tests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ADAPTER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ADAPTER_ROOT.parents[1]
for entry in (str(REPO_ROOT / "tests"), str(REPO_ROOT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from support.host_runner import ask_opt_path
from support.pointwise import build_module

TRITON_PYTHON = os.environ.get("ASK_TEST_TRITON_PYTHON")


@unittest.skipUnless(TRITON_PYTHON, "set ASK_TEST_TRITON_PYTHON to run real Triton export/fusion tests")
class TritonPipelineAndFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = ask_opt_path()
        caps = json.loads(subprocess.check_output([cls.driver, "--capabilities"], text=True))
        if "triton" not in caps.get("third_party_compilers", []):
            raise unittest.SkipTest("this ASK build does not enable Triton")
        cls.worker = str(ADAPTER_ROOT / "worker.py")
        cls.pipeline_mlir = (REPO_ROOT / "tests/ir/pipeline.mlir").read_text(encoding="utf-8")

    def test_export_pipeline_llir_check_and_routing_preflights(self):
        with tempfile.TemporaryDirectory(prefix="ask-triton-pipe-") as directory:
            work = Path(directory)
            dest = work / "exports"
            prepared = work / "prepared.mlir"
            res = subprocess.run(
                [self.driver, str(REPO_ROOT / "tests/ir/pipeline.mlir"), "--export-external",
                 "--export-dir", str(dest), f"--compiler-python=triton={TRITON_PYTHON}",
                 f"--compiler-worker=triton={self.worker}", "-o", str(prepared)],
                text=True, capture_output=True, timeout=180,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
            artifact = json.loads((dest / "external-0.artifact.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["phase"], "exported")
            self.assertEqual(artifact["producer"]["version"], "3.7.1")
            self.assertIn("captures(none)", artifact["payload"])

            chk = subprocess.run([self.driver, str(dest / "external-0.ll"), "--check-llir",
                                  "-o", str(work / "verified.ll")], text=True, capture_output=True)
            self.assertEqual(chk.returncode, 0, chk.stderr)

            # Unsupported worker option must fail without publishing manifest.
            bad_mlir = work / "bad.mlir"
            bad_mlir.write_text(self.pipeline_mlir.replace("num_warps = 4", "num_warps = 16"), encoding="utf-8")
            failed = subprocess.run(
                [self.driver, str(bad_mlir), "--export-external", "--export-dir", str(work / "rej"),
                 f"--compiler-python=triton={TRITON_PYTHON}", f"--compiler-worker=triton={self.worker}"],
                text=True, capture_output=True, timeout=180,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("UnsupportedOptions", failed.stderr)
            self.assertFalse((work / "rej/manifest.json").exists())

            for label, routes in (
                ("missing-py", [f"--compiler-worker=triton={self.worker}"]),
                ("malformed", [f"--compiler-python={TRITON_PYTHON}"]),
                ("disabled", [f"--compiler-worker=not_registered={self.worker}"]),
            ):
                out_dir = work / label
                bad_route = subprocess.run(
                    [self.driver, str(REPO_ROOT / "tests/ir/pipeline.mlir"), "--export-external",
                     "--export-dir", str(out_dir), *routes],
                    text=True, capture_output=True,
                )
                self.assertGreater(bad_route.returncode, 0, label)
                self.assertFalse(out_dir.exists())

    def test_native_fusion_precisions_mixed_compute_and_artifact_rejections(self):
        with tempfile.TemporaryDirectory(prefix="ask-triton-fuse-") as directory:
            work = Path(directory)

            def fuse(name: str, graph: str = self.pipeline_mlir, selected_worker: str = self.worker,
                     expected_error: str | None = None) -> Path:
                dest = work / name
                dest.mkdir()
                (dest / "input.mlir").write_text(graph, encoding="utf-8")
                res = subprocess.run(
                    [self.driver, str(dest / "input.mlir"), "--fuse-external",
                     "--export-dir", str(dest / "exports"),
                     f"--compiler-python=triton={TRITON_PYTHON}",
                     f"--compiler-worker=triton={selected_worker}",
                     "-o", str(dest / "fused.ll"), "--report-json", str(dest / "report.json")],
                    capture_output=True, text=True, timeout=180,
                )
                if expected_error:
                    self.assertGreater(res.returncode, 0, res.stderr)
                    self.assertIn(expected_error, res.stderr)
                    self.assertFalse((dest / "fused.ll").exists())
                else:
                    self.assertEqual(res.returncode, 0, res.stderr)
                return dest

            def replay(name: str, reply: dict, expected_error: str | None, graph: str = self.pipeline_mlir):
                resp = work / f"{name}.json"
                resp.write_text(json.dumps(reply), encoding="utf-8")
                fake_w = work / f"{name}.py"
                fake_w.write_text(
                    "import argparse\nfrom pathlib import Path\n"
                    "p=argparse.ArgumentParser()\np.add_argument('--request')\np.add_argument('--output')\n"
                    f"a=p.parse_args()\nPath(a.output).write_bytes(Path({str(resp)!r}).read_bytes())\n",
                    encoding="utf-8",
                )
                return fuse(name, graph, str(fake_w), expected_error)

            success = fuse("i32-base")
            fused_ll = (success / "fused.ll").read_text(encoding="utf-8")
            report = json.loads((success / "report.json").read_text(encoding="utf-8"))
            artifact = json.loads((success / "exports/external-0.artifact.json").read_text(encoding="utf-8"))
            self.assertEqual((report["phase"], report["fusion"]["inlined_helpers"]), ("fused_llvm", 2))
            self.assertNotIn("ASK_BOUNDARY", fused_ll)

            # Tamper replay rejections against one real exported artifact (zero extra Triton JIT runs).
            for label, mutator, msg in (
                ("producer-version", lambda b: b["producer"].update(version="0.0.0"), "pinned producer"),
                ("adapter-version", lambda b: b["producer"].update(adapter_version="0.3.0"), "pinned producer"),
                ("digest", lambda b: b.update(payload=b["payload"] + "\n; bad\n"), "digest"),
                ("no-contract", lambda b: b.pop("fusion_contract"), "source / instrumentation evidence"),
                ("mask-proof", lambda b: b["fusion_contract"]["proof"].update(same_mask=False), "same_mask"),
                ("layout", lambda b: b["fusion_contract"]["proof"]["layout"].update(matched_uninstrumented_baseline=False),
                 "preserved the producer layout"),
                ("shape", lambda b: b["fusion_contract"].update(shape=[11]), "shape/constexpr"),
            ):
                bad = copy.deepcopy(artifact)
                mutator(bad)
                replay(label, bad, msg)

            for label, old, new, msg in (
                ("stride", "getelementptr i32", "getelementptr i16", "element index"),
                ("pred", "icmp slt i32", "icmp ult i32", "signed index"),
            ):
                bad = copy.deepcopy(artifact)
                bad["payload"] = bad["payload"].replace(old, new)
                bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
                replay(label, bad, msg)

            # Equivalent predicate spelling is accepted.
            eq = copy.deepcopy(artifact)
            eq["payload"], n = re.subn(r"icmp slt i32 (%[\w.]+), 10", r"icmp sle i32 \1, 9", eq["payload"])
            self.assertGreater(n, 0)
            eq["fusion_contract"]["llir_sha256"] = hashlib.sha256(eq["payload"].encode()).hexdigest()
            replay("equiv-pred", eq, None)

            # Float contraction enabled vs disabled.
            for fp_fusion, expect_contract in ((True, True), (False, False)):
                fg = build_module(10, dtype="f32", factor=1.25, external_offset=0.1,
                                  enable_fp_fusion=fp_fusion).to_mlir()
                fd = fuse(f"f32-contract-{fp_fusion}", fg)
                fir = (fd / "fused.ll").read_text(encoding="utf-8")
                self.assertEqual("fmul contract float" in fir, expect_contract)

            # Remaining element types + mixed f16/bf16 -> f32 compute + multi-site BLOCK=256.
            for dtype, arch in (("f16", 80), ("bf16", 80), ("bool", 75), ("i64", 86)):
                tg = build_module(10, dtype=dtype, arch=arch).to_mlir()
                td = fuse(f"type-{dtype}", tg)
                self.assertNotIn("ASK_BOUNDARY", (td / "fused.ll").read_text(encoding="utf-8"))

            for dtype in ("f16", "bf16"):
                mg = build_module(10, dtype=dtype, compute_dtype="f32", factor=1.25,
                                  external_offset=0.1, arch=86).to_mlir()
                md = fuse(f"mixed-{dtype}", mg)
                mrep = json.loads((md / "report.json").read_text(encoding="utf-8"))
                self.assertTrue({dtype, "f32"} <= set(mrep["fusion"]["required_element_types"]))

            multi_src = (
                self.pipeline_mlir
                .replace("tensor<10xi32>", "tensor<19xi32>")
                .replace("array<i64: 10>", "array<i64: 19>")
                .replace("N = 10", "N = 19")
                .replace("BLOCK = 128", "BLOCK = 256")
            )
            multi_d = fuse("multi-site", multi_src)
            multi_rep = json.loads((multi_d / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(multi_rep["fusion"]["removed_boundary_markers"], 4)


if __name__ == "__main__":
    unittest.main()
