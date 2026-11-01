"""Real native fusion and rejection tests; GPU numerical validation is explicit."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

driver, python, worker = sys.argv[1:]
root = Path(__file__).resolve().parents[1]
source = (root / "tests/ir/pipeline.mlir").read_text()

with tempfile.TemporaryDirectory(prefix="ask-native-fusion-") as directory:
    work = Path(directory)

    def fuse(name, graph=source, selected_worker=worker, expected_error=None):
        destination = work / name
        destination.mkdir()
        (destination / "input.mlir").write_text(graph)
        result = subprocess.run([
            driver, str(destination / "input.mlir"), "--fuse-external",
            "--export-dir", str(destination / "exports"), "--triton-python", python,
            "--triton-worker", str(selected_worker), "-o", str(destination / "fused.ll"),
            "--report-json", str(destination / "report.json")],
            capture_output=True, text=True, timeout=180)
        if expected_error:
            assert result.returncode > 0, result.stderr
            assert expected_error in result.stderr, result.stderr
            assert not (destination / "fused.ll").exists()
            assert not (destination / "exports/manifest.json").exists()
        else:
            assert result.returncode == 0, result.stderr
        return destination

    success = fuse("success")
    fused = (success / "fused.ll").read_text()
    report = json.loads((success / "report.json").read_text())
    artifact = json.loads((success / "exports/external-0.artifact.json").read_text())
    assert report["phase"] == "fused_llvm"
    assert report["fusion"]["inlined_helpers"] == 2
    assert report["fusion"]["entry"] == artifact["entry"]
    assert report["fusion"]["removed_boundary_markers"] == 2
    assert "ASK_BOUNDARY" not in fused and "__ask_pre" not in fused
    assert "ld.global.b32" in fused and "st.global.b32" in fused
    assert "!ask.shape_snapshot" in fused
    assert "add i32" in (success / "exports/ask.helpers.ll").read_text()
    output_artifact = json.loads((success / "exports/fused.artifact.json").read_text())
    assert output_artifact["payload"] == fused
    assert output_artifact["invocation"] == artifact["invocation"]
    assert output_artifact["validation"]["llvm_module_verified"]
    assert "fusion_contract" not in output_artifact
    verified = subprocess.run([driver, str(success / "fused.ll"), "--check-llir", "-o",
                               str(success / "verified.ll")], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stderr

    # Test native validation independently from the trusted exporter by replaying
    # one genuine artifact with one inconsistent field changed in each case.
    def replay(name, reply, message, graph=source):
        response = work / (name + ".json")
        response.write_text(json.dumps(reply))
        fake_worker = work / (name + ".py")
        fake_worker.write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser()\np.add_argument('--request')\np.add_argument('--output')\n"
            "a=p.parse_args()\n"
            f"Path(a.output).write_bytes(Path({str(response)!r}).read_bytes())\n")
        return fuse(name, graph, fake_worker, message)

    bad = copy.deepcopy(artifact)
    bad["payload"] += "\n; corruption\n"
    replay("digest", bad, "digest")
    bad = copy.deepcopy(artifact)
    del bad["fusion_contract"]
    replay("missing-contract", bad, "boundary contract")
    bad = copy.deepcopy(artifact)
    del bad["validation"]
    replay("missing-validation", bad, "protocol 1")
    bad = copy.deepcopy(artifact)
    bad["fusion_contract"]["proof"]["same_mask"] = False
    replay("mask-proof", bad, "same_mask")
    bad = copy.deepcopy(artifact)
    bad["fusion_contract"]["proof"]["layout"]["matched_uninstrumented_baseline"] = False
    replay("changed-layout", bad, "preserved the producer layout")
    bad = copy.deepcopy(artifact)
    bad["fusion_contract"]["input"]["marker_count"] += 1
    replay("site-coverage", bad, "count")
    bad = copy.deepcopy(artifact)
    bad["fusion_contract"]["shape"] = [11]
    replay("shape", bad, "shape/constexpr")
    bad = copy.deepcopy(artifact)
    bad["payload"] = bad["payload"].replace("ASK_BOUNDARY_INPUT_V1", "REMOVED_BOUNDARY_INPUT")
    bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
    replay("missing-marker", bad, "retain both")
    for name, old, new, message in [
        ("wrong-element-offset", "getelementptr i32", "getelementptr i16", "element index"),
        ("wrong-predicate", "icmp slt i32", "icmp ult i32", "signed index"),
        ("missing-launch-attribute", '"nvvm.reqntid"="128"', '', "entry ABI"),
    ]:
        bad = copy.deepcopy(artifact)
        assert old in bad["payload"]
        bad["payload"] = bad["payload"].replace(old, new)
        bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
        replay(name, bad, message)
    replay("post-other-input", artifact, "different graph input",
           source.replace('"ask.add"(%um, %threes)', '"ask.add"(%um, %x)'))
    # Equivalent predicates must not depend on the producer's exact spelling.
    import re
    equivalent = copy.deepcopy(artifact)
    equivalent["payload"], changed = re.subn(r'icmp slt i32 (%[\w.]+), 10',
                                              r'icmp sle i32 \1, 9', equivalent["payload"])
    assert changed
    equivalent["fusion_contract"]["llir_sha256"] = hashlib.sha256(equivalent["payload"].encode()).hexdigest()
    replay("equivalent-predicate", equivalent, None)

    # BLOCK > physical threads creates two value sites per logical boundary.
    # N=19 also makes one masked lane group normalize to a sign-bit expression.
    multiple = source.replace("tensor<10xi32>", "tensor<19xi32>").replace(
        "array<i64: 10>", "array<i64: 19>").replace("N = 10", "N = 19").replace(
        "BLOCK = 128", "BLOCK = 256")
    multi = fuse("multiple-sites", multiple)
    multi_report = json.loads((multi / "report.json").read_text())
    assert multi_report["fusion"]["removed_boundary_markers"] == 4
    assert "ASK_BOUNDARY" not in (multi / "fused.ll").read_text()
    print("Native LLVM fusion verified including equivalent masks and multiple sites; 12 inconsistent requests rejected")
