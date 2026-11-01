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
sys.path[:0] = [str(root / "tests"), str(root / "python")]
from support.pointwise import build_module

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

    # f32 uses typed pointer metadata and float LLVM marker values. The
    # pointer-only CUDA entry ABI stays identical to the integer case.
    float_graph = build_module(10, dtype="f32", factor=1.25, external_offset=0.1).to_mlir()
    floating = fuse("float", float_graph)
    float_ir = (floating / "fused.ll").read_text()
    float_helpers = (floating / "exports/ask.helpers.ll").read_text()
    float_artifact = json.loads((floating / "exports/external-0.artifact.json").read_text())
    float_report = json.loads((floating / "report.json").read_text())
    assert float_report["fusion"]["dtype"] == "f32"
    assert "fadd float" in float_helpers and "contract" not in float_helpers
    assert "fadd contract float" in float_ir and "fmul contract float" in float_ir
    assert "ASK_BOUNDARY" not in float_ir
    assert float_artifact["fusion_contract"]["dtype"] == "f32"
    assert all(p["element_type"] == "f32" for p in float_artifact["port_map"])

    bad = copy.deepcopy(float_artifact)
    bad["fusion_contract"]["dtype"] = "i32"
    replay("float-graph-dtype", bad, "dtype does not match", float_graph)
    bad = copy.deepcopy(float_artifact)
    bad["port_map"][0]["element_type"] = "i32"
    replay("float-port-dtype", bad, "port element type", float_graph)
    bad = copy.deepcopy(float_artifact)
    bad["fusion_contract"]["input"]["arguments"]["value"] = 0
    replay("float-marker-argument-map", bad, "argument mapping", float_graph)
    bad = copy.deepcopy(float_artifact)
    bad["payload"], changed = re.subn(r'("=f,r,f,l,r"\(i32 %[^,]+, )float %[^,]+',
                                       r'\1i32 0', bad["payload"])
    assert changed
    bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
    replay("float-marker-value-type", bad, "malformed ASK boundary marker", float_graph)
    bad = copy.deepcopy(float_artifact)
    bad["fusion_contract"]["numerics"]["enable_fp_fusion"] = False
    replay("float-numerical-policy", bad, "numerical policy", float_graph)

    no_contract_graph = build_module(10, dtype="f32", factor=1.25, external_offset=0.1,
                                     enable_fp_fusion=False).to_mlir()
    no_contract = fuse("float-no-contract", no_contract_graph)
    no_contract_ir = (no_contract / "fused.ll").read_text()
    assert "fmul float" in no_contract_ir and "fadd float" in no_contract_ir
    assert "contract float" not in no_contract_ir
    bad = json.loads((no_contract / "exports/external-0.artifact.json").read_text())
    assert "fmul float" in bad["payload"]
    bad["payload"] = bad["payload"].replace("fmul float", "fmul contract float", 1)
    bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
    replay("float-disabled-contraction-flag", bad, "despite enable_fp_fusion=false", no_contract_graph)

    # One typed graph builder is also used by host and GPU numerical matrices.
    # Low-precision boundary types and byte strides are verified independently
    # of the numerical GPU checks; a four-byte address must not pass for fp16.
    for dtype, llvm_type in (("f16", "half"), ("bf16", "bfloat")):
        low_graph = build_module(10, dtype=dtype, factor=1.25, external_offset=0.1).to_mlir()
        lowered = fuse(dtype, low_graph)
        low_ir = (lowered / "fused.ll").read_text()
        low_helpers = (lowered / "exports/ask.helpers.ll").read_text()
        low_artifact = json.loads((lowered / "exports/external-0.artifact.json").read_text())
        low_report = json.loads((lowered / "report.json").read_text())
        assert low_report["fusion"]["dtype"] == dtype
        assert f"fadd {llvm_type}" in low_helpers and "contract" not in low_helpers
        assert "ASK_BOUNDARY" not in low_ir
        assert all(port["element_type"] == dtype for port in low_artifact["port_map"])
        bad = copy.deepcopy(low_artifact)
        assert f"getelementptr {llvm_type}" in bad["payload"]
        bad["payload"] = bad["payload"].replace(f"getelementptr {llvm_type}", "getelementptr i32")
        bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
        replay(dtype + "-wrong-byte-stride", bad, "element index", low_graph)
        bad = copy.deepcopy(low_artifact)
        bad["port_map"][0]["element_type"] = "f32"
        replay(dtype + "-wrong-port-type", bad, "port element type", low_graph)
        bad = copy.deepcopy(low_artifact)
        bad["payload"], changed = re.subn(
            rf'("=h,r,h,l,r"\(i32 %[^,]+, ){llvm_type} %[^,]+', r'\1i16 0', bad["payload"])
        assert changed
        bad["fusion_contract"]["llir_sha256"] = hashlib.sha256(bad["payload"].encode()).hexdigest()
        replay(dtype + "-wrong-marker-type", bad, "malformed ASK boundary marker", low_graph)

    # BLOCK > physical threads creates two value sites per logical boundary.
    # N=19 also makes one masked lane group normalize to a sign-bit expression.
    multiple = source.replace("tensor<10xi32>", "tensor<19xi32>").replace(
        "array<i64: 10>", "array<i64: 19>").replace("N = 10", "N = 19").replace(
        "BLOCK = 128", "BLOCK = 256")
    multi = fuse("multiple-sites", multiple)
    multi_report = json.loads((multi / "report.json").read_text())
    assert multi_report["fusion"]["removed_boundary_markers"] == 4
    assert "ASK_BOUNDARY" not in (multi / "fused.ll").read_text()
    print("Native LLVM fusion verified for i32/f32/f16/bf16, producer contraction, equivalent masks and multiple sites; inconsistent requests rejected")
