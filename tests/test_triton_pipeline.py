"""Real SDK graph -> worker export -> ASK native LLVM reader, without a GPU launch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

driver, python, worker = sys.argv[1:]
root = Path(__file__).resolve().parents[1]


def checked(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args}\n{result.stderr}")
    return result


with tempfile.TemporaryDirectory(prefix="ask-triton-e2e-") as directory:
    destination = Path(directory) / "exports"
    prepared = Path(directory) / "prepared.mlir"
    checked([driver, str(root / "tests/ir/pipeline.mlir"), "--export-external",
             "--export-dir", str(destination), "--triton-python", python,
             "--triton-worker", worker, "-o", str(prepared)])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["phase"] == "exported"
    assert len(manifest["markers"]) == 2
    snapshot = manifest["shape_snapshot"]
    assert snapshot["snapshot_digest"]
    shapes = {item["value_id"]: item["shape"] for item in snapshot["shape_bindings"]}
    assert shapes["v_t"] == [10] and shapes["v_u"] == [10]
    assert manifest["external_requests"][0]["compiler"] == "triton"
    assert manifest["external_requests"][0]["op_id"] == "F"
    request = json.loads((destination / "external-0.request.json").read_text())
    assert set(request) == {"protocol_version", "source", "entry", "signature",
                            "source_bindings", "constants", "fixed_options", "target", "grid"}
    artifact = json.loads((destination / "external-0.artifact.json").read_text())
    original_ir = (destination / "external-0.ll").read_text()
    assert artifact["protocol_version"] == 1 and artifact["format"] == "llvm-ir"
    assert artifact["payload"] == original_ir, "producer IR must not be rewritten during export"
    assert artifact["producer"]["version"] == "3.7.1"
    assert artifact["entry"] == "scale2"
    assert [(port["source_parameter"], port["logical_port"], port["argument"])
            for port in artifact["port_map"]] == [("t", "input:0", 0), ("u", "output:0", 1)]
    assert [arg["binding"] for arg in artifact["invocation"]["arguments"]] == [
        "input:0", "output:0", "null", "null"]
    assert artifact["specializations"]["constants"] == request["constants"]
    assert artifact["invocation"]["grid"] == request["grid"]
    assert not artifact["validation"]["device_execution_performed"]
    assert not artifact["validation"]["llvm_module_verified"]
    assert "captures(none)" in original_ir
    checked([driver, str(destination / "external-0.ll"), "--check-llir",
             "-o", str(Path(directory) / "verified.ll")])
    optimized_llvm = Path(directory) / "optimized.ll"
    optimization_report = Path(directory) / "optimization.json"
    checked([driver, str(destination / "external-0.ll"), "--optimize-llir",
             "-o", str(optimized_llvm), "--report-json", str(optimization_report)])
    checked([driver, str(optimized_llvm), "--check-llir",
             "-o", str(Path(directory) / "optimized-verified.ll")])
    assert json.loads(optimization_report.read_text())["phase"] == "llvm_optimized"
    assert (destination / "external-0.ll").read_text() == original_ir
    assert "asm" in optimized_llvm.read_text(), "real Triton inline PTX must remain represented"
    assert '"ask.external"' in prepared.read_text()
    assert '"ask.mark"' not in prepared.read_text()

    # A failed selected-worker request must not publish an exported manifest.
    bad_graph = Path(directory) / "unsupported-options.mlir"
    bad_graph.write_text((root / "tests/ir/pipeline.mlir").read_text().replace(
        "num_warps = 4", "num_warps = 8"))
    rejected = Path(directory) / "rejected"
    rejected_output = Path(directory) / "rejected.mlir"
    failed = subprocess.run([driver, str(bad_graph), "--export-external",
                             "--export-dir", str(rejected), "--triton-python", python,
                             "--triton-worker", worker, "-o", str(rejected_output)],
                            text=True, capture_output=True, timeout=180)
    assert failed.returncode != 0, "unsupported worker options must fail the driver"
    assert "UnsupportedOptions" in failed.stderr
    assert not (rejected / "manifest.json").exists()
    assert not (rejected / "external-0.artifact.json").exists()
    assert not rejected_output.exists()

    # Start with a real successful reply, then alter only its launch grid. A
    # worker response must remain bound to this request even if its IR is valid.
    wrong_grid = json.loads(json.dumps(artifact))
    wrong_grid["invocation"]["grid"] = [2, 1, 1]
    wrong_reply = Path(directory) / "wrong-grid-reply.json"
    wrong_reply.write_text(json.dumps(wrong_grid))
    fake_worker = Path(directory) / "wrong-grid-worker.py"
    fake_worker.write_text(
        "import argparse\nfrom pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--request', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "args = parser.parse_args()\n"
        f"Path(args.output).write_text(Path({str(wrong_reply)!r}).read_text())\n")
    wrong_destination = Path(directory) / "wrong-grid-exports"
    wrong_prepared = Path(directory) / "wrong-grid-prepared.mlir"
    invalid_response = subprocess.run(
        [driver, str(root / "tests/ir/pipeline.mlir"), "--export-external",
         "--export-dir", str(wrong_destination), "--triton-python", python,
         "--triton-worker", str(fake_worker), "-o", str(wrong_prepared)],
        text=True, capture_output=True, timeout=180)
    assert invalid_response.returncode > 0, "changed-grid response must fail validation"
    assert "grid" in invalid_response.stderr.lower(), invalid_response.stderr
    assert not (wrong_destination / "external-0.ll").exists()
    assert not (wrong_destination / "manifest.json").exists()
    assert not wrong_prepared.exists()
    print("Triton 3.7.1 original LLVM IR exported/verified; optional LLVM cleanup verified; no GPU execution")
