"""Explicit GPU check of producer-independent CUDA ABI v2.

Uses handcrafted native LLVM IR: no external compiler, DSL, adapter or markers.
Run with --llc PATH --ptxas PATH --output DIR; not part of CPU-only CTest.
"""
import argparse
import json
from pathlib import Path

from ask import ExecutionSession
from ask.cuda import CUDAError, compile_cuda, execute
from ask.gpu import CUDAKernel


def kernel_ir(arguments, incoming, outgoing, pointer_type):
    registers = ("tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z")
    declarations = "\n".join(f"declare i32 @llvm.nvvm.read.ptx.sreg.{name}()" for name in registers)
    reads = "\n".join(f"  %{name.replace('.', '_')} = call i32 @llvm.nvvm.read.ptx.sreg.{name}()" for name in registers)
    return f'''target triple = "nvptx64-nvidia-cuda"
@static = internal addrspace(3) global [12 x i32] undef, align 4
@dynamic = external addrspace(3) global [0 x i32], align 4
{declarations}
define ptx_kernel void @generic_kernel({arguments}) {{
{reads}
  %by = mul i32 %ctaid_y, 2
  %bz = mul i32 %ctaid_z, 4
  %bxy = add i32 %ctaid_x, %by
  %b = add i32 %bxy, %bz
  %base = mul i32 %b, 12
  %ty = mul i32 %tid_y, 3
  %tz = mul i32 %tid_z, 6
  %txy = add i32 %tid_x, %ty
  %t = add i32 %txy, %tz
  %index = add i32 %base, %t
  %in = getelementptr i32, {pointer_type} %{incoming}, i32 %index
  %out = getelementptr i32, {pointer_type} %{outgoing}, i32 %index
  %sp = getelementptr [12 x i32], ptr addrspace(3) @static, i32 0, i32 %t
  %dp = getelementptr [0 x i32], ptr addrspace(3) @dynamic, i32 0, i32 %t
  %value = load i32, {pointer_type} %in, align 4
  store volatile i32 %value, ptr addrspace(3) %sp, align 4
  %v1 = load volatile i32, ptr addrspace(3) %sp, align 4
  %plus1 = add i32 %v1, 1
  store volatile i32 %plus1, ptr addrspace(3) %dp, align 4
  %v2 = load volatile i32, ptr addrspace(3) %dp, align 4
  %plus2 = add i32 %v2, 1
  store i32 %plus2, {pointer_type} %out, align 4
  ret void
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llc", required=True)
    parser.add_argument("--ptxas", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    with ExecutionSession() as session:
        cases = [("global-two", ["input:0", "output:0"], "ptr addrspace(1)"),
                 ("global-five", ["null", "output:0", "null", "input:0", "null"], "ptr addrspace(1)"),
                 ("generic-two", ["input:0", "output:0"], "ptr")]
        for name, bindings, pointer_type in cases:
            count = len(bindings)
            directory = args.output / name
            directory.mkdir()
            payload = kernel_ir(", ".join(f"{pointer_type} %arg{i}" for i in range(count)),
                                f"arg{bindings.index('input:0')}", f"arg{bindings.index('output:0')}", pointer_type)
            artifact = {"format": "llvm-ir", "entry": "generic_kernel", "payload": payload,
                        "target": {"backend": "cuda", "arch": 80},
                        "required_element_types": ["i32"], "codegen": {"fp_contract": "off", "ptx_version": 80},
                        "port_map": [{"argument": bindings.index(port), "logical_port": port,
                                      "shape": [2, 2, 2, 12], "layout": "contiguous_row_major", "element_type": "i32"}
                                     for port in ("input:0", "output:0")],
                        "invocation": {"version": 2, "kind": "cuda-kernel", "grid": [2, 2, 2],
                                       "block": [3, 2, 2], "shared_bytes": 48,
                                       "arguments": [{"argument": i, "binding": binding, "llvm_type": pointer_type}
                                                     for i, binding in enumerate(bindings)]}}
            source = directory / "input.ll"
            source.write_text(payload)
            codegen = compile_cuda(source, artifact, directory / "codegen", llc=args.llc, ptxas=args.ptxas)
            values = list(range(-48, 48))
            expected = [value + 2 for value in values]
            one_shot = execute(artifact, codegen, values)
            assert one_shot["output"] == expected
            assert len(one_shot["parameter_layout"]) == count
            package = CUDAKernel(artifact, codegen, {}, {})
            prepared = package.prepare(session)
            incoming = session.from_host(values, shape=(2, 2, 2, 12), dtype="i32")
            outgoing = session.tensor((2, 2, 2, 12), "i32")
            prepared.submit(incoming, outgoing).wait()
            assert outgoing.to_host() == expected
            before = session.stats["launches"]
            native = session.load(Path(codegen["cubin"]).read_bytes(), artifact["entry"],
                                  required_types=["i32"], arch=80, expected_parameters=count)
            parameters = [{"input:0": incoming, "output:0": outgoing, "null": None}[binding] for binding in bindings]
            try:
                session.submit(native, parameters, grid=[2, 2, 2], block=[3, 2, 2], shared_bytes=2**31)
            except CUDAError as error:
                assert "dynamic shared memory" in str(error), str(error)
            else:
                raise AssertionError("oversized shared-memory launch succeeded")
            assert session.stats["launches"] == before
            results.append({"parameters": count, "pointer_type": pointer_type, "grid": [2, 2, 2], "block": [3, 2, 2],
                            "static_shared_bytes": 48, "dynamic_shared_bytes": 48,
                            "one_shot_values_checked": len(expected), "resident_values_checked": len(expected),
                            "oversized_shared_rejected": True, "passed": True})
            (directory / "artifact.json").write_text(json.dumps(artifact, indent=2))
    (args.output / "validation.json").write_text(json.dumps({"passed": True, "cases": results}, indent=2))
    print(json.dumps({"passed": True, "cases": len(results), "values_checked": 192 * len(results)}))


if __name__ == "__main__":
    main()
