"""用一个薄 worker 将现成 PTX 接入 ASK 的三步执行计划。

安装 ASK 后运行：
  python examples/external_ptx.py --llc /path/to/llc --ptxas /path/to/ptxas

计算 y = (x + 1) * 2 + 3；前后两步由 ASK 编译，中间 PTX 由驱动 JIT。
同一脚本的 --probe / --request 模式仅使用标准库，不导入 ASK。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


PTX = '''.version 8.0
.target sm_80
.address_size 64
.visible .entry scale2(.param .u64 input_ptr, .param .u64 output_ptr) {
    .reg .pred %p;
    .reg .b32 %i, %v, %r;
    .reg .b64 %x, %y, %offset, %xp, %yp;
    ld.param.u64 %x, [input_ptr];
    ld.param.u64 %y, [output_ptr];
    mov.u32 %i, %tid.x;
    setp.ge.u32 %p, %i, 10;
    @%p bra done;
    mul.wide.u32 %offset, %i, 4;
    add.u64 %xp, %x, %offset;
    add.u64 %yp, %y, %offset;
    ld.global.u32 %v, [%xp];
    add.u32 %r, %v, %v;
    st.global.u32 [%yp], %r;
done:
    ret;
}
'''


def worker(arguments):
    producer = {"name": "ptx_demo", "version": "1"}
    if arguments.probe:
        print(json.dumps({"producer": producer, "adapter_version": "1"}))
        return
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    if request["protocol_version"] != 2 or request["port_shapes"] != {"input:0": [10], "output:0": [10]}:
        raise ValueError("this example requires protocol v2 and two length-10 tensor ports")
    artifact = {"protocol_version": 2, "format": "ptx", "payload": request["source"],
        "entry": request["entry"], "target": request["target"], "producer": producer,
        "launch": {"grid": request["grid"], "block": [128, 1, 1], "shared_bytes": 0},
        "arguments": ["input:0", "output:0"], "required_element_types": ["i32"]}
    temporary = arguments.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    temporary.replace(arguments.output)


def demo(arguments):
    from ask import Compiler, CompilerEnvironment, ExecutionSession, Module, TensorType

    with tempfile.TemporaryDirectory(prefix="ask-external-ptx-") as directory, ExecutionSession() as session:
        arch = session.capability[0] * 10 + session.capability[1]
        if arch < 80:
            raise RuntimeError("this PTX example requires CUDA compute capability 8.0 or newer")
        module, tensor = Module(), TensorType((10,), "i32")
        semantics = module.semantics("scale2", inputs=[tensor], results=[tensor])
        implementation = module.implementation("ptx_scale2", compiler="ptx_demo", source=PTX,
            entry="scale2", signature={"x": "*i32", "y": "*i32"},
            source_bindings={"x": "input:0", "y": "output:0"}, constants={}, options={},
            target={"backend": "cuda", "arch": arch, "warp_size": 32}, grid=(1, 1, 1))
        graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
        first = graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (10,)))
        middle = graph.external(first, semantics=semantics, implementation=implementation)
        graph.return_(graph.add(middle, graph.splat(graph.constant(3), (10,))))
        root = Path(directory)
        program = Compiler(arguments.ask_opt).compile_cuda(module,
            compilers={"ptx_demo": CompilerEnvironment(python=sys.executable, worker=Path(__file__).resolve())},
            llc=arguments.llc, ptxas=arguments.ptxas,
            cache_dir=root / "cache", output_dir=root / "program")
        steps = [step["kind"] for step in program.plan["steps"]]
        assert steps == ["owned", "external", "owned"], steps
        x = session.from_host(list(range(10)), dtype="i32")
        y = session.tensor((10,), "i32")
        before = session.stats["launches"]
        program.prepare(session).submit(x, y).wait()
        result, launches = y.to_host(), session.stats["launches"] - before
        assert result == [2 * value + 5 for value in range(10)] and launches == 3
        print(json.dumps({"steps": steps, "launches": launches, "output": result}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ask-opt", help="optional ask-opt path; defaults to installed native bindings")
    parser.add_argument("--llc", help="LLVM compiler for the two ASK-owned steps")
    parser.add_argument("--ptxas", help="PTX assembler for the two ASK-owned steps")
    arguments = parser.parse_args()
    if arguments.probe or arguments.request:
        if not arguments.probe and arguments.output is None:
            parser.error("--request requires --output")
        worker(arguments)
    else:
        if not arguments.llc or not arguments.ptxas:
            parser.error("the demo requires --llc and --ptxas for its ASK-owned steps")
        demo(arguments)


if __name__ == "__main__":
    main()
