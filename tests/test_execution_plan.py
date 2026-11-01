"""Plan SSA/ABI checks, portable packages and asynchronous multi-step lifetime."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ask.cuda import CUDAError
from ask.gpu import _unsupported_fusion
from ask.plan import CUDAProgram, validate_plan
from ask.runtime import ExecutionSession
import test_cuda_compilation as compilation_fixtures
from test_runtime_and_cache import ArrayOwner, FakeNativeSession


def sample_plan():
    return {"schema_version": 1, "kind": "ask.cuda.execution_plan",
            "inputs": ["x"], "outputs": ["z"],
            "buffers": [{"id": name, "shape": [2, 5], "element_type": "i32"} for name in ("x", "y", "z")],
            "steps": [{"id": "first", "kind": "owned", "function": "first", "inputs": ["x"],
                       "outputs": ["y"], "depends_on": []},
                      {"id": "second", "kind": "owned", "function": "second", "inputs": ["y"],
                       "outputs": ["z"], "depends_on": ["first"]}],
            "scheduling": {"policy": "sequential", "stream": "session"}}


def session_fixture():
    session = ExecutionSession.__new__(ExecutionSession)
    session._lock, session._pending = threading.RLock(), []
    session._native = FakeNativeSession()
    session._native.load = lambda *args: object()
    return session


class ExecutionPlanTests(unittest.TestCase):
    def make_program(self, directory, plan=None):
        kernel, _ = compilation_fixtures.PreparedExecutableContractTests().make_kernel_package(directory)
        return CUDAProgram(plan or sample_plan(), {"first": kernel, "second": kernel})

    def test_schema_ssa_dependency_and_abi_rejections(self):
        mutations = [
            lambda p: p.update(schema_version=True), lambda p: p.update(kind="future"),
            lambda p: p.update(unrecognized=1),
            lambda p: p["scheduling"].update(stream="default"),
            lambda p: p["buffers"][0].update(shape=[True]),
            lambda p: p["buffers"][0].update(shape=[2**63]),
            lambda p: p["buffers"][0].update(element_type="f64"),
            lambda p: p["buffers"].append(copy.deepcopy(p["buffers"][0])),
            lambda p: p["steps"][0].update(id="../escape"),
            lambda p: p["steps"][0].update(inputs=["y"]),
            lambda p: p["steps"][0].update(outputs=["x"]),
            lambda p: p["steps"][1].update(depends_on=[]),
            lambda p: p["steps"][1].update(depends_on=["first", "first"]),
            lambda p: p.update(outputs=["undefined"]),
        ]
        for mutate in mutations:
            plan = sample_plan()
            mutate(plan)
            with self.subTest(plan=plan), self.assertRaises(CUDAError):
                validate_plan(plan)
        with tempfile.TemporaryDirectory() as directory:
            plan = sample_plan()
            plan["buffers"][1]["element_type"] = "f32"
            with self.assertRaisesRegex(CUDAError, "ABI"):
                self.make_program(Path(directory), plan)

    def test_package_relocation_digests_and_path_confinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = self.make_program(root)
            program.save(root / "package")
            shutil.move(root / "package", root / "relocated")
            loaded = CUDAProgram.load(root / "relocated")
            with session_fixture() as session:
                self.assertEqual(loaded.prepare(session).inputs[0]["shape"], [2, 5])
            manifest = root / "relocated/program.json"
            receipt = json.loads(manifest.read_text())
            original = copy.deepcopy(receipt)
            receipt["steps"]["second"]["files"]["cubin"]["path"] = "../k.cubin"
            manifest.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(CUDAError, "escapes"):
                CUDAProgram.load(root / "relocated")
            manifest.write_text(json.dumps(original))
            (root / "relocated/package/second/kernel.cubin").write_bytes(b"tampered")
            with self.assertRaisesRegex(CUDAError, "digest"):
                CUDAProgram.load(root / "relocated")

    def test_all_ports_checked_before_any_launch(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session, session_fixture() as other:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            for invalid in (other.tensor((2, 5), "i32"), session.tensor((10,), "i32"), x):
                with self.assertRaises(CUDAError):
                    executable.submit(x, invalid)
                self.assertEqual(session._native.events, [])
            owner = ArrayOwner(shape=(2, 5), typestr="<i4")
            borrowed = session.from_cuda_array(owner)
            owner.description["data"] = (4096, True)
            with self.assertRaisesRegex(CUDAError, "changed"):
                executable.submit(borrowed, z)
            self.assertEqual(session._native.events, [])

    def test_temporaries_retained_by_final_event_and_released_after_completion(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            tracked = []
            allocate = session.tensor
            def record(*args):
                tensor = allocate(*args)
                tracked.append(weakref.ref(tensor))
                return tensor
            with patch.object(session, "tensor", side_effect=record), patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                event = executable.submit(x, z)
            self.assertEqual(len(session._native.events), 2)
            session._native.events[0].complete = True
            session.reap()
            del event, executable
            gc.collect()
            self.assertIsNotNone(tracked[0]())
            session._native.events[-1].complete = True
            session.reap()
            gc.collect()
            self.assertIsNone(tracked[0]())

    def test_partial_submission_failure_keeps_queued_temporary_alive(self):
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory)).prepare(session)
            x, z = session.tensor((2, 5), "i32"), session.tensor((2, 5), "i32")
            original = session._native.submit
            calls = 0
            def submit(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise CUDAError("device launch failed")
                return original(*args)
            with patch.object(session._native, "submit", side_effect=submit):
                with self.assertRaisesRegex(CUDAError, "device launch failed"):
                    executable.submit(x, z)
            temporary = weakref.ref(session._pending[0]._references[-1])
            gc.collect()
            self.assertIsNotNone(temporary())
            session.synchronize()
            gc.collect()
            self.assertIsNone(temporary())

    def test_output_aliases_and_zero_step_identity(self):
        plan = sample_plan()
        plan["outputs"] = ["z", "z", "x"]
        with tempfile.TemporaryDirectory() as directory, session_fixture() as session:
            executable = self.make_program(Path(directory), plan).prepare(session)
            x, z, other = [session.tensor((2, 5), "i32") for _ in range(3)]
            with self.assertRaisesRegex(CUDAError, "same device storage"):
                executable.submit_many([x], [z, other, x])
            self.assertFalse(session._native.events)
            executable.submit_many([x], [z, z, x]).wait()
            identity = dict(plan, outputs=["x", "x"], steps=[], buffers=[plan["buffers"][0]])
            complete = CUDAProgram(identity, {}).prepare(session).submit_many([x], [x, x])
            self.assertTrue(complete.query())
            self.assertIs(complete.wait(), complete)

    def test_only_explicit_unsupported_fusion_is_retryable(self):
        self.assertTrue(_unsupported_fusion('ask-opt: UnsupportedFusion: boundary proof failed'))
        self.assertTrue(_unsupported_fusion(json.dumps({"error": {"code": "UnsupportedFusion", "stage": "fusion"}})))
        for error in ('invalid LLVM', 'worker failed: UnsupportedFusion somewhere',
                      '{"error":{"code":"InvalidRequest","stage":"fusion"}}'):
            self.assertFalse(_unsupported_fusion(error))


if __name__ == "__main__":
    unittest.main()
