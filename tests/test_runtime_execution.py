"""Explicit GPU lifecycle/thread checks against an existing verified raw package.

Usage: PYTHONPATH=python:build/core/python python3 tests/test_runtime_execution.py
  --artifact build/raw-bf16-gpu/artifact.json
  --codegen build/raw-bf16-gpu/codegen/codegen.json
  --reference build/raw-bf16-gpu/verification.json --output build/runtime-lifecycle.json

The reference fixture is produced by the independent shared numeric oracle.
This test reuses the supplied cubin and never rebuilds a compiler or kernel.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import gc
import json
import math
from pathlib import Path
import threading
import weakref

from ask import ExecutionSession
from ask.gpu import CUDAKernel


def numeric(value):
    return {"NaN": math.nan, "+Inf": math.inf, "-Inf": -math.inf}[value] if isinstance(value, str) else value


def check(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert (math.isnan(left) and math.isnan(right)) or (
            left == right and (left != 0 or math.copysign(1, left) == math.copysign(1, right))), (left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "codegen", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    codegen = json.loads(args.codegen.read_text())
    reference = json.loads(args.reference.read_text())
    assert reference["passed"]
    values = [numeric(row["input_f32"]) for row in reference["cases"]]
    expected = [numeric(row["expected_f32"]) for row in reference["cases"]]
    package = CUDAKernel(artifact, codegen, {}, {})
    checks = {}
    faulthandler.dump_traceback_later(30, exit=True)
    try:
        with ExecutionSession() as session:
            empty = session.tensor((0, 5), "f32")
            assert empty.data_ptr != 0 and empty.nbytes == 0
            assert empty.__cuda_array_interface__["data"] == (0, False)
            with ExecutionSession() as consumer:
                imported_empty = consumer.from_cuda_array(empty)
                assert imported_empty.data_ptr == 0 and imported_empty.shape == (0, 5)
                assert imported_empty.to_host() == []
            assert empty.data_ptr != 0 and empty.to_host() == []
            checks["empty_tensor_cai_null_pointer_roundtrip"] = True
            prepared = package.prepare(session)
            source = session.from_host(values, dtype="f32")
            output = session.tensor((len(values),), "f32")
            before = session.stats
            for _ in range(20):
                event = prepared.submit(source, output)
            event.wait()
            after = session.stats
            for key in ("allocations", "module_loads", "stream_creations", "primary_context_retains",
                        "host_to_device_copies", "device_to_host_copies", "synchronizations"):
                assert after[key] == before[key], (key, before, after)
            assert after["launches"] - before["launches"] == 20
            check(output.to_host(), expected)
            checks["resident_twenty_submissions"] = {"passed": True, "before": before, "after": after}

            class Owner:
                def __init__(self, tensor):
                    self.tensor = tensor
                @property
                def __cuda_array_interface__(self):
                    return self.tensor.__cuda_array_interface__

            owner = Owner(session.from_host(values, dtype="f32"))
            owner_ref = weakref.ref(owner)
            borrowed = session.from_cuda_array(owner)
            event = prepared.submit(borrowed, output)
            del owner, borrowed
            gc.collect()
            assert owner_ref() is not None
            event.wait()
            gc.collect()
            assert owner_ref() is None
            check(output.to_host(), expected)
            checks["external_owner_retained_until_completion"] = True

            stop = threading.Event()
            barrier = threading.Barrier(4)
            outputs = [session.tensor((len(values),), "f32") for _ in range(3)]
            def submit_and_wait(index):
                barrier.wait(timeout=5)
                for iteration in range(40):
                    completion = prepared.submit(source, outputs[index])
                    if iteration % 4 == 0:
                        completion.wait()
                completion.wait()
                return outputs[index].to_host()
            def read_properties():
                barrier.wait(timeout=5)
                count = 0
                while not stop.is_set():
                    assert not session.closed
                    assert session.stream > 0
                    assert session.stats["resident_modules"] == 1
                    count += 1
                return count
            with ThreadPoolExecutor(max_workers=4) as threads:
                submissions = [threads.submit(submit_and_wait, index) for index in range(3)]
                properties = threads.submit(read_properties)
                try:
                    for future in submissions:
                        check(future.result(timeout=15), expected)
                finally:
                    stop.set()
                polls = properties.result(timeout=5)
            assert polls > 0
            checks["cross_thread_wait_query_properties"] = {"passed": True, "submissions": 120,
                                                            "property_poll_iterations": polls}

        # Python tensors/events refer back weakly. Dropping the session runs its
        # close/wait even when the completion event and native buffers survive.
        disposable = ExecutionSession()
        prepared = package.prepare(disposable)
        source = disposable.from_host(values, dtype="f32")
        output = disposable.tensor((len(values),), "f32")
        completion = prepared.submit(source, output)
        session_ref = weakref.ref(disposable)
        del prepared, disposable, source, output
        gc.collect()
        assert session_ref() is None
        completion.wait()
        assert completion.query()
        checks["event_survives_session_deletion"] = True
    finally:
        faulthandler.cancel_dump_traceback_later()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "checks": checks}, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": list(checks)}))


if __name__ == "__main__":
    main()
