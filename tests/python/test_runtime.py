"""CUDA Array Interface and async owner contracts without a CUDA driver."""
import copy
import gc
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import ask
from ask.cuda import CUDAError
from ask.runtime import ExecutionSession, _cuda_array_description


class ArrayOwner:
    def __init__(self, **changes):
        self.description = {"version": 3, "shape": (2, 5), "typestr": "<f4",
                            "data": (4096, False), "strides": None, "stream": 7}
        self.description.update(changes)
        self.reads = 0

    @property
    def __cuda_array_interface__(self):
        self.reads += 1
        return copy.deepcopy(self.description)


class NativeEvent:
    def __init__(self):
        self.complete = False
        self.query_hook = None

    def query(self):
        if self.query_hook:
            self.query_hook()
        return self.complete

    def wait(self):
        self.complete = True


class NativeSession:
    def __init__(self):
        self.closed = False
        self.stream = 99
        self.device = 0
        self.capability = (8, 6)
        self.stats = {}
        self.log = []
        self.events = []
        self.next_pointer = 8192
        self.close_hook = None

    def check_types(self, types):
        self.log.append(("check_types", types))

    def allocate(self, nbytes):
        pointer = self.next_pointer
        self.next_pointer += max(nbytes, 64)
        return SimpleNamespace(data_ptr=pointer, nbytes=nbytes, readonly=False)

    def import_buffer(self, pointer, nbytes, readonly):
        self.log.append(("import", pointer, nbytes, readonly))
        return SimpleNamespace(data_ptr=pointer, nbytes=nbytes, readonly=readonly)

    def wait_for_external_stream(self, stream):
        self.log.append(("wait", stream))

    def submit(self, kernel, buffers, grid, block, shared_bytes):
        self.log.append(("submit", grid, block))
        event = NativeEvent()
        self.events.append(event)
        return event

    def reap(self):
        pass

    def synchronize(self):
        self.log.append(("synchronize",))
        for event in self.events:
            event.complete = True

    def close(self):
        if self.close_hook:
            self.close_hook()
        self.synchronize()
        self.closed = True


class ArrayInterfaceTests(unittest.TestCase):
    def test_contiguous_description_and_special_stream_values(self):
        for stream in (None, 1, 2, 7340032):
            owner = ArrayOwner(strides=(20, 4), stream=stream)
            self.assertEqual(_cuda_array_description(owner), ((2, 5), "f32", 4096, False, stream))
        self.assertEqual(_cuda_array_description(ArrayOwner(shape=(0, 5), data=(0, False)))[0], (0, 5))
        self.assertEqual(_cuda_array_description(ArrayOwner(shape=(1, 5), strides=(999, 4)))[0], (1, 5))

    def test_v2_requires_a_declared_dependency_or_explicit_override(self):
        owner = ArrayOwner(version=2)
        del owner.description["stream"]
        with self.assertRaisesRegex(CUDAError, "version 2.*producer_stream"):
            _cuda_array_description(owner)
        for stream in (None, 1, 2, 7340032):
            self.assertEqual(_cuda_array_description(owner, producer_stream=stream)[4], stream)
        for stream in (0, False, -1, "stream"):
            with self.assertRaises(CUDAError):
                _cuda_array_description(owner, producer_stream=stream)
        # An explicit stream field is accepted as an extension to v2.
        owner.description["stream"] = 11
        self.assertEqual(_cuda_array_description(owner)[4], 11)
        owner.description["version"] = 3
        del owner.description["stream"]
        self.assertIsNone(_cuda_array_description(owner)[4])

    def test_invalid_storage_and_metadata_are_rejected_before_native_import(self):
        invalid = [
            {"version": True}, {"version": 1}, {"typestr": "<u2"}, {"typestr": "|V2"},
            {"typestr": []}, {"typestr": {}},
            {"typestr": ">f4"}, {"shape": (2, -1)}, {"shape": (True, 5)},
            {"data": (0, False)}, {"data": (4097, False)}, {"data": (4096, 0)},
            {"data": (2**64, False)}, {"data": (2**64 - 4, False)},
            {"strides": (4, 8)}, {"strides": (-20, 4)}, {"strides": (20,)},
            {"mask": object()}, {"stream": 0}, {"stream": True}, {"stream": -1},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises((CUDAError, ValueError)):
                _cuda_array_description(ArrayOwner(**change))


class SessionOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.native = NativeSession()
        extension = SimpleNamespace(CUDASession=SimpleNamespace(create=lambda device: self.native))
        self.native_patch = patch.object(ask, "_native", extension, create=True)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.session = ExecutionSession()
        self.addCleanup(self.session.close)

    def submit(self, *tensors):
        return self.session.submit(object(), list(tensors), grid=(1, 1, 1), block=(128, 1, 1))

    def test_empty_owned_tensor_exports_null_cai_pointer_without_changing_allocation(self):
        tensor = self.session.tensor((0, 5), "f32")
        self.assertGreater(tensor.data_ptr, 0)
        self.assertEqual(tensor.nbytes, 0)
        self.assertEqual(tensor.__cuda_array_interface__["data"], (0, False))
        imported = self.session.from_cuda_array(tensor)
        self.assertEqual(imported.data_ptr, 0)
        self.assertEqual(imported.shape, (0, 5))
        self.assertEqual(imported.nbytes, 0)

    def test_v2_override_persists_and_still_checks_owner_metadata(self):
        owner = ArrayOwner(version=2)
        del owner.description["stream"]
        with self.assertRaisesRegex(CUDAError, "version 2"):
            self.session.from_cuda_array(owner)
        self.assertFalse(any(item[0] == "import" for item in self.native.log))
        tensor = self.session.from_cuda_array(owner, producer_stream=17)
        self.assertEqual(self.native.log[-1], ("wait", 17))
        owner.description["stream"] = 29
        self.submit(tensor)
        self.assertEqual(self.native.log[-2], ("wait", 17))
        owner.description["data"] = (8192, False)
        with self.assertRaisesRegex(CUDAError, "changed after import"):
            self.submit(tensor)
        owner.description["data"] = (4096, False)
        ready = self.session.from_cuda_array(owner, producer_stream=None)
        before = len([item for item in self.native.log if item[0] == "wait"])
        self.submit(ready)
        self.assertEqual(len([item for item in self.native.log if item[0] == "wait"]), before)

    def test_each_external_access_refreshes_producer_stream_before_launch(self):
        owner = ArrayOwner()
        tensor = self.session.from_cuda_array(owner)
        self.assertEqual(self.native.log[-1], ("wait", 7))
        owner.description["stream"] = 11
        self.submit(tensor)
        self.assertEqual(self.native.log[-2:], [("wait", 11), ("submit", (1, 1, 1), (128, 1, 1))])
        self.assertEqual(owner.reads, 2)
        owner.description["stream"] = None
        self.submit(tensor)
        self.assertEqual(owner.reads, 3)
        self.assertEqual([item for item in self.native.log if item[0] == "wait"], [("wait", 7), ("wait", 11)])

    def test_changed_external_metadata_cannot_reuse_original_import(self):
        for field, replacement in (("data", (8192, False)), ("data", (4096, True)),
                                   ("shape", (10,)), ("typestr", "<i4")):
            with self.subTest(field=field, replacement=replacement):
                owner = ArrayOwner()
                tensor = self.session.from_cuda_array(owner)
                owner.description[field] = replacement
                before = len(self.native.events)
                with self.assertRaisesRegex(CUDAError, "changed after import"):
                    self.submit(tensor)
                self.assertEqual(len(self.native.events), before)

    def test_pending_submission_retains_owner_after_user_drops_tensor_and_event(self):
        owner = ArrayOwner()
        retained = weakref.ref(owner)
        tensor = self.session.from_cuda_array(owner)
        event = self.submit(tensor)
        del owner, tensor, event
        gc.collect()
        self.assertIsNotNone(retained())
        self.session.reap()
        self.assertIsNotNone(retained())
        self.native.events[0].complete = True
        self.session.reap()
        gc.collect()
        self.assertIsNone(retained())

    def test_close_waits_before_releasing_pending_external_owners(self):
        owner = ArrayOwner()
        retained = weakref.ref(owner)
        tensor = self.session.from_cuda_array(owner)
        self.submit(tensor)
        del owner, tensor
        self.native.close_hook = lambda: self.assertIsNotNone(retained())
        self.session.close()
        self.native.close_hook = None
        gc.collect()
        self.assertTrue(self.session.closed)
        self.assertIsNone(retained())

    def test_cross_session_bindings_and_readonly_writes_reject_before_launch(self):
        tensor = self.session.tensor((2, 5))
        other = ExecutionSession()
        self.addCleanup(other.close)
        with self.assertRaisesRegex(CUDAError, "different execution session"):
            other.submit(object(), [tensor], grid=(1, 1, 1), block=(128, 1, 1))
        readonly = self.session.from_cuda_array(ArrayOwner(data=(4096, True)))
        with self.assertRaisesRegex(CUDAError, "read-only"):
            self.session.submit(object(), [readonly], grid=(1, 1, 1), block=(128, 1, 1), writable_indices=(0,))
        self.assertFalse(self.native.events)

    def test_reap_serializes_pending_owner_updates_with_other_submissions(self):
        first = self.submit(self.session.tensor((2, 5)))
        entered, release, started, submitted = (threading.Event() for _ in range(4))
        failures = []
        def blocked_query():
            # Only the first thread pauses in the native call. Without the
            # session lock, the producer can query the same event and append
            # while the first thread is rebuilding the pending-owner list.
            if threading.current_thread() is a:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release query")
        self.native.events[0].query_hook = blocked_query
        def reaper():
            try:
                self.session.reap()
            except BaseException as error:
                failures.append(error)
        def producer():
            started.set()
            try:
                self.submit(self.session.tensor((2, 5)))
                submitted.set()
            except BaseException as error:
                failures.append(error)
        a = threading.Thread(target=reaper, daemon=True)
        b = threading.Thread(target=producer, daemon=True)
        a.start()
        self.assertTrue(entered.wait(2))
        b.start()
        self.assertTrue(started.wait(2))
        try:
            self.assertFalse(submitted.wait(0.05))
        finally:
            release.set()
            a.join(5)
            b.join(5)
        self.assertFalse(a.is_alive() or b.is_alive(), "session lock deadlocked")
        self.assertFalse(failures, failures)
        self.assertEqual(len(self.native.events), 2)
        self.assertEqual(len(self.session._pending), 2)
        self.assertFalse(first.query())


if __name__ == "__main__":
    unittest.main()
