"""Persistent content-addressed artifacts, independent of compiler caches.

Callers provide all compilation inputs/fingerprints in the key. This module checks integrity;
it neither authorizes a compiler nor proves an artifact's semantic correctness.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Iterator
import weakref


_FORMAT_VERSION = 1
_REQUIRED_KEY_FIELDS = frozenset({"schema_version", "pipeline_version"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_thread_guard = threading.Lock()
_thread_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_held_locks = threading.local()


class _InvalidEntry(Exception):
    pass


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache key object members must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError("cache keys must contain only JSON values and finite numbers")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _key_bytes(key: dict[str, Any]) -> bytes:
    if not isinstance(key, dict):
        raise TypeError("cache key must be a dictionary")
    missing = _REQUIRED_KEY_FIELDS - key.keys()
    if missing:
        raise ValueError("cache key is missing: " + ", ".join(sorted(missing)))
    if type(key["schema_version"]) is not int or key["schema_version"] < 1:
        raise ValueError("schema_version must be a positive integer")
    for name in ("pipeline_version",):
        if not isinstance(key[name], str) or not key[name]:
            raise ValueError(f"{name} must be a nonempty string")
    return _json_bytes(_json_value(key))


def _artifact_name(name: Any) -> str:
    if not isinstance(name, str) or not name or any(character in name for character in '\\:<>"|?*') or any(ord(character) < 32 for character in name):
        raise ValueError("artifact names must be relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"unsafe artifact path: {name!r}")
    if not path.parts:
        raise ValueError("artifact names must identify a file")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(prefix + suffix for prefix in ("COM", "LPT") for suffix in "123456789¹²³")
    for part in path.parts:
        if part.endswith((".", " ")) or part.split(".", 1)[0].upper() in reserved:
            raise ValueError(f"artifact path is not portable: {name!r}")
    return name


def _validate_names(names: Any) -> set[str]:
    result = {_artifact_name(name) for name in names}
    for name in result:
        if any(parent.as_posix() in result for parent in PurePosixPath(name).parents
               if parent.as_posix() != "."):
            raise ValueError("an artifact file cannot also be another artifact's directory")
    return result


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"cache directory must not be a symlink or file: {path}")


def _regular_bytes(path: Path) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise _InvalidEntry(f"not a regular cache file: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
                         getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if path.is_symlink() or not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise _InvalidEntry(f"not a regular cache file: {path.name}")
        return stream.read()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON manifest member")
        result[key] = value
    return result


def _discard(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _entry_lock(path: Path) -> Iterator[None]:
    name = str(path)
    held = getattr(_held_locks, "paths", set())
    if name in held:
        raise RuntimeError("a cache builder cannot recursively request the same key")
    with _thread_guard:
        lock = _thread_locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[name] = lock
    with lock:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if path.is_symlink() or not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("cache lock must be a regular file")
            if os.name == "posix":
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            elif os.name == "nt":
                import msvcrt
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                while True:
                    try:
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.05)
            else:
                raise NotImplementedError("cache process locks require POSIX or Windows")
            _held_locks.paths = held | {name}
            try:
                yield
            finally:
                _held_locks.paths = held
        finally:
            # Closing releases the OS lock, including when the builder raises.
            # Keep the lock file: unlinking it permits two locked inodes per key.
            os.close(descriptor)


class IRCache:
    """A disk cache of named byte artifacts with per-key process locking.

    ``get_or_create`` returns ``(artifacts, stats)``. Counters describe one
    completed request, not process-global totals. An ``invalid`` result means a
    corrupt entry was detected and rebuilt; it counts as both one miss and one
    invalid entry. Builder exceptions propagate unchanged and publish nothing.

    Key objects must explicitly include every required fingerprint field;
    graph/DSL entries may contain full inputs or their caller-computed digests.
    JSON object order is ignored, tuple/list shapes are equivalent, and NaN,
    infinities, non-string object keys and arbitrary Python objects are rejected.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._entries = self.root / "entries"
        self._locks = self.root / "locks"
        self._temporary = self.root / "tmp"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for path in (self.root, self._entries, self._locks, self._temporary):
            _ensure_directory(path)

    def _read(self, entry: Path, digest: str, key_data: bytes) -> dict[str, bytes] | None:
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            return None
        try:
            if not stat.S_ISDIR(mode):
                raise _InvalidEntry("entry is not a directory")
            if {child.name for child in entry.iterdir()} != {"manifest.json", "manifest.sha256", "artifacts"}:
                raise _InvalidEntry("entry contains missing or unexpected files")
            manifest_data = _regular_bytes(entry / "manifest.json")
            if _regular_bytes(entry / "manifest.sha256") != hashlib.sha256(manifest_data).hexdigest().encode() + b"\n":
                raise _InvalidEntry("manifest checksum mismatch")
            manifest = json.loads(manifest_data, object_pairs_hook=_unique_object)
            if not isinstance(manifest, dict) or set(manifest) != {"format_version", "key", "key_sha256", "artifacts"}:
                raise _InvalidEntry("invalid manifest structure")
            if type(manifest["format_version"]) is not int or manifest["format_version"] != _FORMAT_VERSION:
                raise _InvalidEntry("unsupported cache format version")
            if manifest["key_sha256"] != digest or _key_bytes(manifest["key"]) != key_data:
                raise _InvalidEntry("manifest key mismatch")
            records = manifest["artifacts"]
            if not isinstance(records, dict) or not records:
                raise _InvalidEntry("manifest has no artifact records")
            names = _validate_names(records)
            directory = entry / "artifacts"
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise _InvalidEntry("artifact directory is missing or is a symlink")
            actual_files: set[str] = set()
            actual_directories: set[str] = set()
            def failed_walk(error: OSError) -> None:
                raise error
            for current, subdirectories, files in os.walk(directory, followlinks=False, onerror=failed_walk):
                for name in subdirectories:
                    child = Path(current) / name
                    if not stat.S_ISDIR(child.lstat().st_mode):
                        raise _InvalidEntry("artifact tree contains a symlink")
                    actual_directories.add(child.relative_to(directory).as_posix())
                for name in files:
                    actual_files.add((Path(current) / name).relative_to(directory).as_posix())
            expected_directories = {parent.as_posix() for name in names
                                    for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
            if actual_files != names or actual_directories != expected_directories:
                raise _InvalidEntry("artifact inventory mismatch")
            artifacts = {}
            for name in sorted(names):
                record = records[name]
                if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
                    raise _InvalidEntry(f"invalid artifact record: {name}")
                if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
                    raise _InvalidEntry(f"invalid artifact size: {name}")
                if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
                    raise _InvalidEntry(f"invalid artifact digest: {name}")
                payload = _regular_bytes(directory / name)
                if len(payload) != record["size_bytes"]:
                    raise _InvalidEntry(f"artifact size mismatch: {name}")
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise _InvalidEntry(f"artifact checksum mismatch: {name}")
                artifacts[name] = payload
            return artifacts
        except _InvalidEntry:
            raise
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
            raise _InvalidEntry(f"invalid cache entry: {error}") from error

    def _publish(self, digest: str, key_data: bytes, artifacts: dict[str, bytes]) -> None:
        staging = Path(tempfile.mkdtemp(prefix=digest + ".", dir=self._temporary))
        entry = self._entries / digest
        published = False
        try:
            payload_root = staging / "artifacts"
            payload_root.mkdir(mode=0o700)
            records = {}
            for name, payload in sorted(artifacts.items()):
                destination = payload_root / name
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                records[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            manifest = _json_bytes({"format_version": _FORMAT_VERSION,
                                    "key": json.loads(key_data), "key_sha256": digest,
                                    "artifacts": records})
            for name, payload in (("manifest.json", manifest),
                                  ("manifest.sha256", hashlib.sha256(manifest).hexdigest().encode() + b"\n")):
                with (staging / name).open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            for current, _, _ in os.walk(staging, topdown=False):
                _sync_directory(Path(current))
            # Validate our complete on-disk representation before publication.
            self._read(staging, digest, key_data)
            os.replace(staging, entry)
            published = True
            _sync_directory(self._entries)
        except BaseException:
            if published:
                _discard(entry)
            raise
        finally:
            _discard(staging)

    def get_or_create(self, key: dict[str, Any], builder: Callable[[], dict[str, bytes]]) -> tuple[dict[str, bytes], dict[str, Any]]:
        key_data = _key_bytes(key)
        digest = hashlib.sha256(key_data).hexdigest()
        self._ensure_layout()
        with _entry_lock(self._locks / (digest + ".lock")):
            status, reason = "miss", "entry_not_found"
            try:
                artifacts = self._read(self._entries / digest, digest, key_data)
            except _InvalidEntry as error:
                status, reason = "invalid", str(error)
                artifacts = None
                _discard(self._entries / digest)
            if artifacts is not None:
                status, reason = "hit", None
            else:
                built = builder()
                if not isinstance(built, dict) or not built:
                    raise TypeError("cache builder must return a nonempty dict of artifact bytes")
                _validate_names(built)
                if any(not isinstance(payload, bytes) for payload in built.values()):
                    raise TypeError("cache artifact payloads must be bytes")
                artifacts = dict(built)
                self._publish(digest, key_data, artifacts)
            return artifacts, {
                "key": digest, "status": status, "reason": reason,
                "counters": {"requests": 1, "hits": int(status == "hit"),
                             "misses": int(status != "hit"), "invalid": int(status == "invalid"),
                             "builds": int(status != "hit")},
            }
