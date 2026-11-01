#!/usr/bin/env python3
"""Fetch the pinned Linux x86-64 LLVM/MLIR SDK using only the Python stdlib.

The default destination is <repository>/.deps. --destination changes the parent
directory; the SDK always occupies the directory named in cmake/llvm.lock.json.
An existing SDK must carry a matching receipt and required files. It is never
overwritten to repair a mismatch. Python needs tarfile.data_filter (3.12+ or a
Python release with its security backport).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ".ask-sdk.json"
REQUIRED_FILES = (
    "lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/mlir/MLIRConfig.cmake",
    "include/llvm/IR/Module.h", "include/mlir/IR/MLIRContext.h",
    "lib/libLLVMCore.a", "lib/libMLIRIR.a", "bin/llvm-config",
    "bin/mlir-tblgen", "bin/opt",
)


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "version", "cmake_version", "revision", "platform",
                "directory", "url", "sha256", "size_bytes", "unpacked_size_limit_bytes"}
    if not isinstance(lock, dict) or not required.issubset(lock):
        raise ValueError("LLVM lock is missing required fields")
    if lock["schema_version"] != 1 or not re.fullmatch(r"[a-f0-9]{64}", lock["sha256"]):
        raise ValueError("unsupported lock schema or invalid SHA256")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", lock["directory"]):
        raise ValueError("SDK directory must be one relative path component")
    if not lock["url"].startswith("https://"):
        raise ValueError("SDK downloads require an HTTPS URL")
    if type(lock["size_bytes"]) is not int or lock["size_bytes"] <= 0:
        raise ValueError("invalid archive size")
    if type(lock["unpacked_size_limit_bytes"]) is not int or lock["unpacked_size_limit_bytes"] <= 0:
        raise ValueError("invalid extraction size limit")
    return lock


def receipt_for(lock):
    return {key: lock[key] for key in ("schema_version", "version", "revision", "directory",
                                     "url", "sha256", "size_bytes")}


def verify_archive(path, lock):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != lock["size_bytes"]:
        raise ValueError(f"archive has wrong type or size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != lock["sha256"]:
        raise ValueError(f"archive SHA256 mismatch; leaving it untouched: {path}")


def validate_sdk(path, lock, require_receipt=True):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"SDK path is not a real directory: {path}")
    for name in REQUIRED_FILES:
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError(f"SDK is incomplete: missing {candidate}")
        if not candidate.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"SDK required file points outside its directory: {candidate}")
    config = (path / "lib/cmake/llvm/LLVMConfig.cmake").read_text(encoding="utf-8")
    if not re.search(r"set\(LLVM_PACKAGE_VERSION\s+" + re.escape(lock["version"]) + r"\)", config):
        raise ValueError(f"SDK version does not match lock: {path}")
    if require_receipt:
        try:
            receipt = json.loads((path / RECEIPT).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"existing SDK has no valid receipt; choose an empty destination: {path}") from exc
        if receipt != receipt_for(lock):
            raise ValueError(f"existing SDK belongs to a different lock; refusing overwrite: {path}")


def _download_archive(path, lock):
    if path.exists() or path.is_symlink():
        verify_archive(path, lock)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part",
                                         delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(lock["url"], headers={"User-Agent": "ASK-SDK-fetch/1"})
            count, reported = 0, 0
            with urllib.request.urlopen(request, timeout=60) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("download redirected to a non-HTTPS URL")
                while chunk := response.read(4 * 1024 * 1024):
                    count += len(chunk)
                    if count > lock["size_bytes"]:
                        raise ValueError("download is larger than the locked archive")
                    output.write(chunk)
                    if count - reported >= 64 * 1024 * 1024:
                        print(f"Downloaded {count}/{lock['size_bytes']} bytes", file=sys.stderr)
                        reported = count
        verify_archive(temporary, lock)
        if path.exists() or path.is_symlink():
            verify_archive(path, lock)
        else:
            temporary.rename(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def archive_filter(lock):
    if not hasattr(tarfile, "data_filter"):
        raise ValueError("safe extraction requires Python with tarfile.data_filter")
    root = lock["directory"]
    size = 0

    def checked(member, destination):
        nonlocal size
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != root \
                or "\\" in member.name:
            raise ValueError(f"archive member escapes SDK root: {member.name}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported special archive member: {member.name}")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or "\\" in member.linkname:
                raise ValueError(f"absolute or invalid archive link: {member.name}")
            target = (posixpath.join(str(name.parent), member.linkname)
                      if member.issym() else member.linkname)
            target = PurePosixPath(posixpath.normpath(target))
            if not target.parts or target.parts[0] != root or ".." in target.parts:
                raise ValueError(f"archive link escapes SDK root: {member.name}")
        if member.isfile():
            size += member.size
            if size > lock["unpacked_size_limit_bytes"]:
                raise ValueError("archive exceeds locked extraction size limit")
        return tarfile.data_filter(member, destination)

    return checked


def fetch(lock, destination):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / lock["directory"]
    if target.exists() or target.is_symlink():
        validate_sdk(target, lock)
        return target
    # Check safe-extraction support before a potentially large download.
    archive_filter(lock)
    archive = destination / (lock["directory"] + ".tar.gz")
    _download_archive(archive, lock)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".ask-llvm-extract-") as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(staging, filter=archive_filter(lock))
        extracted = staging / lock["directory"]
        validate_sdk(extracted, lock, require_receipt=False)
        (extracted / RECEIPT).write_text(json.dumps(receipt_for(lock), indent=2) + "\n",
                                       encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise ValueError(f"SDK appeared during extraction; refusing overwrite: {target}")
        extracted.rename(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / ".deps")
    parser.add_argument("--lock", type=Path, default=ROOT / "cmake" / "llvm.lock.json")
    args = parser.parse_args(argv)
    try:
        lock = load_lock(args.lock)
        if lock["platform"] != "linux-x86_64" or sys.platform != "linux" \
                or platform.machine() not in ("x86_64", "AMD64"):
            raise ValueError("this SDK lock supports only Linux x86-64")
        print(fetch(lock, args.destination))
        return 0
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"LLVM SDK fetch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
