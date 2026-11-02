#!/usr/bin/env python3
"""Fetch the pinned ASK LLVM/MLIR SDK into .deps.

Uses only the Python standard library. Fetches the SDK into <repository>/.deps.
Existing directories must carry a matching receipt and required files and are
never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import sys
import tarfile
import tempfile
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
LLVM_RECEIPT = ".ask-sdk.json"
LLVM_REQUIRED_FILES = (
    "lib/cmake/llvm/LLVMConfig.cmake",
    "lib/cmake/mlir/MLIRConfig.cmake",
    "include/llvm/IR/Module.h",
    "include/mlir/IR/MLIRContext.h",
    "lib/libLLVMCore.a",
    "lib/libMLIRIR.a",
    "bin/llvm-config",
    "bin/llc",
    "bin/mlir-tblgen",
    "bin/opt",
)
_EXTRACTALL_SUPPORTS_FILTER = "filter" in inspect.signature(tarfile.TarFile.extractall).parameters


def _validate_common_lock(lock: dict, required_fields: set[str], label: str) -> dict:
    if not isinstance(lock, dict) or not required_fields.issubset(lock):
        raise ValueError(f"{label} lock is missing required fields")
    if lock["schema_version"] != 1 or not re.fullmatch(r"[a-f0-9]{64}", str(lock["sha256"])):
        raise ValueError(f"unsupported {label} lock schema or invalid SHA256")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(lock["directory"])):
        raise ValueError(f"{label} directory must be one relative path component")
    if not str(lock["url"]).startswith("https://"):
        raise ValueError(f"{label} downloads require an HTTPS URL")
    if type(lock["size_bytes"]) is not int or lock["size_bytes"] <= 0:
        raise ValueError(f"invalid {label} archive size")
    return lock


def load_llvm_lock(path: Path | str = ROOT / "cmake" / "llvm.lock.json") -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "version", "platform",
        "directory", "url", "sha256", "size_bytes", "unpacked_size_limit_bytes",
    }
    _validate_common_lock(lock, required, "LLVM")
    if type(lock["unpacked_size_limit_bytes"]) is not int or lock["unpacked_size_limit_bytes"] <= 0:
        raise ValueError("invalid extraction size limit")
    return lock


def llvm_receipt_for(lock: dict) -> dict:
    return {
        key: lock[key]
        for key in ("schema_version", "version", "directory", "url", "sha256", "size_bytes")
    }


def verify_archive(path: Path, lock: dict) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != lock["size_bytes"]:
        raise ValueError(f"archive has wrong type or size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != lock["sha256"]:
        raise ValueError(f"archive SHA256 mismatch; leaving it untouched: {path}")


def _download_archive(path: Path, lock: dict) -> None:
    if path.exists() or path.is_symlink():
        verify_archive(path, lock)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(lock["url"], headers={"User-Agent": "ASK-deps-fetch/1"})
            count, reported = 0, 0
            with urllib.request.urlopen(request, timeout=60) as response:
                geturl = getattr(response, "geturl", None)
                if callable(geturl) and not response.geturl().startswith("https://"):
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


def archive_filter(lock: dict, *, allow_links: bool = True):
    root = lock["directory"]
    limit = lock.get("unpacked_size_limit_bytes", lock["size_bytes"] * 32)
    size = 0

    def checked(member: tarfile.TarInfo, destination: str | Path):
        nonlocal size
        name = PurePosixPath(member.name)
        if (
            name.is_absolute()
            or ".." in name.parts
            or not name.parts
            or name.parts[0] != root
            or "\\" in member.name
        ):
            raise ValueError(f"archive member escapes root: {member.name}")
        is_link = member.issym() or member.islnk()
        if is_link and not allow_links:
            raise ValueError(f"links are not permitted in this archive: {member.name}")
        if not (member.isfile() or member.isdir() or is_link):
            raise ValueError(f"unsupported special archive member: {member.name}")
        if is_link:
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or "\\" in member.linkname:
                raise ValueError(f"absolute or invalid archive link: {member.name}")
            target = (
                posixpath.join(str(name.parent), member.linkname)
                if member.issym()
                else member.linkname
            )
            target = PurePosixPath(posixpath.normpath(target))
            if not target.parts or target.parts[0] != root or ".." in target.parts:
                raise ValueError(f"archive link escapes root: {member.name}")
        if member.isfile():
            size += member.size
            if size > limit:
                raise ValueError("archive exceeds locked extraction size limit")
        if hasattr(tarfile, "data_filter"):
            return tarfile.data_filter(member, str(destination))
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        if member.isdir():
            member.mode = 0o755
        elif member.isfile():
            member.mode = 0o755 if (member.mode & 0o111) else 0o644
        return member

    return checked


def _extract_archive(archive: Path, staging: Path, checker) -> None:
    with tarfile.open(archive, "r:*") as stream:
        if _EXTRACTALL_SUPPORTS_FILTER:
            stream.extractall(staging, filter=checker)
        else:
            members = [checker(member, staging) for member in stream.getmembers()]
            stream.extractall(staging, members=[m for m in members if m is not None])


def validate_llvm_sdk(path: Path, lock: dict, *, require_receipt: bool = True) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"SDK path is not a real directory: {path}")
    for name in LLVM_REQUIRED_FILES:
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError(f"SDK is incomplete: missing {candidate}")
        if not candidate.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"SDK required file points outside its directory: {candidate}")
    config = (path / "lib/cmake/llvm/LLVMConfig.cmake").read_text(encoding="utf-8")
    if not re.search(r'set\(LLVM_PACKAGE_VERSION\s+"?' + re.escape(lock["version"]) + r'"?\)', config):
        raise ValueError(f"SDK version does not match lock: {path}")
    if require_receipt:
        try:
            receipt = json.loads((path / LLVM_RECEIPT).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"existing SDK has no valid receipt; choose an empty destination: {path}") from exc
        if receipt != llvm_receipt_for(lock):
            raise ValueError(f"existing SDK belongs to a different lock; refusing overwrite: {path}")


def fetch_llvm(lock: dict, destination: Path | str) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / lock["directory"]
    if target.exists() or target.is_symlink():
        validate_llvm_sdk(target, lock)
        return target
    archive = destination / PurePosixPath(urllib.parse.urlsplit(lock["url"]).path).name
    _download_archive(archive, lock)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".ask-llvm-extract-") as temporary:
        staging = Path(temporary)
        _extract_archive(archive, staging, archive_filter(lock, allow_links=True))
        extracted = staging / lock["directory"]
        validate_llvm_sdk(extracted, lock, require_receipt=False)
        (extracted / LLVM_RECEIPT).write_text(
            json.dumps(llvm_receipt_for(lock), indent=2) + "\n", encoding="utf-8"
        )
        if target.exists() or target.is_symlink():
            raise ValueError(f"SDK appeared during extraction; refusing overwrite: {target}")
        extracted.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / ".deps")
    parser.add_argument("--only", choices=("llvm",), default="llvm",
                        help="fetch the pinned LLVM/MLIR SDK")
    parser.add_argument("--llvm-lock", type=Path, default=ROOT / "cmake" / "llvm.lock.json")
    args = parser.parse_args(argv)
    try:
        if args.only in (None, "llvm"):
            llvm_lock = load_llvm_lock(args.llvm_lock)
            if (
                llvm_lock["platform"] != "linux-x86_64"
                or sys.platform != "linux"
                or platform.machine() not in ("x86_64", "AMD64")
            ):
                raise ValueError("this LLVM SDK lock supports only Linux x86-64")
            print(fetch_llvm(llvm_lock, args.destination))
        return 0
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"ASK dependency fetch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
