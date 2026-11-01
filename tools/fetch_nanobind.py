#!/usr/bin/env python3
"""Fetch the SHA256-locked nanobind source into .deps, outside CMake configure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(lock, destination):
    if not hasattr(tarfile, "data_filter"):
        raise ValueError("safe extraction requires Python with tarfile.data_filter")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / lock["directory"]
    receipt = target / ".ask-nanobind.json"
    if target.exists():
        if not receipt.is_file() or json.loads(receipt.read_text()) != lock:
            raise ValueError(f"existing nanobind does not match lock: {target}")
        if not (target / "cmake/nanobind-config.cmake").is_file():
            raise ValueError(f"existing nanobind is incomplete: {target}")
        return target
    with tempfile.TemporaryDirectory(dir=destination, prefix=".ask-nanobind-") as directory:
        staging = Path(directory)
        archive = staging / "source.tar.gz"
        with urllib.request.urlopen(lock["url"], timeout=60) as response:
            contents = response.read(lock["size_bytes"] + 1)
        if len(contents) != lock["size_bytes"] or hashlib.sha256(contents).hexdigest() != lock["sha256"]:
            raise ValueError("nanobind archive size or SHA256 does not match lock")
        archive.write_bytes(contents)

        def checked(member, destination):
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != lock["directory"] or ".." in parts or "\\" in member.name:
                raise ValueError("nanobind archive escapes its root")
            if not (member.isfile() or member.isdir()):
                raise ValueError("nanobind source archive has an unsupported entry")
            return tarfile.data_filter(member, destination)

        with tarfile.open(archive) as source:
            source.extractall(staging, filter=checked)
        extracted = staging / lock["directory"]
        if not (extracted / "cmake/nanobind-config.cmake").is_file():
            raise ValueError("nanobind archive is missing CMake configuration")
        (extracted / receipt.name).write_text(json.dumps(lock, indent=2) + "\n")
        extracted.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / ".deps")
    args = parser.parse_args()
    lock = json.loads((ROOT / "cmake/nanobind.lock.json").read_text())
    print(fetch(lock, args.destination))


if __name__ == "__main__":
    main()
