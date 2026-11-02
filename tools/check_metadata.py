#!/usr/bin/env python3
"""Check release metadata and, optionally, the built core sdist and wheel.

Run in the build environment (Python 3.10 uses scikit-build-core's tomli dependency).
"""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_version(source):
    assignments = [node for node in ast.parse(source).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__"
                           for target in node.targets)]
    require(len(assignments) == 1, "expected one __version__ literal")
    version = ast.literal_eval(assignments[0].value)
    require(isinstance(version, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version),
            "release version must be major.minor.patch without leading zeros")
    return version


def check_source():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = read_version((ROOT / "python/ask/_version.py").read_text())
    require("version" not in config["project"] and "version" in config["project"]["dynamic"],
            "pyproject must derive its version from python/ask/_version.py")
    providers = config["tool"]["dynamic-metadata"]
    require(any(p.get("field") == "version" and p.get("input") == "python/ask/_version.py"
                and p.get("provider") == "scikit_build_core.metadata.regex" for p in providers),
            "version metadata provider must read python/ask/_version.py")
    nanobind = json.loads((ROOT / "cmake/nanobind.lock.json").read_text())["version"]
    require(f"nanobind=={nanobind}" in config["build-system"]["requires"],
            "build-system nanobind pin differs from cmake/nanobind.lock.json")
    return version, config["project"]["requires-python"]


def check_message(data, version, requires_python):
    metadata = BytesParser().parsebytes(data)
    for field, value in (("Name", "ask-compiler"), ("Version", version),
                         ("Requires-Python", requires_python),
                         ("Description-Content-Type", "text/markdown")):
        require(metadata[field] == value, f"{field}: expected {value!r}, got {metadata[field]!r}")
    require(not metadata.get_all("Requires-Dist"), "core package must have no runtime Python dependencies")
    require("# ASK" in metadata.get_payload(), "release metadata must contain the README")


def check_dist(directory, version, requires_python):
    sdists, wheels = sorted(directory.glob("*.tar.gz")), sorted(directory.glob("*.whl"))
    require(len(sdists) == len(wheels) == 1, "dist directory must contain exactly one sdist and one core wheel")
    with tarfile.open(sdists[0], "r:gz") as archive:
        files = {"/".join(PurePosixPath(item.name).parts[1:]): item
                 for item in archive.getmembers() if item.isfile()}
        required = {"PKG-INFO", "README.md", "pyproject.toml", "CMakeLists.txt", "CMakePresets.json",
                    "python/ask/_version.py", "cmake/AskVersion.cmake", "cmake/llvm.lock.json",
                    "cmake/nanobind.lock.json", "tools/fetch_deps.py", "python/ask/adapters.py",
                    "schemas/external-compiler-request-v2.schema.json",
                    "schemas/external-kernel-artifact-v2.schema.json"}
        require(required <= files.keys(), f"sdist missing required sources: {sorted(required - files.keys())}")
        for name in files:
            require(not any(part in {".deps", "build", "dist", ".venv", "__pycache__", ".git"}
                            or part.endswith(".egg-info") for part in PurePosixPath(name).parts),
                    f"sdist includes local build state: {name}")
            require(not name.endswith((".pyc", ".pyo")), f"sdist includes Python cache: {name}")
        check_message(archive.extractfile(files["PKG-INFO"]).read(), version, requires_python)
        require(read_version(archive.extractfile(files["python/ask/_version.py"]).read()) == version,
                "sdist source version mismatch")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "wheel must contain one METADATA record")
        check_message(archive.read(metadata[0]), version, requires_python)
        require(read_version(archive.read("ask/_version.py")) == version, "wheel Python version mismatch")
        require("ask/_bin/ask-opt" in names and any(name.startswith("ask/_native.") and name.endswith(".so") for name in names),
                "wheel must contain both the native extension and companion CLI")
        require(not any("_adapter_registry" in name or name.startswith("ask/_adapters/")
                        or name.startswith("ask/share/ask/adapters/") for name in names),
                "ASK wheel must contain only the generic external execution interface")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, help="also verify a core sdist and wheel")
    args = parser.parse_args()
    version, requires_python = check_source()
    if args.dist_dir:
        check_dist(args.dist_dir, version, requires_python)
    print(f"ASK {version}: release metadata consistent")


if __name__ == "__main__":
    main()
