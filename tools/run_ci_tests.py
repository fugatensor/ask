#!/usr/bin/env python3
"""Run native CTest suites and reject missing tests or skipped coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def check_report(path: Path, expected: set[str]) -> None:
    report = ET.parse(path).getroot()
    cases = report.findall(".//testcase")
    names = [case.get("name") for case in cases]
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError(f"CTest suites differ: expected {sorted(expected)}, found {names}")
    for case in cases:
        name = case.get("name")
        if (case.get("status") != "run"
                or any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))):
            raise ValueError(f"CTest suite did not pass: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true", help="also require the hardware CUDA runtime suite")
    parser.add_argument("--cublaslt", action="store_true", help="also require the CUDA runtime and cuBLASLt suites")
    args = parser.parse_args(argv)
    build_dir = args.build_dir.resolve()
    report = build_dir / "ci-tests.xml"
    expected = {"ask-core", "ask-compiler", "ask-pass", "ask-primitives"}
    if "ASK_TARGET_BACKENDS:STRING=host" in (build_dir / "CMakeCache.txt").read_text():
        expected.add("ask-host-blas")
    if args.cuda or args.cublaslt:
        expected.add("ask-runtime")
    if args.cublaslt:
        expected.add("ask-matmul")
    report.unlink(missing_ok=True)
    result = subprocess.run([
        "ctest", "--test-dir", str(build_dir), "--verbose", "--no-tests=error",
        "--test-output-size-passed", "1048576", "--test-output-size-failed", "1048576",
        "--output-junit", str(report), "-R", "^(" + "|".join(sorted(expected)) + ")$",
    ])
    if result.returncode:
        return result.returncode
    try:
        check_report(report, expected)
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"CI test coverage check failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {len(expected)} native CTest suites without skipped tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
