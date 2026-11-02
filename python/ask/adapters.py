"""Compiler-independent subprocess protocol for externally compiled CUDA code.

The caller supplies a worker explicitly. ASK neither discovers compiler packages
nor imports their Python modules; compiler-specific workers belong to callers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .compiler import CompilerError


@dataclass(frozen=True)
class CompilerEnvironment:
    """Explicit worker, optionally run by an isolated Python interpreter."""
    python: str | os.PathLike[str] | None = None
    worker: str | os.PathLike[str] | None = None

    def __post_init__(self):
        for field in ("python", "worker"):
            item = getattr(self, field)
            if item is None and field == "python":
                continue
            if item is None:
                raise ValueError("compiler environment worker is required")
            value = os.fspath(item)
            if not isinstance(value, str) or not value:
                raise ValueError(f"compiler environment {field} must not be empty")
            object.__setattr__(self, field, value)


def _require(condition, message):
    if not condition:
        raise CompilerError(message)


def _read_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def worker_command(environment):
    if not isinstance(environment, CompilerEnvironment):
        raise TypeError("compilers must map IDs to CompilerEnvironment values")
    command = []
    if environment.python is not None:
        interpreter = shutil.which(environment.python)
        _require(interpreter is not None, "compiler Python executable not found")
        # Do not resolve virtualenv interpreter symlinks.
        command.append(str(Path(interpreter).absolute()))
        worker = Path(environment.worker).resolve()
        _require(worker.is_file(), "compiler worker file not found")
    else:
        executable = shutil.which(environment.worker)
        _require(executable is not None, "compiler worker executable not found")
        worker = Path(executable).absolute()
    return [*command, str(worker)]


def _run(command, arguments, timeout):
    try:
        result = subprocess.run([*command, *arguments], capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CompilerError("external compiler worker failed", diagnostics=str(error)) from error
    if result.returncode:
        raise CompilerError("external compiler worker failed", diagnostics=result.stderr,
                            returncode=result.returncode)
    return result.stdout


def worker_identity(command, *, timeout, request=None):
    """Probe transitive producer dependencies before every cache lookup."""
    try:
        with tempfile.TemporaryDirectory(prefix="ask-probe-") as directory:
            arguments = ["--probe"]
            if request is not None:
                path = Path(directory) / "request.json"
                path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
                arguments.extend(["--request", str(path)])
            probe = _read_json(_run(command, arguments, timeout))
    except (ValueError, TypeError) as error:
        raise CompilerError("external compiler probe returned invalid JSON") from error
    _require(isinstance(probe, dict) and isinstance(probe.get("producer"), dict)
             and isinstance(probe.get("adapter_version"), str) and probe["adapter_version"],
             "worker probe requires producer and adapter_version")
    _require(all(isinstance(probe["producer"].get(k), str) and probe["producer"][k]
                 for k in ("name", "version")), "worker producer requires name and version")
    return {"probe": probe, "files": [
        {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for path in command]}


def export_kernel(command, request, *, timeout):
    with tempfile.TemporaryDirectory(prefix="ask-external-") as directory:
        request_path, output_path = Path(directory) / "request.json", Path(directory) / "artifact.json"
        request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        _run(command, ["--request", str(request_path), "--output", str(output_path)], timeout)
        try:
            return normalize_artifact(_read_json(output_path.read_text()), request)
        except (OSError, ValueError, TypeError) as error:
            raise CompilerError("external compiler returned an invalid artifact", diagnostics=str(error)) from error


_ELEMENT_TYPES = {"*i1": ("bool", 1), "*i32": ("i32", 4), "*i64": ("i64", 8),
                  "*fp16": ("f16", 2), "*bf16": ("bf16", 2), "*fp32": ("f32", 4)}


def normalize_artifact(artifact, request):
    """Derive tensor ABI from verified ASK ports, rather than adapter boilerplate."""
    fields = {"protocol_version", "format", "payload", "entry", "target", "producer",
              "launch", "arguments", "required_element_types"}
    _require(isinstance(artifact, dict) and set(artifact) == fields,
             "external artifact must contain exactly the protocol v2 fields")
    _require(type(artifact["protocol_version"]) is int and artifact["protocol_version"] == 2,
             "external artifact requires protocol_version 2")
    _require(artifact["format"] in ("cubin", "ptx"), "external artifact format must be cubin or ptx")
    _require(isinstance(artifact["target"], dict)
             and artifact["target"] == request["target"], "external target differs from the request")
    launch = artifact["launch"]
    _require(isinstance(launch, dict) and set(launch) == {"grid", "block", "shared_bytes"},
             "launch requires grid, block and shared_bytes")
    _require(launch["grid"] == request["grid"], "external launch grid differs from the request")
    from ._abi import scalar_bytes
    arguments = artifact["arguments"]
    _require(isinstance(arguments, list), "external arguments must be an ordered list")
    bindings = request["source_bindings"]
    expected = {binding for name, binding in bindings.items()
                if request["signature"][name].startswith("*")}
    pointers = [argument for argument in arguments if isinstance(argument, str)]
    _require(set(pointers) - {"null"} == expected
             and all(pointers.count(name) == 1 for name in expected),
             "external arguments must bind every logical port exactly once")
    def scalar_key(dtype, value):
        try:
            return dtype, scalar_bytes(dtype, value)
        except (ValueError, TypeError) as error:
            raise CompilerError("invalid external scalar argument", diagnostics=str(error)) from error
    expected_scalars = Counter(scalar_key(request["signature"][name], value)
                              for name, value in bindings.items()
                              if not request["signature"][name].startswith("*"))
    actual_scalars = Counter()
    invocation_arguments = []
    for index, argument in enumerate(arguments):
        if isinstance(argument, str):
            invocation_arguments.append({"argument": index, "binding": argument, "kind": "pointer"})
        else:
            _require(isinstance(argument, dict) and set(argument) == {"type", "value"},
                     "scalar arguments require exactly type and value")
            actual_scalars[scalar_key(argument["type"], argument["value"])] += 1
            invocation_arguments.append({"argument": index, "kind": "scalar", **argument})
    _require(actual_scalars == expected_scalars,
             "external scalar types and values must match the request exactly")
    ports = []
    for parameter, logical_port in bindings.items():
        if not request["signature"][parameter].startswith("*"):
            continue
        dtype, size = _ELEMENT_TYPES[request["signature"][parameter]]
        ports.append({"argument": arguments.index(logical_port), "logical_port": logical_port,
                      "shape": request["port_shapes"][logical_port], "element_type": dtype,
                      "storage_bytes": size, "layout": "contiguous_row_major"})
    normalized = {key: value for key, value in artifact.items() if key not in ("launch", "arguments")}
    normalized["port_map"] = sorted(ports, key=lambda port: port["argument"])
    normalized["invocation"] = {"version": 2, "kind": "cuda-kernel", **launch,
        "completion": "device-event", "arguments": invocation_arguments}
    from .cuda import _kernel_contract
    _kernel_contract(normalized)
    return normalized
