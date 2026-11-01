#!/usr/bin/env python3
"""ASK protocol-1 Triton exporter. No Triton import occurs before a request/probe.

This adapter deliberately accepts only the locked Triton CUDA
pointer ABI: two equally typed data pointers, integer constexprs and two unused scratch
pointers. It exports code and descriptions, never Python/kernel handles.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

_DEPENDENCIES = json.loads(Path(__file__).with_name("dependencies.lock.json").read_text(encoding="utf-8"))
PROTOCOL_VERSION = _DEPENDENCIES["protocol_version"]
ADAPTER_VERSION = _DEPENDENCIES["adapter_version"]
ELEMENT_TYPES = {"*i1": "bool", "*i32": "i32", "*i64": "i64",
                 "*fp32": "f32", "*fp16": "f16", "*bf16": "bf16"}
STORAGE_BYTES = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
SUPPORTED_TRITON_VERSION = _DEPENDENCIES["compiler"]["version"]


class WorkerError(Exception):
    def __init__(self, code, message, stage="request"):
        super().__init__(message)
        self.code = code
        self.stage = stage


def require(condition, message, code="InvalidRequest", stage="request"):
    if not condition:
        raise WorkerError(code, message, stage)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_request(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError("InvalidRequest", str(exc)) from exc
    return validate_request(value)


def validate_request(request):
    require(isinstance(request, dict), "request must be a JSON object")
    fields = {"protocol_version", "source", "entry", "signature",
              "source_bindings", "constants", "fixed_options", "target", "grid"}
    require(fields <= set(request) and set(request) <= fields | {"fusion"},
            f"request fields differ: missing={sorted(fields - set(request))}, "
            f"unknown={sorted(set(request) - fields - {'fusion'})}")
    if "fusion" in request:
        require(isinstance(request["fusion"], dict) and set(request["fusion"]) == {"enabled"}
                and type(request["fusion"]["enabled"]) is bool,
                "fusion must be exactly {enabled: true|false}")
    require(type(request["protocol_version"]) is int and
            request["protocol_version"] == PROTOCOL_VERSION,
            f"only protocol_version={PROTOCOL_VERSION} is supported", "ProtocolMismatch")
    source = request["source"]
    require(isinstance(source, str) and 0 < len(source.encode("utf-8")) <= 1024 * 1024,
            "source must be nonempty UTF-8 text no larger than 1 MiB")
    entry = request["entry"]
    require(isinstance(entry, str) and entry.isidentifier() and not entry.startswith("__"),
            "entry must be a Python identifier")
    for field in ("signature", "source_bindings", "constants", "fixed_options", "target"):
        require(isinstance(request[field], dict), f"{field} must be an object")
    signature = request["signature"]
    require(len(signature) == 2 and all(isinstance(k, str) and k.isidentifier()
                                      for k in signature),
            "initial adapter requires exactly two named data-pointer parameters",
            "UnsupportedSignature")
    require(all(isinstance(v, str) and v in ELEMENT_TYPES for v in signature.values()),
            "adapter supports *i1, *i32, *i64, *fp32, *fp16 or *bf16 data pointers", "UnsupportedSignature")
    require(len(set(signature.values())) == 1,
            "input and output data pointers must have the same element type", "UnsupportedSignature")
    if "*bf16" in signature.values():
        target = request["target"]
        require(target.get("backend") == "cuda" and type(target.get("arch")) is int
                and target["arch"] >= 80,
                "bf16 requires a CUDA target with compute capability 8.0 or newer",
                "UnsupportedDType")
    bindings = request["source_bindings"]
    require(set(bindings) == set(signature),
            "source_bindings must cover exactly the data-pointer signature")
    require(all(isinstance(value, str) for value in bindings.values()) and
            set(bindings.values()) == {"input:0", "output:0"},
            "initial adapter requires one input:0 and one output:0 binding")
    constants = request["constants"]
    require(all(isinstance(k, str) and k.isidentifier() and type(v) is int
                and -(2**63) <= v < 2**63 for k, v in constants.items()),
            "constexpr values must be named signed 64-bit integers")
    require(not set(constants).intersection(signature),
            "data parameters cannot also be constexprs")
    require({"num_warps"} <= set(request["fixed_options"]) <= {"num_warps", "enable_fp_fusion"} and
            type(request["fixed_options"].get("num_warps")) is int and
            request["fixed_options"]["num_warps"] in (4, 8) and
            ("enable_fp_fusion" not in request["fixed_options"] or
             type(request["fixed_options"]["enable_fp_fusion"]) is bool),
            "adapter requires fixed_options num_warps=4|8 and optional boolean enable_fp_fusion",
            "UnsupportedOptions")
    target = request["target"]
    require(set(target) == {"backend", "arch", "warp_size"} and target["backend"] == "cuda" and
            type(target["arch"]) is int and 70 <= target["arch"] <= 999 and
            type(target["warp_size"]) is int and target["warp_size"] == 32,
            "adapter requires cuda, an ordinary integer arch >= 70 and warp_size=32", "UnsupportedTarget")
    grid = request["grid"]
    require(isinstance(grid, list) and len(grid) == 3 and
            all(type(v) is int and 0 < v <= 2**31 - 1 for v in grid),
            "grid must contain three positive signed 32-bit dimensions")
    _validate_source(source, entry, set(signature), set(constants))
    return request


def _validate_source(source, entry, pointer_names, constant_names):
    """Reject unsupported source packaging before importing the module.

    This is a small source-format restriction, not a security sandbox. Kernel
    bodies are compiled by Triton; ASK's semantic registry owns their contract.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkerError("InvalidSource", str(exc), "source") from exc
    functions = []
    imports = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                pair = (alias.name, alias.asname)
                require(pair in {("triton", None), ("triton.language", "tl")},
                        "source may only import triton and triton.language as tl",
                        "UnsupportedSource", "source")
                imports.add(pair)
        elif isinstance(statement, ast.FunctionDef):
            functions.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) \
                and isinstance(statement.value.value, str):
            continue
        else:
            raise WorkerError("UnsupportedSource", "unsupported top-level source statement", "source")
    require(imports == {("triton", None), ("triton.language", "tl")},
            "source must import triton and triton.language as tl", "UnsupportedSource", "source")
    require(len(functions) == 1 and functions[0].name == entry,
            "source must define exactly the requested entry", "UnsupportedSource", "source")
    function = functions[0]
    require(len(function.decorator_list) == 1 and
            ast.dump(function.decorator_list[0]) == ast.dump(ast.parse("triton.jit", mode="eval").body),
            "entry must use exactly @triton.jit", "UnsupportedSource", "source")
    args = function.args
    require(not (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                 or args.defaults or args.kw_defaults),
            "entry cannot use defaults, varargs or keyword-only arguments",
            "UnsupportedSignature", "source")
    require({arg.arg for arg in args.args} == pointer_names | constant_names,
            "source argument names must exactly match signature plus constants",
            "InvalidSourceBinding", "source")
    for arg in args.args:
        if arg.arg in constant_names:
            require(arg.annotation is not None and ast.dump(arg.annotation) ==
                    ast.dump(ast.parse("tl.constexpr", mode="eval").body),
                    f"{arg.arg} must be annotated tl.constexpr", "InvalidSourceBinding", "source")
        else:
            require(arg.annotation is None,
                    f"data pointer {arg.arg} must use the request signature",
                    "UnsupportedSignature", "source")


@contextlib.contextmanager
def compiler_output_to_stderr():
    """Redirect Python and native compiler stdout away from control output."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)


def load_triton():
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
    except ImportError as exc:
        raise WorkerError("DependencyUnavailable", str(exc), "environment") from exc
    require(triton.__version__ == SUPPORTED_TRITON_VERSION,
            f"tested ABI requires Triton {SUPPORTED_TRITON_VERSION}; found {triton.__version__}",
            "UnsupportedCompilerVersion", "environment")
    return triton, ASTSource, GPUTarget


def probe():
    triton, _, _ = load_triton()
    return {"protocol_version": PROTOCOL_VERSION, "compiler_id": "triton",
            "adapter_version": ADAPTER_VERSION, "triton_version": triton.__version__,
            "python_version": platform.python_version(), "formats": ["llvm-ir"],
            "target_constraints": {"backend": "cuda", "minimum_arch": 70, "warp_size": 32,
                                   "arch_encoding": "ordinary-integer-sm",
                                   "requires_successful_compilation": True},
            "supported_element_types": list(ELEMENT_TYPES.values()), "device_execution_tested": False,
            "fusion_supported": True}


def _header_tokens(header):
    # A lexer for LLVM headers only. Inline assembly bodies are never examined.
    token = re.compile(r'\s+|"(?:\\.|[^"\\])*"|[-A-Za-z$._][-A-Za-z$._0-9]*|[0-9]+|[@%#!=(),{}]')
    result, position = [], 0
    while position < len(header):
        match = token.match(header, position)
        require(match is not None, "unsupported token in LLVM entry header",
                "UnsupportedEntryABI", "artifact")
        value = match.group()
        if not value.isspace():
            result.append(value)
        position = match.end()
    return result


def parse_entry_signature(llvm_ir, entry):
    """Read only the pinned pointer-only kernel header, not arbitrary LLVM IR.

    Full LLVM parsing/verifying remains the core's responsibility. Unsupported
    entry forms fail closed instead of inferring an ABI from PTX or source order.
    """
    require(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", entry) is not None,
            "unsupported LLVM entry symbol", "UnsupportedEntryABI", "artifact")
    found = []
    for match in re.finditer(r"(?m)^define\s+", llvm_ir):
        # LLVM function definitions in the accepted subset have no quoted or
        # aggregate types. The opening body brace terminates this header.
        end = llvm_ir.find("{", match.start())
        require(end != -1, "unterminated LLVM definition", "UnsupportedEntryABI", "artifact")
        header = llvm_ir[match.start():end]
        if not re.search(r"@" + re.escape(entry) + r"\s*\(", header):
            continue
        tokens = _header_tokens(header)
        require(tokens[:6] == ["define", "ptx_kernel", "void", "@", entry, "("],
                "expected a ptx_kernel void entry", "UnsupportedEntryABI", "artifact")
        parts, current, depth, index = [], [], 1, 6
        while index < len(tokens) and depth:
            item = tokens[index]
            if item == "(":
                depth += 1
            elif item == ")":
                depth -= 1
            if depth == 0:
                if current:
                    parts.append(current)
                break
            if item == "," and depth == 1:
                require(bool(current), "empty LLVM parameter", "UnsupportedEntryABI", "artifact")
                parts.append(current)
                current = []
            else:
                current.append(item)
            index += 1
        require(depth == 0, "unterminated LLVM parameter list", "UnsupportedEntryABI", "artifact")
        parameters = []
        for ordinal, part in enumerate(parts):
            require(len(part) >= 7 and part[:5] == ["ptr", "addrspace", "(", "1", ")"]
                    and part[-2] == "%" and re.fullmatch(r"[A-Za-z_0-9.$-]+", part[-1]),
                    f"parameter {ordinal} is not a supported address-space-1 pointer",
                    "UnsupportedEntryABI", "artifact")
            attributes = part[5:-2]
            require(attributes in ([], ["readnone", "captures", "(", "none", ")"]),
                    f"unsupported attributes on parameter {ordinal}: {attributes}",
                    "UnsupportedEntryABI", "artifact")
            parameters.append({"argument": ordinal, "llvm_type": "ptr addrspace(1)",
                               "llvm_name": part[-1],
                               "unused_scratch_attributes": bool(attributes)})
        found.append({"calling_convention": "ptx_kernel", "return_type": "void",
                      "parameters": parameters})
    require(len(found) == 1, "expected exactly one matching LLVM kernel definition",
            "UnsupportedEntryABI", "artifact")
    return found[0]


def _json_data(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "non-string metadata key",
                "UnsupportedMetadata", "artifact")
        return {k: _json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(v) for v in value]
    raise WorkerError("UnsupportedMetadata", f"cannot serialize {type(value).__name__}", "artifact")


def compiled_element_types(ttir):
    """Read builtin scalar types from Triton's public MLIR assembly output.

    This is a lexical type inventory, not an operation or effect proof. Ignore
    quoted strings, comments, SSA/symbol names and attribute keys, then accept
    builtin type tokens, including compact shaped types such as 128xbf16.
    LLVM IR alone is too late: BF16 casts may already be lowered to integer/PTX
    instructions, even though the external pointer ABI is entirely f32.
    """
    require(isinstance(ttir, str) and ttir.strip(),
            "compiler did not return asm['ttir'] for computation type requirements",
            "MissingMLIRArtifact", "artifact")
    visible = []
    position = 0
    while position < len(ttir):
        if ttir[position] == '"':
            position += 1
            while position < len(ttir) and ttir[position] != '"':
                position += 2 if ttir[position] == "\\" else 1
            require(position < len(ttir), "unterminated MLIR string",
                    "UnsupportedMLIRArtifact", "artifact")
            position += 1
            visible.append(" ")
        elif ttir.startswith("//", position):
            end = ttir.find("\n", position + 2)
            position = len(ttir) if end == -1 else end + 1
            visible.append("\n")
        elif ttir.startswith("/*", position):
            position += 2
            depth = 1
            while position < len(ttir) and depth:
                if ttir.startswith("/*", position):
                    depth += 1
                    position += 2
                elif ttir.startswith("*/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
            require(depth == 0, "unterminated MLIR comment", "UnsupportedMLIRArtifact", "artifact")
            visible.append(" ")
        else:
            visible.append(ttir[position])
            position += 1
    scalar_types = re.findall(
        r"(?<![-\w.$%@#!])(?:(?:[0-9]+|\?|\[[0-9]+\]|\*)x)*"
        r"(bf16|f16|f32|i1|i32|i64)(?![-\w.$])(?!\s*=)", "".join(visible))
    return {"bool" if name == "i1" else name for name in scalar_types}


def build_artifact(request, compiled, triton_version, source_order):
    require(isinstance(compiled.asm.get("llir"), str) and compiled.asm["llir"].strip(),
            "compiler did not return asm['llir']", "MissingLLVMArtifact", "artifact")
    llvm_ir = compiled.asm["llir"]
    entry = compiled.name
    metadata = _json_data(compiled.metadata._asdict())
    require(metadata.get("name") == entry and metadata.get("triton_version") == triton_version,
            "compiled name/version metadata mismatch", "ArtifactMismatch", "artifact")
    require(metadata.get("target") == request["target"],
            "compiled target differs from requested target", "ArtifactMismatch", "artifact")
    required_types = {ELEMENT_TYPES[signature] for signature in request["signature"].values()}
    required_types.update(compiled_element_types(compiled.asm.get("ttir")))
    require("bf16" not in required_types or request["target"]["arch"] >= 80,
            "bf16 computation requires a CUDA target with compute capability 8.0 or newer",
            "UnsupportedDType", "artifact")
    require(metadata.get("num_warps") == request["fixed_options"]["num_warps"] and
            metadata.get("warp_size") == 32,
            "compiled launch configuration differs from request", "ArtifactMismatch", "artifact")
    if "enable_fp_fusion" in request["fixed_options"]:
        require(metadata.get("enable_fp_fusion") is request["fixed_options"]["enable_fp_fusion"],
                "compiled floating contraction policy differs from request", "ArtifactMismatch", "artifact")
    require(type(metadata.get("enable_fp_fusion")) is bool,
            "compiled floating contraction policy is missing", "ArtifactMismatch", "artifact")
    for name in ("shared", "global_scratch_size", "profile_scratch_size", "tmem_size"):
        require(type(metadata.get(name)) is int and metadata[name] == 0,
                f"initial ABI requires zero {name}; found {metadata.get(name)!r}",
                "UnsupportedResources", "artifact")
    require(metadata.get("num_ctas") == 1 and not metadata.get("tensordesc_meta") and
            metadata.get("launch_cooperative_grid") is False and
            metadata.get("launch_pdl") is False and not metadata.get("instrumentation_mode"),
            "unsupported clustered, descriptor, cooperative or instrumented launch",
            "UnsupportedResources", "artifact")
    for resource in ("global_scratch", "profile_scratch"):
        require(type(metadata.get(resource + "_align")) is int and
                metadata[resource + "_align"] > 0,
                f"missing or invalid {resource} alignment", "UnsupportedResources", "artifact")
    signature = parse_entry_signature(llvm_ir, entry)
    parameters = signature["parameters"]
    require(len(parameters) == len(source_order) + 2,
            f"Triton {SUPPORTED_TRITON_VERSION} ABI must contain data pointers followed by two scratch pointers",
            "UnsupportedEntryABI", "artifact")
    require(all(not p["unused_scratch_attributes"] for p in parameters[:2]) and
            all(p["unused_scratch_attributes"] for p in parameters[2:]),
            "unexpected data/scratch parameter attributes", "UnsupportedEntryABI", "artifact")
    port_map, arguments = [], []
    count = request["constants"].get("N")
    require(type(count) is int and count >= 0,
            "Triton pointwise exporter requires an explicit nonnegative N logical shape",
            "UnsupportedShape", "artifact")
    for ordinal, name in enumerate(source_order):
        binding = request["source_bindings"][name]
        element_type = ELEMENT_TYPES[request["signature"][name]]
        port_map.append({"source_parameter": name, "logical_port": binding,
                         "argument": ordinal, "element_type": element_type,
                         "storage_bytes": STORAGE_BYTES[element_type],
                         "shape": [count], "layout": "contiguous_row_major"})
        arguments.append({"argument": ordinal, "binding": binding,
                          "llvm_type": parameters[ordinal]["llvm_type"]})
    resources = {}
    for ordinal, resource in enumerate(("global_scratch", "profile_scratch"), start=2):
        resources[resource] = {"size_bytes": 0, "alignment": metadata[resource + "_align"]}
        arguments.append({"argument": ordinal, "resource": resource, "binding": "null",
                          "llvm_type": parameters[ordinal]["llvm_type"]})
    layout = re.findall(r'(?m)^target datalayout = "([^"\n]+)"\s*$', llvm_ir)
    require(len(layout) == 1, "missing or ambiguous LLVM data layout", "ArtifactMismatch", "artifact")
    dependencies = []
    for name, path in metadata.get("extern_libs", []):
        try:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as exc:
            raise WorkerError("DependencyUnavailable", f"cannot hash {name}: {exc}", "artifact") from exc
        dependencies.append({"name": name, "path": path, "sha256": digest,
                             "role": "producer_compile_input"})
    return {"protocol_version": PROTOCOL_VERSION, "format": "llvm-ir", "payload": llvm_ir,
            "entry": entry, "entry_signature": signature,
            "producer": {"name": "triton", "version": triton_version,
                         "adapter_version": ADAPTER_VERSION,
                         "python_version": platform.python_version(), "llvm_version": None},
            "target": {**request["target"], "data_layout": layout[0]},
            "port_map": port_map, "required_element_types": sorted(required_types),
            "codegen": {"fp_contract": "fast" if metadata.get("enable_fp_fusion") is True else "off",
                        "ptx_version": 80},
            "producer_metadata": {"num_warps": metadata["num_warps"], "num_ctas": 1,
                                  "resources": resources, "compiler_metadata": metadata},
            "invocation": {"kind": "cuda-kernel", "version": 2,
                           "grid": request["grid"], "block": [metadata["num_warps"] * metadata["warp_size"], 1, 1],
                           "shared_bytes": 0, "arguments": arguments,
                           "completion": "device-event"},
            "dependencies": dependencies,
            "specializations": {"constants": request["constants"],
                                "fixed_options": request["fixed_options"],
                                "source_bindings": request["source_bindings"],
                                "source_sha256": hashlib.sha256(request["source"].encode()).hexdigest()},
            "validation": {"entry_abi": "checked-tested-subset", "llvm_module_verified": False,
                           "unresolved_link_dependencies": "not_analyzed",
                           "device_execution_performed": False}}


def export(request):
    request = validate_request(request)
    fusion = None
    instrumented_source = None
    contract = None
    if request.get("fusion", {}).get("enabled", False):
        fusion_spec = importlib.util.spec_from_file_location(
            "ask_triton_fusion", Path(__file__).with_name("fusion.py"))
        fusion = importlib.util.module_from_spec(fusion_spec)
        fusion_spec.loader.exec_module(fusion)
        try:
            instrumented_source, contract = fusion.instrument(request)
        except fusion.FusionError as exc:
            raise WorkerError("UnsupportedFusion", str(exc), "fusion") from exc
    triton, ASTSource, GPUTarget = load_triton()
    digest = hashlib.sha256(request["source"].encode("utf-8")).hexdigest()
    module_name = "ask_triton_" + digest
    # JITFunction/inspect must see an actual source file throughout compilation.
    with tempfile.TemporaryDirectory(prefix="ask-triton-source-") as directory:
        source_file = Path(directory) / "kernel.py"
        source_file.write_text(request["source"], encoding="utf-8")
        spec = importlib.util.spec_from_file_location(module_name, source_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            function = getattr(module, request["entry"])
            require(isinstance(function, triton.JITFunction),
                    "entry is not a triton.JITFunction", "UnsupportedSource", "source")
            names = list(function.arg_names)
            constexpr_names = {names[index] for index in function.constexprs}
            require(constexpr_names == set(request["constants"]),
                    "public JIT signature disagrees with requested constexprs",
                    "InvalidSourceBinding", "source")
            runtime_names = [name for name in names if name not in constexpr_names]
            require(set(runtime_names) == set(request["signature"]),
                    "public JIT signature disagrees with data-pointer names",
                    "InvalidSourceBinding", "source")
            # Use declared source order, never the JSON object's insertion order.
            signature = {name: request["signature"][name] for name in runtime_names}
            compiled = triton.compile(
                ASTSource(function, signature, constexprs=request["constants"]),
                target=GPUTarget(**request["target"]), options=dict(request["fixed_options"]))
            require(compiled.src.signature == signature,
                    "compiled source signature differs from requested signature",
                    "ArtifactMismatch", "artifact")
            artifact = build_artifact(request, compiled, triton.__version__, runtime_names)
            if fusion is None:
                return artifact
            trace_file = Path(directory) / "instrumented.py"
            trace_file.write_text(instrumented_source, encoding="utf-8")
            trace_name = module_name + "_trace"
            trace_spec = importlib.util.spec_from_file_location(trace_name, trace_file)
            trace_module = importlib.util.module_from_spec(trace_spec)
            sys.modules[trace_name] = trace_module
            try:
                trace_spec.loader.exec_module(trace_module)
                traced_function = getattr(trace_module, request["entry"])
                traced = triton.compile(
                    ASTSource(traced_function, signature, constexprs=request["constants"]),
                    target=GPUTarget(**request["target"]), options=dict(request["fixed_options"]))
                artifact = build_artifact(request, traced, triton.__version__, runtime_names)
                try:
                    artifact["fusion_contract"] = fusion.finalize_contract(
                        contract, compiled, traced, artifact)
                    artifact["required_element_types"] = sorted(set(artifact["required_element_types"]) |
                        set(artifact["fusion_contract"]["required_element_types"]))
                except fusion.FusionError as exc:
                    raise WorkerError("UnsupportedFusion", str(exc), "fusion") from exc
                artifact["instrumented_source"] = instrumented_source
                return artifact
            finally:
                sys.modules.pop(trace_name, None)
        finally:
            sys.modules.pop(module_name, None)


def write_result(path, result):
    path = Path(path)
    # Atomic publication: a failing export never creates a successful response.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.probe:
        if args.request is not None or args.output is not None:
            parser.error("--probe cannot be combined with --request/--output")
    elif args.request is None or args.output is None:
        parser.error("provide --probe or both --request and --output")
    try:
        with compiler_output_to_stderr():
            result = probe() if args.probe else export(read_request(args.request))
        if args.probe:
            print(json.dumps(result, sort_keys=True))
        else:
            write_result(args.output, result)
        return 0
    except WorkerError as exc:
        error = {"protocol_version": PROTOCOL_VERSION,
                 "error": {"code": exc.code, "stage": exc.stage, "message": str(exc)}}
    except Exception as exc:
        error = {"protocol_version": PROTOCOL_VERSION,
                 "error": {"code": "ExportFailed", "stage": "compile",
                           "message": f"{type(exc).__name__}: {exc}"}}
    print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
