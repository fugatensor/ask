"""Trace-only instrumentation for the first provable Triton fusion boundary.

The accepted source has one guarded i32 load, pure pointwise arithmetic, and
one store using the same index and mask. This module never decodes Triton's
memory inline assembly. ASK's LLVM pass validates the identity trace calls.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import re

PROTOCOL = "ask.triton.boundary.v1"
INPUT_ASM = "// ASK_BOUNDARY_INPUT_V1\nmov.b32 $0, $1;"
OUTPUT_ASM = "// ASK_BOUNDARY_OUTPUT_V1\nmov.b32 $0, $1;"
CONSTRAINTS = "=r,r,l,r,r"
MAX_ELEMENTS = 2147483520
ALLOWED_TRITON_ENVIRONMENT = frozenset({
    "TRITON_CACHE_DIR", "TRITON_HOME", "TRITON_ALWAYS_COMPILE",
    "TRITON_KERNEL_DUMP", "TRITON_DUMP_DIR", "TRITON_DISABLE_LINE_INFO",
})


class FusionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FusionError(message)


def check_environment(environment=None):
    """An override must not replace code after its DSL boundary was proved."""
    environment = os.environ if environment is None else environment
    for name, value in sorted(environment.items()):
        if not value:
            continue
        if name == "LLVM_PASS_PLUGIN_PATH" or (name.startswith("TRITON_") and
                                                name not in ALLOWED_TRITON_ENVIRONMENT):
            raise FusionError("fusion does not support compiler environment override: " + name)


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _expression(text):
    return ast.parse(text, mode="eval").body


def _call(node, name):
    return isinstance(node, ast.Call) and _same(node.func, _expression("tl." + name))


def _integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and \
            isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
        return -node.operand.value
    return None


def _assignment(statement, used):
    require(isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
            isinstance(statement.targets[0], ast.Name), "expected a single local assignment")
    name = statement.targets[0].id
    require(name not in used, "reassignment or shadowing is outside the fusion subset: " + name)
    used.add(name)
    return name


def _keywords(call, expected):
    names = [item.arg for item in call.keywords]
    require(set(names) == set(expected) and len(names) == len(expected),
            "unsupported or repeated keyword on boundary memory operation")
    return {item.arg: item.value for item in call.keywords}


def _pointwise(node, values, constants):
    number = _integer(node)
    if number is not None:
        require(-(2**31) <= number < 2**31, "pointwise literal must fit signed i32")
        return
    if isinstance(node, ast.Name):
        require(node.id in values or node.id in constants,
                "pointwise arithmetic may use only loaded values, prior results, and constexprs")
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        _pointwise(node.left, values, constants)
        _pointwise(node.right, values, constants)
        return
    raise FusionError("pointwise subset permits only i32 values, literals, add, sub, and multiply")


def _trace(value, address, index, mask, asm):
    mask_i32 = ast.Call(func=ast.Attribute(value=copy.deepcopy(mask), attr="to", ctx=ast.Load()),
                        args=[_expression("tl.int32")], keywords=[])
    return ast.Call(func=_expression("tl.inline_asm_elementwise"), args=[], keywords=[
        ast.keyword(arg="asm", value=ast.Constant(asm)),
        ast.keyword(arg="constraints", value=ast.Constant(CONSTRAINTS)),
        ast.keyword(arg="args", value=ast.List(elts=[copy.deepcopy(value), copy.deepcopy(address),
                                                    ast.Name(id=index, ctx=ast.Load()), mask_i32],
                                                ctx=ast.Load())),
        ast.keyword(arg="dtype", value=_expression("tl.int32")),
        ast.keyword(arg="is_pure", value=ast.Constant(False)),
        ast.keyword(arg="pack", value=ast.Constant(1)),
    ])


def instrument(request):
    """Validate the entire kernel body before inserting two identity operations."""
    check_environment()
    constants = request["constants"]
    require(set(constants) == {"N", "BLOCK"}, "fusion requires exactly the N and BLOCK constexprs")
    count = constants["N"]
    require(type(count) is int and 1 <= count <= MAX_ELEMENTS,
            f"fusion requires 1 <= N <= {MAX_ELEMENTS}; zero-sized graphs use ordinary execution")
    require(constants["BLOCK"] == 128, "fusion currently requires the existing BLOCK=128 strategy")
    require(request["grid"] == [(count + 127) // 128, 1, 1],
            "fusion requires the existing grid to cover exactly ceil(N / 128) blocks")
    require(request["fixed_options"] == {"num_warps": 4},
            "fusion currently requires the existing num_warps=4 strategy")
    input_name = next(name for name, binding in request["source_bindings"].items()
                      if binding == "input:0")
    output_name = next(name for name, binding in request["source_bindings"].items()
                       if binding == "output:0")
    tree = ast.parse(request["source"])
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef))
    require(function.returns is None and not getattr(function, "type_params", []),
            "fusion does not support function return annotations or type parameters")
    used = {argument.arg for argument in function.args.args}
    require(not used.intersection({"tl", "triton"}), "kernel arguments must not shadow Triton modules")
    used.update({"tl", "triton"})
    statements = function.body
    require(len(statements) >= 3, "fusion requires index, load, and store statements")
    index = _assignment(statements[0], used)
    require(_same(statements[0].value,
                  _expression("tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")),
            "index must be program_id(0) * BLOCK + arange(0, BLOCK)")
    mask_expression = _expression(index + " < N")
    cursor = 1
    mask = mask_expression
    if isinstance(statements[cursor], ast.Assign) and _same(statements[cursor].value, mask_expression):
        mask_name = _assignment(statements[cursor], used)
        mask = ast.Name(id=mask_name, ctx=ast.Load())
        cursor += 1
    require(cursor < len(statements) - 1, "missing guarded input load")
    load_statement = statements[cursor]
    loaded_name = _assignment(load_statement, used)
    load = load_statement.value
    require(_call(load, "load") and len(load.args) == 1,
            "fusion requires exactly one assigned tl.load")
    require(_same(load.args[0], _expression(input_name + " + " + index)),
            "input load address must be input pointer plus the canonical index")
    load_keywords = _keywords(load, {"mask", "other"})
    require(_same(load_keywords["mask"], mask) and _integer(load_keywords["other"]) == 0,
            "input load requires the canonical mask and other=0")

    values = {loaded_name}
    for statement in statements[cursor + 1:-1]:
        name = _assignment(statement, used)
        _pointwise(statement.value, values, constants)
        values.add(name)
    last = statements[-1]
    require(isinstance(last, ast.Expr) and _call(last.value, "store") and len(last.value.args) == 2,
            "fusion requires exactly one final tl.store")
    store = last.value
    require(_same(store.args[0], _expression(output_name + " + " + index)),
            "output store address must use the same canonical index")
    store_keywords = _keywords(store, {"mask"})
    require(_same(store_keywords["mask"], mask), "output store must use the input load mask")
    _pointwise(store.args[1], values, constants)

    # Keep every original computation. Only the load's result and store's value
    # pass through a register-to-register identity with explicit boundary data.
    input_trace = ast.Assign(targets=[ast.Name(id=loaded_name, ctx=ast.Store())],
                             value=_trace(ast.Name(id=loaded_name, ctx=ast.Load()),
                                          load.args[0], index, mask, INPUT_ASM))
    store.args[1] = _trace(store.args[1], store.args[0], index, mask, OUTPUT_ASM)
    function.body.insert(cursor + 1, input_trace)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"

    def boundary(parameter, binding, asm):
        return {"marker_asm": asm, "constraints": CONSTRAINTS,
                "source_parameter": parameter, "source_binding": binding,
                "arguments": {"value": 0, "address": 1, "index": 2, "mask": 3},
                "mask_encoding": "zext_i1_to_i32"}

    contract = {
        "protocol": PROTOCOL, "entry": request["entry"], "dtype": "i32", "shape": [count],
        "block": 128, "grid": list(request["grid"]), "num_warps": 4,
        "original_source_sha256": hashlib.sha256(request["source"].encode()).hexdigest(),
        "instrumented_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input": {**boundary(input_name, "input:0", INPUT_ASM), "other": 0},
        "output": boundary(output_name, "output:0", OUTPUT_ASM),
        "proof": {"source_subset": "canonical-pointwise-i32-v1", "input_loads": 1,
                  "output_stores": 1, "same_index": True, "same_mask": True,
                  "no_pointer_escape": True, "only_output_store_has_effect": True,
                  "requires_disjoint_input_output": True,
                  "index_range": [0, request["grid"][0] * 128], "active_range": [0, count]},
    }
    return source, contract


def _layout(ttgir):
    require(isinstance(ttgir, str), "Triton did not expose TTGIR for layout comparison")
    layouts = set(re.findall(r"#ttg\.blocked<\{([^{}]*)\}>", ttgir))
    canonical = "sizePerThread=[1],threadsPerWarp=[32],warpsPerCTA=[4],order=[0]"
    normalized = {re.sub(r"\s+", "", value) for value in layouts}
    require(normalized == {canonical}, "fusion requires the canonical one-element-per-lane blocked layout")
    require(not re.search(r"#ttg\.(?:slice|dot_op|nvidia_mma|shared)", ttgir),
            "additional layout encodings are outside the fusion subset")
    return canonical


def _llvm_string(text):
    return '"' + ''.join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                         else f"\\{byte:02X}" for byte in text.encode()) + '"'


def finalize_contract(contract, baseline, compiled, artifact):
    """Bind the AST proof to this exact export; LLVM validation is still required."""
    require(_layout(baseline.asm.get("ttgir")) == _layout(compiled.asm.get("ttgir")),
            "identity tracing changed the selected blocked layout")
    for key in ("num_warps", "warp_size", "num_ctas", "shared", "global_scratch_size",
                "profile_scratch_size", "tmem_size"):
        require(getattr(baseline.metadata, key) == getattr(compiled.metadata, key),
                "identity tracing changed the selected execution configuration: " + key)
    payload = artifact["payload"]
    for role in ("input", "output"):
        boundary = contract[role]
        # These are our own exact identity strings, not Triton's memory asm.
        # The core subsequently verifies actual CallInst types/operands and uses.
        require(payload.count(_llvm_string(boundary["marker_asm"])) == 1,
                "expected exactly one surviving " + role + " identity trace")
        binding = next(item for item in artifact["port_map"]
                       if item["source_parameter"] == boundary["source_parameter"])
        require(binding["logical_port"] == boundary["source_binding"],
                "compiled boundary parameter binding changed")
        boundary["kernel_argument"] = binding["argument"]
    contract["llir_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    contract["proof"]["layout"] = {"size_per_thread": [1], "threads_per_warp": [32],
                                      "warps_per_cta": [4], "order": [0],
                                      "matched_uninstrumented_baseline": True}
    return contract
