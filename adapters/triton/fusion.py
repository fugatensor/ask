"""Trace-only instrumentation for the first provable Triton fusion boundary.

The accepted source has one guarded i32/f32 load, pure pointwise arithmetic, and
one store using the same index and mask. This module never decodes Triton's
memory inline assembly. ASK's LLVM pass validates the identity trace calls.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import re

PROTOCOL = "ask.triton.boundary.v1"
INPUT_ASM = "// ASK_BOUNDARY_INPUT_V1\nmov.b32 $0, $1;"
OUTPUT_ASM = "// ASK_BOUNDARY_OUTPUT_V1\nmov.b32 $0, $1;"
CONSTRAINTS = "=r,r,l,r,r"
VALUE_TYPES = {"*i32": ("i32", "tl.int32", CONSTRAINTS),
               "*fp32": ("f32", "tl.float32", "=f,r,f,l,r")}
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


def _assignment(statement, protected):
    require(isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
            isinstance(statement.targets[0], ast.Name), "expected a single local assignment")
    name = statement.targets[0].id
    require(name not in protected, "kernel parameters and compiler modules cannot be shadowed: " + name)
    return name


def _bind(call, parameters):
    """Bind public call parameters, independent of keyword/positional spelling."""
    require(len(call.args) <= len(parameters), "too many memory/index call arguments")
    result = dict(zip(parameters, call.args))
    for item in call.keywords:
        require(item.arg in parameters and item.arg not in result,
                "unknown or repeated memory/index call argument")
        result[item.arg] = item.value
    require(set(result) == set(parameters), "missing explicit memory/index call argument")
    return result


def _linear(pid=0, lane=0, constant=0):
    return ("linear", (pid, lane, constant))


class _Dataflow:
    """Small abstract interpreter for scalar arithmetic and contiguous addresses.

    Affine coefficients describe program_id(0), arange(0, BLOCK), and a constant.
    Values loaded from memory remain opaque numeric values. Every call and
    statement is checked; proving one address never hides an unknown effect.
    """
    def __init__(self, constants, bindings, dtype):
        self.constants = constants
        self.dtype = dtype
        self.names = {name: _linear(constant=value) for name, value in constants.items()}
        self.names.update({name: ("pointer", binding, (0, 0, 0))
                           for name, binding in bindings.items()})
        self.index = (constants["BLOCK"], 1, 0)
        self.mask = ("mask", self.index, constants["N"])
        # Construct an immutable index expression from the proved affine form.
        # Referencing a local name here would be wrong after that local is
        # reassigned between the input and output boundary.
        self.index_ast = _expression("tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")

    def evaluate(self, node):
        number = _integer(node)
        if number is not None:
            require(-(2**31) <= number < 2**31, "literal must fit signed i32")
            return _linear(constant=number)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            require(self.dtype == "f32" and math.isfinite(node.value),
                    "floating literals require f32 data and a finite value")
            return ("float", node.value)
        if isinstance(node, ast.Name):
            require(node.id in self.names, "unknown or forward local reference: " + node.id)
            return self.names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.evaluate(node.operand)
            require(value[0] in ("linear", "value", "float"), "unary arithmetic requires numeric values")
            if isinstance(node.op, ast.UAdd):
                return value
            if value[0] == "linear":
                return ("linear", tuple(-x for x in value[1]))
            return ("float", -value[1]) if value[0] == "float" else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = self.evaluate(node.left), self.evaluate(node.right)
            if left[0] == "pointer" or right[0] == "pointer":
                # Only pointer + element offset (either order), or pointer -
                # offset. Pointer-pointer arithmetic and numeric casts escape.
                if right[0] == "pointer" and isinstance(node.op, ast.Add):
                    left, right = right, left
                require(left[0] == "pointer" and right[0] == "linear" and
                        isinstance(node.op, (ast.Add, ast.Sub)), "unsupported pointer arithmetic")
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                return ("pointer", left[1], tuple(a + sign * b for a, b in zip(left[2], right[1])))
            require(left[0] in ("linear", "value", "float") and right[0] in ("linear", "value", "float"),
                    "arithmetic requires numeric values")
            if left[0] == right[0] == "linear":
                a, b = left[1], right[1]
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    sign = -1 if isinstance(node.op, ast.Sub) else 1
                    return ("linear", tuple(x + sign * y for x, y in zip(a, b)))
                if a[:2] == (0, 0):
                    return ("linear", tuple(a[2] * y for y in b))
                if b[:2] == (0, 0):
                    return ("linear", tuple(b[2] * x for x in a))
            # Non-affine pure arithmetic can compute a stored value, but cannot
            # serve as a proven address or mask in this adapter.
            return ("value",)
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left, right = self.evaluate(node.left), self.evaluate(node.comparators[0])
            if isinstance(node.ops[0], ast.Gt):
                left, right = right, left
            else:
                require(isinstance(node.ops[0], ast.Lt), "boundary predicate must be a strict upper bound")
            require(left[0] == right[0] == "linear" and right[1][:2] == (0, 0),
                    "predicate must compare an affine index with a constexpr bound")
            return ("mask", left[1], right[1][2])
        if _call(node, "program_id"):
            args = _bind(node, ("axis",))
            require(self.evaluate(args["axis"]) == _linear(), "only program_id axis 0 is supported")
            return _linear(pid=1)
        if _call(node, "arange"):
            args = _bind(node, ("start", "end"))
            require(self.evaluate(args["start"]) == _linear() and
                    self.evaluate(args["end"]) == _linear(constant=self.constants["BLOCK"]),
                    "arange must span [0, BLOCK)")
            return _linear(lane=1)
        raise FusionError("unsupported expression or call in boundary dataflow")

    def address(self, node, binding):
        require(self.evaluate(node) == ("pointer", binding, self.index),
                "memory address does not cover the declared port at the logical index")

    def predicate(self, node):
        require(self.evaluate(node) == self.mask, "memory mask must describe the logical index < N")

    def numeric(self, node):
        require(self.evaluate(node)[0] in ("linear", "value", "float"), "stored value must be numeric")

def _trace(value, address, index, mask, asm, dtype, constraints, *, store=False):
    mask_i32 = ast.Call(func=ast.Attribute(value=copy.deepcopy(mask), attr="to", ctx=ast.Load()),
                        args=[_expression("tl.int32")], keywords=[])
    value = copy.deepcopy(value)
    if store:
        # tl.store converts its value to the pointer's element type. Put the
        # same conversion before the identity marker so Q sees the actual
        # stored value even when F's expression has a different numeric type.
        value = ast.Call(func=_expression("tl.cast"), args=[value, _expression(dtype)], keywords=[])
    operands = [value, copy.deepcopy(address), copy.deepcopy(index), mask_i32]
    if dtype == "tl.float32":
        # Triton's public inline-asm broadcaster uses the first argument as
        # its type exemplar and rejects comparing a pointer with a float.
        # A leading integer index permits both without casting either value.
        operands = [operands[2], operands[0], operands[1], operands[3]]
    return ast.Call(func=_expression("tl.inline_asm_elementwise"), args=[], keywords=[
        ast.keyword(arg="asm", value=ast.Constant(asm)),
        ast.keyword(arg="constraints", value=ast.Constant(constraints)),
        ast.keyword(arg="args", value=ast.List(elts=operands, ctx=ast.Load())),
        ast.keyword(arg="dtype", value=_expression(dtype)),
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
    block = constants["BLOCK"]
    require(type(block) is int and 1 <= block <= 1024 and block & (block - 1) == 0,
            "BLOCK must be a power of two no larger than 1024")
    require(request["grid"] == [(count + block - 1) // block, 1, 1] and
            request["grid"][0] * block <= 2**31,
            "grid must cover ceil(N / BLOCK) blocks without overflowing signed i32 indices")
    warps = request["fixed_options"].get("num_warps")
    require({"num_warps"} <= set(request["fixed_options"]) <= {"num_warps", "enable_fp_fusion"} and
            type(warps) is int and warps in (4, 8) and
            type(request["fixed_options"].get("enable_fp_fusion", True)) is bool,
            "fusion supports num_warps=4|8 and optional boolean enable_fp_fusion")
    input_name = next(name for name, binding in request["source_bindings"].items()
                      if binding == "input:0")
    output_name = next(name for name, binding in request["source_bindings"].items()
                       if binding == "output:0")
    signature = request["signature"]
    require(signature.get(input_name) in VALUE_TYPES and
            signature.get(input_name) == signature.get(output_name),
            "fusion requires matching *i32 or *fp32 data pointers")
    dtype, triton_dtype, constraints = VALUE_TYPES[signature[input_name]]
    input_asm, output_asm = INPUT_ASM, OUTPUT_ASM
    arguments = {"value": 0, "address": 1, "index": 2, "mask": 3}
    if dtype == "f32":
        input_asm, output_asm = (text.replace("$1", "$2") for text in (INPUT_ASM, OUTPUT_ASM))
        arguments = {"index": 0, "value": 1, "address": 2, "mask": 3}
    tree = ast.parse(request["source"])
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef))
    require(function.returns is None and not getattr(function, "type_params", []),
            "fusion does not support function return annotations or type parameters")
    used = {argument.arg for argument in function.args.args}
    require(not used.intersection({"tl", "triton"}), "kernel arguments must not shadow Triton modules")
    used.update({"tl", "triton"})
    flow = _Dataflow(constants, request["source_bindings"], dtype)
    rewritten = []
    input_trace = store = None
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            name = _assignment(statement, used)
            if _call(statement.value, "load"):
                require(input_trace is None and store is None, "expected exactly one input load before output store")
                args = _bind(statement.value, ("pointer", "mask", "other"))
                flow.address(args["pointer"], "input:0")
                flow.predicate(args["mask"])
                require(flow.evaluate(args["other"]) in (_linear(), ("float", 0.0)),
                        "input load requires other=0")
                flow.names[name] = ("value",)
                input_trace = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=_trace(ast.Name(id=name, ctx=ast.Load()), args["pointer"],
                                 flow.index_ast, args["mask"], input_asm, triton_dtype, constraints))
                rewritten.extend((statement, input_trace))
                continue
            result = flow.evaluate(statement.value)
            flow.names[name] = result
            rewritten.append(statement)
        elif isinstance(statement, ast.Expr) and _call(statement.value, "store"):
            require(input_trace is not None and store is None, "expected exactly one output store after input load")
            store = statement.value
            args = _bind(store, ("pointer", "value", "mask"))
            flow.address(args["pointer"], "output:0")
            flow.predicate(args["mask"])
            flow.numeric(args["value"])
            traced = _trace(args["value"], args["pointer"], flow.index_ast, args["mask"],
                            output_asm, triton_dtype, constraints, store=True)
            # Normalize call argument binding only; preserve the user's address,
            # value arithmetic, predicate and every original local computation.
            store.args = [args["pointer"], traced]
            store.keywords = [ast.keyword(arg="mask", value=args["mask"])]
            rewritten.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and \
                isinstance(statement.value.value, str):
            rewritten.append(statement)
        else:
            raise FusionError("only local assignments and the output store are supported")
    require(input_trace is not None and store is not None, "both input load and output store are required")
    function.body = rewritten
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"

    def boundary(parameter, binding, asm):
        return {"marker_asm": asm, "constraints": constraints,
                "source_parameter": parameter, "source_binding": binding,
                "arguments": dict(arguments),
                "mask_encoding": "zext_i1_to_i32"}

    contract = {
        "protocol": PROTOCOL, "entry": request["entry"], "dtype": dtype, "shape": [count],
        "block": block, "grid": list(request["grid"]), "num_warps": warps,
        "original_source_sha256": hashlib.sha256(request["source"].encode()).hexdigest(),
        "instrumented_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input": {**boundary(input_name, "input:0", input_asm), "other": 0},
        "output": boundary(output_name, "output:0", output_asm),
        "proof": {"source_subset": "affine-pointwise-typed-v3", "input_loads": 1,
                  "output_stores": 1, "same_index": True, "same_mask": True,
                  "no_pointer_escape": True, "only_output_store_has_effect": True,
                  "requires_disjoint_input_output": True,
                  "index_range": [0, request["grid"][0] * block], "active_range": [0, count]},
    }
    return source, contract


def _layout(ttgir):
    require(isinstance(ttgir, str), "Triton did not expose TTGIR for layout comparison")
    layouts = set(re.findall(r"#ttg\.blocked<\{([^{}]*)\}>", ttgir))
    normalized = {re.sub(r"\s+", "", value) for value in layouts}
    require(bool(normalized), "missing blocked layout encoding")
    require(not re.search(r"#ttg\.(?:slice|dot_op|nvidia_mma|shared)", ttgir),
            "additional layout encodings are outside the fusion subset")
    return sorted(normalized)


def _llvm_string(text):
    return '"' + ''.join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                         else f"\\{byte:02X}" for byte in text.encode()) + '"'


def finalize_contract(contract, baseline, compiled, artifact):
    """Bind the AST proof to this exact export; LLVM validation is still required."""
    require(_layout(baseline.asm.get("ttgir")) == _layout(compiled.asm.get("ttgir")),
            "identity tracing changed the selected blocked layout")
    for key in ("num_warps", "warp_size", "num_ctas", "shared", "global_scratch_size",
                "profile_scratch_size", "tmem_size", "enable_fp_fusion", "enable_reflect_ftz"):
        require(getattr(baseline.metadata, key) == getattr(compiled.metadata, key),
                "identity tracing changed the selected execution configuration: " + key)
    payload = artifact["payload"]
    for role in ("input", "output"):
        boundary = contract[role]
        # These are our own exact identity strings, not Triton's memory asm.
        # The core subsequently verifies actual CallInst types/operands and uses.
        count = payload.count(_llvm_string(boundary["marker_asm"]))
        require(count > 0, "missing surviving " + role + " identity trace")
        boundary["marker_count"] = count
        binding = next(item for item in artifact["port_map"]
                       if item["source_parameter"] == boundary["source_parameter"])
        require(binding["logical_port"] == boundary["source_binding"],
                "compiled boundary parameter binding changed")
        boundary["kernel_argument"] = binding["argument"]
    contract["llir_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    contract["numerics"] = {name: getattr(compiled.metadata, name)
                            for name in ("enable_fp_fusion", "enable_reflect_ftz")}
    contract["proof"]["layout"] = {"blocked_encodings": _layout(compiled.asm.get("ttgir")),
                                      "matched_uninstrumented_baseline": True}
    return contract
