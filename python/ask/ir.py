"""Build and serialize ASK IR. Parsing, verification and preparation are native."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .types import ScalarType, TensorType, Type, as_type, compatible

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)


def identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"expected an ASCII identifier, got {value!r}")
    return value


def mlir_string(value: str) -> str:
    """Quote UTF-8 bytes using MLIR's two-digit hexadecimal string escapes."""
    if not isinstance(value, str):
        raise TypeError("MLIR string attributes require str values")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("MLIR strings must contain valid Unicode") from exc
    return '"' + "".join(
        chr(byte) if 0x20 <= byte <= 0x7E and byte not in (0x22, 0x5C)
        else f"\\{byte:02X}" for byte in encoded
    ) + '"'


def _i64(value: int) -> str:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError(f"expected a signed 64-bit integer, got {value!r}")
    return f"{value} : i64"


def _attribute(value: Any) -> str:
    if isinstance(value, str):
        return mlir_string(value)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return _i64(value)
    if type(value) is float and math.isfinite(value):
        return f"{value:.17e} : f64"
    if isinstance(value, Mapping):
        return _dictionary(value)
    raise TypeError(f"unsupported attribute value: {value!r}")


def _dictionary(values: Mapping[str, Any]) -> str:
    if not isinstance(values, Mapping):
        raise TypeError("expected an attribute mapping")
    return "{" + ", ".join(
        f"{identifier(key)} = {_attribute(value)}" for key, value in sorted(values.items())
    ) + "}"


def _array_i64(values: Sequence[int]) -> str:
    for value in values:
        _i64(value)
    return "array<i64" + (": " + ", ".join(str(value) for value in values) if values else "") + ">"


def _types(types: Sequence[Type]) -> str:
    return "(" + ", ".join(str(type_) for type_ in types) + ")"


@dataclass(frozen=True)
class Semantics:
    name: str
    inputs: tuple[Type, ...]
    results: tuple[Type, ...]
    shape_rule: str
    shape_operand: int
    _module: Module = field(repr=False, compare=False)


@dataclass(frozen=True)
class Implementation:
    name: str
    compiler: str
    _module: Module = field(repr=False, compare=False)


@dataclass(frozen=True)
class Value:
    name: str
    type: Type
    value_id: str
    _graph: Graph = field(repr=False, compare=False)


class Module:
    """A module with symbol declarations and one or more single-block graphs."""

    def __init__(self) -> None:
        self._declarations: list[str] = []
        self._graphs: list[Graph] = []
        self._symbols: set[str] = set()
        self._operation_ids: set[str] = set()

    def _add_symbol(self, name: str) -> None:
        identifier(name)
        if name in self._symbols:
            raise ValueError(f"duplicate module symbol: {name}")
        self._symbols.add(name)

    def semantics(self, name: str, *, inputs: Sequence[Type | str],
                  results: Sequence[Type | str], shape_rule: str = "same_as",
                  shape_operand: int = 0, effects: str = "none") -> Semantics:
        input_types = tuple(as_type(value) for value in inputs)
        result_types = tuple(as_type(value) for value in results)
        if len(result_types) != 1:
            raise ValueError("the initial SDK supports one external result")
        if effects != "none":
            raise ValueError("the initial SDK supports only effects='none'")
        if shape_rule not in ("same_as", "fixed"):
            raise ValueError("shape_rule must be 'same_as' or 'fixed'")
        if shape_rule == "same_as" and (
            type(shape_operand) is not int or not 0 <= shape_operand < len(input_types)
        ):
            raise ValueError("shape_operand must select an input")
        if shape_rule == "same_as" and not compatible(input_types[shape_operand], result_types[0]):
            raise ValueError("same_as input and result types must have compatible shapes")
        if shape_rule == "fixed" and any(
            isinstance(type_, TensorType) and None in type_.shape for type_ in result_types
        ):
            raise ValueError("fixed result shapes must be concrete")
        self._add_symbol(name)
        attrs = [
            f"sym_name = {mlir_string(name)}",
            f"function_type = {_types(input_types)} -> {_types(result_types)}",
            f"shape_rule = {mlir_string(shape_rule)}",
            'effects = "none"',
        ]
        if shape_rule == "same_as":
            attrs.append(f"shape_operand = {_i64(shape_operand)}")
        self._declarations.append('"ask.semantics"() {' + ", ".join(attrs) + "} : () -> ()")
        return Semantics(name, input_types, result_types, shape_rule, shape_operand, self)

    def implementation(self, name: str, *, compiler: str, source: str, entry: str,
                       signature: Mapping[str, str], source_bindings: Mapping[str, str],
                       constants: Mapping[str, Any] | None = None,
                       options: Mapping[str, Any] | None = None,
                       target: Mapping[str, Any] | None = None,
                       grid: Sequence[int] = (1, 1, 1)) -> Implementation:
        identifier(compiler)
        identifier(entry)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("DSL source must be a nonempty string")
        if set(signature) != set(source_bindings):
            raise ValueError("signature and source_bindings must name the same source parameters")
        if any(not isinstance(value, str) for value in signature.values()):
            raise TypeError("signature entries must be strings")
        if any(not isinstance(value, str) for value in source_bindings.values()):
            raise TypeError("source_bindings entries must be strings")
        if not target or not isinstance(target.get("backend"), str) or not target["backend"]:
            raise ValueError("target requires a nonempty backend string")
        grid = tuple(grid)
        if len(grid) != 3 or any(type(dim) is not int or dim < 1 for dim in grid):
            raise ValueError("grid must contain three positive integers")
        attrs = [
            f"sym_name = {mlir_string(name)}", f"compiler = {mlir_string(compiler)}",
            f"source = {mlir_string(source)}", f"entry = {mlir_string(entry)}",
            f"signature = {_dictionary(signature)}",
            f"source_bindings = {_dictionary(source_bindings)}",
            f"constants = {_dictionary(constants or {})}",
            f"options = {_dictionary(options or {})}",
            f"target = {_dictionary(target or {})}", f"grid = {_array_i64(grid)}",
        ]
        self._add_symbol(name)
        self._declarations.append('"ask.implementation"() {' + ", ".join(attrs) + "} : () -> ()")
        return Implementation(name, compiler, self)

    def graph(self, name: str, *, inputs: Mapping[str, Type | str],
              results: Sequence[Type | str]) -> Graph:
        if not isinstance(inputs, Mapping):
            raise TypeError("graph inputs must map argument names to types")
        for arg_name in inputs:
            identifier(arg_name)
        input_types = {key: as_type(value) for key, value in inputs.items()}
        result_types = tuple(as_type(value) for value in results)
        self._add_symbol(name)
        graph = Graph(self, name, input_types, result_types)
        self._graphs.append(graph)
        return graph

    def to_mlir(self) -> str:
        lines = ["module attributes {ask.schema_version = 1 : i64} {"]
        lines.extend("  " + declaration for declaration in self._declarations)
        for graph in self._graphs:
            lines.extend("  " + line for line in graph._serialize())
        lines.append("}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_mlir()


class Graph:
    def __init__(self, module: Module, name: str, inputs: Mapping[str, Type],
                 results: tuple[Type, ...]) -> None:
        self._module = module
        self.name = name
        self.result_types = results
        self.inputs = {key: Value(f"%{key}", type_, f"{name}.arg{index}", self)
                       for index, (key, type_) in enumerate(inputs.items())}
        self._operations: list[str] = []
        self._counter = 0
        self._returned: tuple[Value, ...] | None = None

    def _check_values(self, values: Sequence[Value]) -> None:
        if self._returned is not None:
            raise ValueError("cannot append operations after return")
        if any(not isinstance(value, Value) or value._graph is not self for value in values):
            raise ValueError("all operands must be values from this graph")

    def _emit(self, operation: str, operands: Sequence[Value], result_type: Type,
              attributes: Mapping[str, str], *, id: str | None,
              value_id: str | None = None) -> Value:
        self._check_values(operands)
        op_id = id if id is not None else f"{self.name}.op{self._counter}"
        if not isinstance(op_id, str) or not op_id:
            raise ValueError("operation id must be a nonempty string")
        if op_id in self._module._operation_ids:
            raise ValueError(f"duplicate operation id: {op_id}")
        logical_id = value_id if value_id is not None else f"{op_id}.result0"
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("value_id must be a nonempty string")
        name = f"%ask_v{self._counter}"
        while name in {value.name for value in self.inputs.values()}:
            self._counter += 1
            name = f"%ask_v{self._counter}"
        attrs = dict(attributes)
        attrs["ask.id"] = mlir_string(op_id)
        attrs["ask.value_id"] = mlir_string(logical_id)
        text = (f"{name} = {mlir_string(operation)}(" + ", ".join(value.name for value in operands)
                + ") {" + ", ".join(f"{key} = {value}" for key, value in attrs.items())
                + f"}} : {_types([value.type for value in operands])} -> {result_type}")
        self._module._operation_ids.add(op_id)
        self._operations.append(text)
        self._counter += 1
        return Value(name, result_type, logical_id, self)

    def constant(self, value: int, *, dtype: str = "i32",
                 id: str | None = None, value_id: str | None = None) -> Value:
        result_type = ScalarType(dtype)
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError(f"constant must be a signed i32 integer, got {value!r}")
        literal = f"{value} : {dtype}"
        return self._emit("ask.constant", (), result_type, {"value": literal}, id=id, value_id=value_id)

    def splat(self, value: Value, shape: Sequence[int], *, id: str | None = None,
              value_id: str | None = None) -> Value:
        self._check_values((value,))
        if not isinstance(value.type, ScalarType):
            raise TypeError("splat requires a scalar value")
        result_type = TensorType(tuple(shape), value.type.dtype)
        if None in result_type.shape:
            raise ValueError("splat requires a concrete shape")
        return self._emit("ask.splat", (value,), result_type,
                          {"shape": _array_i64(result_type.shape)}, id=id, value_id=value_id)

    def _binary(self, operation: str, lhs: Value, rhs: Value, *, id: str | None,
                value_id: str | None) -> Value:
        self._check_values((lhs, rhs))
        if not compatible(lhs.type, rhs.type):
            raise TypeError("add/mul require matching operand shapes; use an explicit splat")
        result_type = lhs.type
        if isinstance(lhs.type, TensorType) and isinstance(rhs.type, TensorType):
            result_type = TensorType(tuple(left if left is not None else right
                                           for left, right in zip(lhs.type.shape, rhs.type.shape)),
                                     lhs.type.dtype)
        return self._emit(operation, (lhs, rhs), result_type, {"overflow": '"wrap"'},
                          id=id, value_id=value_id)

    def add(self, lhs: Value, rhs: Value, *, id: str | None = None,
            value_id: str | None = None) -> Value:
        return self._binary("ask.add", lhs, rhs, id=id, value_id=value_id)

    def mul(self, lhs: Value, rhs: Value, *, id: str | None = None,
            value_id: str | None = None) -> Value:
        return self._binary("ask.mul", lhs, rhs, id=id, value_id=value_id)

    def external(self, *operands: Value, semantics: Semantics, implementation: Implementation,
                 result_type: Type | str | None = None, id: str | None = None,
                 value_id: str | None = None) -> Value:
        self._check_values(operands)
        if semantics._module is not self._module or implementation._module is not self._module:
            raise ValueError("semantics and implementation must be registered in this module")
        if len(operands) != len(semantics.inputs) or any(
            not compatible(value.type, declared) for value, declared in zip(operands, semantics.inputs)
        ):
            raise TypeError("external inputs do not match the registered semantic signature")
        declared = semantics.results[0]
        if result_type is not None:
            inferred = as_type(result_type)
        elif semantics.shape_rule == "same_as" and isinstance(declared, TensorType):
            source_type = operands[semantics.shape_operand].type
            if not isinstance(source_type, TensorType):
                raise TypeError("tensor same_as requires a tensor shape operand")
            inferred = TensorType(source_type.shape, declared.dtype)
        else:
            inferred = declared
        if not compatible(inferred, declared):
            raise TypeError("external result type does not match its semantic signature")
        return self._emit("ask.external", operands, inferred,
                          {"semantics": f"@{semantics.name}", "implementation": f"@{implementation.name}"},
                          id=id, value_id=value_id)

    def mark(self, value: Value, *, mark_id: str, role: str = "value",
             port_id: str | None = None, id: str | None = None) -> Value:
        self._check_values((value,))
        if role not in ("input", "output", "value"):
            raise ValueError("mark role must be input, output or value")
        if role != "value" and not port_id:
            raise ValueError("input/output marks require port_id")
        if not isinstance(mark_id, str) or not mark_id:
            raise ValueError("mark_id must be a nonempty string")
        attrs = {"mark_id": mlir_string(mark_id), "role": mlir_string(role),
                 "value_id": mlir_string(value.value_id)}
        if port_id is not None:
            attrs["port_id"] = mlir_string(port_id)
        return self._emit("ask.mark", (value,), value.type, attrs, id=id, value_id=value.value_id)

    def return_(self, *values: Value) -> None:
        self._check_values(values)
        if len(values) != len(self.result_types) or any(
            value.type != declared for value, declared in zip(values, self.result_types)
        ):
            raise TypeError("returned values must exactly match the declared graph result types")
        self._returned = tuple(values)

    def _serialize(self) -> list[str]:
        if self._returned is None:
            raise ValueError(f"graph {self.name!r} has no return")
        arguments = ", ".join(
            f"{value.name}: {value.type} {{ask.value_id = {mlir_string(value.value_id)}}}"
            for value in self.inputs.values()
        )
        result = "" if not self.result_types else " -> " + _types(self.result_types)
        lines = [f"func.func @{self.name}({arguments}){result} {{"]
        lines.extend("  " + operation for operation in self._operations)
        if self._returned:
            lines.append("  func.return " + ", ".join(value.name for value in self._returned)
                         + " : " + ", ".join(str(value.type) for value in self._returned))
        else:
            lines.append("  func.return")
        lines.append("}")
        return lines
