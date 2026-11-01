"""Explicit compiler adapters selected by the native build's registration set.

Loading this module never imports an external compiler or discovers entrypoints.
Build authorization is checked before importing any selected adapter module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module as _import_module
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .compiler import CompilerError


@dataclass(frozen=True)
class CompilerEnvironment:
    """Explicit subprocess environment for a registered compiler exporter."""
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


class CompilerAdapter(ABC):
    """Producer-specific policy for an otherwise generic compilation pipeline."""
    compiler_id: str

    @abstractmethod
    def cache_identity(self, request: dict[str, Any], environment: CompilerEnvironment,
                       *, timeout: float) -> dict[str, Any]:
        """Validate producer policy and return producer/adapter/DSL/numerics identity."""


def _build_registrations():
    from . import __path__ as package_paths
    candidates = [Path(directory) / "_adapter_registry.json" for directory in package_paths]
    found = [path for path in candidates if path.is_file()]
    if not found:
        return {}
    contents = [json.loads(path.read_text()) for path in found]
    if any(value != contents[0] for value in contents[1:]):
        raise CompilerError("multiple ASK builds expose different compiler adapter registrations")
    if not isinstance(contents[0], dict):
        raise CompilerError("compiler adapter registration must be an object keyed by compiler ID")
    return contents[0]


class AdapterRegistry:
    """Only native-enabled IDs may resolve an explicitly registered factory."""
    def __init__(self, capabilities: dict[str, Any], *, registrations=None):
        enabled = capabilities.get("third_party_compilers", [])
        descriptions = capabilities.get("compiler_adapters", [])
        if not isinstance(enabled, list) or not isinstance(descriptions, list):
            raise CompilerError("invalid native compiler adapter capabilities")
        self._enabled = set(enabled)
        self._capabilities = {item["id"]: item for item in descriptions
                              if isinstance(item, dict) and isinstance(item.get("id"), str)}
        self._registrations = registrations

    def require_enabled(self, compiler_id, *, fusion=False):
        if not isinstance(compiler_id, str) or compiler_id not in self._enabled:
            raise CompilerError(f"this ASK build does not enable compiler {compiler_id!r}")
        capability = self._capabilities.get(compiler_id, {})
        if fusion and capability.get("supports_fusion") is not True:
            raise CompilerError(f"compiler {compiler_id!r} does not support native LLIR fusion in this build")

    def require_any(self):
        if not self._enabled:
            raise CompilerError("this ASK build does not enable any third-party compiler")

    def registration(self, compiler_id):
        self.require_enabled(compiler_id)
        if self._registrations is None:
            self._registrations = _build_registrations()
        entry = self._registrations.get(compiler_id)
        if not isinstance(entry, dict) or entry.get("id", compiler_id) != compiler_id:
            raise CompilerError(f"enabled compiler {compiler_id!r} has no matching Python registration")
        if entry.get("protocol_version") != self._capabilities.get(compiler_id, {}).get("protocol_version"):
            raise CompilerError(f"compiler {compiler_id!r} registration protocol differs from the native build")
        return entry

    def load(self, compiler_id):
        entry = self.registration(compiler_id)
        module, factory = entry.get("module"), entry.get("factory")
        if not isinstance(module, str) or not isinstance(factory, str):
            raise CompilerError(f"compiler {compiler_id!r} has an invalid adapter factory registration")
        try:
            adapter = getattr(_import_module(module), factory)()
        except (ImportError, AttributeError) as error:
            raise CompilerError(f"cannot load registered compiler adapter {compiler_id!r}: {error}") from error
        if not isinstance(adapter, CompilerAdapter) or adapter.compiler_id != compiler_id:
            raise CompilerError(f"compiler adapter factory {compiler_id!r} returned an incompatible adapter")
        return adapter

    def environments(self, compilers: Mapping[str, CompilerEnvironment] | None, legacy_options=None):
        self.require_any()
        if compilers is not None and not isinstance(compilers, Mapping):
            raise TypeError("compilers must map compiler IDs to CompilerEnvironment values")
        result = dict(compilers or {})
        for compiler_id, environment in result.items():
            self.require_enabled(compiler_id)
            if not isinstance(environment, CompilerEnvironment):
                raise TypeError(f"compiler {compiler_id!r} requires a CompilerEnvironment")
        legacy = dict(legacy_options or {})
        if legacy:
            # Aliases are inert build metadata, not backend-specific conditionals.
            aliases = {}
            for compiler_id in sorted(self._enabled):
                for name, field in self.registration(compiler_id).get("aliases", {}).items():
                    if name in aliases or field not in ("python", "worker"):
                        raise CompilerError("ambiguous or invalid compiler environment alias registration")
                    aliases[name] = (compiler_id, field)
            grouped = {}
            for name, value in legacy.items():
                if name not in aliases:
                    raise TypeError(f"unknown compiler environment option: {name}")
                compiler_id, field = aliases[name]
                if compiler_id in result:
                    raise TypeError(f"compiler {compiler_id!r} environment was provided twice")
                fields = grouped.setdefault(compiler_id, {})
                if field in fields:
                    raise TypeError(f"compiler {compiler_id!r} environment {field} was provided twice")
                fields[field] = value
            for compiler_id, fields in grouped.items():
                if "worker" not in fields:
                    raise TypeError(f"compiler {compiler_id!r} requires a worker path")
                result[compiler_id] = CompilerEnvironment(**fields)
        return result
