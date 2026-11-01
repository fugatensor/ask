"""Build ASK MLIR and submit it to the native compiler using a small Python SDK."""

from pkgutil import extend_path

# Allow a source checkout and its selected build-tree extension on PYTHONPATH.
__path__ = extend_path(__path__, __name__)

from ._version import __version__
from .compiler import Compiler, CompilerError, PreparedModule
from .ir import Graph, Implementation, Module, Semantics, Value
from .types import ScalarType, TensorType
from .runtime import ExecutionSession, DeviceTensor, Event
from .adapters import CompilerAdapter, CompilerEnvironment

__all__ = [
    "__version__",
    "Compiler", "CompilerError", "PreparedModule", "Graph", "Implementation",
    "Module", "Semantics", "Value", "ScalarType", "TensorType",
    "ExecutionSession", "DeviceTensor", "Event",
    "CompilerAdapter", "CompilerEnvironment",
]
