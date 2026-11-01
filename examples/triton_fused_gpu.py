"""Run ASK/Triton fusion with the shared typed graph test matrix.

  PYTHONPATH=python python3 examples/triton_fused_gpu.py \
    --ask-opt build/triton/bin/ask-opt \
    --triton-python /path/to/triton/python \
    --triton-worker adapters/triton/worker.py \
    --llc .deps/llvm-1f126a6d-ubuntu-x64-1/bin/llc \
    --ptxas /path/to/triton/backends/nvidia/bin/ptxas \
    --cache-dir build/new-cache --output-dir build/new-results \
    --dtypes i32 f32 f16 bf16

Use --cases to select graph templates and --extended for additional shapes.
Graph definitions and numerical references live in tests/support/pointwise.py;
adding a precision does not require another copy of the graph or the runner.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
# Keep the example's programmatic helpers available to existing local scripts.
from support.pointwise import (build_module, expected_values, input_values,
                               json_values, numerically_equal)
from support.gpu_validation import main


if __name__ == "__main__":
    main()
