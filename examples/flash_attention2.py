"""Run a standalone, tiled FlashAttention-2-style forward through ASK.

The kernel source and pinned upstream/license references live in
examples/kernels/flash_attention2.py. Dense FP16/BF16 forward only; no backward,
autotuning, dropout, arbitrary strides or performance claim. CAUSAL is top-left
(key index <= query index), also for unequal query/key lengths.

Example (with an ASK build enabling the Triton adapter):
  python examples/flash_attention2.py --triton-python /path/to/venv/bin/python \
    --llc /path/to/llc --ptxas /path/to/ptxas --dtype bf16 --causal
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from ask import Compiler, CompilerEnvironment, ExecutionSession, Module, TensorType

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = Path(__file__).with_name('kernels') / 'flash_attention2.py'


def build_module(*, batch=1, heads=2, m=37, n=55, d=32, dv=64, dtype='f16',
                 causal=False, arch=86, block_m=32, block_n=32):
    if any(type(value) is not int or value <= 0 for value in (batch, heads, m, n)):
        raise ValueError('batch, heads, query length and key length must be positive integers')
    if d not in (32, 64, 128) or dv not in (32, 64, 128):
        raise ValueError('this example supports head and value dimensions 32, 64 or 128')
    if dtype not in ('f16', 'bf16') or type(causal) is not bool:
        raise ValueError('use dtype=f16|bf16 and a boolean causal flag')
    if block_m != 32 or block_n != 32:
        raise ValueError('this example fixes BLOCK_M=BLOCK_N=32 for reproducibility')
    if type(arch) is not int or arch < 80:
        raise ValueError('this attention example requires sm80 or newer')
    q = TensorType((batch, heads, m, d), dtype)
    k = TensorType((batch, heads, n, d), dtype)
    v = TensorType((batch, heads, n, dv), dtype)
    out = TensorType((batch, heads, m, dv), dtype)
    module = Module()
    semantics = module.semantics('dense_attention_forward', inputs=[q, k, v], results=[out], shape_rule='fixed')
    pointer = '*fp16' if dtype == 'f16' else '*bf16'
    implementation = module.implementation(
        'flash_attention2', compiler='triton', source=KERNEL_PATH.read_text(encoding='utf-8'),
        entry='flash_attention2', signature={name: pointer for name in ('Q', 'K', 'V', 'O')},
        source_bindings={'Q': 'input:0', 'K': 'input:1', 'V': 'input:2', 'O': 'output:0'},
        constants={'M': m, 'N': n, 'D': d, 'DV': dv, 'CAUSAL': int(causal),
                   'BLOCK_M': block_m, 'BLOCK_N': block_n},
        options={'num_warps': 4, 'num_stages': 2},
        target={'backend': 'cuda', 'arch': arch, 'warp_size': 32},
        grid=((m + block_m - 1) // block_m, batch * heads, 1),
    )
    graph = module.graph('main', inputs={'q': q, 'k': k, 'v': v}, results=[out])
    result = graph.external(*(graph.inputs[name] for name in ('q', 'k', 'v')),
                            semantics=semantics, implementation=implementation)
    graph.return_(result)
    return module


def sample_inputs(*, batch, heads, m, n, d, dv, seed=42, scale=1.0):
    rng = random.Random(seed)
    return tuple([rng.uniform(-scale, scale) for _ in range(count)]
                 for count in (batch * heads * m * d, batch * heads * n * d,
                               batch * heads * n * dv))


def reference(q, k, v, *, batch, heads, m, n, d, dv, causal):
    """FP64 host reference over already quantized, flattened input values.

    Deliberately uses a direct row softmax and full key loop, independent of the
    kernel's tiling and online accumulator. Only for small validation examples.
    """
    output = []
    for bh in range(batch * heads):
        for row in range(m):
            keys = min(n, row + 1) if causal else n
            scores = [math.fsum(q[(bh * m + row) * d + dim] *
                                k[(bh * n + col) * d + dim] for dim in range(d)) / math.sqrt(d)
                      for col in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            output.extend(math.fsum(weight * v[(bh * n + col) * dv + dim]
                                    for col, weight in enumerate(weights)) / denominator
                          for dim in range(dv))
    return output


def check_result(actual, expected, dtype):
    atol, rtol = (0.002, 0.002) if dtype == 'f16' else (0.02, 0.02)
    if len(actual) != len(expected):
        raise AssertionError('attention result shape differs from the reference')
    errors = [abs(value - target) for value, target in zip(actual, expected)]
    for index, (value, target, error) in enumerate(zip(actual, expected, errors)):
        if not math.isfinite(value) or error > atol + rtol * abs(target):
            raise AssertionError(f'attention[{index}]={value}, reference={target}, error={error}; '
                                 f'atol={atol}, rtol={rtol}')
    return {'max_abs_error': max(errors, default=0.0), 'atol': atol, 'rtol': rtol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('batch', 1), ('heads', 2), ('m', 37), ('n', 55), ('d', 32), ('dv', 64)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--dtype', choices=('f16', 'bf16'), default='f16')
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--emit-only', action='store_true')
    parser.add_argument('--ask-opt', default=os.environ.get('ASK_OPT'))
    parser.add_argument('--triton-python', default=os.environ.get('ASK_TRITON_PYTHON', sys.executable))
    parser.add_argument('--triton-worker', default=str(ROOT / 'adapters/triton/worker.py'))
    parser.add_argument('--llc', default=os.environ.get('ASK_LLC'))
    parser.add_argument('--ptxas', default=os.environ.get('ASK_PTXAS'))
    args = parser.parse_args()
    shape = {name: getattr(args, name) for name in ('batch', 'heads', 'm', 'n', 'd', 'dv')}
    if args.emit_only:
        print(build_module(**shape, dtype=args.dtype, causal=args.causal).to_mlir(), end='')
        return
    if not args.llc or not args.ptxas:
        parser.error('provide --llc/--ptxas or ASK_LLC/ASK_PTXAS')
    with ExecutionSession() as session, tempfile.TemporaryDirectory(prefix='ask-flash-attention-') as work:
        major, minor = session.capability
        module = build_module(**shape, dtype=args.dtype, causal=args.causal, arch=major * 10 + minor)
        compiler = Compiler(args.ask_opt) if args.ask_opt else Compiler()
        kernel = compiler.compile_cuda(module, llc=args.llc, ptxas=args.ptxas,
            compilers={'triton': CompilerEnvironment(args.triton_python, args.triton_worker)},
            cache_dir=Path(work) / 'cache', output_dir=Path(work) / 'out')
        arrays = sample_inputs(**shape)
        b, h, m, n, d, dv = (shape[name] for name in ('batch', 'heads', 'm', 'n', 'd', 'dv'))
        inputs = [session.from_host(values, shape=dimensions, dtype=args.dtype)
                  for values, dimensions in zip(arrays, ((b, h, m, d), (b, h, n, d), (b, h, n, dv)))]
        output = session.tensor((b, h, m, dv), args.dtype)
        kernel.prepare(session).submit_many(inputs, [output]).wait()
        expected = reference(*(value.to_host() for value in inputs), **shape, causal=args.causal)
        error = check_result(output.to_host(), expected, args.dtype)
        print(json.dumps({'device': f'sm_{major * 10 + minor}', 'dtype': args.dtype,
                          'shape': shape, 'causal': args.causal, 'verified': True, **error}, indent=2))


if __name__ == '__main__':
    main()
