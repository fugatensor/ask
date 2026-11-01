"""Dense FlashAttention-2-style forward, pointer ABI for Ampere and newer.

Adapted from the online-softmax forward algorithm in Triton's fused attention
example (OpenAI kernel team), pinned to Triton 3.7.1 commit
f797708c0626e5f9840ca5b0a98790e2c7cb09ad:
https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/tutorials/06-fused-attention.py
Algorithm: https://arxiv.org/abs/2307.08691v1 (Tri Dao, FlashAttention-2).
Upstream license: LICENSE.triton in this directory (MIT).

One program owns BLOCK_M query rows and streams BLOCK_N key/value rows.
The accumulator, online maximum and normalization sum use FP32; both dot
products use Tensor Cores with FP16/BF16 operands. No quadratic score buffer,
autotuning, dropout, bias, variable lengths, backward or performance claim.
Contiguous Q[B,H,M,D], K[B,H,N,D], V[B,H,N,DV] -> O[B,H,M,DV].
CAUSAL uses top-left alignment: key position <= query position, including M!=N.
"""
import triton
import triton.language as tl


@triton.jit
def flash_attention2(Q, K, V, O, M: tl.constexpr, N: tl.constexpr,
                     D: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    value_dims = tl.arange(0, DV)
    q = tl.load(Q + batch_head * M * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < M, other=0)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DV), tl.float32)
    stop = N
    if CAUSAL:
        stop = tl.minimum(N, (query_block + 1) * BLOCK_M)
    for start in range(0, stop, BLOCK_N):
        positions = start + cols
        k = tl.load(K + batch_head * N * D + positions[None, :] * D + dims[:, None],
                    mask=positions[None, :] < N, other=0)
        scores = tl.dot(q, k) * (D ** -0.5) * 1.4426950408889634
        allowed = positions[None, :] < N
        if CAUSAL:
            allowed = allowed & (rows[:, None] >= positions[None, :])
        scores = tl.where(allowed, scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp2(scores - new_maximum[:, None])
        correction = tl.exp2(maximum - new_maximum)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None]
        v = tl.load(V + batch_head * N * DV + positions[:, None] * DV + value_dims[None, :],
                    mask=positions[:, None] < N, other=0)
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)
        maximum = new_maximum
    result = accumulator / denominator[:, None]
    tl.store(O + batch_head * M * DV + rows[:, None] * DV + value_dims[None, :],
             result.to(O.dtype.element_ty), mask=rows[:, None] < M)
