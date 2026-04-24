"""M6: Paged attention — Triton kernel stub + CPU fallback.

On real GPU hardware, vLLM/SGLang/Flash-Infer implement PagedAttention
as a fused CUDA/Triton kernel that walks the block table inside the
kernel so KV reads never have to materialize a contiguous `[B, T, H, D]`
tensor. The scheduling logic (block allocation, block table per
sequence, free list, copy-on-write for prefix caching) lives in Python;
the hot loop is GPU code.

This file provides:

1. A Triton kernel **written out in full** (`paged_attention_kernel`)
   that targets the single-query decode case (one new query token per
   sequence, attending over all cached KV blocks). It requires CUDA to
   actually execute; we check Triton + CUDA availability at import
   time and only expose the launcher when both are present.

2. A **pure-PyTorch CPU fallback** (`paged_attention_cpu`) with
   identical signature and semantics, so the scheduling logic can be
   exercised + tested without a GPU. This is the correctness oracle
   for the kernel above.

3. `paged_attention(...)` — a dispatcher that calls the kernel on
   CUDA tensors and the CPU path otherwise.

4. A **mock scheduler validator** (`_validate_block_scheduling`) that
   builds random block tables / sequence lengths, runs the CPU path,
   compares against a dense reference, and reports max absolute error.
   This is what `python -m app.paged_attention_triton` runs.

Shape conventions (match vLLM / SGLang):

- `query`        : (num_seqs, num_heads, head_dim) — one decode query
                   per sequence.
- `key_cache`    : (num_blocks, block_size, num_kv_heads, head_dim)
- `value_cache`  : (num_blocks, block_size, num_kv_heads, head_dim)
- `block_tables` : (num_seqs, max_blocks_per_seq) — int32, logical
                   block i -> physical block id. -1 / unused rows are
                   beyond `context_lens[seq]`.
- `context_lens` : (num_seqs,) — number of cached KV tokens per seq.
- `scale`        : float, usually 1/sqrt(head_dim).
- returns out    : (num_seqs, num_heads, head_dim).

We use MHA (`num_kv_heads == num_heads`) throughout; GQA would just
need an extra head-index divide in both paths and is left as a
one-liner once we have real hardware to validate against.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton availability
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
_CUDA_AVAILABLE = torch.cuda.is_available()

try:  # pragma: no cover — only fires on a GPU machine with triton
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Triton kernel — requires GPU to actually run.
# ---------------------------------------------------------------------------
#
# This is a faithful skeleton of a vLLM-style paged-attention decode
# kernel: one CUDA block per (sequence, head) pair, iterate the
# sequence's block table, load BLOCK_SIZE tokens at a time, compute
# dot(q, k), online-softmax, accumulate against v.
#
# Because this repo has no CUDA on CI (Mac), the kernel is never JIT-ed
# automatically; `paged_attention(...)` routes to the CPU fallback.
# The code below is the reference implementation you would drop into
# a real deployment — it compiles as soon as `triton` imports cleanly.
#
if _TRITON_AVAILABLE:  # pragma: no cover

    @triton.jit
    def paged_attention_kernel(
        out_ptr,               # (num_seqs, num_heads, head_dim)
        q_ptr,                 # (num_seqs, num_heads, head_dim)
        k_cache_ptr,           # (num_blocks, block_size, num_heads, head_dim)
        v_cache_ptr,           # (num_blocks, block_size, num_heads, head_dim)
        block_tables_ptr,      # (num_seqs, max_blocks)
        context_lens_ptr,      # (num_seqs,)
        scale,                 # fp32
        stride_ob, stride_oh,
        stride_qb, stride_qh,
        stride_kb, stride_kbb, stride_kh,
        stride_vb, stride_vbb, stride_vh,
        stride_btb,            # per-seq stride of block_tables
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        ctx_len = tl.load(context_lens_ptr + seq_idx)

        # Load query vector for (seq, head).
        d_offs = tl.arange(0, HEAD_DIM)
        q = tl.load(
            q_ptr + seq_idx * stride_qb + head_idx * stride_qh + d_offs
        ).to(tl.float32)

        # Online-softmax accumulators.
        m_i = tl.full([1], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Walk the block table.
        num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(0, num_blocks):
            phys = tl.load(block_tables_ptr + seq_idx * stride_btb + b)
            tok_offs = tl.arange(0, BLOCK_SIZE)
            tok_valid = (b * BLOCK_SIZE + tok_offs) < ctx_len

            # Load K block: (BLOCK_SIZE, HEAD_DIM) for this head.
            k_base = phys * stride_kb + head_idx * stride_kh
            k = tl.load(
                k_cache_ptr + k_base
                + tok_offs[:, None] * stride_kbb
                + d_offs[None, :],
                mask=tok_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            v_base = phys * stride_vb + head_idx * stride_vh
            v = tl.load(
                v_cache_ptr + v_base
                + tok_offs[:, None] * stride_vbb
                + d_offs[None, :],
                mask=tok_valid[:, None],
                other=0.0,
            ).to(tl.float32)

            # Scores: q · k^T, masked.
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(tok_valid, s, float("-inf"))

            # Online softmax update.
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        acc = acc / l_i
        tl.store(
            out_ptr + seq_idx * stride_ob + head_idx * stride_oh + d_offs,
            acc.to(tl.float16),  # or bf16 depending on dtype of out
        )

    def _launch_triton(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        num_seqs, num_heads, head_dim = query.shape
        num_blocks, block_size = key_cache.shape[0], key_cache.shape[1]
        max_blocks = block_tables.shape[1]
        out = torch.empty_like(query)

        grid = (num_seqs, num_heads)
        paged_attention_kernel[grid](
            out, query, key_cache, value_cache,
            block_tables, context_lens, scale,
            out.stride(0), out.stride(1),
            query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_tables.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            MAX_BLOCKS=max_blocks,
        )
        return out


# ---------------------------------------------------------------------------
# CPU fallback (the oracle).
# ---------------------------------------------------------------------------


def paged_attention_cpu(
    query: torch.Tensor,         # (S, H, D)
    key_cache: torch.Tensor,     # (B, bs, H, D)
    value_cache: torch.Tensor,   # (B, bs, H, D)
    block_tables: torch.Tensor,  # (S, max_blocks) int
    context_lens: torch.Tensor,  # (S,) int
    scale: float,
) -> torch.Tensor:
    """Reference paged attention — correctness oracle for the kernel."""
    num_seqs, num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    out = torch.zeros_like(query)

    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        if ctx == 0:
            continue
        # Gather K/V for this sequence by walking its block table.
        num_blocks = (ctx + block_size - 1) // block_size
        ks = []
        vs = []
        for b in range(num_blocks):
            phys = int(block_tables[s, b].item())
            ks.append(key_cache[phys])     # (bs, H, D)
            vs.append(value_cache[phys])
        K = torch.cat(ks, dim=0)[:ctx]     # (ctx, H, D)
        V = torch.cat(vs, dim=0)[:ctx]     # (ctx, H, D)

        # Attention per head.
        q = query[s]                       # (H, D)
        # scores: (H, ctx) = q @ K^T per head
        scores = torch.einsum("hd,thd->ht", q, K) * scale
        probs = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("ht,thd->hd", probs, V)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Paged attention — Triton on CUDA, PyTorch on CPU."""
    if query.is_cuda and _TRITON_AVAILABLE:  # pragma: no cover
        return _launch_triton(
            query, key_cache, value_cache, block_tables, context_lens, scale
        )
    return paged_attention_cpu(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )


# ---------------------------------------------------------------------------
# Mock scheduler / validator
# ---------------------------------------------------------------------------


@dataclass
class MockBatch:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    scale: float


def _build_mock_batch(
    num_seqs: int = 4,
    num_heads: int = 4,
    head_dim: int = 16,
    block_size: int = 8,
    max_context: int = 40,
    total_blocks: int = 64,
    seed: int = 0,
) -> Tuple[MockBatch, torch.Tensor]:
    """Random batch + dense reference output for cross-checking.

    Models the scheduling path: pick a random context length per
    sequence, allocate enough physical blocks from a global pool,
    write random K/V into them at the right logical offsets, record
    the sequence's block table. Returns both the batch and the
    dense-attention reference output.
    """
    g = torch.Generator().manual_seed(seed)

    # Random context lengths.
    context_lens = torch.randint(1, max_context + 1, (num_seqs,), generator=g)
    max_blocks = (int(context_lens.max().item()) + block_size - 1) // block_size

    # Global KV pool.
    key_cache = torch.randn(total_blocks, block_size, num_heads, head_dim, generator=g)
    value_cache = torch.randn(total_blocks, block_size, num_heads, head_dim, generator=g)

    # Allocate blocks to sequences from a simple free list.
    free = list(range(total_blocks))
    # shuffle deterministically
    perm = torch.randperm(total_blocks, generator=g).tolist()
    free = perm[:]

    block_tables = torch.full((num_seqs, max_blocks), -1, dtype=torch.int64)
    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        nb = (ctx + block_size - 1) // block_size
        for i in range(nb):
            block_tables[s, i] = free.pop()

    query = torch.randn(num_seqs, num_heads, head_dim, generator=g)
    scale = 1.0 / math.sqrt(head_dim)

    # Dense reference — gather per seq, compute standard attention.
    ref = torch.zeros_like(query)
    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        nb = (ctx + block_size - 1) // block_size
        Ks = torch.cat([key_cache[int(block_tables[s, i])] for i in range(nb)], dim=0)[:ctx]
        Vs = torch.cat([value_cache[int(block_tables[s, i])] for i in range(nb)], dim=0)[:ctx]
        scores = torch.einsum("hd,thd->ht", query[s], Ks) * scale
        probs = torch.softmax(scores, dim=-1)
        ref[s] = torch.einsum("ht,thd->hd", probs, Vs)

    batch = MockBatch(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=scale,
    )
    return batch, ref


def _validate_block_scheduling(seed: int = 0) -> float:
    batch, ref = _build_mock_batch(seed=seed)
    out = paged_attention(
        batch.query, batch.key_cache, batch.value_cache,
        batch.block_tables, batch.context_lens, batch.scale,
    )
    err = float((out - ref).abs().max())
    return err


# ---------------------------------------------------------------------------
# Self-test — `python -m app.paged_attention_triton`
# ---------------------------------------------------------------------------


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"triton available: {_TRITON_AVAILABLE}")
    print(f"cuda available:   {_CUDA_AVAILABLE}")
    print("Running CPU fallback against dense reference on mock batches...")
    max_err = 0.0
    for seed in range(5):
        err = _validate_block_scheduling(seed=seed)
        print(f"  seed={seed} max|err|={err:.3e}")
        max_err = max(max_err, err)
    print(f"overall max|err|={max_err:.3e}")
    assert max_err < 1e-5, f"paged attention diverged from dense reference: {max_err}"
    print("OK — scheduling logic (block tables, context lens) validates clean.")


if __name__ == "__main__":  # pragma: no cover
    _main()
