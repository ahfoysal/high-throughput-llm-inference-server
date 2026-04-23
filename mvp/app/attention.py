"""Paged attention (pure PyTorch, CPU-friendly) — M3.

On a real GPU deployment this is a fused Triton kernel: one block of
threads per (head, sequence) pair, each walking that sequence's block
table in `PagedKVCache` and computing softmax(Q K^T / sqrt(d)) V
entirely inside SRAM, never materializing the full (B, H, L, D) tensor.

Mac has no CUDA, so we implement the *semantics*: the attention op
reads K,V from the paged cache by gathering through per-sequence block
tables, then does a standard masked attention. This proves the block-
table plumbing round-trips correctly; the speed story is deferred to a
CUDA host.

We also include `reference_attention` (a dense, unpaged attention over
contiguous K,V) so correctness tests can assert the paged and dense
variants produce numerically identical outputs.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from .paged_cache import PagedKVCache


# ---------------------------------------------------------------------------
# Reference dense attention (no paging)
# ---------------------------------------------------------------------------

def reference_attention(
    q: torch.Tensor,        # (H, Tq, D)
    k: torch.Tensor,        # (H, Tk, D)
    v: torch.Tensor,        # (H, Tk, D)
    causal_offset: int = 0, # position of q[0] in the full sequence (for causal masking)
) -> torch.Tensor:
    """Standard multi-head attention for a single sequence.

    Returns (H, Tq, D). If `causal_offset` > 0, queries are assumed to
    start at that absolute position and a lower-triangular mask is
    applied over the full Tk length.
    """
    H, Tq, D = q.shape
    Tk = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    # scores: (H, Tq, Tk)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Causal mask: query at (causal_offset + i) can only attend to keys 0..(causal_offset+i).
    if Tq > 0 and Tk > 0:
        q_pos = torch.arange(Tq, device=q.device).unsqueeze(-1) + causal_offset  # (Tq, 1)
        k_pos = torch.arange(Tk, device=q.device).unsqueeze(0)                   # (1, Tk)
        mask = k_pos > q_pos  # True where disallowed
        if mask.any():
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # (H, Tq, D)
    return out


# ---------------------------------------------------------------------------
# Paged attention
# ---------------------------------------------------------------------------

def paged_attention_single(
    q: torch.Tensor,   # (H, Tq, D) — queries for this sequence/layer
    cache: PagedKVCache,
    seq_id: int,
    layer: int,
) -> torch.Tensor:
    """Attention for one sequence, reading K/V through the paged cache.

    Matches `reference_attention(q, k_full, v_full, causal_offset=Tk-Tq)`
    where k_full/v_full are the *entire* sequence's KV including the
    newly-appended chunk. Caller is responsible for having already
    appended this step's new K/V to the cache before calling.
    """
    k, v = cache.gather(seq_id, layer)  # (H, L, D)
    L = k.shape[1]
    Tq = q.shape[1]
    causal_offset = L - Tq
    if causal_offset < 0:
        raise ValueError(
            f"paged_attention_single: L ({L}) < Tq ({Tq}); append K/V before attending."
        )
    return reference_attention(q, k, v, causal_offset=causal_offset)


def paged_attention_batch(
    qs: List[torch.Tensor],   # list of (H, Tq_i, D) per sequence
    cache: PagedKVCache,
    seq_ids: List[int],
    layer: int,
) -> List[torch.Tensor]:
    """Run paged attention for a variable-length batch.

    Under the hood this is a Python for-loop — on CPU that's already the
    fastest option for ragged shapes (no kernel-launch overhead to
    amortize). On GPU this would become a single fused kernel whose
    grid indexes into (seq, head) tuples.
    """
    outs: List[torch.Tensor] = []
    for q, sid in zip(qs, seq_ids):
        outs.append(paged_attention_single(q, cache, sid, layer))
    return outs


# ---------------------------------------------------------------------------
# Self-test — paged vs dense must match bitwise (float32) / near-bit (float16).
# ---------------------------------------------------------------------------

def _self_test(seed: int = 0) -> None:
    """Quick correctness check. Invoked by `python -m app.attention`."""
    torch.manual_seed(seed)

    H, D = 4, 8
    block_size = 3        # deliberately small + non-power-of-2 to stress block boundaries
    prompt_len = 7        # spans >2 blocks so the last one is partial
    new_tokens = 5        # decode steps: 1 token each
    num_layers = 1

    cache = PagedKVCache(
        num_layers=num_layers,
        num_heads=H,
        head_dim=D,
        block_size=block_size,
        num_blocks=16,
    )

    # Full random K,V over the whole prompt + decode horizon — ground truth.
    total_len = prompt_len + new_tokens
    k_full = torch.randn(H, total_len, D)
    v_full = torch.randn(H, total_len, D)
    q_full = torch.randn(H, total_len, D)

    seq_id = 42
    cache.add_sequence(seq_id)

    # Prefill: append prompt_len tokens, then query the last one.
    cache.append(
        seq_id=seq_id, layer=0,
        keys=k_full[:, :prompt_len, :].transpose(0, 1),   # (T, H, D)
        values=v_full[:, :prompt_len, :].transpose(0, 1),
    )
    cache.advance_length(seq_id, prompt_len)

    # Attend with queries = the whole prompt (prefill-style).
    q_prefill = q_full[:, :prompt_len, :]
    paged_out = paged_attention_single(q_prefill, cache, seq_id, layer=0)
    ref_out = reference_attention(
        q_prefill,
        k_full[:, :prompt_len, :],
        v_full[:, :prompt_len, :],
        causal_offset=0,
    )
    assert torch.allclose(paged_out, ref_out, atol=1e-6), "prefill mismatch"

    # Decode steps: one token at a time.
    for t in range(prompt_len, total_len):
        cache.append(
            seq_id=seq_id, layer=0,
            keys=k_full[:, t:t + 1, :].transpose(0, 1),
            values=v_full[:, t:t + 1, :].transpose(0, 1),
        )
        cache.advance_length(seq_id, 1)

        q_step = q_full[:, t:t + 1, :]
        paged_step = paged_attention_single(q_step, cache, seq_id, layer=0)
        ref_step = reference_attention(
            q_step,
            k_full[:, :t + 1, :],
            v_full[:, :t + 1, :],
            causal_offset=t,
        )
        assert torch.allclose(paged_step, ref_step, atol=1e-6), f"decode step t={t} mismatch"

    # Free and check free-list is restored.
    before_free = cache.num_free_blocks()
    cache.free(seq_id)
    after_free = cache.num_free_blocks()
    assert after_free > before_free, "blocks not returned on free"

    print("paged attention self-test OK "
          f"(prompt={prompt_len}, decode={new_tokens}, block_size={block_size})")


if __name__ == "__main__":
    _self_test()
