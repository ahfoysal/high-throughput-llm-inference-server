"""PagedKVCache — M3 block-paged KV cache (pure PyTorch, CPU-friendly).

Real PagedAttention (vLLM) uses a fused Triton/CUDA kernel to attend over
KV tensors scattered across fixed-size blocks, indexed per-sequence by a
"block table". That kernel requires a GPU, which this MacBook doesn't
have. We implement the *data structures and semantics* in pure PyTorch
so the correctness properties (no fragmentation, per-sequence block
tables, O(1) alloc/free) hold and so the attention code in
`attention.py` really does scatter/gather through the block table.

Layout
------
- Global KV pool, one big tensor per layer:
      key_pool[layer]   : (num_blocks, num_heads, block_size, head_dim)
      value_pool[layer] : (num_blocks, num_heads, block_size, head_dim)
  Logical block id `b` indexes the leading axis. Free blocks are
  tracked in a free-list.

- Per sequence: `BlockTable` — a growing list of block ids plus
  `length` (number of tokens actually written). The last block may be
  partially filled; `length % block_size` is the offset inside it.

- Allocator: O(1) `allocate()` pops from the free-list;
  `free(seq_id)` returns all of the sequence's blocks to the free-list
  and deletes its table.

Semantics match vLLM's paged cache at the block/table level. Attention
in `attention.py` reconstructs the contiguous (length, head_dim) KV for
a sequence by gathering through its block table — that's the paged
equivalent of the kernel's in-place scatter/gather.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class BlockTable:
    """Per-sequence mapping of logical token positions -> physical block ids."""

    block_ids: List[int] = field(default_factory=list)
    length: int = 0  # number of tokens actually written (<= len(block_ids)*block_size)

    def num_blocks(self) -> int:
        return len(self.block_ids)

    def last_block_offset(self, block_size: int) -> int:
        """Offset of the next write position within the last block."""
        return self.length % block_size


class AllocationError(RuntimeError):
    pass


class PagedKVCache:
    """Block-paged KV cache for one transformer layer stack.

    Parameters
    ----------
    num_layers : int
    num_heads  : int
    head_dim   : int
    block_size : int         tokens per block
    num_blocks : int         total blocks in the pool (capacity = num_blocks*block_size tokens)
    dtype      : torch.dtype
    device     : torch.device or str
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_blocks, num_heads, block_size, head_dim)
        self.key_pool: List[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.value_pool: List[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]

        # Free-list: newest-first (stack) for cache locality on reuse.
        self._free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))
        self._tables: Dict[int, BlockTable] = {}

    # ------------------------------------------------------------------
    # Allocator
    # ------------------------------------------------------------------
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def can_allocate(self, tokens: int) -> bool:
        blocks_needed = (tokens + self.block_size - 1) // self.block_size
        return blocks_needed <= len(self._free_blocks)

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self._tables:
            raise AllocationError(f"seq {seq_id} already exists")
        self._tables[seq_id] = BlockTable()

    def free(self, seq_id: int) -> None:
        table = self._tables.pop(seq_id, None)
        if table is None:
            return
        # Return blocks to the free-list. Order doesn't matter for
        # correctness; reverse keeps hot blocks near the top.
        for bid in reversed(table.block_ids):
            self._free_blocks.append(bid)

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise AllocationError("no free blocks")
        return self._free_blocks.pop()

    def _ensure_capacity(self, table: BlockTable, new_tokens: int) -> None:
        """Grow `table` so it can hold `new_tokens` more tokens."""
        needed_total = table.length + new_tokens
        needed_blocks = (needed_total + self.block_size - 1) // self.block_size
        while table.num_blocks() < needed_blocks:
            table.block_ids.append(self._allocate_block())

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def append(
        self,
        seq_id: int,
        layer: int,
        keys: torch.Tensor,   # (T, H, D) new K for this sequence/layer
        values: torch.Tensor, # (T, H, D) new V
    ) -> None:
        """Append T new tokens of K/V for (seq_id, layer) into its blocks.

        For a sequence, callers must append with consistent `layer` calls
        per step (one call per layer per decode step). The table's
        `length` is only advanced once per step — after the *last* layer
        writes — via `advance_length(seq_id, T)`.
        """
        table = self._tables[seq_id]
        # Grow blocks (idempotent across layers because length isn't advanced yet).
        self._ensure_capacity(table, keys.shape[0])

        T = keys.shape[0]
        write_pos = table.length  # start offset in logical token space
        block_size = self.block_size

        k_pool = self.key_pool[layer]
        v_pool = self.value_pool[layer]

        # Scatter into (possibly multiple) blocks.
        written = 0
        while written < T:
            logical = write_pos + written
            block_idx_in_table = logical // block_size
            offset = logical % block_size
            bid = table.block_ids[block_idx_in_table]
            room = block_size - offset
            take = min(room, T - written)
            # keys[written:written+take] is (take, H, D); pool slot is (H, block_size, D).
            # Transpose to match (H, take, D) -> assign into [bid, :, offset:offset+take, :].
            chunk_k = keys[written:written + take].transpose(0, 1)  # (H, take, D)
            chunk_v = values[written:written + take].transpose(0, 1)
            k_pool[bid, :, offset:offset + take, :] = chunk_k
            v_pool[bid, :, offset:offset + take, :] = chunk_v
            written += take

    def advance_length(self, seq_id: int, new_tokens: int) -> None:
        """Bump logical length after all layers have written this step."""
        self._tables[seq_id].length += new_tokens

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def gather(self, seq_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct contiguous K, V of shape (H, L, D) for this sequence/layer."""
        table = self._tables[seq_id]
        L = table.length
        if L == 0:
            empty = torch.empty(
                (self.num_heads, 0, self.head_dim),
                dtype=self.dtype, device=self.device,
            )
            return empty, empty

        k_pool = self.key_pool[layer]
        v_pool = self.value_pool[layer]
        block_size = self.block_size

        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        consumed = 0
        for i, bid in enumerate(table.block_ids):
            remaining = L - consumed
            if remaining <= 0:
                break
            take = min(block_size, remaining)
            k_parts.append(k_pool[bid, :, :take, :])  # (H, take, D)
            v_parts.append(v_pool[bid, :, :take, :])
            consumed += take

        k = torch.cat(k_parts, dim=1)  # (H, L, D)
        v = torch.cat(v_parts, dim=1)
        return k, v

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def length_of(self, seq_id: int) -> int:
        return self._tables[seq_id].length

    def table_of(self, seq_id: int) -> BlockTable:
        return self._tables[seq_id]

    def utilization(self) -> float:
        used = self.num_blocks - len(self._free_blocks)
        return used / max(1, self.num_blocks)

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}, "
            f"blocks={self.num_blocks}, free={len(self._free_blocks)}, "
            f"seqs={len(self._tables)})"
        )
