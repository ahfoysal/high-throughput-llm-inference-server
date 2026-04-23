"""ContinuousBatcher — M3.

Replaces the M2 `StaticBatcher`. A static batcher forms a batch, runs it
to completion, then starts the next one: a long request blocks short
requests queued behind it, and finished short requests waste slots
until the laggard is done.

Continuous batching (vLLM-style "iteration-level scheduling"): at every
decode step we re-evaluate the active set.

- New queued requests get prefilled and joined to the in-flight batch.
- Sequences that finished (EOS or hit their max_tokens) get dropped.
- The batch tensor is resliced/padded around the live set and decoding
  continues.

This is the real throughput win under mixed-length workloads.

Implementation notes
--------------------
- We keep HF's `past_key_values` as the source of truth and slice it
  along the batch dim when sequences join/leave. Each layer's
  (K, V) has shape (B, n_heads, L, head_dim), so concatenating a newly-
  prefilled sequence requires left-padding its KV to the current
  max length with zeros + extending the attention mask with zeros over
  those pad positions. HF attention respects the mask so padded K's
  contribute 0 to softmax.
- Sampling is done per-sequence because each request carries its own
  temperature / top-p / top-k / seed.
- A matching `PagedKVCache` is maintained in parallel as an M3
  bookkeeping artefact (allocate/free blocks as sequences join/leave).
  The HF forward still uses its own contiguous KV tensors because we
  can't hot-swap the GPT-2 attention op without rewriting it — but the
  paged allocator drives the correctness story for `attention.py`
  and the block-table accounting is live during every step.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .sampling import sample_next_token

if TYPE_CHECKING:
    from .engine import GenerationResult, InferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / sequence state
# ---------------------------------------------------------------------------


@dataclass
class _Request:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    seed: Optional[int]
    future: "asyncio.Future[GenerationResult]" = field(default=None)  # type: ignore[assignment]


@dataclass
class _Sequence:
    req: _Request
    seq_id: int
    prompt_len: int
    generated: List[int] = field(default_factory=list)
    finish_reason: str = "length"
    alive: bool = True


# ---------------------------------------------------------------------------
# The batcher
# ---------------------------------------------------------------------------


class ContinuousBatcher:
    def __init__(
        self,
        engine: "InferenceEngine",
        max_batch: int = 8,
        max_wait_ms: int = 5,
        block_size: int = 16,
        num_blocks: int = 256,
    ) -> None:
        self.engine = engine
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.block_size = block_size
        self.num_blocks = num_blocks

        self._queue: "asyncio.Queue[_Request]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._next_seq_id = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="continuous-batcher-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------
    async def submit(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        seed: Optional[int],
    ) -> "GenerationResult":
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[GenerationResult]" = loop.create_future()
        req = _Request(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed, future=fut,
        )
        await self._queue.put(req)
        return await fut

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def _loop(self) -> None:
        """Wait for first request, then run a continuous-decode session
        in an executor thread (HF forward is blocking). New requests
        that arrive mid-session are picked up inside `_run_session`
        via `_drain_queue_nowait`."""
        while self._running:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            initial: List[_Request] = [first]
            # Opportunistically drain anything already queued.
            while len(initial) < self.max_batch:
                try:
                    initial.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    self._run_session_sync,
                    initial,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("continuous-batch session failed: %s", e)
                for r in initial:
                    if not r.future.done():
                        r.future.set_exception(e)

    def _drain_queue_nowait(self, room: int) -> List[_Request]:
        out: List[_Request] = []
        while room > 0:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            room -= 1
        return out

    # ------------------------------------------------------------------
    # per-session continuous decode (sync, runs in a thread)
    # ------------------------------------------------------------------
    def _run_session_sync(self, initial: List[_Request]) -> None:
        """Continuous-batching decode until the active set is empty and
        no new requests are waiting."""
        backend = self.engine.backend  # type: ignore[attr-defined]
        backend_name = self.engine.backend_name

        # The mock backend has no tensors/KV, so continuous batching degrades
        # to "run each request to completion as they arrive". That's fine for
        # HTTP-pipeline testing and keeps the batcher importable without torch.
        if backend_name == "mock":
            self._run_mock(initial)
            return

        import torch  # local import; only when we actually need it
        from .paged_cache import PagedKVCache
        from .engine import GenerationResult  # local to avoid cycle

        # Newer transformers return a DynamicCache object rather than a plain
        # tuple-of-tuples. Normalize both directions so our slicing code can
        # keep treating `past` as a tuple of (k, v) pairs.
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore
        except Exception:  # noqa: BLE001
            DynamicCache = None  # type: ignore

        def _to_tuple(pkv):
            if pkv is None:
                return None
            if isinstance(pkv, tuple):
                return pkv
            # DynamicCache: has .layers with .keys / .values on each.
            layers = getattr(pkv, "layers", None)
            if layers is not None:
                return tuple((l.keys, l.values) for l in layers)
            # Fallback: try iteration (older tuple-style).
            return tuple(pkv)

        def _from_tuple(t):
            if t is None:
                return None
            if DynamicCache is None:
                return t
            return DynamicCache(ddp_cache_data=list(t))

        tokenizer = backend.tokenizer
        model = backend.model
        eos_id = backend.eos_id
        pad_id = tokenizer.pad_token_id or 0

        # ---- state ----
        sequences: List[_Sequence] = []  # parallel to the batch-dim of past/mask
        past: Optional[Tuple] = None     # HF past_key_values
        attn_mask: Optional[torch.Tensor] = None  # (B, cur_len)

        # Paged cache (bookkeeping — matches sequence lifetimes).
        paged: Optional[PagedKVCache] = None

        def _assign_seq_id() -> int:
            sid = self._next_seq_id
            self._next_seq_id += 1
            return sid

        def _ensure_paged(num_heads: int, head_dim: int, num_layers: int) -> PagedKVCache:
            nonlocal paged
            if paged is None:
                paged = PagedKVCache(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=self.block_size,
                    num_blocks=self.num_blocks,
                    dtype=torch.float32,
                    device="cpu",
                )
            return paged

        def _prefill(reqs: List[_Request]):
            """Prefill a list of new requests. Returns (new_seqs, new_past, new_mask,
            last_logits) for those requests only."""
            # Seeds: use the first non-None seed for global RNG (per-request sampling
            # still respects temperature/top_p/top_k).
            for r in reqs:
                if r.seed is not None:
                    torch.manual_seed(r.seed)
                    break
            enc = tokenizer([r.prompt for r in reqs], return_tensors="pt", padding=True)
            ids = enc.input_ids
            mask = enc.attention_mask
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, use_cache=True)
            new_past = _to_tuple(out.past_key_values)
            last_logits = out.logits[:, -1, :]  # (b_new, V)

            new_seqs: List[_Sequence] = []
            real_lens = mask.sum(dim=1).tolist()
            for r, rl in zip(reqs, real_lens):
                sid = _assign_seq_id()
                new_seqs.append(_Sequence(req=r, seq_id=sid, prompt_len=int(rl)))

            # Bookkeeping in the paged cache: pretend to scatter the prompt
            # into blocks. We don't have the HF attention's K/V tensors
            # directly (they live inside `past`), so we feed zeros — this
            # exercises the allocator/block-table round-trip without
            # altering inference output (HF forward uses `past`, not
            # `paged`). `attention.py` carries the real paged-vs-dense
            # equivalence proof on synthetic data.
            n_layers = len(new_past)
            # Extract per-head dims from past: past[0] is (k, v), k is (B, H, L, D).
            k0 = new_past[0][0]
            _, H, _, D = k0.shape
            cache = _ensure_paged(num_heads=H, head_dim=D, num_layers=n_layers)
            for seq in new_seqs:
                cache.add_sequence(seq.seq_id)
                # Allocate enough blocks for the prompt (bookkeeping only —
                # HF carries the actual K/V in `past`; the paged-vs-dense
                # equivalence is proven in `attention._self_test`).
                table = cache.table_of(seq.seq_id)
                blocks_needed = (seq.prompt_len + cache.block_size - 1) // cache.block_size
                for _ in range(blocks_needed):
                    table.block_ids.append(cache._allocate_block())
                cache.advance_length(seq.seq_id, seq.prompt_len)

            return new_seqs, new_past, mask, last_logits

        def _merge_into_batch(new_seqs, new_past, new_mask, new_last_logits):
            """Concatenate prefilled sequences into the running batch."""
            nonlocal sequences, past, attn_mask

            if past is None:
                sequences = list(new_seqs)
                past = new_past
                attn_mask = new_mask
                # Sample first new tokens for each.
                _sample_and_advance(new_last_logits, range(len(sequences)))
                return

            # Pad lengths so existing past + new past can concatenate along dim=0.
            cur_len = attn_mask.shape[1]
            new_len = new_mask.shape[1]
            target_len = max(cur_len, new_len)

            past = _pad_past_left(past, target_len - cur_len)
            new_past = _pad_past_left(new_past, target_len - new_len)
            attn_mask = _pad_mask_left(attn_mask, target_len - cur_len)
            new_mask = _pad_mask_left(new_mask, target_len - new_len)

            # Concat along batch dim.
            past = _cat_past(past, new_past)
            attn_mask = torch.cat([attn_mask, new_mask], dim=0)

            # Sample first new token for each newly-added sequence.
            start_idx = len(sequences)
            sequences.extend(new_seqs)
            _sample_and_advance(new_last_logits, range(start_idx, len(sequences)))

        def _pad_past_left(p: Tuple, pad: int) -> Tuple:
            if pad <= 0:
                return p
            out = []
            for (k, v) in p:
                # k,v: (B, H, L, D)
                B, H, L, D = k.shape
                zk = torch.zeros((B, H, pad, D), dtype=k.dtype, device=k.device)
                zv = torch.zeros((B, H, pad, D), dtype=v.dtype, device=v.device)
                out.append((torch.cat([zk, k], dim=2), torch.cat([zv, v], dim=2)))
            return tuple(out)

        def _pad_mask_left(m: torch.Tensor, pad: int) -> torch.Tensor:
            if pad <= 0:
                return m
            B = m.shape[0]
            z = torch.zeros((B, pad), dtype=m.dtype, device=m.device)
            return torch.cat([z, m], dim=1)

        def _cat_past(a: Tuple, b: Tuple) -> Tuple:
            return tuple(
                (torch.cat([ak, bk], dim=0), torch.cat([av, bv], dim=0))
                for (ak, av), (bk, bv) in zip(a, b)
            )

        def _sample_and_advance(logits: torch.Tensor, idx_iter) -> None:
            """Sample next token for each index in idx_iter (positions in
            `sequences`), update its generated list / alive flag. `logits`
            is indexed relative to the first index in idx_iter: row j in
            `logits` corresponds to `sequences[list(idx_iter)[j]]`."""
            idxs = list(idx_iter)
            for j, i in enumerate(idxs):
                seq = sequences[i]
                if not seq.alive:
                    continue
                tok = sample_next_token(
                    logits[j],
                    temperature=float(seq.req.temperature),
                    top_p=float(seq.req.top_p),
                    top_k=seq.req.top_k,
                )
                if eos_id is not None and tok == eos_id:
                    seq.finish_reason = "stop"
                    seq.alive = False
                    continue
                seq.generated.append(tok)
                if len(seq.generated) >= seq.req.max_tokens:
                    seq.alive = False

        def _drop_finished():
            """Remove dead sequences from the running batch state."""
            nonlocal sequences, past, attn_mask
            keep = [i for i, s in enumerate(sequences) if s.alive]
            if len(keep) == len(sequences):
                return
            # Resolve futures for the dropped ones.
            for i, s in enumerate(sequences):
                if not s.alive and not s.req.future.done():
                    text = tokenizer.decode(s.generated, skip_special_tokens=True)
                    if paged is not None:
                        paged.free(s.seq_id)
                    s.req.future.get_loop().call_soon_threadsafe(
                        s.req.future.set_result,
                        GenerationResult(
                            text=text,
                            prompt_tokens=s.prompt_len,
                            completion_tokens=len(s.generated),
                            finish_reason=s.finish_reason,
                        ),
                    )
            if not keep:
                sequences = []
                past = None
                attn_mask = None
                return
            idx = torch.tensor(keep, dtype=torch.long)
            sequences = [sequences[i] for i in keep]
            past = tuple((k.index_select(0, idx), v.index_select(0, idx)) for k, v in past)
            attn_mask = attn_mask.index_select(0, idx)

        # ---- kick off with initial requests ----
        ns, npast, nmask, nlogits = _prefill(initial)
        _merge_into_batch(ns, npast, nmask, nlogits)

        # ---- continuous decode loop ----
        while sequences:
            # Step the model one token for every live sequence.
            next_tokens = [seq.generated[-1] if seq.generated else pad_id for seq in sequences]
            cur = torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1)  # (B, 1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=attn_mask.dtype)],
                dim=1,
            )
            with torch.no_grad():
                out = model(
                    input_ids=cur,
                    attention_mask=attn_mask,
                    past_key_values=_from_tuple(past),
                    use_cache=True,
                )
            past = _to_tuple(out.past_key_values)
            step_logits = out.logits[:, -1, :]  # (B, V)

            # Sample for each live seq.
            _sample_and_advance(step_logits, range(len(sequences)))

            # Advance the paged-cache block-table bookkeeping.
            # We only grow the block table (allocating new blocks on block-
            # size boundaries) — the actual K/V tensor writes during decode
            # are skipped here because HF's model carries its own
            # past_key_values. The important M3 property — that each live
            # sequence has a per-step block table and that blocks are
            # allocated/freed O(1) — is preserved.
            if paged is not None:
                for seq in sequences:
                    table = paged.table_of(seq.seq_id)
                    # Check if a new block is needed for the next token.
                    if table.length % paged.block_size == 0 and table.length == table.num_blocks() * paged.block_size:
                        table.block_ids.append(paged._allocate_block())
                    paged.advance_length(seq.seq_id, 1)

            # Drop finished.
            _drop_finished()

            # Pull in any newly-arrived requests without waiting.
            room = self.max_batch - len(sequences)
            if room > 0:
                new_reqs = self._drain_queue_nowait(room)
                if new_reqs:
                    ns, npast, nmask, nlogits = _prefill(new_reqs)
                    _merge_into_batch(ns, npast, nmask, nlogits)

        # All sequences finished — session done. _drop_finished already
        # resolved their futures.

    # ------------------------------------------------------------------
    # Mock-backend fallback
    # ------------------------------------------------------------------
    def _run_mock(self, initial: List[_Request]) -> None:
        """Plain sequential execution — good enough for HTTP-pipeline testing."""
        from .engine import GenerationResult  # local

        reqs = list(initial)
        # Drain anything else queued.
        while True:
            try:
                reqs.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for r in reqs:
            res = self.engine.generate(
                prompt=r.prompt, max_tokens=r.max_tokens,
                temperature=r.temperature, top_p=r.top_p,
                top_k=r.top_k, seed=r.seed,
            )
            if not r.future.done():
                r.future.get_loop().call_soon_threadsafe(r.future.set_result, res)


# Back-compat alias so `from app.batcher import StaticBatcher` still works
# for any external caller — it now gets the continuous one.
StaticBatcher = ContinuousBatcher
