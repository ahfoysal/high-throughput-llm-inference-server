"""Inference engine — M2: real KV cache + static batched generation.

Public surface:
- `InferenceEngine.generate(...)`  — single-request, reuses past_key_values
- `InferenceEngine.generate_batch(...)` — N requests, one padded forward per
  decode step, per-sequence KV cache carried forward.

Single-sequence KV cache is a thin wrapper around HF's tuple-of-tuples
`past_key_values`. For batched decoding we pass the whole batched
past_key_values back into the model (one forward for the whole batch
at each step) — this is the actual M2 win: no re-encoding of the prompt,
N sequences advanced in a single matmul.

Padding strategy for batched prefill:
- Left-pad shorter prompts with `pad_token_id` and build an
  `attention_mask`. HF's GPT-2 attention respects the mask so padded
  positions don't contaminate real token queries/keys.
- With left-padding every sequence's last real token sits at index -1,
  so next-token logits are simply `out.logits[:, -1, :]`.

M3 will replace this with true continuous batching (merge new requests
into an in-flight decode step) + PagedAttention.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .sampling import sample_next_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV cache — M2: a real wrapper around HF past_key_values.
# ---------------------------------------------------------------------------


class KVCache:
    """Per-sequence KV cache.

    M2 scope: hold HF's `past_key_values` (a tuple of (k, v) tensors, one
    pair per transformer layer) across decode steps so each step only
    feeds in 1 new token instead of re-encoding the whole prefix.

    M3 will swap this for a block-paged allocator.
    """

    __slots__ = ("past", "length")

    def __init__(self) -> None:
        self.past: Optional[Tuple] = None
        self.length: int = 0  # number of real (non-padded) tokens cached

    def reset(self) -> None:
        self.past = None
        self.length = 0

    def update(self, past: Tuple, new_tokens: int = 1) -> None:
        self.past = past
        self.length += new_tokens

    @property
    def num_layers(self) -> int:
        return 0 if self.past is None else len(self.past)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "length" | "stop"


# ---------------------------------------------------------------------------
# Mock backend — deterministic fallback (unchanged from M1)
# ---------------------------------------------------------------------------


_MOCK_VOCAB = [
    " the", " a", " cat", " dog", " sat", " ran", " on", " under", " mat",
    " and", " then", " quickly", " slowly", " jumped", " over", " fence",
    " hello", " world", " token", " model", " generates", " text", ".",
    ",", " is", " was", " very", " happy", " today", " here",
]


class _MockBackend:
    name = "mock"

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        prompt_tokens = max(1, len(prompt.split()))
        out_tokens: List[str] = []
        for _ in range(max_tokens):
            tok = _MOCK_VOCAB[0] if temperature <= 1e-6 else rng.choice(_MOCK_VOCAB)
            out_tokens.append(tok)
        return GenerationResult(
            text="".join(out_tokens),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(out_tokens),
            finish_reason="length",
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: Sequence[int],
        temperatures: Sequence[float],
        top_ps: Sequence[float],
        top_ks: Sequence[Optional[int]],
        seeds: Sequence[Optional[int]],
    ) -> List[GenerationResult]:
        return [
            self.generate(p, mt, t, tp, tk, sd)
            for p, mt, t, tp, tk, sd in zip(
                prompts, max_tokens, temperatures, top_ps, top_ks, seeds
            )
        ]


# ---------------------------------------------------------------------------
# HuggingFace backend — M2: real KV cache + batched generate_batch
# ---------------------------------------------------------------------------


class _HFBackend:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        logger.info("Loading model %s on CPU...", model_name)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Left-pad: every sequence's last real token sits at index -1,
        # which lets us take next-token logits as out.logits[:, -1, :].
        self.tokenizer.padding_side = "left"
        logger.info("Model ready in %.2fs", time.time() - t0)

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Single-sequence path
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_tokens = int(input_ids.shape[1])

        kv = KVCache()
        generated: List[int] = []
        finish_reason = "length"

        cur_ids = input_ids
        with torch.no_grad():
            for _ in range(max_tokens):
                out = self.model(
                    input_ids=cur_ids,
                    past_key_values=kv.past,
                    use_cache=True,
                )
                logits = out.logits[0, -1, :]
                kv.update(out.past_key_values, new_tokens=cur_ids.shape[1])

                next_id = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if self.eos_id is not None and next_id == self.eos_id:
                    finish_reason = "stop"
                    break
                generated.append(next_id)
                cur_ids = torch.tensor([[next_id]], dtype=input_ids.dtype)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Batched path — M2 main contribution
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: Sequence[int],
        temperatures: Sequence[float],
        top_ps: Sequence[float],
        top_ks: Sequence[Optional[int]],
        seeds: Sequence[Optional[int]],
    ) -> List[GenerationResult]:
        torch = self.torch
        n = len(prompts)
        if n == 0:
            return []
        if n == 1:
            return [
                self.generate(
                    prompts[0], max_tokens[0], temperatures[0],
                    top_ps[0], top_ks[0], seeds[0],
                )
            ]

        for sd in seeds:
            if sd is not None:
                torch.manual_seed(sd)
                break

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,  # left-pad (set in __init__)
        )
        input_ids = enc.input_ids              # (B, L)
        attn_mask = enc.attention_mask         # (B, L)

        prompt_token_counts = attn_mask.sum(dim=1).tolist()
        max_new = max(int(x) for x in max_tokens)

        # Prefill — one forward pass for the whole batch.
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]  # (B, V)

        eos = self.eos_id
        alive = [True] * n
        generated: List[List[int]] = [[] for _ in range(n)]
        finish_reasons: List[str] = ["length"] * n
        remaining = [int(mt) for mt in max_tokens]

        pad_id = self.tokenizer.pad_token_id or 0

        # Sample first new token for each sequence.
        next_ids: List[int] = []
        for i in range(n):
            if remaining[i] <= 0:
                alive[i] = False
                next_ids.append(pad_id)
                continue
            tok = sample_next_token(
                last_logits[i],
                temperature=float(temperatures[i]),
                top_p=float(top_ps[i]),
                top_k=top_ks[i],
            )
            if eos is not None and tok == eos:
                alive[i] = False
                finish_reasons[i] = "stop"
                next_ids.append(pad_id)
            else:
                generated[i].append(tok)
                remaining[i] -= 1
                next_ids.append(tok)
                if remaining[i] <= 0:
                    alive[i] = False

        # Decode loop — one padded forward per step, feeding all B sequences
        # their next token (dead sequences feed pad; outputs are ignored).
        step = 0
        while any(alive) and step < max_new - 1:
            step += 1
            cur = torch.tensor(next_ids, dtype=input_ids.dtype).unsqueeze(1)  # (B, 1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((n, 1), dtype=attn_mask.dtype)], dim=1
            )
            with torch.no_grad():
                out = self.model(
                    input_ids=cur,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            step_logits = out.logits[:, -1, :]  # (B, V)

            new_next: List[int] = []
            for i in range(n):
                if not alive[i]:
                    new_next.append(pad_id)
                    continue
                tok = sample_next_token(
                    step_logits[i],
                    temperature=float(temperatures[i]),
                    top_p=float(top_ps[i]),
                    top_k=top_ks[i],
                )
                if eos is not None and tok == eos:
                    alive[i] = False
                    finish_reasons[i] = "stop"
                    new_next.append(pad_id)
                    continue
                generated[i].append(tok)
                remaining[i] -= 1
                new_next.append(tok)
                if remaining[i] <= 0:
                    alive[i] = False
            next_ids = new_next

        results: List[GenerationResult] = []
        for i in range(n):
            text = self.tokenizer.decode(generated[i], skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=int(prompt_token_counts[i]),
                    completion_tokens=len(generated[i]),
                    finish_reason=finish_reasons[i],
                )
            )
        return results


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------


DEFAULT_MODEL = os.environ.get("MVP_MODEL", "sshleifer/tiny-gpt2")


class InferenceEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        force_mock: bool = False,
    ) -> None:
        self.model_name = model_name

        force_mock = force_mock or os.environ.get("MVP_MOCK", "").lower() in (
            "1", "true", "yes",
        )

        if force_mock:
            logger.warning("MVP_MOCK set — using deterministic mock backend.")
            self.backend: object = _MockBackend()
        else:
            try:
                self.backend = _HFBackend(model_name)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to load HF model '%s' (%s). Falling back to mock.",
                    model_name, e,
                )
                self.backend = _MockBackend()

    @property
    def backend_name(self) -> str:
        return self.backend.name  # type: ignore[attr-defined]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        return self.backend.generate(  # type: ignore[attr-defined]
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: Sequence[int],
        temperatures: Sequence[float],
        top_ps: Sequence[float],
        top_ks: Sequence[Optional[int]],
        seeds: Sequence[Optional[int]],
    ) -> List[GenerationResult]:
        return self.backend.generate_batch(  # type: ignore[attr-defined]
            prompts=prompts,
            max_tokens=max_tokens,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            seeds=seeds,
        )
