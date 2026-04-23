"""Inference engine.

MVP: wraps a HuggingFace causal-LM. Falls back to a deterministic mock
generator if the model can't load (no network, slow download, etc.) so the
API surface is always exercisable on a CPU-only MacBook.

Forward-looking scaffolding:
- `KVCache` — placeholder for M2 (custom paged KV cache).
- `ContinuousBatcher` — placeholder for M3 (vLLM-style continuous batching).
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .sampling import sample_next_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placeholders for later milestones. Intentionally thin — they document the
# architecture we're building toward without implementing it in the MVP.
# ---------------------------------------------------------------------------


class KVCache:
    """Placeholder KV cache.

    TODO(M2): replace HF's built-in `past_key_values` with our own tensor
    pool so we can (a) pre-allocate blocks and (b) later adopt PagedAttention.

    TODO(M3): implement block-level paging (vLLM-style) — each sequence
    references a list of non-contiguous blocks, freed when the sequence ends.
    """

    def __init__(self, max_seqs: int = 1, block_size: int = 16) -> None:
        self.max_seqs = max_seqs
        self.block_size = block_size
        self._hf_past = None  # MVP: stash HF's tuple-of-tuples here.

    def reset(self) -> None:
        self._hf_past = None


class ContinuousBatcher:
    """Placeholder continuous batcher.

    TODO(M3): maintain a running set of in-flight sequences and, at each
    decode step, merge newly-arrived requests into the batch rather than
    waiting for the current batch to finish (static batching).

    MVP behaviour: a global lock that serializes requests. This is enough
    to exercise the API but gives zero throughput benefit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


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
# Mock backend — deterministic, no-dependency fallback
# ---------------------------------------------------------------------------


_MOCK_VOCAB = [
    " the", " a", " cat", " dog", " sat", " ran", " on", " under", " mat",
    " and", " then", " quickly", " slowly", " jumped", " over", " fence",
    " hello", " world", " token", " model", " generates", " text", ".",
    ",", " is", " was", " very", " happy", " today", " here",
]


class _MockBackend:
    """Deterministic pseudo-LLM for offline/fast-iteration use.

    Produces plausible-looking whitespace-separated tokens seeded by the
    prompt so output is reproducible. Not a language model — just enough to
    verify the HTTP pipeline end-to-end.
    """

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
        # crude prompt-token count: whitespace split
        prompt_tokens = max(1, len(prompt.split()))
        out_tokens: List[str] = []
        for _ in range(max_tokens):
            if temperature <= 1e-6:
                tok = _MOCK_VOCAB[0]  # greedy-ish
            else:
                tok = rng.choice(_MOCK_VOCAB)
            out_tokens.append(tok)
        return GenerationResult(
            text="".join(out_tokens),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(out_tokens),
            finish_reason="length",
        )


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------


class _HFBackend:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        import torch  # local import: keeps mock-only path lightweight
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        logger.info("Loading model %s on CPU...", model_name)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info("Model ready in %.2fs", time.time() - t0)

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

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

        generated: List[int] = []
        past = None  # HF KV cache (tuple of tuples). M2 will replace this.
        finish_reason = "length"

        cur_ids = input_ids
        with torch.no_grad():
            for _ in range(max_tokens):
                out = self.model(
                    input_ids=cur_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                logits = out.logits[0, -1, :]  # (vocab,)
                past = out.past_key_values

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


# ---------------------------------------------------------------------------
# Public engine — picks a backend at init time
# ---------------------------------------------------------------------------


DEFAULT_MODEL = os.environ.get("MVP_MODEL", "sshleifer/tiny-gpt2")


class InferenceEngine:
    """High-level engine used by the FastAPI layer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        force_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.kv_cache = KVCache()
        self.batcher = ContinuousBatcher()

        force_mock = force_mock or os.environ.get("MVP_MOCK", "").lower() in (
            "1", "true", "yes",
        )

        if force_mock:
            logger.warning("MVP_MOCK set — using deterministic mock backend.")
            self.backend = _MockBackend()
            return

        try:
            self.backend = _HFBackend(model_name)
        except Exception as e:  # noqa: BLE001 — intentional broad catch
            logger.warning(
                "Failed to load HF model '%s' (%s). Falling back to mock backend.",
                model_name,
                e,
            )
            self.backend = _MockBackend()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        # MVP: serialize through the placeholder batcher. M3 will remove this.
        self.batcher.acquire()
        try:
            return self.backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
        finally:
            self.batcher.release()
