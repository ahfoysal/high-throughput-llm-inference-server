"""M4 / M6: Speculative decoding.

M6 upgrade: real draft/target pair (`distilgpt2` / `gpt2` by default)
plus proper **rejection sampling** against the draft and target
distributions — not just greedy argmax-matching. Also measures and
reports acceptance rate.

Two-model speculative decoding:

  1. Draft model cheaply proposes K candidate tokens autoregressively.
  2. Target model verifies them in a single forward pass (K+1 positions).
  3. We accept the longest prefix whose argmax matches the draft (greedy
     version), or use rejection sampling against draft vs target
     distributions (multinomial version).

In the greedy path this is *exactly* equivalent to running the target
model alone with greedy decoding — every accepted token is
`argmax(target_logits_at_that_position)`. The target forward over K+1
positions additionally gives us a "free" bonus token from the last
position's logits, so even on total rejection we still make 1 token of
progress per round.

For the demo both draft and target default to the same tiny model
(`sshleifer/tiny-gpt2`) — acceptance is ~100% which is the expected
degenerate case and useful for correctness. With distinct draft/target
(e.g. distilgpt2 + gpt2) acceptance drops but you get real wall-clock
speedup because the expensive target forward amortizes over K tokens.

CPU-only, pure PyTorch, leans on HF `past_key_values` for KV reuse on
both models.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class SpecResult:
    tokens: List[int]
    text: str
    rounds: int
    proposed: int
    accepted: int
    bonus: int            # "free" target tokens from the last-slot logits
    wall_time: float

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.proposed) if self.proposed else 0.0


# ---------------------------------------------------------------------------
# KV-cache trimming helpers
# ---------------------------------------------------------------------------
#
# HF `past_key_values` is a tuple of (K, V) per layer; each K,V has shape
# (batch, n_heads, seq_len, head_dim). After we feed K+1 new tokens and
# decide to accept only `a` of them plus 1 bonus, both caches hold more
# state than we want. We slice back to the correct length so the next
# round starts from a consistent state.


def _trim_past(past, new_len: int):
    """Trim a past_key_values cache to `new_len` tokens.

    Supports both the legacy tuple-of-tuples layout and the newer
    transformers `DynamicCache` object (which exposes `.crop(new_len)`).
    """
    # Newer HF: DynamicCache has a `crop` method that truncates in place.
    if hasattr(past, "crop"):
        past.crop(new_len)
        return past
    # Legacy tuple-of-tuples fallback.
    trimmed = []
    for layer in past:
        k, v = layer[0], layer[1]
        trimmed.append((k[:, :, :new_len, :], v[:, :, :new_len, :]))
    return tuple(trimmed)


# ---------------------------------------------------------------------------
# Speculative decoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Greedy speculative decoding with a draft + target HF model."""

    def __init__(
        self,
        target_model_name: str = "sshleifer/tiny-gpt2",
        draft_model_name: Optional[str] = None,
        K: int = 4,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.K = int(K)
        self.target_name = target_model_name
        self.draft_name = draft_model_name or target_model_name

        logger.info("Loading target=%s, draft=%s", self.target_name, self.draft_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.target = AutoModelForCausalLM.from_pretrained(self.target_name).eval()
        if self.draft_name == self.target_name:
            self.draft = self.target
        else:
            self.draft = AutoModelForCausalLM.from_pretrained(self.draft_name).eval()

        # Sanity: both models must share the same vocab size for argmax
        # comparison to be meaningful.
        if self.draft.config.vocab_size != self.target.config.vocab_size:
            raise ValueError(
                f"draft vocab {self.draft.config.vocab_size} != "
                f"target vocab {self.target.config.vocab_size}"
            )

    # ------------------------------------------------------------------
    # Baseline — plain greedy decode with the target alone.
    # Used for correctness comparison.
    # ------------------------------------------------------------------
    def greedy_baseline(self, prompt: str, max_tokens: int) -> Tuple[List[int], float]:
        t0 = time.time()
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        past = None
        cur = ids
        out_tokens: List[int] = []
        with torch.no_grad():
            for _ in range(max_tokens):
                out = self.target(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1, :].argmax().item())
                out_tokens.append(nxt)
                cur = torch.tensor([[nxt]], dtype=ids.dtype)
        return out_tokens, time.time() - t0

    # ------------------------------------------------------------------
    # Real speculative decoding with rejection sampling.
    # ------------------------------------------------------------------
    #
    # This is the *Leviathan et al. 2023* / *Chen et al. 2023* procedure:
    #
    #   For each draft position i in 1..K:
    #     Let q = draft prob over vocab at step i
    #     Let p = target prob over vocab at step i
    #     Draw u ~ Uniform(0,1)
    #     If u < min(1, p(x_i) / q(x_i)):  accept x_i
    #     Else: reject, resample from the residual distribution
    #            p'(x) = max(0, p(x) - q(x)) / Σ max(0, p - q)
    #           break out of the acceptance loop
    #
    # If ALL K draft tokens are accepted, we additionally sample one
    # "free" bonus token from p_{K+1} (target's prediction after the
    # last accepted draft token).
    #
    # This procedure gives output tokens whose marginal distribution is
    # exactly p (the target's distribution), so it's a drop-in
    # replacement for plain target sampling — same quality, fewer
    # target forward passes when draft and target agree a lot.
    # ------------------------------------------------------------------
    def generate_rejection(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SpecResult:
        t0 = time.time()
        if seed is not None:
            torch.manual_seed(seed)

        tau = max(float(temperature), 1e-6)

        def _probs(logits: torch.Tensor) -> torch.Tensor:
            """logits: (V,) -> probability vector (V,) after temp + optional top-k."""
            z = logits / tau
            if top_k is not None and top_k > 0 and top_k < z.shape[-1]:
                vals, idx = torch.topk(z, top_k)
                mask = torch.full_like(z, float("-inf"))
                mask[idx] = vals
                z = mask
            return torch.softmax(z, dim=-1)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = int(ids.shape[1])

        with torch.no_grad():
            d_out = self.draft(input_ids=ids, use_cache=True)
            t_out = self.target(input_ids=ids, use_cache=True)
        draft_past = d_out.past_key_values
        target_past = t_out.past_key_values
        last_token = int(ids[0, -1].item())
        draft_cached = prompt_len
        target_cached = prompt_len

        generated: List[int] = []
        proposed = accepted = bonus = rounds = 0

        with torch.no_grad():
            while len(generated) < max_tokens:
                rounds += 1
                K = min(self.K, max_tokens - len(generated))

                # ----- 1. Draft proposes K tokens + records q_i(x_i) -----
                draft_tokens: List[int] = []
                draft_probs_at_pos: List[torch.Tensor] = []
                d_past = draft_past
                d_cur = torch.tensor([[last_token]], dtype=ids.dtype)
                for _ in range(K):
                    d_step = self.draft(
                        input_ids=d_cur, past_key_values=d_past, use_cache=True
                    )
                    d_past = d_step.past_key_values
                    q = _probs(d_step.logits[0, -1, :])
                    tok = int(torch.multinomial(q, 1).item())
                    draft_tokens.append(tok)
                    draft_probs_at_pos.append(q)
                    d_cur = torch.tensor([[tok]], dtype=ids.dtype)
                proposed += K

                # ----- 2. Target forward over K+1 positions -----
                verify_in = torch.tensor(
                    [[last_token] + draft_tokens], dtype=ids.dtype
                )
                t_step = self.target(
                    input_ids=verify_in,
                    past_key_values=target_past,
                    use_cache=True,
                )
                # logits[i] predicts token at draft position i (0..K-1)
                # logits[K] is the bonus slot.
                target_probs_at_pos: List[torch.Tensor] = [
                    _probs(t_step.logits[0, i, :]) for i in range(K + 1)
                ]

                # ----- 3. Rejection loop -----
                a = 0
                corrective: Optional[int] = None
                for i in range(K):
                    q = draft_probs_at_pos[i]
                    p = target_probs_at_pos[i]
                    x = draft_tokens[i]
                    qx = float(q[x])
                    px = float(p[x])
                    # min(1, p/q); guard q=0 (shouldn't happen post-softmax)
                    ratio = 1.0 if qx <= 0 else min(1.0, px / qx)
                    u = float(torch.rand(1).item())
                    if u < ratio:
                        a += 1
                        continue
                    # Rejected — sample from residual (p - q)+ normalized.
                    residual = (p - q).clamp(min=0.0)
                    s = float(residual.sum())
                    if s > 0:
                        residual = residual / s
                        corrective = int(torch.multinomial(residual, 1).item())
                    else:
                        # Degenerate (p fully covered by q on top). Fall
                        # back to target's own sample at this position.
                        corrective = int(torch.multinomial(p, 1).item())
                    break
                accepted += a

                if corrective is None:
                    # All K accepted → bonus token from p_{K} (slot K).
                    corrective = int(torch.multinomial(target_probs_at_pos[K], 1).item())
                    bonus += 1
                    kept = draft_tokens[:a] + [corrective]
                else:
                    kept = draft_tokens[:a] + [corrective]

                # Truncate if overshooting budget
                remaining = max_tokens - len(generated)
                kept = kept[:remaining]
                generated.extend(kept)

                # ----- 4. KV cache fixup (same bookkeeping as greedy path) -----
                target_cached = target_cached + len(kept)
                target_past = _trim_past(t_step.past_key_values, target_cached)
                draft_commit = draft_cached + max(0, len(kept) - 1)
                draft_past = _trim_past(d_past, draft_commit)
                draft_cached = draft_commit

                last_token = kept[-1] if kept else last_token

                if (
                    self.tokenizer.eos_token_id is not None
                    and last_token == self.tokenizer.eos_token_id
                ):
                    break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return SpecResult(
            tokens=generated,
            text=text,
            rounds=rounds,
            proposed=proposed,
            accepted=accepted,
            bonus=bonus,
            wall_time=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Baseline — plain multinomial sample from the target alone.
    # Useful for apples-to-apples wall-clock comparison with the
    # rejection-sampling path.
    # ------------------------------------------------------------------
    def sample_baseline(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        t0 = time.time()
        if seed is not None:
            torch.manual_seed(seed)
        tau = max(float(temperature), 1e-6)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        past = None
        cur = ids
        out_tokens: List[int] = []
        with torch.no_grad():
            for _ in range(max_tokens):
                out = self.target(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                z = out.logits[0, -1, :] / tau
                if top_k is not None and top_k > 0 and top_k < z.shape[-1]:
                    vals, idx = torch.topk(z, top_k)
                    mask = torch.full_like(z, float("-inf"))
                    mask[idx] = vals
                    z = mask
                p = torch.softmax(z, dim=-1)
                nxt = int(torch.multinomial(p, 1).item())
                out_tokens.append(nxt)
                cur = torch.tensor([[nxt]], dtype=ids.dtype)
        return out_tokens, time.time() - t0

    # ------------------------------------------------------------------
    # Greedy speculative decode.
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int) -> SpecResult:
        t0 = time.time()

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = int(ids.shape[1])

        # Prefill both models on the full prompt.
        with torch.no_grad():
            d_out = self.draft(input_ids=ids, use_cache=True)
            t_out = self.target(input_ids=ids, use_cache=True)
        draft_past = d_out.past_key_values
        target_past = t_out.past_key_values
        # last token fed to either model (needed to seed the next draft step)
        last_token = int(ids[0, -1].item())
        # logical sequence length cached in each model
        draft_cached = prompt_len
        target_cached = prompt_len

        generated: List[int] = []
        proposed = accepted = bonus = rounds = 0

        with torch.no_grad():
            while len(generated) < max_tokens:
                rounds += 1
                K = min(self.K, max_tokens - len(generated))

                # ----- 1. Draft proposes K tokens -----
                draft_tokens: List[int] = []
                d_past = draft_past
                d_cur = torch.tensor([[last_token]], dtype=ids.dtype)
                for _ in range(K):
                    d_step = self.draft(
                        input_ids=d_cur, past_key_values=d_past, use_cache=True
                    )
                    d_past = d_step.past_key_values
                    tok = int(d_step.logits[0, -1, :].argmax().item())
                    draft_tokens.append(tok)
                    d_cur = torch.tensor([[tok]], dtype=ids.dtype)
                proposed += K

                # ----- 2. Target verifies in one forward over K positions -----
                # Input = [last_token, draft[0], ..., draft[K-1]] fed as a
                # single sequence on top of target_past. The logits at
                # position i predict the token that should follow
                # input_seq[i].
                verify_in = torch.tensor(
                    [[last_token] + draft_tokens], dtype=ids.dtype
                )
                t_step = self.target(
                    input_ids=verify_in,
                    past_key_values=target_past,
                    use_cache=True,
                )
                # argmax at each of the K+1 positions
                target_argmax = t_step.logits[0, :, :].argmax(dim=-1).tolist()
                # target_argmax[0] = what target thinks should follow
                #   last_token — i.e. the predicted version of draft[0].
                # target_argmax[i] for 1<=i<=K-1 = target's prediction of
                #   what follows draft[i-1], i.e. version of draft[i].
                # target_argmax[K] = bonus free token after draft[K-1].

                # ----- 3. Accept longest matching prefix -----
                a = 0
                for i in range(K):
                    if target_argmax[i] == draft_tokens[i]:
                        a += 1
                    else:
                        break
                accepted += a

                # Tokens we keep:
                #   - the `a` matched draft tokens
                #   - one corrective/bonus token from target_argmax[a]
                #     (this is what target would have emitted at that slot;
                #     always safe because it's the target's own argmax.)
                corrective = target_argmax[a]
                kept = draft_tokens[:a] + [corrective]
                if a == K:
                    bonus += 1  # all draft tokens accepted + 1 free bonus

                # Truncate if we'd overshoot max_tokens
                remaining = max_tokens - len(generated)
                kept = kept[:remaining]
                generated.extend(kept)

                # ----- 4. Fix up KV caches -----
                # Target currently has (target_cached + K + 1) of state.
                # We committed to target_cached + len(kept) real tokens
                # (the last one being `corrective`).
                target_cached = target_cached + len(kept)
                target_past = _trim_past(t_step.past_key_values, target_cached)

                # Draft currently has (draft_cached + K) of state. We
                # committed to draft_cached + len(kept) — but the last
                # committed token (`corrective`) was NOT fed to the draft,
                # so we trim to draft_cached + (len(kept) - 1) and the
                # corrective token becomes the next round's seed.
                draft_commit = draft_cached + max(0, len(kept) - 1)
                draft_past = _trim_past(d_past, draft_commit)
                draft_cached = draft_commit

                last_token = kept[-1] if kept else last_token

                # Stop on EOS
                if (
                    self.tokenizer.eos_token_id is not None
                    and last_token == self.tokenizer.eos_token_id
                ):
                    break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return SpecResult(
            tokens=generated,
            text=text,
            rounds=rounds,
            proposed=proposed,
            accepted=accepted,
            bonus=bonus,
            wall_time=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Self-test / mini-benchmark — `python -m app.speculative`
# ---------------------------------------------------------------------------


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = os.environ.get("SPEC_TARGET", "sshleifer/tiny-gpt2")
    draft = os.environ.get("SPEC_DRAFT", target)
    K = int(os.environ.get("SPEC_K", "4"))
    max_tokens = int(os.environ.get("SPEC_TOKENS", "32"))
    prompt = os.environ.get("SPEC_PROMPT", "Hello, my name is")

    dec = SpeculativeDecoder(target_model_name=target, draft_name=draft, K=K) \
        if False else SpeculativeDecoder(
            target_model_name=target, draft_model_name=draft, K=K,
        )

    # Baseline
    base_tokens, base_t = dec.greedy_baseline(prompt, max_tokens)
    print(f"[baseline] {len(base_tokens)} tok in {base_t:.3f}s "
          f"-> {len(base_tokens)/base_t:.1f} tok/s")

    # Speculative
    res = dec.generate(prompt, max_tokens)
    print(f"[spec]     {len(res.tokens)} tok in {res.wall_time:.3f}s "
          f"-> {len(res.tokens)/res.wall_time:.1f} tok/s")
    print(f"  rounds={res.rounds} proposed={res.proposed} "
          f"accepted={res.accepted} bonus={res.bonus} "
          f"acceptance={res.acceptance_rate*100:.1f}%")

    # Correctness: with draft==target, speculative MUST match baseline exactly.
    if draft == target:
        prefix = min(len(base_tokens), len(res.tokens))
        match = base_tokens[:prefix] == res.tokens[:prefix]
        print(f"  correctness (draft==target) prefix-match: {match}")
        assert match, "speculative output diverged from greedy baseline!"

    print(f"  text: {res.text!r}")

    # ---- Rejection-sampling path (M6) --------------------------------
    temp = float(os.environ.get("SPEC_TEMP", "1.0"))
    top_k = int(os.environ.get("SPEC_TOP_K", "50"))
    seed = int(os.environ.get("SPEC_SEED", "0"))

    base_rs_tokens, base_rs_t = dec.sample_baseline(
        prompt, max_tokens, temperature=temp, top_k=top_k, seed=seed,
    )
    print(f"[sample-baseline] {len(base_rs_tokens)} tok in {base_rs_t:.3f}s "
          f"-> {len(base_rs_tokens)/base_rs_t:.1f} tok/s  (T={temp} top_k={top_k})")

    rs = dec.generate_rejection(
        prompt, max_tokens, temperature=temp, top_k=top_k, seed=seed,
    )
    print(f"[spec-rejection] {len(rs.tokens)} tok in {rs.wall_time:.3f}s "
          f"-> {len(rs.tokens)/rs.wall_time:.1f} tok/s")
    print(f"  rounds={rs.rounds} proposed={rs.proposed} "
          f"accepted={rs.accepted} bonus={rs.bonus} "
          f"acceptance={rs.acceptance_rate*100:.1f}%")
    print(f"  text: {rs.text!r}")


if __name__ == "__main__":  # pragma: no cover
    _main()
