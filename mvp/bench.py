"""Throughput benchmark: single-stream vs static batching (B=1,4,8,16).

Measures tokens/sec using the in-process engine (no HTTP). Prompts are
short and homogeneous so padding overhead doesn't dominate.

Run:
    cd mvp && source venv/bin/activate
    python bench.py
"""
from __future__ import annotations

import time
from statistics import mean

from app.engine import InferenceEngine


PROMPTS = [
    "The quick brown fox jumps over",
    "Once upon a time there was",
    "In a hole in the ground",
    "It was the best of times",
    "Hello, my name is Claude and",
    "The capital of France is",
    "Python is a programming language",
    "To be or not to",
    "All happy families are alike",
    "Call me Ishmael. Some years",
    "Mr and Mrs Dursley of number",
    "The sun was shining on the",
    "It is a truth universally acknowledged",
    "Happy families are all alike",
    "In the beginning there was",
    "Space: the final frontier",
]

MAX_TOKENS = 32
TEMPERATURE = 0.0  # greedy — deterministic, no sampling jitter
TOP_P = 1.0


def bench_no_kv_cache(engine: InferenceEngine, max_tokens: int = MAX_TOKENS) -> float:
    """Token-by-token generation that re-encodes the full prefix each step
    (i.e. no KV cache). Used to prove the KV cache actually helps.

    Only runs with the HF backend; returns 0.0 otherwise.
    """
    if engine.backend_name != "transformers":
        return 0.0
    import torch  # local import
    from app.sampling import sample_next_token

    backend = engine.backend  # type: ignore[attr-defined]
    tok = backend.tokenizer
    model = backend.model
    prompt = PROMPTS[0]

    input_ids = tok(prompt, return_tensors="pt").input_ids
    t0 = time.perf_counter()
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # NOTE: no past_key_values, no use_cache -> full re-encode every step
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1, :]
            nid = sample_next_token(logits, temperature=0.0, top_p=1.0)
            generated.append(nid)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[nid]], dtype=input_ids.dtype)], dim=1
            )
    dt = time.perf_counter() - t0
    return len(generated) / dt


def bench_single(engine: InferenceEngine, n: int) -> float:
    """Run n prompts one-by-one (sequential single-stream)."""
    t0 = time.perf_counter()
    total_tokens = 0
    for i in range(n):
        r = engine.generate(
            prompt=PROMPTS[i % len(PROMPTS)],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=42,
        )
        total_tokens += r.completion_tokens
    dt = time.perf_counter() - t0
    return total_tokens / dt


def bench_batch(engine: InferenceEngine, batch_size: int) -> float:
    """One padded forward-per-step, batch of `batch_size`."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
    t0 = time.perf_counter()
    rs = engine.generate_batch(
        prompts=prompts,
        max_tokens=[MAX_TOKENS] * batch_size,
        temperatures=[TEMPERATURE] * batch_size,
        top_ps=[TOP_P] * batch_size,
        top_ks=[None] * batch_size,
        seeds=[42] * batch_size,
    )
    dt = time.perf_counter() - t0
    total_tokens = sum(r.completion_tokens for r in rs)
    return total_tokens / dt


def main() -> None:
    print(f"Loading engine (max_tokens={MAX_TOKENS}, greedy)...")
    engine = InferenceEngine()
    print(f"Backend: {engine.backend_name}  Model: {engine.model_name}")

    # Warm-up (first forward pass is slow: torch JIT + allocator).
    engine.generate(
        prompt=PROMPTS[0], max_tokens=4, temperature=0.0, top_p=1.0, seed=0,
    )

    print("\n== KV-cache ablation (B=1) ==")
    no_cache = bench_no_kv_cache(engine)
    with_cache = bench_batch(engine, 1)
    print(f"  no KV cache (re-encode every step): {no_cache:7.2f} tok/s")
    print(f"  with KV cache                     : {with_cache:7.2f} tok/s")
    if no_cache > 0:
        print(f"  speedup: {with_cache / no_cache:.2f}x")

    print("\n== Single-stream (sequential) ==")
    tps = bench_single(engine, n=4)
    print(f"  N=4 sequential: {tps:7.2f} tok/s")

    print("\n== Static batching ==")
    rows = []
    for B in (1, 4, 8, 16):
        # average over 2 runs to smooth out noise
        runs = [bench_batch(engine, B) for _ in range(2)]
        tps = mean(runs)
        rows.append((B, tps))
        print(f"  B={B:<2d}:  {tps:7.2f} tok/s  (runs: {[f'{x:.1f}' for x in runs]})")

    base = rows[0][1]
    print("\nSpeedup vs B=1:")
    for B, tps in rows:
        print(f"  B={B:<2d}:  {tps/base:5.2f}x")


if __name__ == "__main__":
    main()
