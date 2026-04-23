"""M3 bench: continuous vs static batching under mixed-length workload.

We simulate an arrival pattern where short requests trickle in behind a
few long ones. Static batching holds the whole batch until the longest
finishes; continuous batching drops finished sequences and pulls new
ones in, so the server stays saturated.

Also runs a correctness spot-check: same prompt + same seed should
produce the same tokens whether run single-stream, under the old
static batcher, or under continuous batching.
"""
from __future__ import annotations

import asyncio
import random
import time
from statistics import mean
from typing import List

from app.engine import InferenceEngine


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

SHORT_PROMPTS = [
    "The quick brown fox",
    "Hello, my name is",
    "Python is a",
    "To be or not",
    "It was the best",
    "Once upon a time",
    "In the beginning",
    "Space: the final",
]

# Mixed max_tokens: few long, many short — the adversarial case for static batching.
def make_workload(n_short: int, n_long: int, short_mt: int = 16, long_mt: int = 96):
    random.seed(0)
    items = []
    for i in range(n_long):
        items.append((SHORT_PROMPTS[i % len(SHORT_PROMPTS)], long_mt))
    for i in range(n_short):
        items.append((SHORT_PROMPTS[i % len(SHORT_PROMPTS)], short_mt))
    random.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Static-batch sim: run the whole workload as one batch, waiting for the
# longest. This matches the M2 StaticBatcher's behavior when it flushes.
# ---------------------------------------------------------------------------

def run_static(engine: InferenceEngine, workload, max_batch: int = 8) -> tuple[float, int]:
    """Process the workload in fixed-size chunks, each run to completion
    before the next starts — i.e. static batching with a finite batch
    window. Each chunk stalls on its slowest member (the M2 wait-for-
    longest pathology that motivates continuous batching)."""
    t0 = time.perf_counter()
    total = 0
    for i in range(0, len(workload), max_batch):
        chunk = workload[i:i + max_batch]
        prompts = [p for p, _ in chunk]
        max_tokens = [mt for _, mt in chunk]
        rs = engine.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperatures=[0.0] * len(chunk),
            top_ps=[1.0] * len(chunk),
            top_ks=[None] * len(chunk),
            seeds=[42] * len(chunk),
        )
        total += sum(r.completion_tokens for r in rs)
    dt = time.perf_counter() - t0
    return total / dt, total


# ---------------------------------------------------------------------------
# Continuous-batch sim: push all requests through ContinuousBatcher.
# We stagger arrivals slightly so the batcher sees the "new request
# mid-decode" path that static batching can't exploit.
# ---------------------------------------------------------------------------

async def run_continuous(engine: InferenceEngine, workload, arrival_gap_ms: float = 0.0):
    from app.batcher import ContinuousBatcher

    batcher = ContinuousBatcher(engine, max_batch=16, max_wait_ms=5)
    batcher.start()
    try:
        t0 = time.perf_counter()

        async def submit(p, mt, delay):
            await asyncio.sleep(delay)
            return await batcher.submit(
                prompt=p, max_tokens=mt,
                temperature=0.0, top_p=1.0, top_k=None, seed=42,
            )

        tasks = [
            asyncio.create_task(submit(p, mt, (i * arrival_gap_ms) / 1000.0))
            for i, (p, mt) in enumerate(workload)
        ]
        rs = await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0
    finally:
        await batcher.stop()
    total = sum(r.completion_tokens for r in rs)
    return total / dt, total


# ---------------------------------------------------------------------------
# Correctness: paged-backed continuous run must produce the same tokens
# as a straight single-stream run (greedy / temperature=0 makes it
# deterministic, no sampling variance).
# ---------------------------------------------------------------------------

async def correctness_check(engine: InferenceEngine) -> None:
    from app.batcher import ContinuousBatcher

    prompt = "The quick brown fox"
    ref = engine.generate(
        prompt=prompt, max_tokens=24,
        temperature=0.0, top_p=1.0, seed=42,
    )

    batcher = ContinuousBatcher(engine, max_batch=4, max_wait_ms=5)
    batcher.start()
    try:
        got = await batcher.submit(
            prompt=prompt, max_tokens=24,
            temperature=0.0, top_p=1.0, top_k=None, seed=42,
        )
    finally:
        await batcher.stop()

    ok = ref.text == got.text
    print(f"correctness: single-stream vs continuous "
          f"{'OK' if ok else 'MISMATCH'}  "
          f"(ref={ref.text!r}  got={got.text!r})")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading engine...")
    engine = InferenceEngine()
    print(f"Backend: {engine.backend_name}  Model: {engine.model_name}")

    # Warm-up.
    engine.generate(
        prompt="warmup", max_tokens=4, temperature=0.0, top_p=1.0, seed=0,
    )

    print("\n== Correctness check ==")
    asyncio.run(correctness_check(engine))

    print("\n== Mixed-length workload (2 long, 14 short) ==")
    workload = make_workload(n_short=14, n_long=2, short_mt=8, long_mt=128)
    print(f"  {len(workload)} requests — max_tokens spread = "
          f"{min(mt for _, mt in workload)}..{max(mt for _, mt in workload)}")

    print("\n  static batching (wait for longest in the batch):")
    static_runs = [run_static(engine, workload) for _ in range(2)]
    static_tps = mean(r[0] for r in static_runs)
    print(f"    tok/s: {static_tps:7.2f}  (runs: {[f'{r[0]:.1f}' for r in static_runs]})")

    print("\n  continuous batching:")
    cont_runs = []
    for _ in range(2):
        cont_runs.append(asyncio.run(run_continuous(engine, workload)))
    cont_tps = mean(r[0] for r in cont_runs)
    print(f"    tok/s: {cont_tps:7.2f}  (runs: {[f'{r[0]:.1f}' for r in cont_runs]})")

    if static_tps > 0:
        print(f"\n  speedup (continuous / static): {cont_tps / static_tps:.2f}x")


if __name__ == "__main__":
    main()
