"""M5 — Prometheus metrics.

Zero-dependency text-exposition implementation. We avoid a hard
dependency on `prometheus_client` to keep `requirements.txt` slim —
Prometheus's text format is trivial to emit by hand, and we only need
counters + histograms.

Tracked:
- `llm_requests_total{endpoint,model,status}` — counter
- `llm_prompt_tokens_total{model}` — counter
- `llm_completion_tokens_total{model}` — counter
- `llm_request_latency_seconds{endpoint,model}` — histogram
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

_LOCK = threading.Lock()

# (name, labels_tuple_sorted) -> value
_COUNTERS: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}

# Histograms: (name, labels) -> (bucket_counts per bucket, sum, count)
_HIST_BUCKETS: List[float] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
]
_HISTS: Dict[
    Tuple[str, Tuple[Tuple[str, str], ...]],
    Tuple[List[int], float, int],
] = {}


def _label_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def inc_counter(name: str, labels: Dict[str, str], value: float = 1.0) -> None:
    key = (name, _label_key(labels))
    with _LOCK:
        _COUNTERS[key] = _COUNTERS.get(key, 0.0) + value


def observe_histogram(name: str, labels: Dict[str, str], value: float) -> None:
    key = (name, _label_key(labels))
    with _LOCK:
        buckets, total, count = _HISTS.get(
            key, ([0] * len(_HIST_BUCKETS), 0.0, 0)
        )
        buckets = list(buckets)
        for i, ub in enumerate(_HIST_BUCKETS):
            if value <= ub:
                buckets[i] += 1
        _HISTS[key] = (buckets, total + value, count + 1)


@contextmanager
def time_request(endpoint: str, model: str) -> Iterator[Dict[str, str]]:
    """Context manager — records latency + request count.

    Yields a mutable status dict; set `status["status"]` before exit.
    """
    t0 = time.perf_counter()
    status: Dict[str, str] = {"status": "ok"}
    try:
        yield status
    except Exception:
        status["status"] = "error"
        raise
    finally:
        elapsed = time.perf_counter() - t0
        labels = {"endpoint": endpoint, "model": model}
        observe_histogram("llm_request_latency_seconds", labels, elapsed)
        inc_counter(
            "llm_requests_total",
            {**labels, "status": status["status"]},
        )


def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    inc_counter("llm_prompt_tokens_total", {"model": model}, prompt_tokens)
    inc_counter("llm_completion_tokens_total", {"model": model}, completion_tokens)


def _fmt_labels(labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


def render() -> str:
    """Render Prometheus text-exposition format."""
    lines: List[str] = []
    with _LOCK:
        # Counters — group by metric name for proper HELP/TYPE.
        counter_names = sorted({name for name, _ in _COUNTERS})
        for name in counter_names:
            lines.append(f"# HELP {name} {name}")
            lines.append(f"# TYPE {name} counter")
            for (n, labels), val in _COUNTERS.items():
                if n != name:
                    continue
                lines.append(f"{name}{_fmt_labels(labels)} {val}")

        hist_names = sorted({name for name, _ in _HISTS})
        for name in hist_names:
            lines.append(f"# HELP {name} {name}")
            lines.append(f"# TYPE {name} histogram")
            for (n, labels), (buckets, total, count) in _HISTS.items():
                if n != name:
                    continue
                cum = 0
                for ub, c in zip(_HIST_BUCKETS, buckets):
                    cum += c
                    le_labels = labels + (("le", str(ub)),)
                    lines.append(f"{name}_bucket{_fmt_labels(le_labels)} {cum}")
                inf_labels = labels + (("le", "+Inf"),)
                lines.append(f"{name}_bucket{_fmt_labels(inf_labels)} {count}")
                lines.append(f"{name}_sum{_fmt_labels(labels)} {total}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {count}")

    lines.append("")
    return "\n".join(lines)
