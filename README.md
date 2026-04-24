# LLM Inference Engine

**Stack:** Python 3.12 + PyTorch 2.5 (baseline) · CUDA 12 + Triton (custom kernels) · C++/CUDA for PagedAttention · FastAPI (OpenAI-compatible API) · `safetensors` · `tokenizers` (HF) · NCCL for multi-GPU

## Full Vision
vLLM-class: PagedAttention KV cache, continuous batching, tensor+pipeline parallelism, speculative decoding, FP8/INT4 quant, prefix caching, OpenAI-compatible API.

## MVP (1 weekend)
Load Llama-3-8B, naive token-by-token generation, HTTP `/v1/completions` endpoint.

## M5 Status — shipped

Pivoted from tensor-parallel multi-GPU (not possible on this Mac) to a
**production-ready OpenAI-compatible API**. Additions in
[`mvp/app/`](./mvp/app/):

- **`/v1/chat/completions`** — OpenAI chat schema (messages array,
  roles, `tools`). Non-chat-tuned backing model gets a minimal
  ChatML-ish template applied in [`chat.py`](./mvp/app/chat.py).
- **SSE streaming** — both completion endpoints accept `stream=true`
  and emit `data: {...}\n\n` deltas terminated by `data: [DONE]`.
- **Function / tool calling** — `tools` array goes into a system
  preamble; generated text is sniffed for
  `{"name": "...", "arguments": {...}}` objects and surfaced as
  structured `tool_calls` on the response, matching OpenAI's shape.
- **Bearer-token auth** — [`auth.py`](./mvp/app/auth.py). Set
  `MVP_API_KEYS=key1,key2`; unset means auth is off (warned at
  startup). `/health` and `/metrics` stay unauthenticated.
- **Prometheus `/metrics`** — [`metrics.py`](./mvp/app/metrics.py).
  Counters for requests / prompt-tokens / completion-tokens, latency
  histogram, all labelled by model + endpoint. Zero extra
  dependency — text format is emitted by hand.
- **Multi-model routing** — `MVP_MODELS=a,b,c` loads N engines at
  startup and dispatches by request `model` field. Short aliases
  (`tiny-gpt2` → `sshleifer/tiny-gpt2`) resolve automatically.
  Unknown models fall back to the first loaded model so existing
  clients keep working.

### Curl examples

```bash
export MVP_API_KEYS=dev-key
export MVP_MODELS=sshleifer/tiny-gpt2
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Chat completion:

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Authorization: Bearer dev-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"tiny-gpt2",
    "messages":[{"role":"user","content":"Say hi"}],
    "max_tokens":16,"seed":42
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8765/v1/chat/completions \
  -H 'Authorization: Bearer dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt2","stream":true,
       "messages":[{"role":"user","content":"stream me"}],
       "max_tokens":16,"seed":1}'
```

Tool call:

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Authorization: Bearer dev-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"tiny-gpt2",
    "messages":[{"role":"user","content":"What is the weather in Tokyo?"}],
    "tools":[{"type":"function","function":{
       "name":"get_weather",
       "description":"Get weather by city",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
    "max_tokens":32
  }'
```

Unauth returns 401:

```bash
curl -i http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt2","messages":[{"role":"user","content":"hi"}]}'
# HTTP/1.1 401 Unauthorized
```

Metrics:

```bash
curl -s http://127.0.0.1:8765/metrics | head
# # HELP llm_requests_total llm_requests_total
# # TYPE llm_requests_total counter
# llm_requests_total{endpoint="chat_completions",model="sshleifer/tiny-gpt2",status="ok"} 3
```

## M3 Status — shipped

Block-paged KV cache + continuous batching (vLLM-style iteration-level
scheduling). Real PagedAttention kernels are Triton/CUDA so this repo
implements the *semantics* in pure PyTorch — correctness-focused, not
speed-focused — and the throughput story comes from continuous batching,
which is the real M3 win anyway.

Pieces:

- **`app/paged_cache.py`** — `PagedKVCache`: a global KV pool split into
  fixed-size blocks (default `block_size=16`), a free-list allocator,
  and per-sequence `BlockTable`s mapping logical token positions to
  physical block ids. `append` / `gather` scatter/gather through the
  table.
- **`app/attention.py`** — paged attention that reads K/V through the
  block table, plus a reference dense attention. A self-test
  (`python -m app.attention`) asserts paged output matches dense within
  `1e-6` over prefill + decode on non-power-of-2 block sizes and partial
  final blocks.
- **`app/batcher.py`** — `ContinuousBatcher` replaces the M2 static
  batcher. At every decode step it drops finished sequences, pulls
  newly-arrived requests from the queue, prefills them, and splices
  them into the in-flight batch by slicing/padding HF's
  `past_key_values` along the batch dim. Each live sequence also
  maintains a real `PagedKVCache` block table (allocator round-trips
  every step) so the M3 invariants hold end-to-end.

Correctness: greedy decode over the same prompt/seed produces identical
tokens under single-stream generation and under the continuous batcher
(see `bench_m3.py → correctness_check`, which asserts equality and exits
non-zero on mismatch).

Throughput on MacBook CPU · `sshleifer/tiny-gpt2` · 16 requests (2 long
`max_tokens=128`, 14 short `max_tokens=8`), greedy:

| Mode                                        | tok/s  | Speedup |
| ------------------------------------------- | ------ | ------- |
| Static batching (chunked, wait for longest) | 1890.8 | 1.00x   |
| Continuous batching                         | 3392.6 | 1.79x   |

Continuous wins because the short requests don't stall behind the two
`max_tokens=128` stragglers — their slots get reused for new arrivals
mid-decode.

Reproduce: `cd mvp && source venv/bin/activate && python bench_m3.py`.
Numbers vary ±15% run-to-run on CPU.

## M2 Status — shipped

Real per-sequence `KVCache` (wraps HF `past_key_values`) + server-side
`StaticBatcher` (async queue, flushes on N queued OR T ms elapsed) that
runs one padded forward pass per decode step across the batch.

Throughput on MacBook CPU · `sshleifer/tiny-gpt2` · 32 new tokens, greedy:

| Mode                                    | tok/s  | Speedup |
| --------------------------------------- | ------ | ------- |
| No KV cache (re-encode prefix each step)|  476.7 | 1.00x   |
| KV cache, B=1                           | 1757.4 | 3.69x   |
| Static batch, B=4                       | 1838.8 | 3.86x   |
| Static batch, B=8                       | 3028.2 | 6.35x   |
| Static batch, B=16                      | 6255.9 | 13.1x   |

Reproduce: `cd mvp && source venv/bin/activate && python bench.py`.
Numbers vary ±15% run-to-run on CPU. The server wires the batcher into
`/v1/completions` so concurrent HTTP clients get merged into batches
automatically (configurable via `MVP_MAX_BATCH`, `MVP_MAX_WAIT_MS`).

## MVP Status — shipped

A working slice lives in [`mvp/`](./mvp/): FastAPI + PyTorch CPU,
OpenAI-compatible `/v1/completions`, greedy / temperature / top-p / top-k
sampling, plus `KVCache` and `ContinuousBatcher` placeholder classes
pointing at M2 / M3. Runs on a MacBook CPU in seconds with the tiny
`sshleifer/tiny-gpt2` model (or an instant deterministic mock backend via
`MVP_MOCK=1`).

```bash
cd mvp
python3.14 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

```bash
curl -s -X POST http://127.0.0.1:8765/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, my name is","max_tokens":12,"temperature":0.8,"top_p":0.9,"seed":42}'
# {"id":"cmpl-...","object":"text_completion","model":"sshleifer/tiny-gpt2",
#  "choices":[{"text":" Juno fairy nuances poison guessed ...","finish_reason":"length"}],
#  "usage":{"prompt_tokens":5,"completion_tokens":12,"total_tokens":17}}
```

See [`mvp/README.md`](./mvp/README.md) for config flags, mock-backend
details, and what's stubbed for later milestones.

## Milestones
- **M1 (Week 1):** Single-GPU HF-transformers baseline + sampling (top-k/p/temp)
- **M2 (Week 3):** Custom KV cache + batched generation  ✅ shipped (see "M2 Status" above)
- **M3 (Week 6):** PagedAttention + continuous batching  ✅ shipped (pure-PyTorch paged simulation, see "M3 Status" above)
- **M4 (Week 9):** Speculative decoding + INT4 quantization (AWQ/GPTQ)
- **M5 (Week 12):** Production-ready OpenAI-compatible API (chat, SSE streaming, tool calling, auth, Prometheus metrics, multi-model routing) ✅ shipped — tensor-parallel multi-GPU deferred (no multi-GPU hardware available)

## Key References
- vLLM paper (Kwon et al., 2023 — PagedAttention)
- FlashAttention 2 + 3
- Llama 3 architecture
