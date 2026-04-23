# LLM Inference Engine

**Stack:** Python 3.12 + PyTorch 2.5 (baseline) · CUDA 12 + Triton (custom kernels) · C++/CUDA for PagedAttention · FastAPI (OpenAI-compatible API) · `safetensors` · `tokenizers` (HF) · NCCL for multi-GPU

## Full Vision
vLLM-class: PagedAttention KV cache, continuous batching, tensor+pipeline parallelism, speculative decoding, FP8/INT4 quant, prefix caching, OpenAI-compatible API.

## MVP (1 weekend)
Load Llama-3-8B, naive token-by-token generation, HTTP `/v1/completions` endpoint.

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
- **M3 (Week 6):** PagedAttention (Triton kernel) + continuous batching
- **M4 (Week 9):** Speculative decoding + INT4 quantization (AWQ/GPTQ)
- **M5 (Week 12):** Tensor-parallel multi-GPU + OpenAI-compatible API

## Key References
- vLLM paper (Kwon et al., 2023 — PagedAttention)
- FlashAttention 2 + 3
- Llama 3 architecture
