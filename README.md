# LLM Inference Engine

**Stack:** Python 3.12 + PyTorch 2.5 (baseline) · CUDA 12 + Triton (custom kernels) · C++/CUDA for PagedAttention · FastAPI (OpenAI-compatible API) · `safetensors` · `tokenizers` (HF) · NCCL for multi-GPU

## Full Vision
vLLM-class: PagedAttention KV cache, continuous batching, tensor+pipeline parallelism, speculative decoding, FP8/INT4 quant, prefix caching, OpenAI-compatible API.

## MVP (1 weekend)
Load Llama-3-8B, naive token-by-token generation, HTTP `/v1/completions` endpoint.

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
- **M2 (Week 3):** Custom KV cache + batched generation
- **M3 (Week 6):** PagedAttention (Triton kernel) + continuous batching
- **M4 (Week 9):** Speculative decoding + INT4 quantization (AWQ/GPTQ)
- **M5 (Week 12):** Tensor-parallel multi-GPU + OpenAI-compatible API

## Key References
- vLLM paper (Kwon et al., 2023 — PagedAttention)
- FlashAttention 2 + 3
- Llama 3 architecture
