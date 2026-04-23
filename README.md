# 06 — LLM Inference Engine

**Stack:** Python 3.12 + PyTorch 2.5 (baseline) · CUDA 12 + Triton (custom kernels) · C++/CUDA for PagedAttention · FastAPI (OpenAI-compatible API) · `safetensors` · `tokenizers` (HF) · NCCL for multi-GPU

## Full Vision
vLLM-class: PagedAttention KV cache, continuous batching, tensor+pipeline parallelism, speculative decoding, FP8/INT4 quant, prefix caching, OpenAI-compatible API.

## MVP (1 weekend)
Load Llama-3-8B, naive token-by-token generation, HTTP `/v1/completions` endpoint.

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
