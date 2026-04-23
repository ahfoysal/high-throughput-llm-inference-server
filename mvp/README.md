# MVP — OpenAI-compatible LLM Inference Server

Minimum viable slice of the full vision (see repo-root `README.md`).
Focus: exercise the **architecture** — HTTP surface, sampling, batching +
KV-cache scaffolding — not benchmark a real LLM. Runs on a MacBook CPU in
seconds.

## Layout

```
mvp/
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── main.py      # FastAPI app, /v1/completions, /health
    ├── engine.py    # InferenceEngine + KVCache + batched HF forward (M2)
    ├── batcher.py   # StaticBatcher: async queue, size/time flush (M2)
    ├── sampling.py  # greedy / temperature / top-p / top-k
    └── schemas.py   # Pydantic OpenAI-compatible request/response models
```

## Install & run

```bash
cd mvp
python3.14 -m venv venv            # 3.12+ works too
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

First startup downloads `sshleifer/tiny-gpt2` (~500 KB) and loads it on CPU
in a few seconds.

## Try it

```bash
curl -s -X POST http://127.0.0.1:8765/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, my name is","max_tokens":12,"temperature":0.8,"top_p":0.9,"seed":42}'
```

Health check:

```bash
curl -s http://127.0.0.1:8765/health
# {"status":"ok","model":"sshleifer/tiny-gpt2","backend":"transformers"}
```

## Configuration

| Env var      | Default                | Meaning                                              |
| ------------ | ---------------------- | ---------------------------------------------------- |
| `MVP_MODEL`  | `sshleifer/tiny-gpt2`  | HF model id to load                                  |
| `MVP_MOCK`   | unset                  | If `1`/`true`, skip HF and use the mock backend      |
| `MVP_MAX_BATCH`   | `8`               | Max requests fused into one batched forward pass     |
| `MVP_MAX_WAIT_MS` | `20`              | Max ms to wait for more requests before flushing     |

### Mock fallback

`tiny-gpt2` is already tiny but still needs a one-time HF download. If
you're offline or want instant startup, run:

```bash
MVP_MOCK=1 uvicorn app.main:app --port 8765
```

The mock backend is a **deterministic pseudo-generator** seeded by the
prompt. It produces plausible-looking whitespace tokens so the HTTP
pipeline and request/response shapes can be exercised end-to-end with
zero network and zero torch inference. It is not a language model —
outputs are intentionally nonsensical. Engine also auto-falls-back to
mock if HF load throws (e.g. no network).

## What's supported vs TODO

Supported in MVP:
- `POST /v1/completions` with OpenAI-compatible shape (`prompt`,
  `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop` parsed
  but not yet enforced in the loop)
- `GET /health`, `GET /`
- Greedy / temperature / nucleus / top-k sampling (see `sampling.py`)
- Single-prompt, single-sequence generation
- HF `past_key_values` KV-cache reuse (so each step decodes one token,
  not the full prefix)

Shipped in M2:
- Real `KVCache` — holds HF `past_key_values` across decode steps so
  each step decodes 1 new token (prefix never re-encoded). ~3.7x vs
  no-cache on CPU / tiny-gpt2.
- `StaticBatcher` (`app/batcher.py`) — async queue; flushes on N
  queued OR T ms elapsed; merges concurrent `/v1/completions` requests
  into a single padded forward per decode step. ~5.5x at B=16 vs B=1
  on CPU / tiny-gpt2. Configurable via `MVP_MAX_BATCH`/`MVP_MAX_WAIT_MS`.
- `python bench.py` reproduces the numbers locally.

Scaffolded, not implemented (see TODOs in code):
- **M3**: block-level PagedAttention for `KVCache`, true continuous
  batching (merge new requests into an in-flight decode step instead
  of static batching that waits for the batch to finish).
- Streaming responses (SSE) — stubbed out with a 400.
- Batched prompts (`prompt: [..., ...]`) — stubbed out with a 400.
- `stop` sequences — accepted in schema, not enforced.
- Speculative decoding, quantization, tensor parallelism — M4+.

## Sample output

With the real `tiny-gpt2` backend (random-init weights; output is
gibberish but confirms the pipeline):

```json
{
  "id": "cmpl-8665618976fe4e1fb443f527",
  "object": "text_completion",
  "model": "sshleifer/tiny-gpt2",
  "choices": [{
    "text": " Juno fairy nuances poison guessed Request dst ...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}
}
```
