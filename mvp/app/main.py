"""FastAPI entry point — M5: production-ready OpenAI-compatible API.

New in M5 (vs M3):
- `/v1/chat/completions` (messages array)
- SSE streaming (`stream=true`) on both completions endpoints
- Function / tool calling (regex-sniffed from generated text)
- Bearer-token auth (`MVP_API_KEYS`)
- Prometheus `/metrics`
- Multi-model routing — load models listed in `MVP_MODELS` at startup
  and dispatch by the request's `model` field.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from . import metrics
from .auth import auth_enabled, require_api_key
from .batcher import ContinuousBatcher
from .chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatResponseMessage,
    ChatUsage,
    build_prompt,
    extract_tool_calls,
    sse_chunk,
    sse_done,
)
from .engine import InferenceEngine
from .schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mvp")


# ---------------------------------------------------------------------------
# Model manager — multiple (engine, batcher) pairs keyed by model id.
# ---------------------------------------------------------------------------


class ModelManager:
    """Owns one engine + batcher per configured model.

    The request's `model` field selects the backend. If no model matches,
    we fall back to the default so existing `{"model":"tiny-gpt2"}`
    clients keep working.
    """

    def __init__(self) -> None:
        self.engines: Dict[str, InferenceEngine] = {}
        self.batchers: Dict[str, ContinuousBatcher] = {}
        self.default_model: Optional[str] = None

    async def start(self) -> None:
        raw = os.environ.get("MVP_MODELS", "").strip()
        model_names = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
        if not model_names:
            # single-model legacy path
            model_names = [os.environ.get("MVP_MODEL", "sshleifer/tiny-gpt2")]

        max_batch = int(os.environ.get("MVP_MAX_BATCH", "8"))
        max_wait_ms = int(os.environ.get("MVP_MAX_WAIT_MS", "20"))

        for name in model_names:
            logger.info("Loading model %s ...", name)
            engine = InferenceEngine(model_name=name)
            batcher = ContinuousBatcher(
                engine, max_batch=max_batch, max_wait_ms=max_wait_ms
            )
            batcher.start()
            self.engines[name] = engine
            self.batchers[name] = batcher
            # Register under short alias too (basename after last '/').
            alias = name.rsplit("/", 1)[-1]
            if alias != name and alias not in self.engines:
                self.engines[alias] = engine
                self.batchers[alias] = batcher
            if self.default_model is None:
                self.default_model = name

    async def stop(self) -> None:
        seen: set = set()
        for b in self.batchers.values():
            if id(b) in seen:
                continue
            seen.add(id(b))
            await b.stop()

    def resolve(self, requested: str) -> str:
        if requested in self.engines:
            return requested
        assert self.default_model is not None
        logger.debug("Model %r not loaded — falling back to %s", requested, self.default_model)
        return self.default_model

    def engine(self, model: str) -> InferenceEngine:
        return self.engines[self.resolve(model)]

    def batcher(self, model: str) -> ContinuousBatcher:
        return self.batchers[self.resolve(model)]


manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = ModelManager()
    await manager.start()
    logger.info(
        "M5 ready — models=%s auth=%s",
        list(manager.engines.keys()),
        "on" if auth_enabled() else "OFF",
    )
    yield
    logger.info("Shutting down.")
    if manager is not None:
        await manager.stop()


app = FastAPI(
    title="High-Throughput LLM Inference Server (M5)",
    version="0.5.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health / metrics — unauthenticated.
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert manager is not None and manager.default_model is not None
    eng = manager.engine(manager.default_model)
    return HealthResponse(
        status="ok",
        model=eng.model_name,
        backend=eng.backend_name,
    )


@app.get("/metrics")
def prom_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/v1/models")
def list_models(_: str = Depends(require_api_key)) -> dict:
    assert manager is not None
    created = int(time.time())
    seen: set = set()
    data = []
    for name, eng in manager.engines.items():
        if id(eng) in seen:
            continue
        seen.add(id(eng))
        data.append({"id": name, "object": "model", "created": created, "owned_by": "local"})
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# /v1/completions — now with streaming.
# ---------------------------------------------------------------------------


@app.post("/v1/completions")
async def completions(
    req: CompletionRequest,
    _: str = Depends(require_api_key),
):
    if manager is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    model_id = manager.resolve(req.model)
    engine = manager.engine(model_id)
    batcher = manager.batcher(model_id)

    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            raise HTTPException(
                status_code=400, detail="Only a single prompt per request is supported"
            )
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    if req.stream:
        return StreamingResponse(
            _stream_completion(
                engine=engine,
                batcher=batcher,
                prompt=prompt,
                req=req,
                model_id=model_id,
            ),
            media_type="text/event-stream",
        )

    with metrics.time_request("completions", model_id):
        result = await batcher.submit(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
    metrics.record_tokens(model_id, result.prompt_tokens, result.completion_tokens)

    return CompletionResponse(
        model=engine.model_name,
        choices=[
            CompletionChoice(
                text=result.text,
                index=0,
                finish_reason=result.finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _stream_completion(
    engine: InferenceEngine,
    batcher: ContinuousBatcher,
    prompt: str,
    req: CompletionRequest,
    model_id: str,
):
    """SSE pseudo-stream.

    The batcher's public API returns one `GenerationResult` at the end
    of decode rather than emitting token-by-token callbacks, so M5's
    streaming is an approximation: we chunk the final text and emit
    each chunk as a delta. Still matches the OpenAI wire format and
    lets clients wire up incremental rendering; a truer per-token
    stream is an engine-level change punted to M6.
    """
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    with metrics.time_request("completions_stream", model_id):
        result = await batcher.submit(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
    metrics.record_tokens(model_id, result.prompt_tokens, result.completion_tokens)

    text = result.text
    # Emit ~one word per chunk so clients see progressive output.
    tokens = text.split(" ")
    for idx, piece in enumerate(tokens):
        delta_text = piece if idx == 0 else " " + piece
        payload = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": engine.model_name,
            "choices": [{"text": delta_text, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0)

    final = {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": engine.model_name,
        "choices": [{"text": "", "index": 0, "finish_reason": result.finish_reason}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield sse_done()


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    _: str = Depends(require_api_key),
):
    if manager is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    model_id = manager.resolve(req.model)
    engine = manager.engine(model_id)
    batcher = manager.batcher(model_id)

    prompt = build_prompt(req.messages, req.tools)

    if req.stream:
        return StreamingResponse(
            _stream_chat(
                engine=engine,
                batcher=batcher,
                prompt=prompt,
                req=req,
                model_id=model_id,
            ),
            media_type="text/event-stream",
        )

    with metrics.time_request("chat_completions", model_id):
        result = await batcher.submit(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
    metrics.record_tokens(model_id, result.prompt_tokens, result.completion_tokens)

    tool_calls = extract_tool_calls(result.text) if req.tools else None
    message = ChatResponseMessage(
        role="assistant",
        content=None if tool_calls else result.text,
        tool_calls=tool_calls,
    )
    finish = "tool_calls" if tool_calls else result.finish_reason

    return ChatCompletionResponse(
        model=engine.model_name,
        choices=[ChatChoice(index=0, message=message, finish_reason=finish)],
        usage=ChatUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _stream_chat(
    engine: InferenceEngine,
    batcher: ContinuousBatcher,
    prompt: str,
    req: ChatCompletionRequest,
    model_id: str,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # First chunk announces the role.
    yield sse_chunk(engine.model_name, chunk_id, created, {"role": "assistant"})

    with metrics.time_request("chat_completions_stream", model_id):
        result = await batcher.submit(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
    metrics.record_tokens(model_id, result.prompt_tokens, result.completion_tokens)

    tool_calls = extract_tool_calls(result.text) if req.tools else None

    if tool_calls:
        # Emit a single chunk with the tool_calls delta.
        tc_payload = [
            {
                "index": i,
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for i, tc in enumerate(tool_calls)
        ]
        yield sse_chunk(
            engine.model_name, chunk_id, created, {"tool_calls": tc_payload}
        )
        yield sse_chunk(
            engine.model_name, chunk_id, created, {}, finish_reason="tool_calls"
        )
    else:
        for idx, piece in enumerate(result.text.split(" ")):
            delta_text = piece if idx == 0 else " " + piece
            yield sse_chunk(
                engine.model_name, chunk_id, created, {"content": delta_text}
            )
            await asyncio.sleep(0)
        yield sse_chunk(
            engine.model_name,
            chunk_id,
            created,
            {},
            finish_reason=result.finish_reason,
        )

    yield sse_done()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {
        "service": "high-throughput-llm-inference-server",
        "stage": "M5",
        "endpoints": [
            "/health",
            "/metrics",
            "/v1/models",
            "/v1/completions",
            "/v1/chat/completions",
        ],
        "auth": "bearer" if auth_enabled() else "disabled",
    }
