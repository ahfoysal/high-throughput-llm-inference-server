"""FastAPI entry point — OpenAI-compatible /v1/completions."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import os

from .batcher import StaticBatcher
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


engine: InferenceEngine | None = None
batcher: StaticBatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, batcher
    logger.info("Initializing InferenceEngine...")
    engine = InferenceEngine()
    max_batch = int(os.environ.get("MVP_MAX_BATCH", "8"))
    max_wait_ms = int(os.environ.get("MVP_MAX_WAIT_MS", "20"))
    batcher = StaticBatcher(engine, max_batch=max_batch, max_wait_ms=max_wait_ms)
    batcher.start()
    logger.info(
        "Engine ready — backend=%s model=%s batch<=%d wait<=%dms",
        engine.backend_name, engine.model_name, max_batch, max_wait_ms,
    )
    yield
    logger.info("Shutting down.")
    if batcher is not None:
        await batcher.stop()


app = FastAPI(
    title="High-Throughput LLM Inference Server (MVP)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert engine is not None
    return HealthResponse(
        status="ok",
        model=engine.model_name,
        backend=engine.backend_name,
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest) -> CompletionResponse:
    if engine is None or batcher is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if req.stream:
        # TODO(M1.1): SSE streaming. Out of MVP scope.
        raise HTTPException(status_code=400, detail="stream=true not supported in MVP")

    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            # TODO(M3): batched prompts in one request. The batcher merges
            # concurrent single-prompt requests into a batch server-side.
            raise HTTPException(status_code=400, detail="MVP supports a single prompt per request")
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    # Submit through the static batcher — concurrent requests get merged
    # into a single padded forward per decode step.
    result = await batcher.submit(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
    )

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


@app.get("/")
def root() -> dict:
    return {
        "service": "high-throughput-llm-inference-server",
        "stage": "MVP",
        "endpoints": ["/health", "/v1/completions"],
    }
