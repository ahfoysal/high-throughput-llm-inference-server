"""FastAPI entry point — OpenAI-compatible /v1/completions."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing InferenceEngine...")
    engine = InferenceEngine()
    logger.info("Engine ready — backend=%s model=%s", engine.backend_name, engine.model_name)
    yield
    logger.info("Shutting down.")


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
def completions(req: CompletionRequest) -> CompletionResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if req.stream:
        # TODO(M1.1): SSE streaming. Out of MVP scope.
        raise HTTPException(status_code=400, detail="stream=true not supported in MVP")

    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            # TODO(M3): batched prompts — handled by ContinuousBatcher.
            raise HTTPException(status_code=400, detail="MVP supports a single prompt only")
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    result = engine.generate(
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
