"""Pydantic request/response models — OpenAI-compatible shape.

Reference: https://platform.openai.com/docs/api-reference/completions
We match the minimal subset the MVP needs; extra fields are ignored.
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = Field(default="tiny-gpt2")
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=16, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    n: int = Field(default=1, ge=1, le=1)  # MVP: n=1 only
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[dict] = None
    finish_reason: str = "length"  # "length" | "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str  # "transformers" | "mock"
