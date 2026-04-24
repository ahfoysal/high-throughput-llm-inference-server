"""M5 — OpenAI-compatible /v1/chat/completions support.

Builds a flat prompt string from a `messages` array using a minimal
ChatML-ish template (the tiny-gpt2 backing model isn't chat-tuned, so
any template is cosmetic — but the shape matches real chat-tuned
models enough that a client treating us as OpenAI-compatible "just
works").

Tool calling: if the request includes `tools`, we prepend a brief
system instruction listing them. When the generated text contains a
JSON object with the shape `{"name": "...", "arguments": {...}}` we
surface it as a structured `tool_calls` entry on the response message
(mirroring OpenAI's shape). This is a pragmatic sniff, not a
trained-in protocol — good enough to demonstrate the wiring.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="tiny-gpt2")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, per OpenAI spec


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(messages: List[ChatMessage], tools: Optional[List[Tool]]) -> str:
    parts: List[str] = []
    if tools:
        tool_lines = [
            f"- {t.function.name}: {t.function.description or ''} "
            f"params={json.dumps(t.function.parameters or {})}"
            for t in tools
        ]
        parts.append(
            "<|system|>\nYou have access to the following tools. "
            "To call a tool, reply with a single JSON object of the form "
            '{"name": "<tool_name>", "arguments": {...}}.\n'
            + "\n".join(tool_lines)
            + "\n"
        )
    for m in messages:
        role = m.role
        content = m.content or ""
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tool-call sniffing
# ---------------------------------------------------------------------------


def _iter_balanced_json_objects(text: str):
    """Yield substrings that look like balanced `{...}` objects.

    Naive brace-matching is fine here — we're sniffing model output, not
    parsing adversarial input. String literals are tracked so `{` / `}`
    inside a JSON string don't confuse the counter.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def extract_tool_calls(text: str) -> Optional[List[ToolCall]]:
    """Sniff generated text for a `{"name": ..., "arguments": ...}` call.

    Returns `None` if nothing tool-shaped is present. Designed to be
    conservative — false positives just mean the assistant's text gets
    re-interpreted as a tool call, which is visible to the client.
    """
    calls: List[ToolCall] = []
    for chunk in _iter_balanced_json_objects(text):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(name, str) and args is not None:
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    function=ToolCallFunction(name=name, arguments=args_str),
                )
            )
    return calls or None


# ---------------------------------------------------------------------------
# SSE chunk helpers
# ---------------------------------------------------------------------------


def sse_chunk(
    model: str,
    chunk_id: str,
    created: int,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
