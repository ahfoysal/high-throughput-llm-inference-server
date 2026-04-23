"""Static batcher: async queue + time/size flush.

M2 implementation. Requests arrive on `submit()` and are buffered. A
background task flushes the buffer to `engine.generate_batch(...)` when
either:

- `max_batch` requests are queued, OR
- `max_wait_ms` has elapsed since the first queued request.

Each waiter awaits its own `asyncio.Future` so a slow request doesn't
block faster ones beyond the flush window.

This is *static* batching — once a batch starts, no new requests join
until it finishes. M3 will replace this with continuous batching.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .engine import GenerationResult, InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class _BatchRequest:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    seed: Optional[int]
    future: "asyncio.Future[GenerationResult]" = field(default=None)  # type: ignore[assignment]


class StaticBatcher:
    def __init__(
        self,
        engine: "InferenceEngine",
        max_batch: int = 8,
        max_wait_ms: int = 20,
    ) -> None:
        self.engine = engine
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self._queue: "asyncio.Queue[_BatchRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="static-batcher-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def submit(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        seed: Optional[int],
    ) -> "GenerationResult":
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[GenerationResult]" = loop.create_future()
        req = _BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            future=fut,
        )
        await self._queue.put(req)
        return await fut

    async def _loop(self) -> None:
        wait_s = self.max_wait_ms / 1000.0
        while self._running:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            batch: List[_BatchRequest] = [first]
            deadline = time.monotonic() + wait_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            await self._run_batch(batch)

    async def _run_batch(self, batch: List[_BatchRequest]) -> None:
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                self.engine.generate_batch,
                [r.prompt for r in batch],
                [r.max_tokens for r in batch],
                [r.temperature for r in batch],
                [r.top_p for r in batch],
                [r.top_k for r in batch],
                [r.seed for r in batch],
            )
            for req, res in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(res)
        except Exception as e:  # noqa: BLE001
            logger.exception("batch failed: %s", e)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
