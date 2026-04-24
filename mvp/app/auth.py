"""M5 — bearer-token auth.

Single-process API-key middleware. Keys are loaded once at import time
from the `MVP_API_KEYS` env var (comma-separated). If unset, auth is
disabled — convenient for local dev but logged loudly.

Usage:
    from fastapi import Depends
    from .auth import require_api_key

    @app.post("/v1/...", dependencies=[Depends(require_api_key)])
    async def handler(...): ...
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Set

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


def _load_keys() -> Set[str]:
    raw = os.environ.get("MVP_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


_API_KEYS: Set[str] = _load_keys()

if not _API_KEYS:
    logger.warning(
        "MVP_API_KEYS is unset — /v1/* endpoints are UNAUTHENTICATED. "
        "Set MVP_API_KEYS=key1,key2 to enable bearer-token auth."
    )
else:
    logger.info("Loaded %d API key(s) from MVP_API_KEYS.", len(_API_KEYS))


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
) -> str:
    """FastAPI dependency — enforces `Authorization: Bearer <key>`.

    Returns the authenticated key (or `"anonymous"` if auth is disabled)
    so downstream handlers can tag metrics / logs per-tenant.
    """
    if not _API_KEYS:
        return "anonymous"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    key = parts[1]
    if key not in _API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return key


def auth_enabled() -> bool:
    return bool(_API_KEYS)
