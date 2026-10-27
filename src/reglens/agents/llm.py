"""LLM behind a one-method protocol: swap providers or use a fake in tests."""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")

# 429 = rate limited, 5xx = provider overloaded or down: worth retrying.
# 4xx like 400/401/403 are our fault (bad request, bad key): retrying wastes time.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLM(Protocol):
    def complete_json(self, prompt: str) -> str:
        """Return the model's raw text, expected to be a JSON object."""
        ...


class LLMConfigError(RuntimeError):
    pass


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def with_retry(
    fn: Callable[[], T],
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Exponential backoff with full jitter, retrying only transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            status = _status(exc)
            if status not in RETRYABLE_STATUS or attempt == attempts:
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            log.warning(
                "LLM %s on attempt %d/%d, retrying in %.1fs", status, attempt, attempts, delay
            )
            sleep(delay)
    raise AssertionError("unreachable")


class GeminiLLM:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMConfigError("GEMINI_API_KEY is not set")
        from google import genai  # lazy: optional dependency
        from google.genai import types

        self._client = genai.Client(api_key=key)
        self._model = model
        # temperature 0: compliance answers should be reproducible, not creative
        self._config = types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"
        )

    def complete_json(self, prompt: str) -> str:
        resp = with_retry(
            lambda: self._client.models.generate_content(
                model=self._model, contents=prompt, config=self._config
            )
        )
        return resp.text or ""
