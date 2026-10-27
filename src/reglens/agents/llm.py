"""LLM behind a one-method protocol: swap providers or use a fake in tests."""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete_json(self, prompt: str) -> str:
        """Return the model's raw text, expected to be a JSON object."""
        ...


class LLMConfigError(RuntimeError):
    pass


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
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt, config=self._config
        )
        return resp.text or ""
