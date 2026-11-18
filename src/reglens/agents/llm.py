"""LLM behind a one-method protocol: swap providers or use a fake in tests.

Providers are chained: `REGLENS_LLM="groq:llama-3.3-70b-versatile,gemini:gemini-2.5-flash"`
tries Groq first and moves to Gemini for good once Groq's daily quota is spent. A spent
daily quota is detected and never retried, because waiting seconds cannot refill it.
"""

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


class LLMHTTPError(RuntimeError):
    """HTTP failure from a provider; `code` carries the status so retry logic can read it."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class QuotaExhausted(RuntimeError):
    """Every configured provider has spent its daily quota: stop, do not keep retrying."""


# Phrases providers use for a per-day cap (Gemini: "PerDay", Groq: "tokens per day (TPD)").
_DAILY_QUOTA = ("perday", "per day", "per_day", "daily limit", "daily quota")


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def is_daily_quota(exc: Exception) -> bool:
    """A 429 for a daily cap: retrying within this run is pointless."""
    if _status(exc) != 429:
        return False
    text = str(exc).lower().replace(" ", "")
    return any(p.replace(" ", "") in text for p in _DAILY_QUOTA)


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
            if status not in RETRYABLE_STATUS or attempt == attempts or is_daily_quota(exc):
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


# OpenAI-compatible chat endpoints: one client covers hosted and local models.
PRESETS: dict[str, tuple[str, str | None, str]] = {
    # name: (base_url, api key env var or None for local, default model)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b-instruct"),
}
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", **{k: v[2] for k, v in PRESETS.items()}}


def _post_json(url: str, headers: dict, body: dict, timeout_s: float) -> dict:
    import requests  # already a core dependency

    resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    if resp.status_code >= 400:
        raise LLMHTTPError(resp.status_code, resp.text[:500])
    return resp.json()


class OpenAICompatLLM:
    """Groq, OpenRouter, Ollama or any OpenAI-compatible server, using plain HTTP."""

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        post: Callable[[str, dict, dict, float], dict] = _post_json,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if provider not in PRESETS:
            raise LLMConfigError(f"unknown provider {provider!r}")
        default_url, key_env, default_model = PRESETS[provider]
        key = api_key or (os.getenv(key_env) if key_env else None)
        if key_env and not key:
            raise LLMConfigError(f"{key_env} is not set")
        self.name = f"{provider}:{model or default_model}"
        self._url = (base_url or default_url).rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"
        self._model = model or default_model
        self._timeout = timeout_s
        self._post = post
        self._sleep = sleep

    def complete_json(self, prompt: str) -> str:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,  # compliance answers should be reproducible
            "response_format": {"type": "json_object"},
        }
        data = with_retry(
            lambda: self._post(self._url, self._headers, body, self._timeout), sleep=self._sleep
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMHTTPError(502, f"unexpected response shape: {str(data)[:200]}") from exc


class FallbackLLM:
    """Try providers in order; once one runs out of daily quota, move on for good."""

    def __init__(self, llms: list, log_fn: Callable[[str], None] = log.warning) -> None:
        if not llms:
            raise LLMConfigError("no LLM provider configured")
        self._llms = list(llms)
        self._i = 0
        self._log = log_fn

    @property
    def active(self) -> str:
        return getattr(self._llms[self._i], "name", type(self._llms[self._i]).__name__)

    @property
    def names(self) -> list[str]:
        return [getattr(m, "name", type(m).__name__) for m in self._llms]

    def complete_json(self, prompt: str) -> str:
        while True:
            try:
                return self._llms[self._i].complete_json(prompt)
            except Exception as exc:
                spent = is_daily_quota(exc) or _status(exc) in RETRYABLE_STATUS
                if not spent:
                    raise
                if self._i + 1 >= len(self._llms):
                    raise QuotaExhausted(f"all providers exhausted, last: {exc}"[:300]) from exc
                self._log(f"{self.active} unavailable ({_status(exc)}), switching provider")
                self._i += 1


def build_llm(spec: str, factories: dict[str, Callable[[str], object]] | None = None):
    """Build a FallbackLLM from "provider:model,provider:model". Unconfigured ones are skipped."""
    factories = factories or {
        "gemini": lambda m: _named(GeminiLLM(m), f"gemini:{m}"),
        **{p: (lambda m, p=p: OpenAICompatLLM(p, m)) for p in PRESETS},
    }
    llms, skipped = [], []
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        provider, _, model = item.partition(":")
        if provider not in factories:
            raise LLMConfigError(f"unknown provider {provider!r} in REGLENS_LLM")
        try:
            llms.append(factories[provider](model or DEFAULT_MODELS.get(provider, "")))
        except LLMConfigError as exc:
            skipped.append(f"{item} ({exc})")
    if not llms:
        raise LLMConfigError("no usable LLM provider: " + "; ".join(skipped))
    for s in skipped:
        log.warning("skipping LLM provider %s", s)
    return FallbackLLM(llms)


def _named(llm, name: str):
    llm.name = name
    return llm
