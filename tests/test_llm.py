import pytest

from reglens.agents.llm import with_retry


class APIError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


def flaky(failures, code):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise APIError(code)
        return "ok"

    return fn, calls


def test_retries_transient_503_then_succeeds():
    fn, calls = flaky(2, 503)
    sleeps = []
    assert with_retry(fn, sleep=sleeps.append) == "ok"
    assert calls["n"] == 3 and len(sleeps) == 2


def test_backoff_is_bounded():
    fn, _ = flaky(4, 429)
    sleeps = []
    with_retry(fn, attempts=5, base_delay=2, max_delay=5, sleep=sleeps.append)
    assert all(0 <= s <= 5 for s in sleeps)


def test_gives_up_after_max_attempts():
    fn, calls = flaky(10, 503)
    with pytest.raises(APIError):
        with_retry(fn, attempts=3, sleep=lambda s: None)
    assert calls["n"] == 3


def test_does_not_retry_client_errors():
    fn, calls = flaky(1, 401)  # bad API key: retrying cannot help
    with pytest.raises(APIError):
        with_retry(fn, sleep=lambda s: None)
    assert calls["n"] == 1


from reglens.agents.llm import (  # noqa: E402
    FallbackLLM,
    LLMConfigError,
    LLMHTTPError,
    OpenAICompatLLM,
    QuotaExhausted,
    build_llm,
    is_daily_quota,
)


def test_daily_quota_detected_for_gemini_and_groq_messages():
    gemini = APIError(429)
    gemini.args = ("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier",)
    groq = LLMHTTPError(429, "Rate limit reached on tokens per day (TPD): Limit 100000")
    per_minute = LLMHTTPError(429, "Rate limit reached on requests per minute (RPM)")
    assert is_daily_quota(gemini) and is_daily_quota(groq)
    assert not is_daily_quota(per_minute) and not is_daily_quota(LLMHTTPError(500, "per day"))


def test_daily_quota_is_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise LLMHTTPError(429, "tokens per day exceeded")

    with pytest.raises(LLMHTTPError):
        with_retry(fn, sleep=lambda s: None)
    assert calls["n"] == 1


def test_openai_compatible_client_sends_deterministic_json_request():
    seen = {}

    def post(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    llm = OpenAICompatLLM("groq", api_key="k", post=post)
    assert llm.complete_json("Return JSON") == '{"ok": true}'
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert llm.name == "groq:llama-3.3-70b-versatile"


def test_local_ollama_needs_no_key_and_hosted_providers_do(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        OpenAICompatLLM("groq")
    assert OpenAICompatLLM("ollama").name.startswith("ollama:")


def test_bad_response_shape_is_a_clear_error():
    llm = OpenAICompatLLM("ollama", post=lambda *a: {"error": "x"}, sleep=lambda s: None)
    with pytest.raises(LLMHTTPError):
        llm.complete_json("p")


class Named:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, fail, 0

    def complete_json(self, prompt):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.name


def test_fallback_switches_for_good_after_quota_and_raises_when_all_spent():
    spent = LLMHTTPError(429, "tokens per day")
    a, b = Named("a", spent), Named("b")
    chain = FallbackLLM([a, b], log_fn=lambda m: None)
    assert chain.complete_json("p") == "b" and chain.complete_json("p") == "b"
    assert a.calls == 1 and chain.active == "b"

    with pytest.raises(QuotaExhausted):
        FallbackLLM([Named("x", spent)], log_fn=lambda m: None).complete_json("p")


def test_fallback_does_not_hide_real_bugs():
    chain = FallbackLLM([Named("a", ValueError("bad prompt")), Named("b")], log_fn=lambda m: None)
    with pytest.raises(ValueError):
        chain.complete_json("p")


def test_build_llm_skips_unconfigured_providers():
    def missing(model):
        raise LLMConfigError("GROQ_API_KEY is not set")

    factories = {"groq": missing, "ollama": lambda m: Named(f"ollama:{m}")}
    chain = build_llm("groq:m1, ollama", factories)
    assert chain.names == ["ollama:qwen2.5:7b-instruct"]
    with pytest.raises(LLMConfigError):
        build_llm("groq", {"groq": missing})
    with pytest.raises(LLMConfigError):
        build_llm("nope:x", {})
