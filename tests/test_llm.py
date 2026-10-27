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
