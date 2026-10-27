import pytest
import requests

from reglens.ingest.download import download_all, fetch_with_retry
from reglens.ingest.manifest import Source

PDF = b"%PDF-1.7 fake"


def no_sleep(_s: float) -> None:
    pass


def http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(response=resp)


def flaky(errors: list[Exception], body: bytes = PDF):
    """Fake fetch: raises each queued error once, then returns body."""
    calls = {"n": 0}

    def fetch(url: str, timeout: float) -> bytes:
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return body

    return fetch, calls


def test_retries_dropped_connection_then_succeeds():
    # the exact failure seen in CI: SEBI closed the connection mid-file
    fetch, calls = flaky([requests.exceptions.ChunkedEncodingError("IncompleteRead")])
    assert fetch_with_retry("https://x", fetch=fetch, sleep=no_sleep) == PDF
    assert calls["n"] == 2


def test_retries_server_errors():
    fetch, calls = flaky([http_error(503), http_error(429)])
    assert fetch_with_retry("https://x", fetch=fetch, sleep=no_sleep) == PDF
    assert calls["n"] == 3


def test_does_not_retry_404():
    fetch, calls = flaky([http_error(404)])
    with pytest.raises(requests.HTTPError):
        fetch_with_retry("https://x", fetch=fetch, sleep=no_sleep)
    assert calls["n"] == 1


def test_gives_up_after_max_attempts():
    fetch, calls = flaky([requests.ConnectionError()] * 10)
    with pytest.raises(requests.ConnectionError):
        fetch_with_retry("https://x", attempts=3, fetch=fetch, sleep=no_sleep)
    assert calls["n"] == 3


def test_download_all_uses_cache_and_rejects_non_pdf(tmp_path):
    src = Source("circ-001", "t", "2026-01-15", "https://x")
    fetch, calls = flaky([])
    download_all([src], tmp_path, fetch=fetch, sleep=no_sleep)
    download_all([src], tmp_path, fetch=fetch, sleep=no_sleep)
    assert calls["n"] == 1  # second run served from disk

    bad = Source("circ-002", "t", "2026-01-15", "https://y")
    html, _ = flaky([], body=b"<html>blocked</html>")
    with pytest.raises(ValueError):
        download_all([bad], tmp_path, fetch=html, sleep=no_sleep)
    assert not (tmp_path / "circ-002.pdf").exists()
