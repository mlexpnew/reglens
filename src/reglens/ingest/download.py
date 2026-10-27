"""Polite, idempotent downloader: skips cached files, retries flaky transfers, verifies PDFs."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

import requests

from .manifest import Source

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "RegLens/0.1 (research project; contact via GitHub mlexpnew)"}

# Network blips (dropped connection mid-file, timeouts) and server-side 429/5xx are transient.
# A 404 or a non-PDF body is a real problem with the manifest: fail fast, never retry.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _fetch(url: str, timeout_s: float) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def fetch_with_retry(
    url: str,
    timeout_s: float = 30.0,
    attempts: int = 4,
    base_delay: float = 2.0,
    fetch: Callable[[str, float], bytes] = _fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Download with exponential backoff and jitter, retrying only transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch(url, timeout_s)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS or attempt == attempts:
                raise
            reason = f"HTTP {status}"
        except RETRYABLE_ERRORS as exc:
            if attempt == attempts:
                raise
            reason = type(exc).__name__
        delay = random.uniform(base_delay / 2, base_delay * 2 ** (attempt - 1))
        log.warning("%s on attempt %d/%d, retrying in %.1fs", reason, attempt, attempts, delay)
        sleep(delay)
    raise AssertionError("unreachable")


def download_all(
    sources: list[Source],
    raw_dir: Path,
    delay_s: float = 1.0,
    timeout_s: float = 30.0,
    fetch: Callable[[str, float], bytes] = _fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for src in sources:
        dest = raw_dir / src.filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("skip %s (cached)", src.doc_id)
            paths.append(dest)
            continue
        content = fetch_with_retry(src.url, timeout_s, fetch=fetch, sleep=sleep)
        if not content.startswith(b"%PDF"):
            raise ValueError(f"{src.doc_id}: url did not return a PDF")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(content)
        tmp.replace(dest)  # atomic: never leaves a half-written PDF
        log.info("saved %s (%d KB)", src.doc_id, len(content) // 1024)
        paths.append(dest)
        sleep(delay_s)
    return paths
