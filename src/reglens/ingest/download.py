"""Polite, idempotent downloader: skips files already on disk, verifies they are PDFs."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .manifest import Source

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "RegLens/0.1 (research project; contact via GitHub mlexpnew)"}


def download_all(
    sources: list[Source], raw_dir: Path, delay_s: float = 1.0, timeout_s: float = 30.0
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for src in sources:
        dest = raw_dir / src.filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("skip %s (cached)", src.doc_id)
            paths.append(dest)
            continue
        resp = requests.get(src.url, headers=HEADERS, timeout=timeout_s)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise ValueError(f"{src.doc_id}: url did not return a PDF")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)  # atomic: never leaves a half-written PDF
        log.info("saved %s (%d KB)", src.doc_id, len(resp.content) // 1024)
        paths.append(dest)
        time.sleep(delay_s)
    return paths
