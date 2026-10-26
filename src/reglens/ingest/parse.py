"""PDF to clean per-page text, with repeated headers/footers stripped."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

_WS = re.compile(r"[ \t ]+")
_PAGE_NUM = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)


def _normalize(text: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def strip_boilerplate(pages: list[str], min_share: float = 0.6) -> list[str]:
    """Drop lines that repeat on most pages (letterheads, footers) and bare page numbers."""
    if len(pages) < 3:
        return ["\n".join(ln for ln in p.splitlines() if not _PAGE_NUM.match(ln)) for p in pages]
    counts = Counter(ln for p in pages for ln in set(p.splitlines()))
    threshold = max(2, int(len(pages) * min_share))
    repeated = {ln for ln, n in counts.items() if n >= threshold}
    return [
        "\n".join(ln for ln in p.splitlines() if ln not in repeated and not _PAGE_NUM.match(ln))
        for p in pages
    ]


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = [_normalize(page.extract_text() or "") for page in reader.pages]
    return strip_boilerplate(pages)
