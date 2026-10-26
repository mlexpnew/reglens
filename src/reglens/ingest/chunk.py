"""Paragraph-aware chunking that keeps page provenance for citations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_long(paragraph: str, size: int) -> list[str]:
    """Split one oversized paragraph on word boundaries."""
    words, parts, cur = paragraph.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        parts.append(cur)
    return parts


def chunk_pages(
    pages: list[str], doc_meta: dict, size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    # (text, page_number) units, pages are 1-indexed for human citations
    units: list[tuple[str, int]] = []
    for page_no, page in enumerate(pages, start=1):
        for para in page.split("\n"):
            para = para.strip()
            if not para:
                continue
            for piece in _split_long(para, size) if len(para) > size else [para]:
                units.append((piece, page_no))

    chunks: list[Chunk] = []
    buf: list[tuple[str, int]] = []

    def flush() -> None:
        text = "\n".join(t for t, _ in buf)
        idx = len(chunks)
        digest = hashlib.sha256(f"{doc_meta['doc_id']}|{idx}|{text}".encode()).hexdigest()
        chunks.append(
            Chunk(
                id=f"{doc_meta['doc_id']}-{idx:04d}-{digest[:10]}",
                text=text,
                metadata={
                    **doc_meta,
                    "chunk_index": idx,
                    "page_start": buf[0][1],
                    "page_end": buf[-1][1],
                },
            )
        )

    for unit in units:
        cur_len = sum(len(t) + 1 for t, _ in buf)
        if buf and cur_len + len(unit[0]) > size:
            flush()
            # carry tail units forward as overlap so context spans chunk borders
            tail, tail_len = [], 0
            for u in reversed(buf):
                if tail_len + len(u[0]) > overlap:
                    break
                tail.insert(0, u)
                tail_len += len(u[0]) + 1
            buf = tail
        buf.append(unit)
    if buf:
        flush()
    return chunks
