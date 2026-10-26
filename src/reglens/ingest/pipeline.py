"""Ingestion pipeline: manifest -> PDFs -> pages -> chunks -> vector store."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunk import chunk_pages
from .manifest import Source
from .parse import extract_pages
from .store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    skipped_empty: int = 0
    store_total: int = 0
    elapsed_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def ingest(
    sources: list[Source], raw_dir: Path, store: VectorStore, size: int, overlap: int
) -> IngestReport:
    t0 = time.perf_counter()
    report = IngestReport()
    for src in sources:
        pdf = raw_dir / src.filename
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} missing; run with --download first")
        pages = extract_pages(pdf)
        if not any(p.strip() for p in pages):
            # scanned PDF with no text layer: flag it, never index empty noise
            log.warning("%s has no extractable text (needs OCR), skipped", src.doc_id)
            report.skipped_empty += 1
            continue
        meta = {
            "doc_id": src.doc_id,
            "title": src.title,
            "date": src.date,
            "url": src.url,
            "circular_no": src.circular_no,
        }
        chunks = chunk_pages(pages, meta, size=size, overlap=overlap)
        store.upsert(chunks)
        report.documents += 1
        report.pages += len(pages)
        report.chunks += len(chunks)
        log.info("%s: %d pages -> %d chunks", src.doc_id, len(pages), len(chunks))
    report.store_total = store.count()
    report.elapsed_s = round(time.perf_counter() - t0, 3)
    return report
