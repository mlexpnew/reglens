import pytest

from reglens.ingest.manifest import Source
from reglens.ingest.pipeline import ingest
from reglens.ingest.store import MemoryStore


def test_end_to_end_and_idempotent(sample_pdf):
    src = Source("circ-001", "Disclosure norms", "2026-01-15", "https://www.sebi.gov.in/x.pdf")
    store = MemoryStore()
    r1 = ingest([src], sample_pdf.parent, store, size=120, overlap=30)
    assert r1.documents == 1 and r1.pages == 3 and r1.chunks >= 2
    c = next(iter(store.items.values()))
    assert c.metadata["url"].startswith("https://") and c.metadata["page_start"] >= 1

    r2 = ingest([src], sample_pdf.parent, store, size=120, overlap=30)
    assert r2.store_total == r1.store_total  # re-run adds no duplicates


def test_missing_pdf_fails_loudly(tmp_path):
    src = Source("nope", "t", "2026-01-15", "https://x")
    with pytest.raises(FileNotFoundError):
        ingest([src], tmp_path, MemoryStore(), size=100, overlap=10)
