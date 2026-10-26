import pytest

from reglens.ingest.chunk import chunk_pages

META = {
    "doc_id": "circ-001",
    "title": "t",
    "date": "2026-01-01",
    "url": "https://x",
    "circular_no": "",
}


def test_small_doc_is_one_chunk_with_pages():
    chunks = chunk_pages(["alpha", "beta"], META, size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2


def test_chunks_respect_size_and_overlap_carries_context():
    paras = [f"para {i} " + "word " * 30 for i in range(10)]
    chunks = chunk_pages(["\n".join(paras)], META, size=400, overlap=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    # last paragraph of chunk N reappears at the start of chunk N+1
    first_tail = chunks[0].text.split("\n")[-1]
    assert chunks[1].text.startswith(first_tail)


def test_oversized_paragraph_splits_on_words():
    chunks = chunk_pages(["word " * 500], META, size=300, overlap=0)
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(not c.text.startswith(" ") for c in chunks)


def test_ids_are_stable_and_unique():
    a = chunk_pages(["x " * 400], META, size=200, overlap=50)
    b = chunk_pages(["x " * 400], META, size=200, overlap=50)
    assert [c.id for c in a] == [c.id for c in b]  # re-ingest = upsert, never duplicates
    assert len({c.id for c in a}) == len(a)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages(["x"], META, size=100, overlap=100)
