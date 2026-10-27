from reglens.ingest.store import collection_name
from reglens.retrieve import CrossEncoderReranker, Passage


class FakeCrossEncoder:
    """Scores a passage by how many query words it contains (logit scale)."""

    def predict(self, pairs):
        return [sum(w in p.lower() for w in q.lower().split()) * 2.0 - 3.0 for q, p in pairs]


def P(i, text, page=1):
    meta = {
        "doc_id": f"d{i}",
        "title": f"Circular {i}",
        "date": "2026-10-07",
        "page_start": page,
        "page_end": page,
        "url": "https://www.sebi.gov.in/x.pdf",
    }
    return Passage(id=f"p{i}", text=text, meta=meta)


def test_rerank_orders_by_relevance_and_truncates():
    rr = CrossEncoderReranker("fake", model=FakeCrossEncoder())
    ps = [P(1, "unrelated text"), P(2, "credit risk meter for debt"), P(3, "credit rating")]
    out = rr.rerank("credit risk meter", ps, k=2)
    assert [p.id for p in out] == ["p2", "p3"]
    assert all(0.0 <= p.score <= 1.0 for p in out)  # sigmoid-normalised


def test_rerank_empty_is_safe():
    assert CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank("q", [], 3) == []


def test_citation_format_single_and_range():
    assert P(1, "x").citation == "Circular 1 (2026-10-07), p.1"
    p = Passage("a", "x", {"title": "T", "date": "d", "page_start": 2, "page_end": 3})
    assert p.citation.endswith("pp.2-3")


def test_collection_is_namespaced_per_embedding_model():
    a = collection_name("sebi", "BAAI/bge-small-en-v1.5")
    b = collection_name("sebi", "all-MiniLM-L6-v2")
    assert a != b and a.startswith("sebi__") and len(a) <= 63
