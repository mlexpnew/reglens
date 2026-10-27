import json

import pytest

pytest.importorskip("langgraph")

from reglens.agents.graph import build_graph  # noqa: E402
from reglens.retrieve import Passage  # noqa: E402


def P(i, text, score):
    meta = {
        "doc_id": f"d{i}",
        "title": f"C{i}",
        "date": "2026-10-07",
        "page_start": 1,
        "page_end": 1,
        "url": "https://x",
    }
    return Passage(id=f"p{i}", text=text, meta=meta, score=score)


class FakeRetriever:
    def __init__(self, passages):
        self.passages, self.calls = passages, 0

    def search(self, query, k):
        self.calls += 1
        return self.passages[:k]


class PassThroughReranker:
    def rerank(self, query, passages, k):
        return sorted(passages, key=lambda p: p.score, reverse=True)[:k]


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt):
        self.calls += 1
        return json.dumps({"claims": [{"text": "Answer.", "sources": ["S1"]}]})


def test_relevant_question_flows_to_analyst():
    llm = CountingLLM()
    g = build_graph(FakeRetriever([P(1, "a", 0.9), P(2, "b", 0.05)]), PassThroughReranker(), llm)
    state = g.invoke({"question": "q"})
    assert llm.calls == 1
    assert [p.id for p in state["passages"]] == ["p1"]  # low-relevance passage filtered
    assert not state["answer"].refused


def test_irrelevant_question_refuses_without_llm_call():
    llm = CountingLLM()
    g = build_graph(FakeRetriever([P(1, "a", 0.02)]), PassThroughReranker(), llm)
    state = g.invoke({"question": "Who won the cricket world cup?"})
    assert state["answer"].refused and llm.calls == 0  # saves cost, no hallucination risk
