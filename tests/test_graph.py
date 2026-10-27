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
    """Returns passages whose text mentions a word of the query (or all, if none match)."""

    def __init__(self, passages):
        self.passages, self.queries = passages, []

    def search(self, query, k):
        self.queries.append(query)
        words = set(query.lower().split())
        hits = [p for p in self.passages if words & set(p.text.lower().split())]
        return (hits or self.passages)[:k]


class PassThroughReranker:
    def rerank(self, query, passages, k):
        return sorted(passages, key=lambda p: p.score, reverse=True)[:k]


class ScriptedLLM:
    """Routes each prompt to the agent that sent it and replays queued responses."""

    def __init__(self, planner=None, analyst=(), checker=()):
        self.planner = planner if planner is not None else {"sub_queries": []}
        self.analyst, self.checker = list(analyst), list(checker)
        self.calls = {"planner": 0, "analyst": 0, "checker": 0}
        self.analyst_prompts = []

    def complete_json(self, prompt):
        if "PLANNER" in prompt:
            self.calls["planner"] += 1
            return json.dumps(self.planner)
        if "CHECKER" in prompt:
            self.calls["checker"] += 1
            return json.dumps(self.checker.pop(0))
        self.calls["analyst"] += 1
        self.analyst_prompts.append(prompt)
        return json.dumps(self.analyst.pop(0))


def claims(*items):
    return {"claims": [{"text": t, "sources": s} for t, s in items]}


def verdicts(*supported):
    return {"verdicts": [{"claim": i, "supported": ok} for i, ok in enumerate(supported, 1)]}


DOCS = [
    P(1, "risk meter shows issuer credit risk from 2026", 0.9),
    P(2, "brokers display awareness messages", 0.8),
    P(3, "unrelated text", 0.05),
]


def test_irrelevant_question_refuses_with_zero_llm_calls():
    llm = ScriptedLLM()
    g = build_graph(FakeRetriever([P(1, "a", 0.02)]), PassThroughReranker(), llm)
    state = g.invoke({"question": "Who won the cricket world cup?"})
    assert state["answer"].refused
    assert llm.calls == {"planner": 0, "analyst": 0, "checker": 0}


def test_verified_answer_happy_path():
    llm = ScriptedLLM(
        analyst=[claims(("The meter shows issuer credit risk.", ["S1"]))],
        checker=[verdicts(True)],
    )
    state = build_graph(FakeRetriever(DOCS), PassThroughReranker(), llm).invoke(
        {"question": "what does the risk meter show"}
    )
    ans = state["answer"]
    assert not ans.refused and ans.verified and ans.revisions == 0
    assert "p3" not in [p.id for p in state["passages"]]  # below relevance gate
    assert llm.calls == {"planner": 1, "analyst": 1, "checker": 1}


def test_planner_sub_queries_each_get_evidence():
    retriever = FakeRetriever(DOCS)
    llm = ScriptedLLM(
        planner={"sub_queries": ["risk meter", "brokers display", "risk meter"]},
        analyst=[claims(("A.", ["S1"]), ("B.", ["S2"]))],
        checker=[verdicts(True, True)],
    )
    state = build_graph(retriever, PassThroughReranker(), llm).invoke(
        {"question": "risk meter and brokers display?"}
    )
    assert state["sub_queries"] == ["risk meter", "brokers display"]  # deduped
    assert retriever.queries[1:] == ["risk meter", "brokers display"]
    assert {"p1", "p2"} <= {p.id for p in state["passages"]}


def test_checker_rejection_triggers_one_revision_then_verifies():
    llm = ScriptedLLM(
        analyst=[
            claims(("Applies to everyone.", ["S1"])),
            claims(("The meter shows issuer credit risk.", ["S1"])),
        ],
        checker=[verdicts(False), verdicts(True)],
    )
    state = build_graph(FakeRetriever(DOCS), PassThroughReranker(), llm).invoke(
        {"question": "what does the risk meter show"}
    )
    ans = state["answer"]
    assert ans.verified and ans.revisions == 1
    assert "REJECTED" in llm.analyst_prompts[1] and "Applies to everyone." in llm.analyst_prompts[1]
    assert llm.calls["analyst"] == 2 and llm.calls["checker"] == 2


def test_unsupported_claims_removed_after_max_revisions():
    llm = ScriptedLLM(
        analyst=[claims(("Good.", ["S1"]), ("Bad.", ["S1"]))] * 2,
        checker=[verdicts(True, False), verdicts(True, False)],
    )
    ans = build_graph(FakeRetriever(DOCS), PassThroughReranker(), llm).invoke(
        {"question": "what does the risk meter show"}
    )["answer"]
    assert [c.text for c in ans.claims] == ["Good."]
    assert ans.removed_by_checker == 1 and ans.verified and not ans.refused


def test_nothing_verifiable_means_refusal():
    llm = ScriptedLLM(
        analyst=[claims(("Bad.", ["S1"]))] * 2,
        checker=[verdicts(False), verdicts(False)],
    )
    ans = build_graph(FakeRetriever(DOCS), PassThroughReranker(), llm).invoke(
        {"question": "what does the risk meter show"}
    )["answer"]
    assert ans.refused and ans.citations == {}


def test_hallucinated_number_caught_without_trusting_the_reviewer():
    # the reviewer says "supported", but 2031 appears nowhere in the source
    llm = ScriptedLLM(
        analyst=[claims(("Applies from 2031.", ["S1"])), claims(("Applies from 2026.", ["S1"]))],
        checker=[verdicts(True), verdicts(True)],
    )
    ans = build_graph(FakeRetriever(DOCS), PassThroughReranker(), llm).invoke(
        {"question": "what does the risk meter show"}
    )["answer"]
    assert ans.revisions == 1 and ans.claims[0].text == "Applies from 2026."


def test_agents_can_be_switched_off_for_ablation():
    llm = ScriptedLLM(analyst=[claims(("A.", ["S1"]))])
    state = build_graph(
        FakeRetriever(DOCS), PassThroughReranker(), llm, use_planner=False, use_checker=False
    ).invoke({"question": "what does the risk meter show"})
    assert not state["answer"].refused and not state["answer"].verified
    assert llm.calls == {"planner": 0, "analyst": 1, "checker": 0}
