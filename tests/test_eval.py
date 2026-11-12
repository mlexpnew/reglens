import csv
from pathlib import Path

from reglens.agents.analyst import Answer, Claim
from reglens.eval import (
    CountingLLM,
    Question,
    _fmt,
    details_markdown,
    evaluate,
    fact_hits,
    load_questions,
    normalize,
    percentile,
    report_markdown,
    retrieval_rows,
    score,
    summarize,
    summarize_retrieval,
)
from reglens.retrieve import Passage

ROOT = Path(__file__).resolve().parents[1]


def P(i, doc, text, score=0.9):
    meta = {"doc_id": doc, "title": doc, "date": "2026-10-07", "page_start": 1, "page_end": 1}
    return Passage(id=f"p{i}", text=text, meta=meta, score=score)


Q_LIMIT = Question(
    id="q1",
    category="single",
    question="What is the broad commodity limit?",
    gold_docs=("cpl",),
    facts=(("2%",), ("deliverable supply",)),
)
Q_OFF = Question(id="q2", category="off_topic", question="Who won the cup?")


def answered_state(text, sources, passages):
    ans = Answer(question="q", claims=[Claim(text=text, sources=sources)], verified=True)
    return {"answer": ans, "passages": passages}


def refused_state():
    return {"answer": Answer(question="q", refused=True, reason="no evidence"), "passages": []}


def test_normalize_handles_indian_number_format_and_case():
    assert normalize("Rs.2,00,000  or  LOWER") == "rs.200000 or lower"
    assert normalize("31’s – x") == "31's - x"


def test_fact_hits_accepts_any_listed_phrasing():
    facts = (("seventeen", "17"), ("aa-",), ("march 31, 2027",))
    assert fact_hits("A maximum of 17 ISINs, rated AA- or above.", facts) == 2


def test_percentile_is_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 95) == 5
    assert percentile([], 50) is None


def test_score_answered_row_tracks_citations_facts_and_grounding():
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply"), P(2, "other", "nothing")]
    state = answered_state("The limit is 2% of the deliverable supply.", ["S1"], passages)
    row = score(Q_LIMIT, state, latency_s=1.234, llm_calls=3, llm_chars=400)
    assert row["cited_docs"] == ["cpl"] and row["fact_hits"] == 2
    assert row["ungrounded_claims"] == 0 and row["approx_tokens"] == 100
    assert row["latency_s"] == 1.23 and not row["refused"]


def test_score_flags_number_missing_from_cited_source():
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    row = score(Q_LIMIT, answered_state("The limit is 5%.", ["S1"], passages), 1.0)
    assert row["ungrounded_claims"] == 1 and row["fact_hits"] == 0


def test_score_refusal_and_error_rows():
    assert score(Q_LIMIT, refused_state(), 1.0)["fact_hits"] == 0
    err = score(Q_LIMIT, None, 1.0, error="RuntimeError: 429")
    assert err["error"] and "refused" not in err


def test_summarize_metrics():
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply"), P(2, "x", "Other 9%")]
    rows = [
        score(Q_LIMIT, answered_state("2% of the deliverable supply.", ["S1"], passages), 2.0),
        score(Q_LIMIT, answered_state("It is 2%.", ["S1", "S2"], passages), 4.0),
        score(Q_LIMIT, refused_state(), 1.0),
        score(Q_OFF, refused_state(), 0.5),
        score(Q_OFF, answered_state("Someone won.", ["S2"], passages), 3.0),
        score(Q_LIMIT, None, 9.0, error="boom"),
    ]
    s = summarize(rows)
    assert s["questions"] == 6 and s["errors"] == 1
    assert s["answer_rate"] == round(2 / 3, 4)
    assert s["fact_recall"] == round((1 + 0.5 + 0) / 3, 4)
    assert s["fully_correct"] == round(1 / 3, 4)
    assert s["citation_precision"] == 0.75  # (1/1 + 1/2) / 2
    assert s["citation_recall"] == 1.0
    assert s["refusal_accuracy"] == 0.5
    assert s["latency_p50_s"] == 2.0


class FakeGraph:
    def __init__(self, llm, states):
        self.llm, self.states = llm, list(states)

    def invoke(self, inp):
        self.llm.complete_json("prompt")
        nxt = self.states.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class EchoLLM:
    def complete_json(self, prompt):
        return "{}"


def test_evaluate_counts_calls_and_survives_errors():
    llm = CountingLLM(EchoLLM())
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    graph = FakeGraph(llm, [answered_state("2%.", ["S1"], passages), RuntimeError("429")])
    ticks = iter([0.0, 1.5, 10.0, 10.25])
    rows = evaluate(
        [Q_LIMIT, Q_OFF],
        graph,
        llm,
        retry_cooldown_s=-1,
        clock=lambda: next(ticks),
        log=lambda m: None,
    )
    assert [r["latency_s"] for r in rows] == [1.5, 0.25]
    assert rows[0]["llm_calls"] == 1 and rows[0]["approx_tokens"] == (6 + 2) // 4
    assert rows[1]["error"].startswith("RuntimeError")


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, k):
        return self.passages[:k]


class ByScoreReranker:
    def rerank(self, query, passages, k):
        return sorted(passages, key=lambda p: p.score, reverse=True)[:k]


def test_retrieval_recall_and_gate():
    passages = [P(1, "a", "x", 0.4), P(2, "cpl", "y", 0.9), P(3, "b", "z", 0.05)]
    rows = retrieval_rows([Q_LIMIT], FakeRetriever(passages), ByScoreReranker(), 12, 2)
    rows += retrieval_rows([Q_OFF], FakeRetriever([P(4, "a", "x", 0.02)]), ByScoreReranker(), 12, 2)
    s = summarize_retrieval(rows, min_relevance=0.10)
    assert s["recall_at_1"] == 1.0 and s["recall_at_2"] == 1.0
    assert s["gate_pass_answerable"] == 1.0 and s["gate_block_negative"] == 1.0


def test_report_lists_every_config():
    md = report_markdown(
        {"full": summarize([]), "no_checker": summarize([])}, {"recall_at_1": 0.5}, 0, {}
    )
    assert "| Metric | full | no_checker |" in md and "| recall_at_1 | 50% |" in md


def test_gold_question_set_is_consistent():
    questions = load_questions(ROOT / "data/eval/questions.jsonl")
    with open(ROOT / "data/sources.csv", newline="", encoding="utf-8") as f:
        known = {row["doc_id"] for row in csv.DictReader(f)}
    assert len(questions) >= 30
    for q in questions:
        assert q.category in {"single", "multi", "unanswerable", "off_topic"}
        if q.answerable:
            assert q.gold_docs and q.facts, q.id
            assert set(q.gold_docs) <= known, q.id
        else:
            assert not q.gold_docs and not q.facts, q.id
    assert sum(not q.answerable for q in questions) >= 5


def test_evaluate_with_real_graph_and_ablation():
    import json

    import pytest

    pytest.importorskip("langgraph")
    from reglens.agents.graph import build_graph

    class Scripted:
        def complete_json(self, prompt):
            if "PLANNER" in prompt:
                return json.dumps({"sub_queries": []})
            if "CHECKER" in prompt:
                return json.dumps({"verdicts": [{"claim": 1, "supported": True}]})
            return json.dumps(
                {"claims": [{"text": "2% of the deliverable supply.", "sources": ["S1"]}]}
            )

    passages = [P(1, "cpl", "Broad 2% of the deliverable supply", 0.9)]
    calls = {}
    for name, checker in (("full", True), ("no_checker", False)):
        llm = CountingLLM(Scripted())
        graph = build_graph(FakeRetriever(passages), ByScoreReranker(), llm, use_checker=checker)
        (row,) = evaluate([Q_LIMIT], graph, llm, log=lambda m: None)
        assert row["fact_hits"] == 2 and row["cited_docs"] == ["cpl"]
        calls[name] = row["llm_calls"]
    assert calls == {"full": 3, "no_checker": 2}  # planner + analyst (+ checker)


def test_failed_questions_are_retried_once_after_cooldown():
    llm = CountingLLM(EchoLLM())
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    graph = FakeGraph(llm, [RuntimeError("429"), answered_state("2%.", ["S1"], passages)])
    waits = []
    (row,) = evaluate([Q_LIMIT], graph, llm, sleep=waits.append, log=lambda m: None)
    assert waits == [60.0]
    assert row["error"] is None and row["retried"] and row["fact_hits"] == 1


def test_formatting_and_per_question_details():
    assert _fmt(2.6667) == "2.7" and _fmt(1444.33) == "1,444" and _fmt(3) == "3"
    assert _fmt(0.5, pct=True) == "50%" and _fmt(None) == "n/a"
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    rows = [
        score(Q_LIMIT, answered_state("2%.", ["S1"], passages), 1.0),
        score(Q_OFF, refused_state(), 0.5),
        score(Q_LIMIT, None, 2.0, error="RuntimeError: 429"),
    ]
    md = details_markdown(rows)
    assert "| q1 | single | verified | 1/2 | cpl | 1.0 |" in md
    assert "refused (no evidence)" in md and "error: RuntimeError: 429" in md


Q3 = Question(id="q3", category="single", question="Third?", gold_docs=("cpl",), facts=(("2%",),))


def test_quota_exhaustion_skips_the_rest_without_hammering_the_provider():
    from reglens.agents.llm import QuotaExhausted

    llm = CountingLLM(EchoLLM())
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    graph = FakeGraph(
        llm, [answered_state("2%.", ["S1"], passages), QuotaExhausted("all providers exhausted")]
    )
    waits = []
    rows = evaluate([Q_LIMIT, Q_OFF, Q3], graph, llm, sleep=waits.append, log=lambda m: None)
    assert rows[0]["error"] is None
    assert rows[1]["error"].startswith("QuotaExhausted")
    assert rows[2]["error"].startswith("skipped: ") and rows[2]["llm_calls"] == 0
    assert waits == []  # no cooldown retry: waiting cannot refill a daily quota
    s = summarize(rows)
    assert s["completed"] == 1 and s["coverage"] == round(1 / 3, 4)
    assert s["errors"] == 1 and s["skipped"] == 1


def test_resume_reuses_checkpoint_and_runs_only_missing(tmp_path):
    passages = [P(1, "cpl", "Broad 2% of the deliverable supply")]
    ckpt = tmp_path / "results_full.jsonl"

    llm = CountingLLM(EchoLLM())
    first = FakeGraph(llm, [answered_state("2%.", ["S1"], passages), RuntimeError("429")])
    evaluate([Q_LIMIT, Q3], first, llm, retry_cooldown_s=-1, checkpoint=ckpt, log=lambda m: None)
    assert ckpt.exists()

    llm2 = CountingLLM(EchoLLM())
    second = FakeGraph(llm2, [answered_state("2%.", ["S1"], passages)])
    rows = evaluate([Q_LIMIT, Q3], second, llm2, checkpoint=ckpt, resume=True, log=lambda m: None)
    assert [r["error"] for r in rows] == [None, None]
    assert second.states == []  # exactly one new graph call: only q3 re-ran
    assert len(ckpt.read_text().splitlines()) == 2


def test_report_flags_incomplete_runs_and_shows_coverage():
    rows = [score(Q_LIMIT, None, 0.0, error="skipped: quota"), score(Q_OFF, refused_state(), 1.0)]
    md = report_markdown(
        {"full": summarize(rows)}, None, 2, {}, min_coverage=0.9, llm_name="groq:x"
    )
    assert "Incomplete run" in md and "| **Coverage (questions run)** | 1/2 |" in md
    assert "groq:x" in md
