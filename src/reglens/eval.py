"""Evaluation harness: run RegLens on a labeled question set and score every answer.

Scoring is deterministic (no LLM judge), so results are reproducible and free to compute:

- retrieval recall@k: share of gold circulars found in the top-k reranked passages
- answer rate: answerable questions that got an answer instead of a refusal
- fact recall: share of expected facts (dates, limits, terms) present in the answer
- citation precision / recall: cited circulars versus the gold circulars
- refusal accuracy: unanswerable and off-topic questions that were refused
- ungrounded claim rate: claims containing a number that is absent from their cited pages
- latency p50 / p95, LLM calls and approximate tokens per question

Each agent configuration (full pipeline, no Checker, ...) runs separately, so the report
shows what every agent adds (ablation).

    python -m reglens.eval --configs full,no_checker
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ANSWERABLE = {"single", "multi"}

CONFIGS: dict[str, dict[str, bool]] = {
    "full": {"use_planner": True, "use_checker": True},
    "no_checker": {"use_planner": True, "use_checker": False},
    "no_planner": {"use_planner": False, "use_checker": True},
    "baseline": {"use_planner": False, "use_checker": False},
}


@dataclass(frozen=True)
class Question:
    id: str
    category: str  # single | multi | unanswerable | off_topic
    question: str
    gold_docs: tuple[str, ...] = ()
    facts: tuple[tuple[str, ...], ...] = ()  # each fact: acceptable phrasings

    @property
    def answerable(self) -> bool:
        return self.category in ANSWERABLE


def load_questions(path: Path | str) -> list[Question]:
    questions: list[Question] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        questions.append(
            Question(
                id=d["id"],
                category=d["category"],
                question=d["question"],
                gold_docs=tuple(d.get("gold_docs") or ()),
                facts=tuple(tuple(f) for f in d.get("facts") or ()),
            )
        )
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    return questions


def normalize(text: str) -> str:
    """Lowercase, unify quotes and dashes, drop thousands separators, squeeze spaces."""
    text = text.lower()
    for a, b in (("’", "'"), ("‘", "'"), ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 2,00,000 -> 200000
    return re.sub(r"\s+", " ", text).strip()


def fact_hits(answer_text: str, facts: tuple[tuple[str, ...], ...]) -> int:
    t = normalize(answer_text)
    return sum(any(normalize(alt) in t for alt in alts) for alts in facts)


class CountingLLM:
    """Wraps an LLM to count calls and characters per question."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.chars = 0

    def complete_json(self, prompt: str) -> str:
        self.calls += 1
        out = self.inner.complete_json(prompt) or ""
        self.chars += len(prompt) + len(out)
        return out


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: list[float]) -> float | None:
    return _ratio(sum(values), len(values))


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile: always one of the observed values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without math import
    return ordered[int(rank) - 1]


def score(
    q: Question,
    state: dict | None,
    latency_s: float,
    llm_calls: int = 0,
    llm_chars: int = 0,
    error: str | None = None,
) -> dict:
    """One result row. Refused or failed answers score zero facts and zero citations."""
    from .agents.analyst import label_sources
    from .agents.checker import ungrounded_numbers

    row = {
        "id": q.id,
        "category": q.category,
        "question": q.question,
        "gold_docs": list(q.gold_docs),
        "facts": len(q.facts),
        "latency_s": round(latency_s, 2),
        "llm_calls": llm_calls,
        "approx_tokens": llm_chars // 4,
        "error": error,
    }
    if error or state is None:
        return row

    ans = state["answer"]
    labeled = label_sources(state.get("passages") or [])
    claims = [] if ans.refused else ans.claims
    cited = sorted(
        {labeled[s].meta.get("doc_id", "") for c in claims for s in c.sources if s in labeled}
    )
    ungrounded = sum(
        1
        for c in claims
        if ungrounded_numbers(c.text, [labeled[s] for s in c.sources if s in labeled])
    )
    row.update(
        refused=ans.refused,
        verified=ans.verified,
        answer=ans.text,
        reason=ans.reason,
        cited_docs=cited,
        claims=len(claims),
        ungrounded_claims=ungrounded,
        fact_hits=0 if ans.refused else fact_hits(ans.text, q.facts),
        revisions=ans.revisions,
        removed_by_checker=ans.removed_by_checker,
        sub_queries=list(state.get("sub_queries") or []),
    )
    return row


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    answerable = [r for r in ok if r["category"] in ANSWERABLE]
    negative = [r for r in ok if r["category"] not in ANSWERABLE]
    answered = [r for r in answerable if not r["refused"]]
    with_facts = [r for r in answerable if r["facts"]]
    claims = sum(r["claims"] for r in ok)
    latencies = [r["latency_s"] for r in ok]

    def overlap(r: dict) -> int:
        return len(set(r["cited_docs"]) & set(r["gold_docs"]))

    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "answer_rate": _mean([not r["refused"] for r in answerable]),
        "fact_recall": _mean([r["fact_hits"] / r["facts"] for r in with_facts]),
        "fully_correct": _mean([r["fact_hits"] == r["facts"] for r in with_facts]),
        "citation_precision": _mean(
            [overlap(r) / len(r["cited_docs"]) for r in answered if r["cited_docs"]]
        ),
        "citation_recall": _mean([overlap(r) / len(r["gold_docs"]) for r in answered]),
        "refusal_accuracy": _mean([r["refused"] for r in negative]),
        "ungrounded_claim_rate": _ratio(sum(r["ungrounded_claims"] for r in ok), claims),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "llm_calls_mean": _mean([r["llm_calls"] for r in ok]),
        "approx_tokens_mean": _mean([r["approx_tokens"] for r in ok]),
    }

# Release gate: the Eval workflow fails when the gated config misses any threshold.
# Set each limit a little below your first real score, then raise it as RegLens improves.
GATES: dict[str, tuple[str, float]] = {
    "fact_recall": ("min", 0.70),
    "citation_precision": ("min", 0.80),
    "refusal_accuracy": ("min", 0.80),
    "ungrounded_claim_rate": ("max", 0.05),
}
MAX_ERROR_SHARE = 0.10  # more failed questions than this means the run itself is not trusted


def check_gates(summary: dict, gates: dict[str, tuple[str, float]] = GATES) -> list[str]:
    """Return one message per broken threshold; an empty list means the gate passed."""
    failures = []
    for key, (kind, limit) in gates.items():
        value = summary.get(key)
        if value is None:
            failures.append(f"{key}: no value")
        elif (kind == "min" and value < limit) or (kind == "max" and value > limit):
            failures.append(f"{key}: {value:.2f} ({kind} {limit:.2f})")
    if summary.get("questions") and summary["errors"] / summary["questions"] > MAX_ERROR_SHARE:
        failures.append(f"errors: {summary['errors']} of {summary['questions']} questions")
    return failures

def retrieval_rows(
    questions: list[Question], retriever, reranker, retrieve_k: int, rerank_k: int
) -> list[dict]:
    """Retrieval and reranking only: no LLM, so it is free to run on every change."""
    rows = []
    for q in questions:
        ranked = reranker.rerank(q.question, retriever.search(q.question, retrieve_k), rerank_k)
        docs = [p.meta.get("doc_id", "") for p in ranked]
        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "gold_docs": list(q.gold_docs),
                "ranked_docs": docs,
                "relevance": [p.score for p in ranked],
            }
        )
    return rows


def summarize_retrieval(rows: list[dict], min_relevance: float) -> dict:
    answerable = [r for r in rows if r["category"] in ANSWERABLE]
    negative = [r for r in rows if r["category"] not in ANSWERABLE]

    def recall(r: dict, k: int) -> float:
        return len(set(r["ranked_docs"][:k]) & set(r["gold_docs"])) / len(r["gold_docs"])

    def passes(r: dict) -> bool:
        return any(s >= min_relevance for s in r["relevance"])

    k = max((len(r["ranked_docs"]) for r in rows), default=0)
    return {
        "recall_at_1": _mean([recall(r, 1) for r in answerable if r["gold_docs"]]),
        f"recall_at_{k}": _mean([recall(r, k) for r in answerable if r["gold_docs"]]),
        "gate_pass_answerable": _mean([passes(r) for r in answerable]),
        "gate_block_negative": _mean([not passes(r) for r in negative]),
    }


def evaluate(
    questions: list[Question],
    graph,
    llm: CountingLLM,
    sleep_s: float = 0.0,
    retry_cooldown_s: float = 60.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> list[dict]:
    """Run one configuration. A failing question is recorded, never fatal.

    Questions that fail (typically a provider rate limit) are retried once after a
    cooldown, so a burst of 429s does not count as a wrong answer.
    """

    def run(q: Question) -> dict:
        llm.reset()
        t0 = clock()
        state, error = None, None
        try:
            state = graph.invoke({"question": q.question})
        except Exception as exc:  # record the failure and keep going
            error = f"{type(exc).__name__}: {exc}"[:300]
        return score(q, state, clock() - t0, llm.calls, llm.chars, error)

    def status(row: dict) -> str:
        return "ERROR" if row["error"] else ("refused" if row["refused"] else "answered")

    rows = []
    for i, q in enumerate(questions, start=1):
        rows.append(run(q))
        log(f"[{i}/{len(questions)}] {q.id}: {status(rows[-1])} in {rows[-1]['latency_s']}s")
        if sleep_s and i < len(questions):
            sleep(sleep_s)

    failed = [i for i, r in enumerate(rows) if r["error"]]
    if failed and retry_cooldown_s >= 0:
        log(f"retrying {len(failed)} failed question(s) after {retry_cooldown_s:g}s")
        sleep(retry_cooldown_s)
        for i in failed:
            rows[i] = run(questions[i])
            rows[i]["retried"] = True
            log(f"[retry] {questions[i].id}: {status(rows[i])} in {rows[i]['latency_s']}s")
            if sleep_s:
                sleep(sleep_s)
    return rows


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:.0f}%"
    if isinstance(value, int):
        return str(value)
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def details_markdown(rows: list[dict]) -> str:
    """Per-question outcome table, so every score can be traced to an answer."""
    out = [
        "| Question | Category | Outcome | Facts | Cited circulars | Latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            outcome, facts, cited = f"error: {r['error'][:60]}", "", ""
        else:
            outcome = "refused" if r["refused"] else ("verified" if r["verified"] else "answered")
            if r["refused"] and r.get("reason"):
                outcome += f" ({r['reason'][:60]})"
            facts = f"{r['fact_hits']}/{r['facts']}" if r["facts"] else "-"
            cited = ", ".join(r["cited_docs"]) or "-"
        out.append(
            f"| {r['id']} | {r['category']} | {outcome} | {facts} | {cited} | {r['latency_s']} |"
        )
    return "\n".join(out) + "\n"


METRIC_ROWS = [
    ("Answer rate (answerable)", "answer_rate", True),
    ("Fact recall", "fact_recall", True),
    ("Fully correct answers", "fully_correct", True),
    ("Citation precision", "citation_precision", True),
    ("Citation recall", "citation_recall", True),
    ("Refusal accuracy (unanswerable + off-topic)", "refusal_accuracy", True),
    ("Claims with ungrounded numbers", "ungrounded_claim_rate", True),
    ("Latency p50 (s)", "latency_p50_s", False),
    ("Latency p95 (s)", "latency_p95_s", False),
    ("LLM calls per question", "llm_calls_mean", False),
    ("Approx. tokens per question", "approx_tokens_mean", False),
    ("Errors", "errors", False),
]


def report_markdown(
    summaries: dict[str, dict],
    retrieval: dict | None,
    n_questions: int,
    counts: dict,
    rows_by_config: dict[str, list[dict]] | None = None,
) -> str:
    mix = ", ".join(f"{v} {k}" for k, v in counts.items())
    out = [f"### RegLens evaluation ({n_questions} questions: {mix})", ""]
    if retrieval:
        out += ["**Retrieval** (no LLM)", "", "| Metric | Score |", "|---|---|"]
        out += [f"| {k} | {_fmt(v, pct=True)} |" for k, v in retrieval.items()]
        out.append("")
    if summaries:
        names = list(summaries)
        out += ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for label, key, pct in METRIC_ROWS:
            cells = " | ".join(_fmt(summaries[n][key], pct) for n in names)
            out.append(f"| {label} | {cells} |")
    for name, rows in (rows_by_config or {}).items():
        out += ["", f"<details><summary>Per-question results: {name}</summary>", ""]
        out += [details_markdown(rows), "</details>"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate RegLens on a labeled question set")
    p.add_argument("--questions", default="data/eval/questions.jsonl")
    p.add_argument("--configs", default="full,no_checker", help=f"from: {', '.join(CONFIGS)}")
    p.add_argument("--out", default="eval_results")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")
    p.add_argument(
        "--retry-cooldown",
        type=float,
        default=60.0,
        help="seconds to wait before retrying failed questions once (-1 disables)",
    )
    p.add_argument("--retrieval-only", action="store_true", help="skip all LLM calls")
    args = p.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        p.error(f"unknown config(s): {', '.join(unknown)}")

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    from .config import settings
    from .ingest.store import ChromaStore
    from .retrieve import ChromaRetriever, CrossEncoderReranker

    store = ChromaStore(settings.chroma_dir, settings.collection, settings.embed_model)
    if store.count() == 0:
        print("Index is empty: run `make ingest` first.", file=sys.stderr)
        return 2
    retriever = ChromaRetriever(store)
    reranker = CrossEncoderReranker(settings.rerank_model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r_rows = retrieval_rows(questions, retriever, reranker, settings.retrieve_k, settings.rerank_k)
    retrieval = summarize_retrieval(r_rows, settings.min_relevance)
    (out / "retrieval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in r_rows))

    summaries: dict[str, dict] = {}
    rows_by_config: dict[str, list[dict]] = {}
    if not args.retrieval_only:
        from .agents.graph import build_graph
        from .agents.llm import GeminiLLM

        llm = CountingLLM(GeminiLLM(settings.llm_model))
        for name in configs:
            graph = build_graph(
                retriever,
                reranker,
                llm,
                retrieve_k=settings.retrieve_k,
                rerank_k=settings.rerank_k,
                min_relevance=settings.min_relevance,
                max_sub_queries=settings.max_sub_queries,
                max_revisions=settings.max_revisions,
                **CONFIGS[name],
            )
            print(f"== config: {name}", file=sys.stderr)
            rows = evaluate(
                questions, graph, llm, sleep_s=args.sleep, retry_cooldown_s=args.retry_cooldown
            )
            rows_by_config[name] = rows
            (out / f"results_{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            )
            summaries[name] = summarize(rows)

    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    summary = {"retrieval": retrieval, "configs": summaries, "llm_model": settings.llm_model}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = report_markdown(summaries, retrieval, len(questions), counts, rows_by_config)
    (out / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
