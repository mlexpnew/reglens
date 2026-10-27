"""LangGraph workflow.

    retrieve -> gate --(nothing relevant)--> refuse
                  \\-> plan --(sub-queries)--> expand --> analyze -> check --(verified)--> END
                         \\-------------------------------^    ^         |
                                                             \\-revise--/  (at most max_revisions)

Design choices:
- The relevance gate runs before any LLM call: off-topic questions cost nothing.
- The Planner only widens recall; each sub-query is reranked against itself so every
  part of a multi-part question gets its own evidence.
- The Checker sends rejected claims back to the Analyst once. Claims still unsupported
  after the last revision are removed; if none survive, the answer is refused.
"""

from __future__ import annotations

from typing import TypedDict

from ..retrieve import Passage, Reranker, Retriever
from .analyst import Answer, analyze, attach_citations, label_sources
from .checker import check, feedback_text
from .llm import LLM
from .planner import plan as plan_queries


class State(TypedDict, total=False):
    question: str
    candidates: list[Passage]
    passages: list[Passage]
    sub_queries: list[str]
    answer: Answer
    feedback: str | None
    revisions: int
    done: bool


def build_graph(
    retriever: Retriever,
    reranker: Reranker,
    llm: LLM,
    retrieve_k: int = 12,
    rerank_k: int = 4,
    min_relevance: float = 0.10,
    use_planner: bool = True,
    use_checker: bool = True,
    max_sub_queries: int = 3,
    per_query_k: int = 2,
    max_passages: int = 8,
    max_revisions: int = 1,
):
    from langgraph.graph import END, START, StateGraph  # lazy: optional dependency

    def relevant(query: str, passages: list[Passage], k: int) -> list[Passage]:
        return [p for p in reranker.rerank(query, passages, k) if p.score >= min_relevance]

    def retrieve(state: State) -> State:
        return {"candidates": retriever.search(state["question"], retrieve_k), "revisions": 0}

    def gate(state: State) -> State:
        return {"passages": relevant(state["question"], state["candidates"], rerank_k)}

    def route_gate(state: State) -> str:
        if not state["passages"]:
            return "refuse"
        return "plan" if use_planner else "analyze"

    def refuse(state: State) -> State:
        return {
            "answer": Answer(
                question=state["question"],
                refused=True,
                reason=f"No passage reached relevance {min_relevance:.2f}.",
            )
        }

    def planner(state: State) -> State:
        return {"sub_queries": plan_queries(state["question"], llm, max_sub_queries)}

    def route_plan(state: State) -> str:
        return "expand" if state.get("sub_queries") else "analyze"

    def expand(state: State) -> State:
        merged = {p.id: p for p in state["passages"]}
        for q in state["sub_queries"]:
            for p in relevant(q, retriever.search(q, retrieve_k), per_query_k):
                if p.id not in merged or p.score > merged[p.id].score:
                    merged[p.id] = p
        ranked = sorted(merged.values(), key=lambda p: p.score, reverse=True)
        return {"passages": ranked[:max_passages]}

    def analyst(state: State) -> State:
        ans = analyze(state["question"], state["passages"], llm, state.get("feedback"))
        ans.revisions = state.get("revisions", 0)
        return {"answer": ans, "feedback": None}

    def checker(state: State) -> State:
        ans = state["answer"]
        if ans.refused or not use_checker:
            return {"done": True}
        labeled = label_sources(state["passages"])
        rejections = check(ans, labeled, llm)
        if not rejections:
            ans.verified = True
            return {"answer": ans, "done": True}
        if state.get("revisions", 0) < max_revisions:
            return {
                "feedback": feedback_text(ans, rejections),
                "revisions": state.get("revisions", 0) + 1,
                "done": False,
            }
        # out of revisions: keep only the claims that passed
        bad = {r.index for r in rejections}
        ans.claims = [c for i, c in enumerate(ans.claims) if i not in bad]
        ans.removed_by_checker = len(bad)
        if not ans.claims:
            ans.refused = True
            ans.citations, ans.urls = {}, {}
            ans.reason = "Checker could not verify any claim against its sources."
        else:
            attach_citations(ans, labeled)
            ans.verified = True
            ans.reason = f"{len(bad)} unverified claim(s) removed by the Checker."
        return {"answer": ans, "done": True}

    def route_check(state: State) -> str:
        return "end" if state.get("done") else "analyze"

    g = StateGraph(State)
    for name, fn in [
        ("retrieve", retrieve),
        ("gate", gate),
        ("refuse", refuse),
        ("plan", planner),
        ("expand", expand),
        ("analyze", analyst),
        ("check", checker),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges(
        "gate", route_gate, {"refuse": "refuse", "plan": "plan", "analyze": "analyze"}
    )
    g.add_conditional_edges("plan", route_plan, {"expand": "expand", "analyze": "analyze"})
    g.add_edge("expand", "analyze")
    g.add_edge("analyze", "check")
    g.add_conditional_edges("check", route_check, {"analyze": "analyze", "end": END})
    g.add_edge("refuse", END)
    return g.compile()
