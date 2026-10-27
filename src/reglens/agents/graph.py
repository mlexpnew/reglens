"""LangGraph workflow: retrieve -> rerank -> (refuse | analyze).

The relevance gate runs *before* the LLM: if nothing relevant was retrieved we refuse
without spending a model call, and the model never sees irrelevant context.
"""

from __future__ import annotations

from typing import TypedDict

from ..retrieve import Passage, Reranker, Retriever
from .analyst import Answer, analyze
from .llm import LLM


class State(TypedDict, total=False):
    question: str
    candidates: list[Passage]
    passages: list[Passage]
    answer: Answer


def build_graph(
    retriever: Retriever,
    reranker: Reranker,
    llm: LLM,
    retrieve_k: int = 12,
    rerank_k: int = 4,
    min_relevance: float = 0.10,
):
    from langgraph.graph import END, START, StateGraph  # lazy: optional dependency

    def retrieve(state: State) -> State:
        return {"candidates": retriever.search(state["question"], retrieve_k)}

    def rerank(state: State) -> State:
        ranked = reranker.rerank(state["question"], state["candidates"], rerank_k)
        return {"passages": [p for p in ranked if p.score >= min_relevance]}

    def route(state: State) -> str:
        return "analyze" if state["passages"] else "refuse"

    def refuse(state: State) -> State:
        return {
            "answer": Answer(
                question=state["question"],
                refused=True,
                reason=f"No passage reached relevance {min_relevance:.2f}.",
            )
        }

    def analyst(state: State) -> State:
        return {"answer": analyze(state["question"], state["passages"], llm)}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("refuse", refuse)
    g.add_node("analyze", analyst)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", route, {"analyze": "analyze", "refuse": "refuse"})
    g.add_edge("analyze", END)
    g.add_edge("refuse", END)
    return g.compile()
