"""Two-stage retrieval: wide vector recall, then precise cross-encoder reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    meta: dict
    score: float = 0.0  # vector similarity, then rerank relevance (0..1)

    @property
    def citation(self) -> str:
        m = self.meta
        pages = (
            f"p.{m['page_start']}"
            if m.get("page_start") == m.get("page_end")
            else f"pp.{m.get('page_start')}-{m.get('page_end')}"
        )
        return f"{m.get('title', m.get('doc_id', '?'))} ({m.get('date', '')}), {pages}"


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Passage]: ...


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[Passage], k: int) -> list[Passage]: ...


class ChromaRetriever:
    def __init__(self, store) -> None:
        self._store = store

    def search(self, query: str, k: int) -> list[Passage]:
        res = self._store.query(query, k=k)
        return [
            Passage(id=i, text=d, meta=m, score=round(1 - dist, 4))
            for i, d, m, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class CrossEncoderReranker:
    """Hugging Face cross-encoder: reads query and passage together, far more precise
    than comparing two independent vectors."""

    def __init__(self, model_name: str, model=None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder  # lazy: pulls torch

            model = CrossEncoder(model_name)
        self._model = model

    def rerank(self, query: str, passages: list[Passage], k: int) -> list[Passage]:
        if not passages:
            return []
        logits = self._model.predict([(query, p.text) for p in passages])
        scored = [
            replace(p, score=round(_sigmoid(float(s)), 4))
            for p, s in zip(passages, logits, strict=True)
        ]
        return sorted(scored, key=lambda p: p.score, reverse=True)[:k]
