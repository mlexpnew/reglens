"""Planner agent: splits multi-part questions into focused search queries.

It runs only after the question has passed the relevance gate, so off-topic questions
still cost zero LLM calls. Any planner failure falls back to "no extra queries": the
planner can widen recall, never block an answer.
"""

from __future__ import annotations

import json
import logging

from .analyst import extract_json
from .llm import LLM

log = logging.getLogger(__name__)

PROMPT = """You are the PLANNER for a search engine over SEBI regulatory circulars.
If the QUESTION asks about several distinct things, split it into at most {n} short,
standalone search queries, one per thing. If it asks about one thing, return an
empty list. Do not answer the question.

Return ONLY a JSON object: {{"sub_queries": ["<query>", ...]}}

QUESTION: {question}
"""


def plan(question: str, llm: LLM, max_queries: int = 3) -> list[str]:
    try:
        data = extract_json(llm.complete_json(PROMPT.format(question=question, n=max_queries)))
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("planner output unusable (%s); using the original question only", exc)
        return []
    queries: list[str] = []
    seen = {question.strip().casefold()}
    for q in data.get("sub_queries") or []:
        q = str(q).strip()
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            queries.append(q)
    return queries[:max_queries]
