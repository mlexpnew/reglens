"""Compliance Checker: verifies every claim against the exact sources it cites.

Two layers:
1. Numeric grounding (code, free): every number in a claim must appear in its cited
   sources. Hallucinated figures are the costliest compliance error and the easiest
   to catch deterministically.
2. Entailment (LLM): a strict reviewer judges whether each claim is supported.

The Checker fails closed: a claim with no verdict, or unreadable reviewer output,
counts as unsupported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..retrieve import Passage
from .analyst import Answer, extract_json
from .llm import LLM

_NUM = re.compile(r"\d+(?:[.,]\d+)*")

PROMPT = """You are a strict SEBI compliance reviewer (CHECKER).
For each CLAIM decide whether it is fully supported by the text of the sources it cites.
A claim is UNSUPPORTED if it adds facts, numbers, dates, entities or conditions that the
cited text does not state, or if it contradicts that text. Faithful paraphrase is fine.

Return ONLY a JSON object:
{{"verdicts": [{{"claim": 1, "supported": true, "reason": "<short reason>"}}]}}

{claims}
"""


@dataclass(frozen=True)
class Rejection:
    index: int  # 0-based position in answer.claims
    reason: str


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def ungrounded_numbers(claim_text: str, cited: list[Passage]) -> set[str]:
    source_numbers: set[str] = set()
    for p in cited:
        source_numbers |= _numbers(p.text)
    return _numbers(claim_text) - source_numbers


def build_prompt(answer: Answer, labeled: dict[str, Passage]) -> str:
    blocks = []
    for i, claim in enumerate(answer.claims, start=1):
        sources = "\n".join(f"[{s}] {labeled[s].text}" for s in claim.sources)
        blocks.append(f"CLAIM {i}: {claim.text}\nCITED SOURCES:\n{sources}")
    return PROMPT.format(claims="\n\n".join(blocks))


def check(answer: Answer, labeled: dict[str, Passage], llm: LLM) -> list[Rejection]:
    """Return the claims that failed verification (empty list = all verified)."""
    rejected: dict[int, str] = {}

    for i, claim in enumerate(answer.claims):
        missing = ungrounded_numbers(claim.text, [labeled[s] for s in claim.sources])
        if missing:
            rejected[i] = f"numbers not found in cited sources: {', '.join(sorted(missing))}"

    try:
        data = extract_json(llm.complete_json(build_prompt(answer, labeled)))
        verdicts = {int(v["claim"]) - 1: v for v in data.get("verdicts") or [] if "claim" in v}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        verdicts = {}

    for i in range(len(answer.claims)):
        if i in rejected:
            continue
        v = verdicts.get(i)
        if v is None:
            rejected[i] = "no reviewer verdict (fail closed)"
        elif v.get("supported") is not True:
            rejected[i] = str(v.get("reason") or "not supported by cited sources")

    return [Rejection(i, r) for i, r in sorted(rejected.items())]


def feedback_text(answer: Answer, rejections: list[Rejection]) -> str:
    return "\n".join(f'- "{answer.claims[r.index].text}" (reason: {r.reason})' for r in rejections)
