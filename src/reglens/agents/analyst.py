"""Analyst agent: drafts an answer where every claim must cite a retrieved passage.

The LLM proposes citations; this module *verifies* them in code. A claim citing a
source that was never retrieved is dropped. No surviving claims means refusal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..retrieve import Passage
from .llm import LLM

REFUSAL = "Insufficient evidence in the indexed SEBI circulars to answer this question."

PROMPT = """You are a SEBI compliance research analyst.
Answer the QUESTION using ONLY the numbered SOURCES. Rules:
1. Every claim must cite one or more source labels, e.g. ["S1"].
2. Never use outside knowledge. If the sources do not answer the question,
   set "insufficient_evidence" to true and return no claims.
3. Quote figures, dates and timelines exactly as written in the sources.
4. Keep the answer short and factual.

Return ONLY a JSON object:
{{"insufficient_evidence": false,
  "claims": [{{"text": "<one factual sentence>", "sources": ["S1"]}}]}}

QUESTION: {question}

SOURCES:
{sources}
"""


@dataclass
class Claim:
    text: str
    sources: list[str]


@dataclass
class Answer:
    question: str
    claims: list[Claim] = field(default_factory=list)
    citations: dict[str, str] = field(default_factory=dict)  # label -> human citation
    urls: dict[str, str] = field(default_factory=dict)
    refused: bool = False
    reason: str = ""
    dropped_claims: int = 0  # claims removed because their citations were invalid

    @property
    def text(self) -> str:
        if self.refused:
            return REFUSAL
        return " ".join(f"{c.text} [{', '.join(c.sources)}]" for c in self.claims)

    def to_dict(self) -> dict:
        return {**asdict(self), "text": self.text}

    def to_markdown(self) -> str:
        out = [f"**Q:** {self.question}", "", f"**A:** {self.text}"]
        if not self.refused and self.citations:
            out += ["", "**Sources**"]
            out += [
                f"- [{lbl}] [{c}]({self.urls.get(lbl, '')})" for lbl, c in self.citations.items()
            ]
        if self.reason:
            out += ["", f"_{self.reason}_"]
        return "\n".join(out)


def label_sources(passages: list[Passage]) -> dict[str, Passage]:
    return {f"S{i}": p for i, p in enumerate(passages, start=1)}


def build_prompt(question: str, labeled: dict[str, Passage]) -> str:
    sources = "\n\n".join(f"[{lbl}] {p.citation}\n{p.text}" for lbl, p in labeled.items())
    return PROMPT.format(question=question, sources=sources)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):  # tolerate fenced output
        raw = raw.strip("`").removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(raw[start : end + 1])


def validate(question: str, raw: str, labeled: dict[str, Passage]) -> Answer:
    ans = Answer(question=question)
    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        ans.refused, ans.reason = True, "Model returned malformed output; refused safely."
        return ans

    if data.get("insufficient_evidence"):
        ans.refused, ans.reason = True, "Analyst found no supporting evidence."
        return ans

    for item in data.get("claims") or []:
        text = str(item.get("text", "")).strip()
        cited = [s for s in item.get("sources") or [] if s in labeled]
        if text and cited:
            ans.claims.append(Claim(text=text, sources=cited))
        else:
            ans.dropped_claims += 1

    if not ans.claims:
        ans.refused, ans.reason = True, "No claim had a valid citation."
        return ans

    used = sorted({s for c in ans.claims for s in c.sources}, key=lambda s: int(s[1:]))
    ans.citations = {s: labeled[s].citation for s in used}
    ans.urls = {s: labeled[s].meta.get("url", "") for s in used}
    if ans.dropped_claims:
        ans.reason = f"{ans.dropped_claims} uncited claim(s) removed by citation check."
    return ans


def analyze(question: str, passages: list[Passage], llm: LLM) -> Answer:
    labeled = label_sources(passages)
    return validate(question, llm.complete_json(build_prompt(question, labeled)), labeled)
