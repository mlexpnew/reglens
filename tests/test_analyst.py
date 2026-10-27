import json

from reglens.agents.analyst import REFUSAL, analyze, build_prompt, label_sources, validate
from reglens.retrieve import Passage


def P(i, text):
    meta = {
        "doc_id": f"d{i}",
        "title": f"Circular {i}",
        "date": "2026-10-07",
        "page_start": i,
        "page_end": i,
        "url": f"https://www.sebi.gov.in/{i}.pdf",
    }
    return Passage(id=f"p{i}", text=text, meta=meta, score=0.9)


PASSAGES = [P(1, "The Credit Risk-o-Meter shows issuer credit risk."), P(2, "Applies from Jan 1.")]
LABELED = label_sources(PASSAGES)


class FakeLLM:
    def __init__(self, payload):
        self.payload, self.prompts = payload, []

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def test_prompt_labels_every_source_with_citation():
    prompt = build_prompt("What is it?", LABELED)
    assert "[S1] Circular 1 (2026-10-07), p.1" in prompt
    assert "[S2]" in prompt and "What is it?" in prompt


def test_valid_claims_keep_citations_and_urls():
    raw = {
        "insufficient_evidence": False,
        "claims": [
            {"text": "It shows issuer credit risk.", "sources": ["S1"]},
            {"text": "It applies from Jan 1.", "sources": ["S2"]},
        ],
    }
    ans = analyze("What is it?", PASSAGES, FakeLLM(raw))
    assert not ans.refused and len(ans.claims) == 2
    assert ans.citations == {
        "S1": "Circular 1 (2026-10-07), p.1",
        "S2": "Circular 2 (2026-10-07), p.2",
    }
    assert ans.urls["S2"].endswith("/2.pdf")
    assert "[S1]" in ans.text


def test_hallucinated_citation_is_dropped():
    raw = {
        "claims": [
            {"text": "Real.", "sources": ["S1"]},
            {"text": "Invented fact.", "sources": ["S9"]},
            {"text": "No source at all.", "sources": []},
        ]
    }
    ans = validate("q", json.dumps(raw), LABELED)
    assert [c.text for c in ans.claims] == ["Real."]
    assert ans.dropped_claims == 2 and "removed" in ans.reason


def test_all_claims_invalid_means_refusal():
    ans = validate("q", json.dumps({"claims": [{"text": "x", "sources": ["S7"]}]}), LABELED)
    assert ans.refused and ans.text == REFUSAL


def test_model_flagged_insufficient_evidence():
    ans = validate("q", json.dumps({"insufficient_evidence": True, "claims": []}), LABELED)
    assert ans.refused


def test_malformed_and_fenced_output():
    assert validate("q", "sorry, I cannot", LABELED).refused
    fenced = '```json\n{"claims":[{"text":"Ok.","sources":["S1"]}]}\n```'
    assert not validate("q", fenced, LABELED).refused


def test_markdown_lists_sources():
    raw = {"claims": [{"text": "It shows issuer credit risk.", "sources": ["S1"]}]}
    md = analyze("What is it?", PASSAGES, FakeLLM(raw)).to_markdown()
    assert "**Sources**" in md and "https://www.sebi.gov.in/1.pdf" in md
