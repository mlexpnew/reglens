import json

from reglens.agents.analyst import Answer, Claim, label_sources
from reglens.agents.checker import check, feedback_text, ungrounded_numbers
from reglens.retrieve import Passage


def P(i, text):
    meta = {"doc_id": f"d{i}", "title": f"C{i}", "page_start": 1, "page_end": 1}
    return Passage(id=f"p{i}", text=text, meta=meta, score=0.9)


LABELED = label_sources([P(1, "Limit is 2% of deliverable supply, from 1,000 lots."), P(2, "x")])


class FakeLLM:
    def __init__(self, raw):
        self.raw, self.prompts = raw, []

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self.raw if isinstance(self.raw, str) else json.dumps(self.raw)


def answer(*claims):
    return Answer(question="q", claims=[Claim(t, s) for t, s in claims])


def test_numbers_must_appear_in_cited_sources():
    cited = [LABELED["S1"]]
    assert ungrounded_numbers("Limit is 2% from 1000 lots.", cited) == set()  # commas ignored
    assert ungrounded_numbers("Limit is 5% of supply.", cited) == {"5"}


def test_all_supported_returns_no_rejections():
    llm = FakeLLM({"verdicts": [{"claim": 1, "supported": True}]})
    assert check(answer(("Limit is 2%.", ["S1"])), LABELED, llm) == []
    assert "CHECKER" in llm.prompts[0] and "deliverable supply" in llm.prompts[0]


def test_reviewer_rejection_and_numeric_rejection_combine():
    a = answer(("Limit is 2%.", ["S1"]), ("Limit is 9%.", ["S1"]))
    llm = FakeLLM({"verdicts": [{"claim": 1, "supported": False, "reason": "adds scope"}]})
    rej = check(a, LABELED, llm)
    assert [(r.index, r.reason) for r in rej][0] == (0, "adds scope")
    assert rej[1].index == 1 and "9" in rej[1].reason
    assert "Limit is 9%." in feedback_text(a, rej)


def test_fails_closed_on_malformed_or_missing_verdicts():
    a = answer(("Limit is 2%.", ["S1"]))
    assert len(check(a, LABELED, FakeLLM("not json"))) == 1
    assert len(check(a, LABELED, FakeLLM({"verdicts": []}))) == 1
