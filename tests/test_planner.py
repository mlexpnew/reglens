import json

from reglens.agents.planner import plan


class FakeLLM:
    def __init__(self, raw):
        self.raw = raw

    def complete_json(self, prompt):
        assert "PLANNER" in prompt
        return self.raw if isinstance(self.raw, str) else json.dumps(self.raw)


def test_splits_dedupes_and_caps():
    raw = {"sub_queries": ["A", "a", " B ", "", "C", "D", "What is A and B?"]}
    assert plan("What is A and B?", FakeLLM(raw), max_queries=3) == ["A", "B", "C"]


def test_single_topic_returns_no_extra_queries():
    assert plan("What is A?", FakeLLM({"sub_queries": []})) == []


def test_bad_output_falls_back_safely():
    assert plan("q", FakeLLM("oops")) == []
    assert plan("q", FakeLLM({"other": 1})) == []
