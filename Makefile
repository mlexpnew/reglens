.PHONY: install install-full test lint ingest eval eval-retrieval

# Lightweight: tests and CI only (models and LLM are faked)
install:
	pip install -e ".[dev]"

# Everything needed to run ingest and ask locally (pulls PyTorch, ~1 GB)
install-full:
	pip install -e ".[dev,hf,agents]"

test:
	pytest -q

lint:
	ruff check .

ingest:
	python -m reglens.ingest.cli --download --query "What are the disclosure timelines for mutual funds?"

# Score RegLens on data/eval/questions.jsonl (needs GEMINI_API_KEY and an index)
eval:
	python -m reglens.eval --configs full,no_checker

# Retrieval metrics only: no LLM calls, no API key needed
eval-retrieval:
	python -m reglens.eval --retrieval-only
