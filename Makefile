.PHONY: install install-full test lint ingest

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
