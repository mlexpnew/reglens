.PHONY: install test lint ingest

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

ingest:
	python -m reglens.ingest.cli --download --query "What are the disclosure timelines for mutual funds?"
