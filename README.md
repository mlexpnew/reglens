# RegLens

![CI](https://github.com/mlexpnew/reglens/actions/workflows/ci.yml/badge.svg?branch=main)

**Ask a question about SEBI regulations. Get a cited answer, checked by a compliance agent, with eval scores on every release.**

> Status: Day 1 of 7. Ingestion pipeline complete. Agents, evaluation and API land next.

## Why
Compliance teams cannot trust an answer they cannot trace. RegLens refuses to answer without a citation, and every release is gated on measured faithfulness.

## Architecture (target)
```
question -> Planner -> Retriever (ChromaDB) -> Analyst -> Compliance Checker -> cited answer
                                                   ^              |
                                                   +-- revise ----+ (max 1 loop, else "insufficient evidence")
```
Served by FastAPI with API key auth, rate limiting, PII-masked JSON logs and `/metrics`.

## Ingestion (done)
| Step | Design choice |
|---|---|
| Sources | `data/sources.csv` manifest of 12 real SEBI circulars; every chunk carries title, date, URL |
| Download | Idempotent, atomic writes, verifies `%PDF` header, polite rate limit |
| Parse | Strips repeated letterheads and page numbers; flags scanned PDFs instead of indexing noise |
| Chunk | Paragraph-aware, 1000 chars, 150 overlap, page range kept for citations |
| Store | ChromaDB (cosine), deterministic chunk IDs so re-runs upsert, never duplicate |

## Evaluation (coming Day 4)
| Metric | Score |
|---|---|
| Faithfulness | [ADD METRIC] |
| Answer relevance | [ADD METRIC] |
| Retrieval recall@5 | [ADD METRIC] |
| p95 latency | [ADD METRIC] |
| Cost per query | [ADD METRIC] |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
make install
make test
make ingest   # downloads circulars, builds the index, runs a sample query
```

## Project layout
```
src/reglens/ingest/   manifest, download, parse, chunk, store, pipeline, cli
tests/                unit + end-to-end tests (synthetic PDFs, no network)
data/sources.csv      source manifest
```

## Limitations
Scanned PDFs without a text layer are skipped (OCR planned). Not legal advice.
