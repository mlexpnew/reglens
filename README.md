# RegLens

![CI](https://github.com/mlexpnew/reglens/actions/workflows/ci.yml/badge.svg?branch=main)

**Ask a question about SEBI regulations. Get a cited answer, checked by a compliance agent, with eval scores on every release.**

> Status: Day 2 of 7. Hugging Face retrieval and cited-answer agents live and verified. Planner, Checker, evaluation and API land next.

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

### Verified run ([Ingest #1](https://github.com/mlexpnew/reglens/actions/runs/40330415529))
| Documents | Pages | Chunks | Failed docs | Time |
|---|---|---|---|---|
| 12 | 46 | 82 | 0 | 8.6 s |

Retrieval smoke test: *"What is the Credit Risk-o-Meter for debt securities?"* returned the correct circular for all top 3 results (cosine similarity 0.70 to 0.73).

Re-run anytime: **Actions > Ingest > Run workflow**.

## Retrieval + agents (Day 2, done)
```
question -> retrieve (bge-small, top 12) -> rerank (bge-reranker, top 4) -> relevance gate
                                                                          |-- below 0.10 -> refuse
                                                                          '-- pass -> Analyst (Gemini) -> citation check -> answer
```
| Piece | Design choice |
|---|---|
| Embeddings | Hugging Face `BAAI/bge-small-en-v1.5`; one Chroma collection per model so vectors never mix |
| Reranker | Hugging Face `BAAI/bge-reranker-base` cross-encoder, then a relevance gate *before* any LLM call |
| Orchestration | LangGraph: `retrieve -> rerank -> (refuse \| analyze)` |
| Analyst | Gemini at temperature 0, JSON output; every claim must cite a source label |
| Citation check | Done in code, not trusted to the model: claims citing unknown sources are dropped; no valid claim means refusal |
| Resilience | Exponential backoff with jitter for Gemini 429/5xx and for flaky PDF downloads |

### Verified demo ([Ask #6](https://github.com/mlexpnew/reglens/actions/runs/40463751205))
| Question | Result | Cited source | Top rerank relevance | Latency |
|---|---|---|---|---|
| What is the Credit Risk-o-Meter and which securities does it apply to? | Answered | Credit Risk-o-Meter circular, 07 Oct 2026, pp. 1-7 | 0.97 | 10.4 s |
| What are the revised position limits for clients in commodity derivatives? | Answered | Commodity position limits circular, 09 Sep 2026, p. 4 | 0.85 | 5.7 s |
| What must stock brokers display on their trading apps under Project Jagrook? | Answered | Project Jagrook circular, 01 Oct 2026, pp. 1-2 | 1.00 | 12.5 s |
| Who won the 2011 Cricket World Cup? (off-topic) | **Refused** | none | below 0.10 | 6.0 s |

All 3 in-scope answers cited the correct circular; the off-topic question was refused without calling the LLM.
Ask your own question: **Actions > Ask > Run workflow**.

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
export GEMINI_API_KEY=...   # free key from aistudio.google.com
reglens-ask "What is the Credit Risk-o-Meter?"
```

## Project layout
```
src/reglens/ingest/   manifest, download, parse, chunk, store, pipeline, cli
src/reglens/agents/   LangGraph graph, Analyst with citation check, Gemini client
src/reglens/retrieve.py  vector retrieval + cross-encoder reranking
src/reglens/ask.py    question CLI
tests/                unit + end-to-end tests with fakes (no network, no API key)
data/sources.csv      source manifest
```

## Limitations
Scanned PDFs without a text layer are skipped (OCR planned). Not legal advice.
