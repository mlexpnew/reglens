# RegLens

![CI](https://github.com/mlexpnew/reglens/actions/workflows/ci.yml/badge.svg?branch=main)

**Ask a question about SEBI regulations. Get a cited answer, checked by a compliance agent, with eval scores on every release.**

> Status: Day 3 of 7. Full agent loop live: Planner, Analyst and Compliance Checker with verified citations. Evaluation and API land next.

## Demo

![RegLens answering a multi-part SEBI question with verified citations, then refusing an off-topic question](docs/demo.gif)

_Real run, not sped up: about 20 s of model loading, then a verified answer in 16 s; the off-topic question is refused in about 1 s with no LLM call._

**Try it in 1 minute, no install:** Actions > Ask > Run workflow, type a question, read the cited answer in the job log ([example run](https://github.com/mlexpnew/reglens/actions/runs/40467064514)).

**Record a terminal demo** (a real run, needs `GEMINI_API_KEY` and `make ingest` first):

```bash
brew install vhs        # or: go install github.com/charmbracelet/vhs@latest
vhs docs/demo.tape      # writes docs/demo.gif
```

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

## Planner + Compliance Checker (Day 3, done)
```
retrieve -> gate --(nothing relevant)--> refuse                         (0 LLM calls)
              \-> plan --(sub-queries)--> expand -> analyze -> check --(verified)--> answer
                     \--------------------------------^    ^        |
                                                          \-revise-/  (max 1)
```
| Piece | Design choice |
|---|---|
| Gate first | Planner runs only after the relevance gate passes, so off-topic questions stay free |
| Planner | Splits multi-part questions into at most 3 sub-queries; each is reranked against itself so every part gets its own evidence |
| Checker, layer 1 | Code: every number in a claim must appear in its cited source. Catches invented figures even if the reviewer model says "supported" |
| Checker, layer 2 | LLM reviewer judges each claim against only the sources it cites |
| Fail closed | Missing verdict or unreadable reviewer output counts as unsupported |
| Revise loop | Rejected claims go back to the Analyst once with reasons; still unsupported claims are removed; nothing verified means refusal |
| Ablation | `REGLENS_USE_PLANNER` / `REGLENS_USE_CHECKER` switch agents off, so Day 4 can measure what each adds |

### Verified demo ([Ask #7](https://github.com/mlexpnew/reglens/actions/runs/40467064514))
| Question | Result | Sub-queries | Sources cited | Latency |
|---|---|---|---|---|
| What is the Credit Risk-o-Meter and which securities does it apply to? | Verified | 2 | Credit Risk-o-Meter circular | 22.3 s |
| What are the revised position limits for clients in commodity derivatives? | Verified | 0 | Position limits circular, p. 4 | 11.7 s |
| What must stock brokers display on their trading apps under Project Jagrook? | Verified | 0 | Project Jagrook circular | 22.6 s |
| What is the Credit Risk-o-Meter, **and** what are the position limits for broad commodities? | Verified, both parts answered | 2 | Credit Risk-o-Meter **and** Position limits circulars | 22.0 s |
| Who won the 2011 Cricket World Cup? (off-topic) | **Refused**, 0 LLM calls | 0 | none | 6.7 s |

Trade-off measured: the Checker adds an LLM call, so verified answers take roughly 2x longer than Day 2 (12 to 23 s vs 6 to 12 s). Day 4 evaluation quantifies whether the accuracy gain is worth it.

## Evaluation (coming Day 4)

Day 4 measures RegLens on a labeled set of SEBI questions: faithfulness, answer relevance, retrieval recall@5, p95 latency and cost per query. Each metric is run with the Planner and Checker switched on and off (ablation), so the README will show what every agent actually adds. Scores replace this paragraph when they exist.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
make install-full        # dev + Hugging Face models + agents (pulls PyTorch, ~1 GB)
make test
make ingest   # downloads circulars, builds the index, runs a sample query
export GEMINI_API_KEY=...   # free key from aistudio.google.com
reglens-ask "What is the Credit Risk-o-Meter?"
```

## Project layout
```
src/reglens/ingest/   manifest, download, parse, chunk, store, pipeline, cli
src/reglens/agents/   LangGraph graph, Planner, Analyst, Checker, Gemini client
src/reglens/retrieve.py  vector retrieval + cross-encoder reranking
src/reglens/ask.py    question CLI
tests/                unit + end-to-end tests with fakes (no network, no API key)
data/sources.csv      source manifest
```

## Limitations
Scanned PDFs without a text layer are skipped (OCR planned). Not legal advice.
