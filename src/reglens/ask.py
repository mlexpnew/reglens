"""CLI: python -m reglens.ask "your question" [--json]"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask RegLens a SEBI compliance question")
    p.add_argument("question", nargs="+")
    p.add_argument("--json", action="store_true", help="print the full answer as JSON")
    args = p.parse_args(argv)
    question = " ".join(args.question)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    from .agents.graph import build_graph
    from .agents.llm import GeminiLLM
    from .ingest.store import ChromaStore
    from .retrieve import ChromaRetriever, CrossEncoderReranker

    store = ChromaStore(settings.chroma_dir, settings.collection, settings.embed_model)
    if store.count() == 0:
        print(
            "Index is empty: run `python -m reglens.ingest.cli --download` first.", file=sys.stderr
        )
        return 2

    graph = build_graph(
        ChromaRetriever(store),
        CrossEncoderReranker(settings.rerank_model),
        GeminiLLM(settings.llm_model),
        retrieve_k=settings.retrieve_k,
        rerank_k=settings.rerank_k,
        min_relevance=settings.min_relevance,
    )
    t0 = time.perf_counter()
    state = graph.invoke({"question": question})
    latency = round(time.perf_counter() - t0, 2)
    answer = state["answer"]

    if args.json:
        out = answer.to_dict()
        out["latency_s"] = latency
        out["top_passages"] = [
            {"id": p.id, "relevance": p.score, "citation": p.citation}
            for p in state.get("passages", [])
        ]
        print(json.dumps(out, indent=2))
    else:
        print(answer.to_markdown())
        print(f"\n_latency {latency}s_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
