"""CLI: `python -m reglens.ingest.cli --download` then query with `--query`."""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import settings
from .download import download_all
from .manifest import load_manifest
from .pipeline import ingest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest SEBI circulars into ChromaDB")
    p.add_argument("--download", action="store_true", help="fetch PDFs listed in the manifest")
    p.add_argument("--query", help="run a quick retrieval check after ingesting")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args(argv)

    # human logs to stderr, one JSON summary line to stdout (machine-readable)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")

    from .store import ChromaStore

    sources = load_manifest(settings.manifest)
    if args.download:
        download_all(sources, settings.raw_dir)
    store = ChromaStore(settings.chroma_dir, settings.collection)
    report = ingest(sources, settings.raw_dir, store, settings.chunk_chars, settings.chunk_overlap)
    print(report.to_json())

    if args.query:
        res = store.query(args.query, k=args.k)
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            cite = f"{meta['doc_id']} p.{meta['page_start']}-{meta['page_end']}"
            print(f"[{1 - dist:.3f}] {cite}: {doc[:160]!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
