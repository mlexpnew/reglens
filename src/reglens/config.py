"""Central settings, overridable with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = Path(os.getenv("REGLENS_RAW_DIR", "data/raw"))
    manifest: Path = Path(os.getenv("REGLENS_MANIFEST", "data/sources.csv"))
    chroma_dir: Path = Path(os.getenv("REGLENS_CHROMA_DIR", "data/chroma"))
    collection: str = os.getenv("REGLENS_COLLECTION", "sebi_circulars")
    chunk_chars: int = int(os.getenv("REGLENS_CHUNK_CHARS", "1000"))
    chunk_overlap: int = int(os.getenv("REGLENS_CHUNK_OVERLAP", "150"))

    # Hugging Face models: small, CPU friendly, strong on retrieval benchmarks
    embed_model: str = os.getenv("REGLENS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("REGLENS_RERANK_MODEL", "BAAI/bge-reranker-base")

    # Retrieval: recall wide with vectors, then rerank precisely
    retrieve_k: int = int(os.getenv("REGLENS_RETRIEVE_K", "12"))
    rerank_k: int = int(os.getenv("REGLENS_RERANK_K", "4"))
    min_relevance: float = float(os.getenv("REGLENS_MIN_RELEVANCE", "0.10"))

    llm_model: str = os.getenv("REGLENS_LLM_MODEL", "gemini-2.5-flash")

    # Agents: switchable so evaluation can measure what each one adds (ablation)
    use_planner: bool = os.getenv("REGLENS_USE_PLANNER", "1") == "1"
    use_checker: bool = os.getenv("REGLENS_USE_CHECKER", "1") == "1"
    max_sub_queries: int = int(os.getenv("REGLENS_MAX_SUB_QUERIES", "3"))
    max_revisions: int = int(os.getenv("REGLENS_MAX_REVISIONS", "1"))


settings = Settings()
