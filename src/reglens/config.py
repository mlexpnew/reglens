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


settings = Settings()
