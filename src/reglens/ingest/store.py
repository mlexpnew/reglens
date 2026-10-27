"""Vector store behind a tiny protocol, so tests run without ChromaDB."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from .chunk import Chunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def count(self) -> int: ...


def collection_name(base: str, embed_model: str) -> str:
    """One collection per embedding model: vectors from different models never mix."""
    slug = re.sub(r"[^a-z0-9]+", "-", embed_model.lower()).strip("-")
    return f"{base}__{slug}"[:63]


class ChromaStore:
    """Persistent ChromaDB collection embedded with a Hugging Face sentence-transformer."""

    def __init__(self, path: Path, collection: str, embed_model: str) -> None:
        import chromadb  # lazy: heavy dependency
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection_name(collection, embed_model),
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=embed_model, normalize_embeddings=True
            ),
            metadata={"hnsw:space": "cosine", "embed_model": embed_model},
        )

    def upsert(self, chunks: list[Chunk], batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                metadatas=[c.metadata for c in part],
            )

    def count(self) -> int:
        return self._col.count()

    def query(self, text: str, k: int = 5) -> dict:
        return self._col.query(query_texts=[text], n_results=k)


class MemoryStore:
    """In-memory store for tests and dry runs."""

    def __init__(self) -> None:
        self.items: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.items[c.id] = c

    def count(self) -> int:
        return len(self.items)
