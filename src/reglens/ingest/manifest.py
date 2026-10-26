"""Source manifest: one row per circular, so every chunk traces back to a real document."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("doc_id", "title", "date", "url")
_DOC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    date: str  # ISO YYYY-MM-DD
    url: str
    circular_no: str = ""

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.pdf"


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> list[Source]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest missing columns: {missing}")

        sources: list[Source] = []
        seen: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            doc_id = (row["doc_id"] or "").strip().lower()
            if not _DOC_ID.match(doc_id):
                raise ManifestError(f"line {line_no}: invalid doc_id {doc_id!r}")
            if doc_id in seen:
                raise ManifestError(f"line {line_no}: duplicate doc_id {doc_id!r}")
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", (row["date"] or "").strip()):
                raise ManifestError(f"line {line_no}: date must be YYYY-MM-DD")
            url = (row["url"] or "").strip()
            if not url.startswith("https://"):
                raise ManifestError(f"line {line_no}: url must be https")
            seen.add(doc_id)
            sources.append(
                Source(
                    doc_id=doc_id,
                    title=(row["title"] or "").strip(),
                    date=row["date"].strip(),
                    url=url,
                    circular_no=(row.get("circular_no") or "").strip(),
                )
            )
    return sources
