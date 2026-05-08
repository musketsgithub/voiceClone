"""Typed records passed between data-pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuthorDocument:
    author_id: str
    doc_id: str
    text: str
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    author_id: str
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TripletRecord:
    passage_id: str
    author_id: str
    real_passage: str
    neutral_draft: str
    doc_id: str
    token_count: int
    structure_summary: str | None = None
    style_guide: str | None = None
    style_regenerated_draft: str | None = None
    negative_type: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
