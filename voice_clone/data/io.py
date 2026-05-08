"""Filesystem IO for corpus and JSONL pipeline artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .cleaning import normalize_text
from .models import AuthorDocument


def load_author_documents(raw_dir: str | Path) -> list[AuthorDocument]:
    """Load `raw_dir/author_id/*.txt` into AuthorDocument records."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {root}")

    documents: list[AuthorDocument] = []
    for author_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        author_id = author_dir.name
        for path in sorted(author_dir.glob("*.txt")):
            text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                documents.append(
                    AuthorDocument(
                        author_id=author_id,
                        doc_id=path.stem,
                        text=text,
                        source_path=str(path),
                    )
                )
    return documents


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)
