"""Stage 1 data pipeline: raw corpus -> passages -> training triplets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

from .chunking import chunk_document
from .gutenberg import download_gutenberg_corpus
from .io import load_author_documents, read_jsonl, write_jsonl
from .models import Passage, TripletRecord
from .neutral_drafts import generate_neutral_draft


@dataclass(frozen=True)
class PipelineStats:
    authors: int
    documents: int
    passages: int
    avg_passage_tokens: float
    passages_per_author: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "authors": self.authors,
            "documents": self.documents,
            "passages": self.passages,
            "avg_passage_tokens": self.avg_passage_tokens,
            "passages_per_author": self.passages_per_author,
        }


def build_passages(
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    target_tokens: int = 1000,
    min_tokens: int = 250,
) -> PipelineStats:
    documents = load_author_documents(raw_dir)
    passages: list[Passage] = []
    for document in documents:
        passages.extend(
            chunk_document(
                document,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
            )
        )

    write_jsonl(output_path, (passage.to_dict() for passage in passages))
    return summarize_passages(passages, document_count=len(documents))


def download_seed_corpus(
    output_dir: str | Path,
    *,
    max_authors: int | None = None,
    docs_per_author: int | None = None,
    sleep_seconds: float = 0.5,
) -> list[Path]:
    return download_gutenberg_corpus(
        output_dir,
        max_authors=max_authors,
        docs_per_author=docs_per_author,
        sleep_seconds=sleep_seconds,
    )


def load_passages(path: str | Path) -> list[Passage]:
    return [Passage(**row) for row in read_jsonl(path)]


def summarize_passages(passages: list[Passage], *, document_count: int | None = None) -> PipelineStats:
    author_counts = Counter(p.author_id for p in passages)
    avg_tokens = sum(p.token_count for p in passages) / len(passages) if passages else 0.0
    return PipelineStats(
        authors=len(author_counts),
        documents=document_count if document_count is not None else len({p.doc_id for p in passages}),
        passages=len(passages),
        avg_passage_tokens=avg_tokens,
        passages_per_author=dict(sorted(author_counts.items())),
    )


def summarize_passages_file(path: str | Path) -> PipelineStats:
    passages = load_passages(path)
    return summarize_passages(passages)


def build_neutral_triplets(
    passages_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "gpt-4.1-mini",
    limit: int | None = None,
    max_per_author: int | None = None,
    seed: int = 7,
    dry_run: bool = False,
) -> list[TripletRecord]:
    passages = load_passages(passages_path)
    if max_per_author is not None:
        passages = select_balanced_passages(passages, max_per_author=max_per_author, seed=seed)
    if limit is not None:
        passages = passages[:limit]

    records: list[TripletRecord] = []
    for passage in passages:
        neutral_draft = (
            f"[DRY RUN neutral rewrite placeholder for {passage.passage_id}]\n\n{passage.text}"
            if dry_run
            else generate_neutral_draft(passage.text, model=model)
        )
        records.append(
            TripletRecord(
                passage_id=passage.passage_id,
                author_id=passage.author_id,
                real_passage=passage.text,
                neutral_draft=neutral_draft,
                doc_id=passage.doc_id,
                token_count=passage.token_count,
            )
        )

    write_jsonl(output_path, (record.to_dict() for record in records))
    return records


def select_balanced_passages(
    passages: list[Passage],
    *,
    max_per_author: int,
    seed: int = 7,
) -> list[Passage]:
    rng = random.Random(seed)
    grouped: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        grouped[passage.author_id].append(passage)

    selected: list[Passage] = []
    for author_id in sorted(grouped):
        author_passages = list(grouped[author_id])
        rng.shuffle(author_passages)
        selected.extend(author_passages[:max_per_author])
    return selected


def author_passage_index(records: list[TripletRecord]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, record in enumerate(records):
        index[record.author_id].append(row_index)
    return dict(index)
