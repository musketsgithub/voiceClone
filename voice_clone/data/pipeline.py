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
from voice_clone.regeneration import regenerate_draft
from voice_clone.structure import extract_structure
from voice_clone.style_guide import generate_style_guide


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
    verbose: bool = False,
) -> list[TripletRecord]:
    passages = load_passages(passages_path)
    if max_per_author is not None:
        passages = select_balanced_passages(passages, max_per_author=max_per_author, seed=seed)
    if limit is not None:
        passages = passages[:limit]

    records: list[TripletRecord] = []
    for index, passage in enumerate(passages, start=1):
        if verbose:
            print(f"[neutral] {index}/{len(passages)} author={passage.author_id} passage={passage.passage_id}", flush=True)
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


def build_style_regen_triplets(
    passages_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "gpt-4.1-mini",
    max_per_author: int = 2,
    max_authors: int | None = None,
    seed: int = 7,
    source_chars: int = 1200,
    guide_examples: int = 3,
    guide_chars: int = 900,
    guide_tokens: int = 450,
    structure_tokens: int = 180,
    draft_tokens: int = 260,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[TripletRecord]:
    passages = load_passages(passages_path)
    selected = select_balanced_passages(
        passages,
        max_per_author=max_per_author,
        max_authors=max_authors,
        seed=seed,
    )
    by_author: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        by_author[passage.author_id].append(passage)

    style_guides: dict[str, str] = {}
    records: list[TripletRecord] = []
    if verbose:
        print(f"[style-regen] selected {len(selected)} passages", flush=True)

    for index, passage in enumerate(selected, start=1):
        if verbose:
            print(f"[style-regen] {index}/{len(selected)} author={passage.author_id} passage={passage.passage_id}", flush=True)
        if passage.author_id not in style_guides:
            if verbose:
                print(f"[style-regen] building style guide author={passage.author_id}", flush=True)
            guide_source = by_author[passage.author_id][:guide_examples]
            examples = [clip_text(p.text, guide_chars) for p in guide_source]
            style_guides[passage.author_id] = (
                dry_run_style_guide(passage.author_id)
                if dry_run
                else generate_style_guide(examples, model=model, max_tokens=guide_tokens)
            )

        source = clip_text(passage.text, source_chars)
        if verbose:
            print(f"[style-regen] extracting structure passage={passage.passage_id}", flush=True)
        structure_summary = (
            dry_run_structure(passage.passage_id)
            if dry_run
            else extract_structure(source, model=model, max_tokens=structure_tokens)
        )
        if verbose:
            print(f"[style-regen] regenerating draft passage={passage.passage_id}", flush=True)
        regenerated = (
            dry_run_regeneration(passage.passage_id, source)
            if dry_run
            else regenerate_draft(
                structure_summary,
                style_guides[passage.author_id],
                model=model,
                max_tokens=draft_tokens,
            )
        )

        records.append(
            TripletRecord(
                passage_id=passage.passage_id,
                author_id=passage.author_id,
                real_passage=passage.text,
                neutral_draft=regenerated,
                doc_id=passage.doc_id,
                token_count=passage.token_count,
                structure_summary=structure_summary,
                style_guide=style_guides[passage.author_id],
                style_regenerated_draft=regenerated,
                negative_type="style_regeneration",
            )
        )

    write_jsonl(output_path, (record.to_dict() for record in records))
    if verbose:
        print(f"[style-regen] wrote {len(records)} triplets to {output_path}", flush=True)
    return records


def select_balanced_passages(
    passages: list[Passage],
    *,
    max_per_author: int,
    max_authors: int | None = None,
    seed: int = 7,
) -> list[Passage]:
    rng = random.Random(seed)
    grouped: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        grouped[passage.author_id].append(passage)

    selected: list[Passage] = []
    author_ids = sorted(grouped)
    if max_authors is not None:
        author_ids = author_ids[:max_authors]

    for author_id in author_ids:
        author_passages = list(grouped[author_id])
        rng.shuffle(author_passages)
        selected.extend(author_passages[:max_per_author])
    return selected


def clip_text(text: str, max_chars: int) -> str:
    return " ".join(text[:max_chars].split())


def dry_run_style_guide(author_id: str) -> str:
    return f"DRY RUN style guide for {author_id}: preserve discourse flow and author-like pacing."


def dry_run_structure(passage_id: str) -> str:
    return f"DRY RUN structure summary for {passage_id}: preserve claim order and paragraph purpose."


def dry_run_regeneration(passage_id: str, source: str) -> str:
    return f"[DRY RUN style-regenerated draft placeholder for {passage_id}]\n\n{source}"


def author_passage_index(records: list[TripletRecord]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, record in enumerate(records):
        index[record.author_id].append(row_index)
    return dict(index)
