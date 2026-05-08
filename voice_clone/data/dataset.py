"""PyTorch Dataset for embedder triplet training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .models import TripletRecord
from .pipeline import author_passage_index


class TripletTextDataset:
    """Samples anchor/positive/negative text triplets.

    Anchor and positive are different real passages by the same author.
    Negative is the neutral rewrite of the anchor, preserving topic while
    stripping author style.
    """

    def __init__(self, records: list[TripletRecord], *, seed: int | None = None) -> None:
        self.records = records
        self.author_index = author_passage_index(records)
        self.rng = random.Random(seed)
        self.valid_indices = [
            idx
            for idx, record in enumerate(records)
            if len(self.author_index[record.author_id]) >= 2
        ]
        if not self.valid_indices:
            raise ValueError("Need at least one author with two passages to sample triplets.")

    @classmethod
    def from_jsonl(cls, path: str | Path, *, seed: int | None = None) -> "TripletTextDataset":
        return cls([triplet_from_row(row) for row in read_jsonl(path)], seed=seed)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor_index = self.valid_indices[item % len(self.valid_indices)]
        anchor = self.records[anchor_index]
        candidates = [idx for idx in self.author_index[anchor.author_id] if idx != anchor_index]
        different_doc_candidates = [
            idx for idx in candidates if self.records[idx].doc_id != anchor.doc_id
        ]
        if different_doc_candidates:
            candidates = different_doc_candidates
        positive = self.records[self.rng.choice(candidates)]
        return {
            "author_id": anchor.author_id,
            "anchor_id": anchor.passage_id,
            "positive_id": positive.passage_id,
            "anchor_doc_id": anchor.doc_id,
            "positive_doc_id": positive.doc_id,
            "anchor": anchor.real_passage,
            "positive": positive.real_passage,
            "negative": anchor.style_regenerated_draft or anchor.neutral_draft,
            "negative_type": anchor.negative_type,
        }


def triplet_from_row(row: dict[str, Any]) -> TripletRecord:
    allowed = TripletRecord.__dataclass_fields__.keys()
    return TripletRecord(**{key: value for key, value in row.items() if key in allowed})


def as_torch_dataset(records: list[TripletRecord], *, seed: int | None = None):
    """Return a torch Dataset subclass when PyTorch is installed."""
    try:
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to use this as a torch Dataset.") from exc

    class TorchTripletDataset(Dataset):
        def __init__(self) -> None:
            self.inner = TripletTextDataset(records, seed=seed)

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, item: int) -> dict[str, Any]:
            return self.inner[item]

    return TorchTripletDataset()
