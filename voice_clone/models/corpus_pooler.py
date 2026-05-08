"""Corpus-level pooling for author style embeddings."""

from __future__ import annotations

from .style_embedder import AttentionPool, _require_torch


torch, nn, F = _require_torch()


class CorpusStylePooler(nn.Module):
    """Attention-pool passage embeddings into one author embedding."""

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.pool = AttentionPool(embedding_dim)

    def forward(self, passage_embeddings, mask=None):
        pooled = self.pool(passage_embeddings, mask)
        return F.normalize(pooled, p=2, dim=-1)
