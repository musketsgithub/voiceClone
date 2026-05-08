"""Long-context style embedder architecture."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to use style embedder models.") from exc
    return torch, nn, F


torch, nn, F = _require_torch()


class AttentionPool(nn.Module):
    """Learned attention pooling over a masked sequence."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, values, mask=None):
        scores = self.score(torch.tanh(self.proj(values))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class StyleEmbedder(nn.Module):
    """Encoder + attention pooling + 512d normalized projection.

    This implementation uses learned attention over token embeddings first.
    Sentence/paragraph grouping can be layered on once the basic data/model
    loop is verified.
    """

    def __init__(self, encoder, hidden_size: int, projection_dim: int = 512) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = AttentionPool(hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "allenai/longformer-base-4096",
        *,
        projection_dim: int = 512,
        gradient_checkpointing: bool = False,
    ) -> "StyleEmbedder":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install `transformers` to load pretrained encoders.") from exc

        encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        return cls(
            encoder=encoder,
            hidden_size=encoder.config.hidden_size,
            projection_dim=projection_dim,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)


def count_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": trainable}
