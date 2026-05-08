"""Style-conditioned causal LM wrapper."""

from __future__ import annotations

from .style_embedder import _require_torch


torch, nn, F = _require_torch()


class StyleConditionedCausalLM(nn.Module):
    """Freeze a base CausalLM and inject a style vector into decoder layers."""

    def __init__(self, base_model, style_dim: int = 512) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = get_hidden_size(base_model)
        self.style_projection = nn.Linear(style_dim, hidden_size)

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        style_dim: int = 512,
        torch_dtype=None,
        device_map=None,
        trust_remote_code: bool = True,
    ) -> "StyleConditionedCausalLM":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install `transformers` to load causal LMs.") from exc

        kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return cls(base_model, style_dim=style_dim)

    def forward(self, input_ids, attention_mask=None, style_embedding=None, labels=None):
        if style_embedding is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        style_shift = self.style_projection(style_embedding)
        layers = get_decoder_layers(self.base_model)
        handles = []

        def make_hook():
            def hook(_module, inputs):
                hidden_states = inputs[0]
                shifted = hidden_states + style_shift[:, None, :].to(hidden_states.dtype)
                return (shifted, *inputs[1:])

            return hook

        try:
            for layer in layers:
                handles.append(layer.register_forward_pre_hook(make_hook()))
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            for handle in handles:
                handle.remove()


def get_hidden_size(model) -> int:
    config = model.config
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from model config.")


def get_decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        current = model
        for part in path:
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None:
            return current
    raise ValueError("Could not find decoder layers for style injection.")
