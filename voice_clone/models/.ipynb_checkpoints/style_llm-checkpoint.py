"""Style-conditioned causal LM wrapper."""

from __future__ import annotations

from .style_embedder import _require_torch


torch, nn, F = _require_torch()


class StyleConditionedCausalLM(nn.Module):
    """Freeze a base CausalLM and inject a style vector into every decoder layer."""

    def __init__(self, base_model, style_dim: int = 512) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = get_hidden_size(base_model)
        self.style_projection = nn.Linear(style_dim, hidden_size)
        # Zero-init so the model starts at baseline loss rather than disrupted.
        nn.init.zeros_(self.style_projection.weight)
        nn.init.zeros_(self.style_projection.bias)

        self._style_shift = None  # set per forward call, read by hooks during recompute

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        layers = get_decoder_layers(self.base_model)
        self._num_layers = len(layers)
        # Persistent hooks: registered once, never removed, so gradient-checkpointing
        # recompute passes also fire them (unlike temporary hooks that are torn down
        # in a finally block before backward runs).
        for layer in layers:
            layer.register_forward_pre_hook(self._inject_style)

    def _inject_style(self, _module, inputs):
        if self._style_shift is None:
            return
        hidden_states = inputs[0]
        # Divide by num_layers so the total gradient magnitude is independent of depth.
        # Without this, gradients from all 28 layers accumulate and the effective LR
        # is ~28x the nominal value, causing divergence.
        shift = (self._style_shift / self._num_layers)[:, None, :].to(hidden_states.dtype)
        return (hidden_states + shift, *inputs[1:])

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        style_dim: int = 512,
        torch_dtype=None,
        device_map=None,
        quantization_config=None,
        trust_remote_code: bool = True,
        token=None,
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
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if token is not None:
            kwargs["token"] = token

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return cls(base_model, style_dim=style_dim)

    def forward(self, input_ids, attention_mask=None, style_embedding=None, labels=None):
        # Set before calling base_model so hooks read the current shift.
        # Not cleared after: the recompute pass during backward fires hooks while
        # loss.backward() is still running, so _style_shift must still be set then.
        self._style_shift = (
            self.style_projection(style_embedding) if style_embedding is not None else None
        )
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


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
