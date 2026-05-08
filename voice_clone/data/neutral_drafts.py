"""Neutral draft generation for Stage 1 triplets."""

from __future__ import annotations

from voice_clone.llm import complete_chat
from voice_clone.prompts import NEUTRAL_DRAFT_PROMPT


def generate_neutral_draft(text: str, *, model: str = "gpt-4.1-mini") -> str:
    return complete_chat(
        [
            {"role": "system", "content": NEUTRAL_DRAFT_PROMPT},
            {"role": "user", "content": text},
        ],
        model=model,
        temperature=0.2,
    )
