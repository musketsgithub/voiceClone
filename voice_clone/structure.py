"""Discourse-structure extraction."""

from __future__ import annotations

from .llm import async_complete_chat, complete_chat
from .prompts import STRUCTURE_PROMPT


def extract_structure(
    text: str,
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    return complete_chat(
        [
            {"role": "system", "content": STRUCTURE_PROMPT},
            {"role": "user", "content": text},
        ],
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
    )


async def async_extract_structure(
    text: str,
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    return await async_complete_chat(
        [
            {"role": "system", "content": STRUCTURE_PROMPT},
            {"role": "user", "content": text},
        ],
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
    )
