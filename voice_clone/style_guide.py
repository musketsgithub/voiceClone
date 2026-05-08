"""Symbolic author style-guide generation."""

from __future__ import annotations

from .llm import async_complete_chat, complete_chat
from .prompts import STYLE_GUIDE_PROMPT


def generate_style_guide(
    corpus_texts: list[str],
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    if not corpus_texts:
        raise ValueError("corpus_texts must contain at least one passage.")

    corpus = "\n\n--- PASSAGE ---\n\n".join(corpus_texts)
    return complete_chat(
        [
            {"role": "system", "content": STYLE_GUIDE_PROMPT},
            {"role": "user", "content": corpus},
        ],
        model=model,
        temperature=0.4,
        max_tokens=max_tokens,
    )


async def async_generate_style_guide(
    corpus_texts: list[str],
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    if not corpus_texts:
        raise ValueError("corpus_texts must contain at least one passage.")

    corpus = "\n\n--- PASSAGE ---\n\n".join(corpus_texts)
    return await async_complete_chat(
        [
            {"role": "system", "content": STYLE_GUIDE_PROMPT},
            {"role": "user", "content": corpus},
        ],
        model=model,
        temperature=0.4,
        max_tokens=max_tokens,
    )
