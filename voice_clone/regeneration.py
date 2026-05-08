"""Intermediate draft regeneration from structure plus style guide."""

from __future__ import annotations

from .llm import async_complete_chat, complete_chat
from .prompts import REGENERATION_PROMPT


def regenerate_draft(
    structure_summary: str,
    style_guide: str,
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    user_prompt = f"""
==============================
STYLE GUIDE
==============================

{style_guide}


==============================
DISCOURSE STRUCTURE
==============================

{structure_summary}


==============================
TASK
==============================

Generate a passage that follows:
- the discourse structure
- the reasoning flow
- the emphasis ordering

while partially reflecting the style guide.

The output should feel like:
"a structurally accurate draft written by someone approximating the author's style."

Do NOT output bullet points.
Write natural prose.
""".strip()

    return complete_chat(
        [
            {"role": "system", "content": REGENERATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.8,
        max_tokens=max_tokens,
    )


async def async_regenerate_draft(
    structure_summary: str,
    style_guide: str,
    *,
    model: str = "gpt-4.1-mini",
    max_tokens: int | None = None,
) -> str:
    user_prompt = f"""
==============================
STYLE GUIDE
==============================

{style_guide}


==============================
DISCOURSE STRUCTURE
==============================

{structure_summary}


==============================
TASK
==============================

Generate a passage that follows:
- the discourse structure
- the reasoning flow
- the emphasis ordering

while partially reflecting the style guide.

The output should feel like:
"a structurally accurate draft written by someone approximating the author's style."

Do NOT output bullet points.
Write natural prose.
""".strip()

    return await async_complete_chat(
        [
            {"role": "system", "content": REGENERATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.8,
        max_tokens=max_tokens,
    )
