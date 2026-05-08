"""Small OpenAI chat wrapper used by data-generation stages."""

from __future__ import annotations

import os
from typing import Iterable


Message = dict[str, str]


def complete_chat(
    messages: Iterable[Message],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> str:
    """Call the configured OpenAI chat model and return text content."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the `openai` package to call model APIs.") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for model generation.")

    client = OpenAI(api_key=key)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": list(messages),
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned an empty response.")
    return content.strip()
