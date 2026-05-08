"""Small OpenAI chat wrapper used by data-generation stages."""

from __future__ import annotations

import asyncio
import os
import random as _random
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


async def async_complete_chat(
    messages: Iterable[Message],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    api_key: str | None = None,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> str:
    """Async version of complete_chat with exponential backoff retry."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import AsyncOpenAI, RateLimitError, APIStatusError
    except ImportError as exc:
        raise RuntimeError("Install the `openai` package to call model APIs.") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for model generation.")

    client = AsyncOpenAI(api_key=key)
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "messages": list(messages),
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Model returned an empty response.")
            return content.strip()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt) + _random.random(), 60.0)
            await asyncio.sleep(delay)
        except APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 529) and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt) + _random.random(), 60.0)
                await asyncio.sleep(delay)
            else:
                raise

    raise RuntimeError("Exhausted retries without success.")
