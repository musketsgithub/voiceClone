"""Passage chunking with a tokenizer-free fallback."""

from __future__ import annotations

import re

from .models import AuthorDocument, Passage


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CHAPTER_HEADING_RE = re.compile(r"^\s*(?:chapter|book|volume)\b", re.I | re.M)


def rough_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def is_probably_front_matter(text: str) -> bool:
    """Filter title pages and tables of contents that survive source cleaning."""
    prefix = text[:2500].lower()
    chapter_heading_count = len(CHAPTER_HEADING_RE.findall(text[:2500]))
    return "contents" in prefix or chapter_heading_count >= 6


def chunk_text(
    text: str,
    *,
    target_tokens: int = 1000,
    min_tokens: int = 250,
) -> list[str]:
    """Chunk text by paragraph while keeping chunks near target_tokens."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = rough_token_count(paragraph)
        if current and current_tokens + paragraph_tokens > target_tokens:
            chunk = "\n\n".join(current).strip()
            if rough_token_count(chunk) >= min_tokens:
                chunks.append(chunk)
            current = []
            current_tokens = 0

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        chunk = "\n\n".join(current).strip()
        if rough_token_count(chunk) >= min_tokens:
            chunks.append(chunk)

    return [chunk for chunk in chunks if not is_probably_front_matter(chunk)]


def chunk_document(
    document: AuthorDocument,
    *,
    target_tokens: int = 1000,
    min_tokens: int = 250,
) -> list[Passage]:
    chunks = chunk_text(document.text, target_tokens=target_tokens, min_tokens=min_tokens)
    passages: list[Passage] = []
    for index, chunk in enumerate(chunks):
        passage_id = f"{document.author_id}:{document.doc_id}:{index:04d}"
        passages.append(
            Passage(
                passage_id=passage_id,
                author_id=document.author_id,
                doc_id=document.doc_id,
                text=chunk,
                token_count=rough_token_count(chunk),
                chunk_index=index,
                source_path=document.source_path,
            )
        )
    return passages
