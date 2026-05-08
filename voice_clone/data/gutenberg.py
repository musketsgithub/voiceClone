"""Small Project Gutenberg corpus fetcher for pipeline smoke tests."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .cleaning import normalize_text


GUTENBERG_TEXT_URLS = (
    "https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    "https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
)


@dataclass(frozen=True)
class GutenbergBook:
    author_id: str
    title: str
    book_id: int

    @property
    def filename(self) -> str:
        return f"{slugify(self.title)}__pg{self.book_id}.txt"


SMOKE_GUTENBERG_BOOKS = [
    GutenbergBook("jane_austen", "Pride and Prejudice", 1342),
    GutenbergBook("jane_austen", "Emma", 158),
    GutenbergBook("charles_dickens", "A Tale of Two Cities", 98),
    GutenbergBook("charles_dickens", "Great Expectations", 1400),
    GutenbergBook("mark_twain", "Adventures of Huckleberry Finn", 76),
    GutenbergBook("mark_twain", "The Adventures of Tom Sawyer", 74),
    GutenbergBook("h_g_wells", "The Time Machine", 35),
    GutenbergBook("h_g_wells", "The War of the Worlds", 36),
    GutenbergBook("mary_shelley", "Frankenstein", 84),
    GutenbergBook("mary_shelley", "The Last Man", 18247),
    GutenbergBook("oscar_wilde", "The Picture of Dorian Gray", 174),
    GutenbergBook("oscar_wilde", "The Importance of Being Earnest", 844),
    GutenbergBook("arthur_conan_doyle", "The Adventures of Sherlock Holmes", 1661),
    GutenbergBook("arthur_conan_doyle", "The Hound of the Baskervilles", 2852),
    GutenbergBook("jack_london", "The Call of the Wild", 215),
    GutenbergBook("jack_london", "White Fang", 910),
]


def slugify(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    return "_".join("".join(chars).split("_")).strip("_")


def fetch_gutenberg_text(book_id: int, *, timeout: float = 30.0) -> str:
    errors: list[str] = []
    for template in GUTENBERG_TEXT_URLS:
        url = template.format(book_id=book_id)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "voiceClone data pipeline smoke test"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not fetch Project Gutenberg book {book_id}:\n" + "\n".join(errors))


def download_gutenberg_corpus(
    output_dir: str | Path,
    *,
    books: list[GutenbergBook] | None = None,
    max_authors: int | None = None,
    docs_per_author: int | None = None,
    sleep_seconds: float = 0.5,
    skip_existing: bool = True,
) -> list[Path]:
    selected_books = select_books(
        books or SMOKE_GUTENBERG_BOOKS,
        max_authors=max_authors,
        docs_per_author=docs_per_author,
    )

    output_root = Path(output_dir)
    written: list[Path] = []
    for index, book in enumerate(selected_books):
        author_dir = output_root / book.author_id
        author_dir.mkdir(parents=True, exist_ok=True)
        output_path = author_dir / book.filename
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            written.append(output_path)
            continue

        text = normalize_text(fetch_gutenberg_text(book.book_id))
        output_path.write_text(text, encoding="utf-8")
        written.append(output_path)

        if sleep_seconds and index < len(selected_books) - 1:
            time.sleep(sleep_seconds)

    return written


def select_books(
    books: list[GutenbergBook],
    *,
    max_authors: int | None = None,
    docs_per_author: int | None = None,
) -> list[GutenbergBook]:
    author_order: list[str] = []
    author_counts: dict[str, int] = {}
    selected: list[GutenbergBook] = []
    for book in books:
        if book.author_id not in author_counts:
            if max_authors is not None and len(author_order) >= max_authors:
                continue
            author_order.append(book.author_id)
            author_counts[book.author_id] = 0

        if docs_per_author is not None and author_counts[book.author_id] >= docs_per_author:
            continue

        selected.append(book)
        author_counts[book.author_id] += 1
    return selected
