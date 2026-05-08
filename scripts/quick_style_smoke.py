"""Tiny one-author style pipeline smoke test.

Runs:
1. style guide generation from two short excerpts
2. structure extraction from one short held-out excerpt
3. draft regeneration from style guide + structure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_clone.data.pipeline import load_passages
from voice_clone.llm import complete_chat
from voice_clone.regeneration import regenerate_draft
from voice_clone.structure import extract_structure


SMOKE_STYLE_GUIDE_PROMPT = """
Create a concise but complete generation-oriented style guide for this author.

Return exactly these sections:
1. Discourse flow
2. Paragraph/sentence behavior
3. Dialogue or exposition habits
4. What to avoid in generated drafts

Use 2 short bullets per section. Do not add an introduction or conclusion.
""".strip()


def clip(text: str, chars: int) -> str:
    return " ".join(text[:chars].split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny style pipeline smoke test.")
    parser.add_argument("--passages", default="data/processed/passages.jsonl")
    parser.add_argument("--author", default="jane_austen")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--excerpt-chars", type=int, default=900)
    parser.add_argument("--guide-tokens", type=int, default=420)
    parser.add_argument("--structure-tokens", type=int, default=120)
    parser.add_argument("--draft-tokens", type=int, default=180)
    args = parser.parse_args()

    passages = [p for p in load_passages(args.passages) if p.author_id == args.author]
    if len(passages) < 3:
        raise RuntimeError(f"Need at least 3 passages for author {args.author!r}.")

    style_examples = [clip(passages[0].text, args.excerpt_chars), clip(passages[1].text, args.excerpt_chars)]
    source = clip(passages[2].text, args.excerpt_chars)

    style_guide = complete_chat(
        [
            {"role": "system", "content": SMOKE_STYLE_GUIDE_PROMPT},
            {"role": "user", "content": "\n\n--- PASSAGE ---\n\n".join(style_examples)},
        ],
        model=args.model,
        temperature=0.3,
        max_tokens=args.guide_tokens,
    )
    structure = extract_structure(
        source,
        model=args.model,
        max_tokens=args.structure_tokens,
    )
    draft = regenerate_draft(
        structure,
        style_guide,
        model=args.model,
        max_tokens=args.draft_tokens,
    )

    print("=" * 80)
    print(f"AUTHOR: {args.author}")
    print("=" * 80)
    print("STYLE GUIDE\n")
    print(style_guide)
    print("\n" + "=" * 80)
    print("STRUCTURE\n")
    print(structure)
    print("\n" + "=" * 80)
    print("REGENERATED DRAFT\n")
    print(draft)


if __name__ == "__main__":
    main()
