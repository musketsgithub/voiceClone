"""CLI for Stage 1 data pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_clone.data.dataset import TripletTextDataset
from voice_clone.data.pipeline import (
    build_neutral_triplets,
    build_passages,
    build_style_regen_triplets,
    download_seed_corpus,
    summarize_passages_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build style-transfer data artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gutenberg = subparsers.add_parser("gutenberg", help="Download a small multi-author Gutenberg corpus.")
    gutenberg.add_argument("--out-dir", default="data/raw")
    gutenberg.add_argument("--max-authors", type=int, default=8)
    gutenberg.add_argument("--docs-per-author", type=int, default=2)
    gutenberg.add_argument("--sleep-seconds", type=float, default=0.5)

    passages = subparsers.add_parser("passages", help="Clean and chunk raw author corpus.")
    passages.add_argument("--raw-dir", required=True)
    passages.add_argument("--out", default="data/processed/passages.jsonl")
    passages.add_argument("--target-tokens", type=int, default=1000)
    passages.add_argument("--min-tokens", type=int, default=250)

    inspect = subparsers.add_parser("inspect", help="Print passage dataset statistics.")
    inspect.add_argument("--passages", default="data/processed/passages.jsonl")

    neutral = subparsers.add_parser("neutral", help="Generate neutral drafts for passages.")
    neutral.add_argument("--passages", default="data/processed/passages.jsonl")
    neutral.add_argument("--out", default="data/processed/triplets.jsonl")
    neutral.add_argument("--model", default="gpt-4.1-mini")
    neutral.add_argument("--limit", type=int)
    neutral.add_argument("--max-per-author", type=int)
    neutral.add_argument("--seed", type=int, default=7)
    neutral.add_argument("--dry-run", action="store_true")

    style_regen = subparsers.add_parser(
        "style-regen",
        help="Generate harder negatives using style guide + structure regeneration.",
    )
    style_regen.add_argument("--passages", default="data/processed/passages.jsonl")
    style_regen.add_argument("--out", default="data/processed/triplets_style_regen.jsonl")
    style_regen.add_argument("--model", default="gpt-4.1-mini")
    style_regen.add_argument("--max-per-author", type=int, default=2)
    style_regen.add_argument("--max-authors", type=int)
    style_regen.add_argument("--seed", type=int, default=7)
    style_regen.add_argument("--source-chars", type=int, default=1200)
    style_regen.add_argument("--guide-examples", type=int, default=3)
    style_regen.add_argument("--guide-chars", type=int, default=900)
    style_regen.add_argument("--guide-tokens", type=int, default=450)
    style_regen.add_argument("--structure-tokens", type=int, default=180)
    style_regen.add_argument("--draft-tokens", type=int, default=260)
    style_regen.add_argument("--dry-run", action="store_true")

    sample = subparsers.add_parser("sample", help="Print sampled anchor/positive/negative triplets.")
    sample.add_argument("--triplets", default="data/processed/triplets.jsonl")
    sample.add_argument("--count", type=int, default=3)
    sample.add_argument("--chars", type=int, default=500)
    sample.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.command == "gutenberg":
        paths = download_seed_corpus(
            args.out_dir,
            max_authors=args.max_authors,
            docs_per_author=args.docs_per_author,
            sleep_seconds=args.sleep_seconds,
        )
        print(json.dumps({"downloaded_files": len(paths), "output_dir": args.out_dir}, indent=2))
    elif args.command == "passages":
        stats = build_passages(
            args.raw_dir,
            args.out,
            target_tokens=args.target_tokens,
            min_tokens=args.min_tokens,
        )
        print(json.dumps(stats.to_dict(), indent=2))
    elif args.command == "inspect":
        stats = summarize_passages_file(args.passages)
        print(json.dumps(stats.to_dict(), indent=2))
    elif args.command == "neutral":
        records = build_neutral_triplets(
            args.passages,
            args.out,
            model=args.model,
            limit=args.limit,
            max_per_author=args.max_per_author,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print(json.dumps({"triplets": len(records), "output": args.out}, indent=2))
    elif args.command == "style-regen":
        records = build_style_regen_triplets(
            args.passages,
            args.out,
            model=args.model,
            max_per_author=args.max_per_author,
            max_authors=args.max_authors,
            seed=args.seed,
            source_chars=args.source_chars,
            guide_examples=args.guide_examples,
            guide_chars=args.guide_chars,
            guide_tokens=args.guide_tokens,
            structure_tokens=args.structure_tokens,
            draft_tokens=args.draft_tokens,
            dry_run=args.dry_run,
        )
        print(json.dumps({"triplets": len(records), "output": args.out}, indent=2))
    elif args.command == "sample":
        dataset = TripletTextDataset.from_jsonl(args.triplets, seed=args.seed)
        for index in range(min(args.count, len(dataset))):
            item = dataset[index]
            print("=" * 80)
            print(f"author_id: {item['author_id']}")
            print(f"anchor_id: {item['anchor_id']}")
            print(f"positive_id: {item['positive_id']}")
            print(f"anchor_doc_id: {item['anchor_doc_id']}")
            print(f"positive_doc_id: {item['positive_doc_id']}")
            print(f"negative_type: {item['negative_type']}")
            print("\nANCHOR\n" + item["anchor"][: args.chars])
            print("\nPOSITIVE\n" + item["positive"][: args.chars])
            print("\nNEGATIVE\n" + item["negative"][: args.chars])


if __name__ == "__main__":
    main()
