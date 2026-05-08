from voice_clone.data.chunking import chunk_text, is_probably_front_matter, rough_token_count
from voice_clone.data.dataset import TripletTextDataset
from voice_clone.data.gutenberg import SMOKE_GUTENBERG_BOOKS, select_books, slugify
from voice_clone.data.models import TripletRecord
from voice_clone.data.pipeline import build_style_regen_triplets, select_balanced_passages
from voice_clone.data.models import Passage


def test_chunk_text_keeps_paragraphs_and_counts_tokens():
    text = "\n\n".join(["one two three four five"] * 6)

    chunks = chunk_text(text, target_tokens=12, min_tokens=5)

    assert len(chunks) >= 2
    assert all(rough_token_count(chunk) >= 5 for chunk in chunks)


def test_triplet_dataset_samples_same_author_positive_and_neutral_negative():
    records = [
        TripletRecord("a:one:0000", "a", "real a one", "neutral a one", "one", 3),
        TripletRecord("a:two:0000", "a", "real a two", "neutral a two", "two", 3),
        TripletRecord("b:one:0000", "b", "real b one", "neutral b one", "one", 3),
    ]

    dataset = TripletTextDataset(records, seed=7)
    item = dataset[0]

    assert item["author_id"] == "a"
    assert item["anchor_id"] != item["positive_id"]
    assert item["anchor_doc_id"] != item["positive_doc_id"]
    assert item["negative"].startswith("neutral a")


def test_gutenberg_selection_limits_authors_and_docs():
    selected = select_books(SMOKE_GUTENBERG_BOOKS, max_authors=3, docs_per_author=1)

    assert len(selected) == 3
    assert len({book.author_id for book in selected}) == 3


def test_slugify_makes_safe_names():
    assert slugify("The Importance of Being Earnest!") == "the_importance_of_being_earnest"


def test_front_matter_filter_detects_contents_chunk():
    text = "CONTENTS\n\n" + "\n".join(f"CHAPTER {i}" for i in range(10))

    assert is_probably_front_matter(text)


def test_select_balanced_passages_limits_each_author():
    passages = [
        Passage(f"{author}:doc:{i}", author, "doc", "text", 1, i)
        for author in ["a", "b"]
        for i in range(5)
    ]

    selected = select_balanced_passages(passages, max_per_author=2, seed=7)

    assert len(selected) == 4
    assert sum(p.author_id == "a" for p in selected) == 2
    assert sum(p.author_id == "b" for p in selected) == 2


def test_style_regen_triplets_work_in_dry_run(tmp_path):
    passages_path = tmp_path / "passages.jsonl"
    output_path = tmp_path / "style_regen.jsonl"
    rows = [
        Passage(f"{author}:doc:{i}", author, "doc", "enough text " * 50, 100, i).to_dict()
        for author in ["a", "b"]
        for i in range(3)
    ]
    passages_path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")

    records = build_style_regen_triplets(
        passages_path,
        output_path,
        max_per_author=1,
        dry_run=True,
    )

    assert len(records) == 2
    assert records[0].negative_type == "style_regeneration"
    assert records[0].style_regenerated_draft
