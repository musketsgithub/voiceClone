# Data Pipeline

Stage 1 is now package code instead of notebook-only code. The notebooks can
import these functions for demos, but the durable path is:

```text
data/raw/
  author_id_a/
    document_1.txt
    document_2.txt
  author_id_b/
    document_1.txt
```

Download a small multi-author Project Gutenberg corpus:

```bash
.venv/bin/python scripts/build_data_pipeline.py gutenberg \
  --out-dir data/raw \
  --max-authors 8 \
  --docs-per-author 2
```

Build cleaned/chunked passages:

```bash
.venv/bin/python scripts/build_data_pipeline.py passages \
  --raw-dir data/raw \
  --out data/processed/passages.jsonl \
  --target-tokens 1000 \
  --min-tokens 250
```

Inspect passage stats:

```bash
.venv/bin/python scripts/build_data_pipeline.py inspect \
  --passages data/processed/passages.jsonl
```

Generate neutral drafts for triplet training:

```bash
.venv/bin/python scripts/build_data_pipeline.py neutral \
  --passages data/processed/passages.jsonl \
  --out data/processed/triplets.jsonl \
  --model gpt-4.1-mini
```

Generate harder style-regenerated negatives:

```bash
.venv/bin/python scripts/build_data_pipeline.py style-regen \
  --passages data/processed/passages.jsonl \
  --out data/processed/triplets_style_regen_small.jsonl \
  --max-authors 8 \
  --max-per-author 2 \
  --source-chars 1200 \
  --guide-examples 3 \
  --guide-chars 900 \
  --guide-tokens 450 \
  --structure-tokens 180 \
  --draft-tokens 260 \
  --model gpt-4.1-mini
```

This creates:

```text
real_passage
structure_summary
style_guide
style_regenerated_draft
negative_type = style_regeneration
```

The embedder dataset automatically uses `style_regenerated_draft` as the
negative when it exists.

Print sampled triplets:

```bash
.venv/bin/python scripts/build_data_pipeline.py sample \
  --triplets data/processed/triplets.jsonl \
  --count 3
```

Next notebooks:

```text
notebooks/04_embedder_forward_pass.ipynb
notebooks/05_embedder_tiny_training.ipynb
notebooks/06_author_style_embeddings.ipynb
notebooks/07_style_llm_forward_pass.ipynb
```

All model training should happen in notebooks. The package contains reusable
model/data code, but the embedder training loop lives in
`05_embedder_tiny_training.ipynb`.

Use `--dry-run` while testing the pipeline without spending API calls:

```bash
.venv/bin/python scripts/build_data_pipeline.py neutral \
  --passages data/processed/passages.jsonl \
  --out data/processed/triplets.jsonl \
  --dry-run
```

The resulting triplets are:

```text
author_id, real_passage, neutral_draft
```

`voice_clone.data.dataset.TripletTextDataset` samples:

```text
anchor   = real passage
positive = different real passage by same author
negative = style_regenerated_draft if present, otherwise neutral_draft
```

The structure/style-guide/regeneration prompt code from the early notebooks now
lives in:

- `voice_clone.structure`
- `voice_clone.style_guide`
- `voice_clone.regeneration`
- `voice_clone.prompts`
