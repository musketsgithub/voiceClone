"""Prompts shared by notebook demos and data-generation jobs."""

STRUCTURE_PROMPT = """
Analyze the passage and extract:

- core claims
- reasoning order
- paragraph purpose
- causal flow
- emphasis structure

Do NOT preserve wording, cadence, syntax, rhetorical phrasing,
or stylistic realization.

The output should preserve discourse structure while stripping
away author-specific prose realization.

Return a clean structured outline.
""".strip()

STYLE_GUIDE_PROMPT = """
Analyze this author's corpus and produce a generation-oriented style guide.

Capture:
- discourse structure tendencies
- argument flow habits
- pacing
- rhetorical patterns
- paragraph dynamics
- sentence tendencies

Avoid vague aesthetic labels. Describe behaviors that a generation system
could actually follow.
""".strip()

NEUTRAL_DRAFT_PROMPT = """
Rewrite this passage in neutral, clear prose.

Preserve all meaning, factual content, paragraph-level structure, and reasoning
order, but remove idiosyncratic author style. Do not summarize. Do not add new
ideas. Return only the rewritten passage.
""".strip()

REGENERATION_PROMPT = """
You are generating an INTERMEDIATE stylistic draft.

Your task is to reconstruct a passage from:
1. A discourse-level structure representation
2. A detailed style guide describing the target author's writing behavior

The goal is NOT exact imitation.

The goal is:
- preserve semantic and discourse structure
- preserve reasoning progression
- preserve paragraph intent
- partially express the target style
- leave room for later refinement by a downstream model

IMPORTANT:
Do NOT copy phrases mechanically.
Do NOT aggressively imitate wording.
Do NOT produce generic AI prose.

The output should:
- feel structurally aligned to the original passage
- reflect higher-level stylistic tendencies
- preserve conceptual pacing
- preserve rhetorical flow

But it should NOT yet contain:
- full stylistic fidelity
- exact cadence replication
- exact lexical mimicry

Think of this as:
"structurally faithful, partially stylized regeneration."
""".strip()
