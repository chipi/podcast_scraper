# Grounded Insights Guide

This guide explains **grounded insights**: structured takeaways from podcast episodes that are linked to verbatim quotes as evidence. It covers how to enable the feature, where output is written, how to use the CLI, and how to validate and troubleshoot.

---

## What Are Grounded Insights?

Grounded insights are **key takeaways** extracted from episode content, each with an explicit **grounding status** and optional **supporting quotes** from the transcript.

- **Insight**: A short, clear statement (e.g. "AI regulation will lag behind innovation by several years").
- **Quote**: A **verbatim** span from the transcript used as evidence. Quotes are first-class: they have character offsets, timestamps, and optional speaker attribution.
- **Grounding contract**: Every insight declares `grounded=true` (has ≥1 supporting quote) or `grounded=false` (extracted but no quote linked). This makes it clear which insights are evidence-backed.

This evidence-first design supports trust, quality metrics, and downstream use (e.g. RAG, search, citation).

---

## Enabling Grounded Insights

Grounded insights are produced by the **Grounded Insight Layer (GIL)** pipeline stage. Enable it in config or via CLI:

- **`generate_gi`**: Set to `true` in your config file to run GIL extraction. Default: `false`. You can also pass **`--generate-gi`** on the command line when running the main pipeline (same effect as `generate_gi: true` in config).
- **`gi_insight_source`**: Source of insight texts. One of:
  - **`stub`** (default): Single placeholder insight (original behaviour).
  - **`summary_bullets`**: Use the first N bullets from the episode summary as insight texts (requires `generate_summaries` and summary metadata with bullets).
  - **`provider`**: Call the summarization provider’s optional **`generate_insights(transcript, ...)`** to extract key takeaways. LLM providers that implement it (OpenAI, Ollama, Anthropic, Gemini, Mistral, Grok, DeepSeek) return a list of short statements; ML/hybrid providers return an empty list so GIL falls back to stub.
- **`gi_max_insights`**: Maximum number of insights when using `provider` or `summary_bullets` (default: `5`). Capped in code for safety.

You can set **`--gi-insight-source`** and **`--gi-max-insights`** on the command line to override config.

Providers that support **`generate_insights`** (optional on the summarization protocol) produce real insight text; the pipeline then runs the evidence stack (QA + NLI) per insight to find grounded quotes. If a provider does not implement it or the call fails, GIL falls back to a single stub insight.

**Pipeline order:** The main pipeline runs: scrape → transcribe → (optional) summarization → write metadata. **GIL runs as an optional next step after metadata** for each episode. So you can run a pipeline that “ends with summaries” (and metadata) and treat GI as an optional add-on: same run, no separate stage—when `generate_gi` is true, each episode gets its metadata file written first, then `gi.json` is written alongside it. GI is **after** summaries (and after the metadata file that contains them), not in parallel.

Evidence stack config (used when GIL is enabled) is already available:

- **`embedding_model`**: Model for sentence embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2` or alias `minilm-l6`).
- **`embedding_device`**: Device for embedding model (`cpu`, `cuda`, `mps`, or `null` for auto).
- **`extractive_qa_model`**: Model for extractive question-answering (e.g. `deepset/roberta-base-squad2` or alias `roberta-squad2`).
- **`extractive_qa_device`**: Device for QA model.
- **`nli_model`**: Model for entailment scoring (e.g. `cross-encoder/nli-deberta-v3-base` or alias `nli-deberta-base`).
- **`nli_device`**: Device for NLI model.

Models load **lazily** when GIL (or another feature that uses them) is first used.

### Provider-based evidence (QA + NLI)

GIL can use **either** the local evidence stack (extractive QA + NLI models above) **or** the same provider backends used for summarization (Option B: two separate capabilities per provider).

- **`quote_extraction_provider`**: Provider used for **quote extraction (QA)** — "find a span that supports this insight." Same backends as `summary_provider`: `transformers`, `hybrid_ml`, `openai`, `gemini`, `mistral`, `grok`, `deepseek`, `anthropic`, `ollama`. Default: `transformers` (local extractive QA).
- **`entailment_provider`**: Provider used for **entailment (NLI)** — "score premise (quote) vs hypothesis (insight)." Same backends; default: `transformers` (local NLI model).

When both are set and the workflow passes the corresponding provider instances into GIL, the pipeline calls:

1. **`quote_extraction_provider.extract_quotes(transcript, insight_text)`** → list of candidate spans (char_start, char_end, text, qa_score).
2. For each candidate, **`entailment_provider.score_entailment(premise=quote.text, hypothesis=insight_text)`** → float in [0, 1].
3. Candidates that pass the QA and NLI thresholds become `GroundedQuote` nodes and `SUPPORTED_BY` edges.

If `quote_extraction_provider` or `entailment_provider` is not passed (e.g. older callers), GIL falls back to the **legacy path**: direct use of local `extractive_qa` and `nli_loader` with `gi_qa_model` and `gi_nli_model`. So existing config and behaviour remain valid.

**Resilience and fallback:**

- **On exception:** If the provider path raises (e.g. `extract_quotes` or `score_entailment` fails, or the provider throws), the pipeline catches the exception and falls back to the legacy path for that episode, so the run continues and an artifact is still produced (possibly with no grounded quotes for affected insights).
- **Malformed returns:** Provider return types are validated. If `extract_quotes` returns a non-list (e.g. dict or null), the provider path treats it as no candidates and does not crash. Each candidate must have `char_start`, `char_end`, `text`, and `qa_score`; candidates missing these are skipped. If `score_entailment` fails for a candidate, that candidate is skipped and the rest are still processed. Invalid data never crashes the pipeline.

**Summary:** Default remains local (transformers) for both. You can set `quote_extraction_provider` and `entailment_provider` to any summarization backend (e.g. `openai`) to use that provider’s `extract_quotes` and `score_entailment` implementations; the same provider instance used for summarization is reused when the config matches.

---

## Output Artifact: gi.json

When GIL is enabled, each episode output directory receives a **`gi.json`** file alongside the existing **metadata document** (e.g. `metadata/<episode>.metadata.json`). The metadata document is **extended for consistency**: it includes an optional **`grounded_insights`** section with provenance only (artifact path, insight count, generated_at, schema_version). The full GI graph stays in `gi.json`; the metadata model stays the single place that describes all episode artifacts (summary + GI index). For full insight content and quotes, use `gi.json`; for a quick index and discovery, use the metadata file’s `grounded_insights` field.

The `gi.json` file contains:

- **Schema and provenance**: `schema_version`, `model_version`, `prompt_version`, `episode_id`.
- **Nodes**: Podcast, Episode, Speaker, Topic, **Insight**, **Quote**.
- **Edges**: e.g. `HAS_INSIGHT` (Episode → Insight), `SUPPORTED_BY` (Insight → Quote), `ABOUT` (Insight → Topic).

The file is co-located with the transcript and summary. The logical “full” set of grounded insights across the project is the union of all per-episode `gi.json` files (no global store in v1).

---

## Schema and Validation

- **Ontology**: [GIL Ontology](../gi/ontology.md) — node/edge types, required properties, grounding contract, ID rules.
- **JSON Schema**: [gi.schema.json](../gi/gi.schema.json) — machine-readable validation.

Generated `gi.json` files must conform to the schema. Validation utilities (e.g. in the `gi` package) can be used in tests and CI.

---

## CLI

Available commands (RFC-050):

- **`gi inspect`**: Inspect a single episode’s grounded insights. Use `--episode-path <path>` to point to a `.gi.json` file, or `--output-dir <dir>` and `--episode-id <id>` to locate the artifact. Options: `--format pretty|json`, `--show` (full insight text and quotes), `--stats` / `--no-stats`, `--strict` (strict schema validation).
- **`gi show-insight`**: Show one insight by ID with its quotes and evidence spans. Use `--id INSIGHT_ID` (required). Locate the episode via `--episode-path <path>` or `--output-dir <dir>` (scans for the artifact containing that insight). Options: `--format pretty|json`, `--context-chars N` (transcript context around quotes).
- **`gi explore`**: Cross-episode query. Use `--output-dir <path>` (required). Optional `--topic <label>` to filter by topic (Topic label or substring in insight text). Options: `--limit N`, `--grounded-only`, `--min-confidence`, `--format pretty|json`, `--out <path>`, `--strict`. Exit codes: 0 success, 2 invalid args, 3 no artifacts, 4 no matching insights.

Entrypoint: `podcast_scraper gi inspect --episode-path /path/to/ep.gi.json`, `gi show-insight --id insight:ep:0 --output-dir /path/to/output`, or `gi explore --output-dir /path/to/output [--topic "AI"]`.

---

## Use Cases

- **Topic research**: Find all insights (and evidence) for a topic across episodes.
- **Speaker mapping**: See which speakers said which quotes and which insights they support.
- **Evidence-backed retrieval**: Filter for grounded-only insights; jump from insight to quote to timestamp in the transcript.
- **Insight Explorer**: Browse insights by topic with quotes and episode context (RFC-050).

---

## Troubleshooting

- **No gi.json**: Ensure `generate_gi` is `true` and the pipeline ran past the GIL stage. Check that the episode has a transcript and that the GIL stage did not error (logs).
- **Validation errors**: Run artifact validation against `docs/gi/gi.schema.json`; fix any missing required fields or invalid types (see ontology for rules).
- **Ungrounded insights**: Some insights may be `grounded=false` if no quote passed the evidence thresholds. This is intentional transparency; you can tune QA/NLI thresholds in config when available.
- **Evidence stack / model load**: If embedding, QA, or NLI models fail to load, check device (e.g. `embedding_device`, `nli_device`) and that `sentence-transformers` and `transformers` are installed (e.g. `pip install -e ".[ml]"`).
- **Provider-based evidence**: If you set `quote_extraction_provider` or `entailment_provider` to an LLM (e.g. `openai`), ensure that provider is initialized (same as for summarization) and the required API key is set. LLM providers implement `extract_quotes` and `score_entailment` via their chat API; ML providers use the local QA and NLI models (see `gi_qa_model`, `gi_nli_model`).

---

## Related Documentation

- [GIL Ontology](../gi/ontology.md) — full ontology and grounding contract.
- [GIL Schema](../gi/gi.schema.json) — JSON schema for `gi.json`.
- [PRD-019: Knowledge Graph Layer (KG)](../prd/PRD-019-knowledge-graph-layer.md) — **separate feature** from GIL (`kg` vs `gi`; entities/linking, not evidence-first insights).
- [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) — where GIL fits in the pipeline.
- [Architecture](../ARCHITECTURE.md) — planned GIL extraction and artifact layout.
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) — config keys and provider options.
- [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md) — GIL core concepts and evidence stack; includes implementation note on provider-based QA/NLI (quote_extraction_provider, entailment_provider).
