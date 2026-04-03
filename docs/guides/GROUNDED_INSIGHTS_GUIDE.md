# Grounded Insights Guide

This guide explains **grounded insights**: structured takeaways from podcast episodes that are linked to verbatim quotes as evidence. It covers how to enable the feature, where output is written, how to use the CLI, and how to validate and troubleshoot.

---

## Summaries, KG, and grounded insights (how they fit together)

Episode **summaries**, the **Knowledge Graph (KG)**, and **grounded insights (GIL)** address different jobs. A practical way to use them together:

| Layer | Role |
| --- | --- |
| **Summaries** ([PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md)) | **Consume** quickly: skim what an episode is about without reading the transcript. |
| **KG** ([Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md)) | **Navigate** a large library: entities, themes, and how episodes connect. |
| **Grounded insights (GIL, this guide)** | **Focus on value and trust**: short takeaways with optional **verbatim quotes** when grounding succeeds. |

Summaries compress nuance and are not a full substitute for primary sources when stakes are high. **GI** adds **traceability to the transcript** where grounding works. **KG** does not replace summaries or GI; it helps you **find** where to read or drill down (see the same stack from the KG side in [Knowledge Graph Guide § How KG fits with summaries and grounded insights](KNOWLEDGE_GRAPH_GUIDE.md#how-kg-fits-with-summaries-and-grounded-insights)).

---

## What Are Grounded Insights?

Grounded insights are **key takeaways** extracted from episode content, each with an explicit **grounding status** and optional **supporting quotes** from the transcript.

- **Insight**: A short, clear statement (e.g. "AI regulation will lag behind innovation by several years").
- **Quote**: A **verbatim** span from the transcript used as evidence. Quotes are first-class: they have character offsets, timestamps, and optional **speaker attribution** (`speaker_id`). When **`.segments.json`** exists next to the transcript and segments include **`speaker`** or **`speaker_id`** (e.g. diarized pipelines), GIL fills **`speaker_id`** from the segment overlapping the quote span; otherwise it is **`null`** (see ontology). **UC2** / **`gi explore --speaker`** are **best effort** when diarization is missing—see [Recorded product decisions (v1, issue 460)](#recorded-product-decisions-v1-issue-460) below.
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
- **`gi_max_insights`**: Maximum number of insights when using `provider` or `summary_bullets` (default: `20`, range `1`–`50`). Capped in code for safety.

You can set **`--gi-insight-source`** and **`--gi-max-insights`** on the command line to override config.

### ML summarization and GIL insight wording

If **`summary_provider`** is **`transformers`** or **`hybrid_ml`**, treat **`generate_insights`** as **unsupported** for meaningful free-form takeaways: those providers do not populate insight wording via the LLM path. For **real insight text** with ML summaries, set **`gi_insight_source: summary_bullets`**, enable **`generate_summaries`**, and ensure the summary pipeline produces **bullets** (your map/reduce or provider config must emit them). Alternatively, switch **`summary_provider`** to an **LLM** and use **`gi_insight_source: provider`** (or still use **`summary_bullets`** if you prefer bullets as the insight list).

The default **`gi_insight_source: stub`** is appropriate for **tests and smoke runs** only. When **`generate_gi`** is on and **`gi_insight_source`** is **`stub`**, the CLI logs a **warning** (outside pytest) so production configs are less likely to ship with placeholder insights by accident.

### ML bullets, `gi_require_grounding`, and local NLI

With **`summary_provider: transformers`** (or **`hybrid_ml`**) and **`gi_insight_source: summary_bullets`**, insight text comes from **map/reduce (or heuristic) bullets**, not a JSON bullet schema. **`gi_require_grounding: true`** is still **best-effort**: grounded quotes need extractive QA + **NLI** scores above configured thresholds on **noisy transcripts** (e.g. Whisper). If you see **zero grounded quotes**, check logs for evidence-step failures; local **CrossEncoder** NLI models return **multi-class logits**—the pipeline maps them to **P(entailment)** using the model’s **`id2label`**. For stricter or cleaner bullets, prefer an **LLM `summary_provider`** or relax thresholds / disable **`gi_require_grounding`** for exploratory ML runs.

### Topic nodes, `ABOUT` edges, and `gi explore --topic`

When **`<output_dir>/search/vectors.faiss`** exists (semantic corpus index, RFC-061),
**`gi explore --topic`** ranks **Insight** rows by embedding similarity to the topic
string first; if the index is missing, errors, or yields no hits after filters, it falls
back to the behavior below. See [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md).

The GIL **ontology** includes **Topic** nodes and **ABOUT** (Insight → Topic) edges. **Current pipeline output** does not yet emit those nodes/edges; artifacts are **Episode, Insight, Quote** plus **SUPPORTED_BY**. Therefore **`gi explore --topic`** matches **(1)** Topic labels **if** present in a hand-edited or future-enriched artifact, and **(2)** always **substring match on insight text** (case-insensitive). Until Topic/ABOUT are produced automatically, treat **`--topic`** as **“search insights (and optional Topic labels)”**, not as a dedicated topic-model facet. A follow-up change may add Topic extraction to `gi.json` without changing this filter semantics. The formal v1 milestone wording for this row is in **§ Recorded product decisions (v1, issue 460)** below.

## Recorded product decisions (v1, issue 460) {#recorded-product-decisions-v1-issue-460}

[GitHub #460](https://github.com/chipi/podcast_scraper/issues/460) tracks **closing ambiguity** for shallow v1—not implementing every GIL enhancement in one release. This table is the **operator-facing record** of what v1 promises.

| Decision area | v1 choice |
| --- | --- |
| **Insight wording + ML** | **`transformers`** / **`hybrid_ml`** do not supply meaningful free-form text via `generate_insights`. For real insight wording with ML summaries, use **`gi_insight_source: summary_bullets`** with **`generate_summaries`** and summary **bullets**, or switch to an **LLM** `summary_provider` and use **`provider`** or **`summary_bullets`**. **`stub`** is for **tests and smoke** only. The CLI emits a **warning** when **`generate_gi`** and **`stub`** run together outside pytest. Details: [ML summarization and GIL insight wording](#ml-summarization-and-gil-insight-wording). |
| **Topic nodes + `gi explore --topic`** | The pipeline does **not** yet emit **Topic** nodes and **`ABOUT`** edges automatically. **`gi explore --topic`** matches **substring on insight text** and any Topic labels already in the artifact. Richer topic graphing is **post–v1** (see [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466)). |
| **Speaker / UC2** | **`speaker_id`** on quotes comes from overlapping **segments** when **`speaker` / `speaker_id`** exists; otherwise **`null`**. **`gi explore --speaker`** and **`gi query`** speaker phrasing match **graph-backed speaker names** or **`speaker_id`** substrings when the graph has that data—**expect weak or empty results** if the run had no speaker signals. |
| **Scale / SQL** | Consumption stays **file-based** (`gi export`, `gi explore`, `gi query`). **Postgres** projection ([PRD-018](../prd/PRD-018-database-projection-gil-kg.md), [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)) is **separate** work—track with a dedicated issue if none exists. |
| **Consumption CLI** | **`gi explore`** / **`gi query`** implement RFC-050-style use cases with **deterministic pattern mapping**, not open-ended LLM question answering. Scope: [GitHub #439](https://github.com/chipi/podcast_scraper/issues/439). |

**KG (same release):** If you also enable **`generate_kg`**, see [Knowledge Graph Guide § Recorded product decisions (v1, KG shallow)](KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg) for extraction modes, CLI scope, and shared **Postgres** deferral.

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

GIL grounding always goes through **`find_grounded_quotes_via_providers`** in `gi/pipeline.py`: for each insight it calls **`extract_quotes`** then **`score_entailment`** on the configured backends. **`quote_extraction_provider`** and **`entailment_provider`** (config enums, same set as `summary_provider`: `transformers`, `hybrid_ml`, `openai`, `gemini`, `mistral`, `grok`, `deepseek`, `anthropic`, `ollama`) choose **which** implementations run. Defaults **`transformers`** use local extractive QA and NLI (aligned with `gi_qa_model` / `gi_nli_model` and the embedding/QA/NLI settings above); LLM backends use chat-style quote and entailment calls.

- **`quote_extraction_provider`**: Backend for **quote extraction (QA)** — find a span that supports the insight.
- **`entailment_provider`**: Backend for **entailment (NLI)** — score premise (quote) vs hypothesis (insight).

The workflow usually passes provider instances into **`build_artifact`**. If either instance is omitted, **`create_gil_evidence_providers(cfg, summary_provider=...)`** in **`gi/deps.py`** constructs clients from config and reuses **`summary_provider`** when the configured types match (so callers do not need to wire four objects by hand).

For each insight the pipeline then:

1. **`extract_quotes(transcript, insight_text)`** → list of candidate spans (char_start, char_end, text, qa_score).
2. For each candidate, **`score_entailment(premise=quote.text, hypothesis=insight_text)`** → float in [0, 1].
3. Candidates that pass the QA and NLI thresholds become `GroundedQuote` nodes and `SUPPORTED_BY` edges.

**Tests and advanced use:** **`find_grounded_quotes`** in `gi/grounding.py` calls local QA/NLI loaders directly without a provider wrapper; it is **not** used inside **`build_artifact`**.

**Quote spans (LLM extractors):** The model returns `quote_text` JSON; the pipeline maps it to **`char_start` / `char_end`** in the transcript with `resolve_llm_quote_span` (`gi/grounding.py`): exact match, apostrophe normalization (`'` vs `’`), whitespace-tolerant token regex, and longest contiguous sub-phrase when the model drops a leading word (e.g. `And`). If nothing matches, that candidate is dropped (no fake offsets). Among equal-length regex matches, the **earliest** occurrence in the transcript wins.

**Retries:** Config **`gi_evidence_extract_retries`** (default `1`) adds extra **`extract_quotes`** calls when the first attempt yields no list of candidates. Later attempts append a short “verbatim substring” reminder to the insight text passed to the extractor only; **NLI** still uses the original insight string.

**Thresholds (config):** **`gi_qa_score_min`** (default `0.3`) and **`gi_nli_entailment_min`** (default `0.5`) gate candidates before and after NLI. Lower them slightly for noisy ML **`summary_bullets`** + Whisper transcripts if you want more grounded quotes at the cost of precision.

**Windowed local QA (long transcripts):** When **`gi_qa_window_chars`** &gt; `0` and the transcript is longer, local extractive QA runs on **overlapping windows** of that size and keeps the **best-scoring** span (then NLI as usual). **`gi_qa_window_overlap_chars`** controls overlap (must be **&lt;** the window size). Default windowing is **on** (`1800` / `300`); set **`gi_qa_window_chars: 0`** to restore a **single** QA pass over the full transcript (often weak on very long episodes).

**Resilience and degraded artifacts:**

- **Strict grounding (`GILGroundingUnsatisfiedError`):** When **`gi_require_grounding`** is true and the stack produced **zero** grounded quotes while there are insights to ground, the pipeline logs a warning and sets **`gi_grounding_degraded`** on metrics when present. If **`gi_fail_on_missing_grounding`** is true, it **raises** `GILGroundingUnsatisfiedError` and the episode fails.
- **Provider failures:** **`GILGroundingUnsatisfiedError`** is always **re-raised** if it escapes the evidence stack. Any **other** exception from the stack is caught, logged at debug, and the pipeline emits a **stub** or **multi-insight artifact with empty quote lists** (insights ungrounded) so the run can continue when strict mode is off.
- **Malformed returns:** Provider return types are validated. If `extract_quotes` returns a non-list (e.g. dict or null), that pass yields no candidates and does not crash. Each candidate must have `char_start`, `char_end`, `text`, and `qa_score`; candidates missing these are skipped. If `score_entailment` fails for a candidate, that candidate is skipped and the rest are still processed.

**Defaults vs API summaries:** Raw defaults for **`quote_extraction_provider`** and **`entailment_provider`** are **`transformers`** (local extractive QA + CrossEncoder NLI; install **`.[ml]`** / **`sentence-transformers`**). When **`gil_evidence_match_summary_provider`** is **`true`** (default) and **`generate_gi`** is on, **`Config`** rewrites both fields from **`transformers`** to **`summary_provider`** if **`summary_provider`** is an API LLM (**`openai`**, **`gemini`**, **`anthropic`**, **`mistral`**, **`deepseek`**, **`grok`**, **`ollama`**) or **`hybrid_ml`** — so a typical “OpenAI + summary bullets” run uses **OpenAI for grounding** without extra keys. Set **`gil_evidence_match_summary_provider: false`** to keep **local** evidence with API summaries (intentional hybrid). If you override only one evidence field to an API and leave the other on **`transformers`**, the CLI logs a **WARNING** about local NLI deps.

---

## GIL evidence: capability × provider matrix {#gil-evidence-provider-matrix}

Each **`summary_provider`** / evidence backend implements **`extract_quotes`** and **`score_entailment`** for GIL (see `gi/grounding.py`, provider modules). **`generate_insights`** is separate: used only when **`gi_insight_source: provider`**.

| Backend | `extract_quotes` | `score_entailment` | `generate_insights` (insight wording) | Notes |
| --- | --- | --- | --- | --- |
| **`transformers`** | Yes (local QA) | Yes (CrossEncoder NLI) | No (returns `[]`) | Requires **`.[ml]`**; `gi_qa_model` / `gi_nli_model`. |
| **`hybrid_ml`** | Yes | Yes | No | Same local evidence stack; map/reduce summaries. |
| **`openai`** | Yes (LLM JSON span) | Yes (LLM score) | Yes | API key; optional **`openai_insight_model`** for insights only. |
| **`gemini`** | Yes | Yes | Yes | `GEMINI_API_KEY`. |
| **`anthropic`** | Yes | Yes | Yes | `ANTHROPIC_API_KEY`. |
| **`mistral`** | Yes | Yes | Yes | `MISTRAL_API_KEY`. |
| **`deepseek`** | Yes | Yes | Yes | `DEEPSEEK_API_KEY`. |
| **`grok`** | Yes | Yes | Yes | `GROK_API_KEY`. |
| **`ollama`** | Yes | Yes | Yes | Local Ollama server. |

Mixing backends (e.g. OpenAI summaries + local entailment) is **supported**; use **`gil_evidence_match_summary_provider`** and explicit **`quote_extraction_provider`** / **`entailment_provider`** to make the choice deliberate. See [CONFIGURATION — GIL evidence providers](../api/CONFIGURATION.md#grounded-insights-gil-evidence-providers).

---

## Output Artifact: gi.json

When GIL is enabled, each episode output directory receives a **`gi.json`** file alongside the existing **metadata document** (e.g. `metadata/<episode>.metadata.json`). The metadata document is **extended for consistency**: it includes an optional **`grounded_insights`** section with provenance only (artifact path, insight count, generated_at, schema_version). The full GI graph stays in `gi.json`; the metadata model stays the single place that describes all episode artifacts (summary + GI index). For full insight content and quotes, use `gi.json`; for a quick index and discovery, use the metadata file’s `grounded_insights` field.

The `gi.json` file contains:

- **Schema and provenance**: `schema_version`, `model_version`, `prompt_version`, `episode_id`.
- **Nodes**: Podcast, Episode, Speaker, Topic, **Insight**, **Quote**.
- **Edges**: e.g. `HAS_INSIGHT` (Episode → Insight), `SUPPORTED_BY` (Insight → Quote), `ABOUT` (Insight → Topic).

### Insight text provenance in `gi.json` (`model_version`)

There is **no** separate `gi_insight_model` config field. Top-level **`model_version`** records **which model lineage produced the insight strings** for that artifact, derived at build time by `podcast_scraper.gi.provenance.resolve_gil_artifact_model_version`:

- **`gi_insight_source: stub`** → `"stub"`.
- **`summary_bullets`** → the summarization model (from the live `summary_provider.summary_model` when present, otherwise from `Config` — e.g. `summary_model` / `openai_summary_model` per provider).
- **`provider`** → the model used for **`generate_insights`**. For OpenAI, that is **`openai_insight_model`** when set, otherwise **`openai_summary_model`** (same instance exposes `insight_model` on the provider after init).

Use this field for audits and metrics, not a second hand-maintained model id in YAML.

The file is co-located with the transcript and summary. The logical “full” set of grounded insights across the project is the union of all per-episode `gi.json` files (no global store in v1).

---

## Schema and Validation

- **Ontology**: [GIL Ontology](../gi/ontology.md) — node/edge types, required properties, grounding contract, ID rules.
- **JSON Schema**: [gi.schema.json](../gi/gi.schema.json) — machine-readable validation.

Generated `gi.json` files must conform to the schema. Validation utilities (e.g. in the `gi` package) can be used in tests and CI.

- **CLI**: `podcast_scraper gi validate <paths…> [--strict]` — same idea as `kg validate` (see [CLI API](../api/CLI.md)).
- **Makefile / script**: `make validate-gi-schema [ARTIFACTS_DIR=path]` or `scripts/tools/validate_gi_schema.py` (always strict) for batch checks in CI.
- **PRD-017 metrics**: `make gil-quality-metrics DIR=<run_root>` or `scripts/tools/gil_quality_metrics.py` — aggregation over `.gi.json` (grounding rate, quote span/timestamp validity, per-episode density). Use `--enforce` and `--min-*` flags to gate releases.

**Two views of “quote validity”:** The **file-based** quality script checks **schema** (spans, `transcript_ref`, timestamps) and does not load transcript files. During a **pipeline run**, `Metrics` / `record_gi_success_counts` can also compute **verbatim** agreement between quote text and the transcript slice when the transcript is available. Use the same view when comparing numbers; they are related but not identical.

### ML vs LLM evidence — outcome benchmark (v1)

For **comparing outcomes** on the **same episodes** (not YAML threshold parity), use a
fixed episode set and two runs — e.g. local transformers evidence vs OpenAI
`extract_quotes` / `score_entailment` with the same `summary_provider` and
`gi_insight_source`. Pair configs live under `config/manual/gil_paired_benchmark_*.yaml`.
After runs, compare with **`make compare-gil-runs REF=<ref_run_root> CAND=<cand_run_root>`**
or **`python scripts/tools/compare_gil_runs.py <ref_run_root> <cand_run_root>`** — per-episode
quote/grounded counts and a short agreement summary.

Methodology and limits: [WIP: GIL ML vs OpenAI outcome benchmark](../wip/gil-ml-vs-openai-outcome-benchmark.md).

---

## CLI

Available commands (RFC-050):

- **`gi validate`**: Validate one or more `.gi.json` files or directories (recursive). Options: `--strict` (full JSON Schema), `-q` / `--quiet` (only failures). Exit **3** if no artifacts are found; **1** if any file fails validation (aligned with `kg validate`).
- **`gi export`**: Export all `*.gi.json` under `--output-dir` as **`--format ndjson`** (default) or **`merged`** (single `gi_corpus_bundle` JSON). Options: `--out PATH` or stdout, `--strict` (fail on schema errors). Mirrors **`kg export`** (RFC-056 symmetry).
- **`gi inspect`**: Inspect a single episode’s grounded insights. Use `--episode-path <path>` to point to a `.gi.json` file, or `--output-dir <dir>` and `--episode-id <id>` to locate the artifact. Options: `--format pretty|json`, `--show` (full insight text and quotes), `--stats` / `--no-stats`, `--strict` (strict schema validation).
- **`gi show-insight`**: Show one insight by ID with its quotes and evidence spans. Use `--id INSIGHT_ID` (required). Locate the episode via `--episode-path <path>` or `--output-dir <dir>` (scans for the artifact containing that insight). Options: `--format pretty|json`, `--context-chars N` (transcript context around quotes).
- **`gi explore`**: Cross-episode query (RFC-050 UC5). Use `--output-dir <path>` (required). Optional `--topic <label>` (Topic label or substring in insight text) and `--speaker <substring>` (substring match on quote **`speaker_id`** from diarization **or** on graph-backed **`speaker_name`** resolved from **SPOKEN_BY → Speaker** when the quote has no id). Options: `--limit N`, `--grounded-only`, `--min-confidence`, `--sort confidence|time`, `--format pretty|json`, `--out <path>`, `--strict`. JSON output follows the RFC-050 Insight Explorer shape (`topic` object when `--topic` is set, nested `episode` and `supporting_quotes`, `top_speakers`). Exit codes: 0 success, 2 invalid args, 3 no artifacts / bad output dir, 4 no matching insights when `--topic` or `--speaker` is set, 5 `--strict` validation failed on an artifact.
- **`gi query`**: RFC-050 UC4 — maps a **fixed set of English question patterns** (not open-ended NL interpretation) to **`gi explore`** or a **topic leaderboard**, then prints an envelope (`question`, `answer`, `explanation`). Use `--output-dir <path>` and `--question "<text>"` (required). Optional `--limit N`, `--format pretty|json` (default **json**), `--strict`. Supported patterns include **topic** phrasing (`What insights about …?`, `What insights are there about …?`, `What are insights about …?`, `Insights about …`, `Show me insights about …`, `Tell me about insights on …`), **speaker** phrasing (`What did <name> say?`), **combined** speaker + topic (`What did <name> say about <topic>?`), and **topic ranking** (`Which topics have the most insights?`, `Top topics`, etc.). Leaderboard answers use `answer.topics` (insight counts per topic label) instead of the explore `insights` list. Exit codes: same as explore for load/validation, plus **2** when the question **does not match** any supported pattern (or required args are missing). See [Recorded product decisions (v1, issue 460)](#recorded-product-decisions-v1-issue-460).

Entrypoint: `podcast_scraper gi validate ./output/metadata --strict`, `gi export --output-dir /path/to/output --format ndjson`, `gi inspect --episode-path /path/to/ep.gi.json`, `gi show-insight --id insight:<id-from-gi.json> --output-dir /path/to/output`, `gi explore --output-dir /path/to/output [--topic "AI"] [--speaker "HOST"]`, or `gi query --output-dir /path/to/output --question 'What insights about inflation?'`.

### Browser visualization (prototype)

To explore **`gi.json`** (and optionally **`kg.json`**) as an interactive graph — filters,
metrics, **Chart.js** bars, and either **vis-network** or **Cytoscape.js** — run
`make serve-gi-kg-viz` from the repo root and open `http://127.0.0.1:8765/`. Prefer this
over opening the HTML files as `file://`, so browser CDNs load reliably.

Full usage and layout of the static app: [Development Guide — GI / KG browser
viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) and
[`web/gi-kg-viz/README.md`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viz/README.md).

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

- [Recorded product decisions (v1, issue 460)](#recorded-product-decisions-v1-issue-460) — shallow v1 scope table (ML, topics, speakers, SQL deferral, CLI).
- [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md) — summaries as the fast consumption layer above transcripts.
- [Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md) — KG as a separate navigation layer (`kg` vs `gi`).
- [GIL Ontology](../gi/ontology.md) — full ontology and grounding contract.
- [GIL Schema](../gi/gi.schema.json) — JSON schema for `gi.json`.
- [PRD-019: Knowledge Graph Layer (KG)](../prd/PRD-019-knowledge-graph-layer.md) — **separate feature** from GIL (`kg` vs `gi`; entities/linking, not evidence-first insights).
- [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) — where GIL fits in the pipeline.
- [Development Guide — GI / KG browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) — optional local UI for `gi.json` / `kg.json`.
- [Architecture](../ARCHITECTURE.md) — GIL extraction and artifact layout.
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) — config keys and provider options.
- [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md) — GIL core concepts and evidence stack; includes implementation note on provider-based QA/NLI (quote_extraction_provider, entailment_provider).
