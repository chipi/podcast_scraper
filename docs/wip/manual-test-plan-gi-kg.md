# Manual test plan: Grounded Insights (GI) and Knowledge Graph (KG)

**Purpose:** Close-the-loop validation on a feed you know well: run the real pipeline on
2–3 episodes, then inspect `*.gi.json` and `*.kg.json` for plausibility, schema health,
and CLI ergonomics.

**Use this doc as a hub:** each section points to deeper guides so you can read **why**
something works while you **run** the steps.

---

## Documentation map (learn while you test)

| Topic | Where to read |
| --- | --- |
| **How GI, KG, and summaries relate** | [Grounded Insights Guide — Summaries, KG, and grounded insights](../guides/GROUNDED_INSIGHTS_GUIDE.md#summaries-kg-and-grounded-insights-how-they-fit-together); [Knowledge Graph Guide — How KG fits with summaries and grounded insights](../guides/KNOWLEDGE_GRAPH_GUIDE.md#how-kg-fits-with-summaries-and-grounded-insights) |
| **Shallow v1 scope (what ships vs deferred)** | GIL: [Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460); KG: [Recorded product decisions (v1, KG shallow)](../guides/KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg); Depth roadmap: [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) |
| **Product intent** | [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md); [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md); [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md) (baseline for bullets and consumption) |
| **Design / use cases** | [RFC-049: GIL core](../rfc/RFC-049-grounded-insight-layer-core.md); [RFC-050: GIL use cases](../rfc/RFC-050-grounded-insight-layer-use-cases.md); [RFC-055: KG core](../rfc/RFC-055-knowledge-graph-layer-core.md); [RFC-056: KG use cases](../rfc/RFC-056-knowledge-graph-layer-use-cases.md) |
| **Enable flags and config tables** | [CONFIGURATION.md — Knowledge Graph (KG)](../api/CONFIGURATION.md#knowledge-graph-kg); [CONFIGURATION.md — Grounded Insights (GIL)](../api/CONFIGURATION.md#grounded-insights-gil); [CONFIGURATION.md — GIL evidence providers](../api/CONFIGURATION.md#grounded-insights-gil-evidence-providers) |
| **GIL evidence × provider (matrix, API vs local)** | [GROUNDED_INSIGHTS_GUIDE — GIL evidence provider matrix](../guides/GROUNDED_INSIGHTS_GUIDE.md#gil-evidence-provider-matrix) |
| **CLI examples** | [CLI — Grounded insights (`gi`) subcommands](../api/CLI.md#grounded-insights-gi-subcommands); [CLI — Knowledge Graph (`kg`) subcommands](../api/CLI.md#knowledge-graph-kg-subcommands) |
| **Pipeline placement** | [Grounded Insights Guide — Pipeline order](../guides/GROUNDED_INSIGHTS_GUIDE.md#enabling-grounded-insights) (GIL after metadata); [Knowledge Graph Guide — Enabling KG](../guides/KNOWLEDGE_GRAPH_GUIDE.md#enabling-kg); [Pipeline and Workflow Guide](../guides/PIPELINE_AND_WORKFLOW.md) |
| **Artifacts and shapes** | [Grounded Insights Guide — Output artifact: gi.json](../guides/GROUNDED_INSIGHTS_GUIDE.md#output-artifact-gijson); [Knowledge Graph Guide — Output artifacts](../guides/KNOWLEDGE_GRAPH_GUIDE.md#output-artifacts) |
| **Ontology and JSON Schema** | [GIL ontology](../architecture/gi/ontology.md); [gi.schema.json](../architecture/gi/gi.schema.json); [KG ontology](../architecture/kg/ontology.md); [kg.schema.json](../architecture/kg/kg.schema.json) |
| **Provider / API setup** | [Provider configuration quick reference](../guides/PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) |
| **Automated test context (optional)** | [TESTING_STRATEGY.md](../architecture/TESTING_STRATEGY.md); acceptance index at repo root: `config/acceptance/README.md` |
| **Transcript-only GIL/KG eval (stub, `data/eval`)** | `data/eval/configs/README.md` (repo root); sample YAML: `gil_eval_stub_curated_5feeds_smoke_v1.yaml`, `kg_eval_stub_curated_5feeds_smoke_v1.yaml`; [EXPERIMENT_GUIDE — GIL/KG experiments](../guides/EXPERIMENT_GUIDE.md#grounded-insights-gil-and-knowledge-graph-kg-experiments) |
| **Manual GI+KG configs (this workflow)** | Index: `config/manual/README.md` at repo root — NPR Planet Money `510289`, OpenAI-first presets |

---

## Ready-made configs: NPR feeds and OpenAI-first

Feeds match **GI/KG acceptance**: NPR Planet Money
`https://feeds.npr.org/510289/podcast.xml` (same as
`config/acceptance/full/acceptance_planet_money_*.yaml`).

**Index and file list:** `config/manual/README.md` (repository root).

| Step | What you exercise | Config file (repo root) | `output_dir` in YAML (use as `$OUT` below) |
| --- | --- | --- | --- |
| **A (optional)** | Summaries + metadata only; no GI/KG | `config/manual/manual_planet_money_openai_summaries_only.yaml` | `.test_outputs/manual/planet_money_openai_summaries_only` |
| **B (recommended)** | OpenAI stack; **GI + KG from summary bullets** (fewer LLM calls than `provider`) | `config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml` | `.test_outputs/manual/planet_money_openai_gi_kg_summary_bullets` |
| **C** | OpenAI stack; **GI + KG via `provider`** (`generate_insights`, `extract_kg_graph`) | `config/manual/manual_planet_money_openai_gi_kg_provider.yaml` | `.test_outputs/manual/planet_money_openai_gi_kg_provider` |
| **D (optional)** | **Local ML** summaries; GI+KG bullets (no API keys) | `config/manual/manual_planet_money_ml_gi_kg_summary_bullets.yaml` | `.test_outputs/manual/planet_money_ml_gi_kg_summary_bullets` |

**Run (direct CLI — output goes to the YAML `output_dir`):**

```bash
python -m podcast_scraper.cli --config config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml
```

**Run (acceptance runner — session layout under `.test_outputs/acceptance/` unless you pass `OUTPUT_DIR`):**

```bash
make test-acceptance CONFIGS="config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml"
```

OpenAI steps need **`OPENAI_API_KEY`** (see `examples/.env.example`).

**Step B (OpenAI + summary bullets GI/KG):** With **`gil_evidence_match_summary_provider: true`** (default),
`Config` aligns **`quote_extraction_provider`** and **`entailment_provider`** to **`summary_provider`**
when they would otherwise stay **`transformers`**, so this preset uses **OpenAI for GIL grounding**
and does **not** require **`.[ml]`** for NLI. Set **`gil_evidence_match_summary_provider: false`**
if you want **local** HF QA/NLI with API summaries (install **`pip install -e ".[ml]"`**). See
[GROUNDED_INSIGHTS_GUIDE — GIL evidence provider matrix](../guides/GROUNDED_INSIGHTS_GUIDE.md#gil-evidence-provider-matrix)
and [CONFIGURATION.md — GIL evidence providers](../api/CONFIGURATION.md#grounded-insights-gil-evidence-providers).

For ML step **D**, set `whisper_device` / `summary_device` to **`cpu`** in the YAML if you are not
on Apple Silicon or MPS is unavailable.

### Acceptance configs (full pipeline, CI-style)

Same NPR / WSJ feeds and provider matrix as automation; each YAML runs **summaries + GI +
KG + semantic index** (`summary_bullets` for GI and KG — not stub-only splits). See
`config/acceptance/README.md` at the repository root.

| Use case | Example config |
| --- | --- |
| Full pipeline, OpenAI stack | `config/acceptance/full/acceptance_planet_money_openai.yaml` |
| Full pipeline, local ML (dev / faster models) | `config/acceptance/full/acceptance_planet_money_ml_dev.yaml` |
| Full pipeline, local ML (prod-style models) | `config/acceptance/full/acceptance_planet_money_ml_prod.yaml` |
| Full pipeline, another cloud provider | `config/acceptance/full/acceptance_planet_money_<provider>.yaml` (e.g. `gemini`, `anthropic`) |

For **The Journal** (WSJ feed), mirror with `config/acceptance/full/acceptance_the_journal_*.yaml`.

For **stub** GI, **`provider`**-mode KG, summaries-only, or other layer-specific presets,
use **`config/manual/`** (see [Ready-made configs](#ready-made-configs-npr-feeds-and-openai-first)) or pytest E2E tests.

**Runner reference:** `scripts/acceptance/README.md` (repository root).

---

## 0. Decide your “truth source”

Pick **one RSS URL** you have actually listened to (or can skim). You will judge:

- Do **summary bullets** and **KG topics/entities** match what you remember?  
  → See [PRD-005](../prd/PRD-005-episode-summarization.md) and [how summaries relate to GI/KG](../guides/GROUNDED_INSIGHTS_GUIDE.md#summaries-kg-and-grounded-insights-how-they-fit-together).
- Do **GI insights** read like real takeaways (not stub text)?  
  → See [enabling grounded insights](../guides/GROUNDED_INSIGHTS_GUIDE.md#enabling-grounded-insights) and [`gi_insight_source`](../api/CONFIGURATION.md#grounded-insights-gil).
- For **grounded** insights, do **quotes** appear in the transcript and support the claim?  
  → See [grounding contract / ontology](../architecture/gi/ontology.md) and [provider-based evidence (QA + NLI)](../guides/GROUNDED_INSIGHTS_GUIDE.md#provider-based-evidence-qa-nli).

Write down the episode titles or IDs you will use so you can find the right files under
`metadata/`.

---

## 1. Choose extraction stack (once per run)

Prefer picking a **row in the table above** (`config/manual/*.yaml`) instead of hand-rolling
flags. Mapping:

| Your goal | Config |
| --- | --- |
| **OpenAI + meaningful GI/KG with minimal extra LLM surface** | Step **B** — `manual_planet_money_openai_gi_kg_summary_bullets.yaml` |
| **OpenAI + full LLM insight + graph extraction** | Step **C** — `manual_planet_money_openai_gi_kg_provider.yaml` |
| **No API keys; local ML** | Step **D** — `manual_planet_money_ml_gi_kg_summary_bullets.yaml` |

**Read alongside:** [ML summarization and GIL insight wording](../guides/GROUNDED_INSIGHTS_GUIDE.md#ml-summarization-and-gil-insight-wording); [KG extraction modes](../guides/KNOWLEDGE_GRAPH_GUIDE.md#enabling-kg) (`stub` / `summary_bullets` / `provider`).

**Gotchas (from guides):**

- **`gi_insight_source: stub`** — placeholder insights; fine for CI-style smoke, not for
  your “does GI make sense?” pass. Details: [Enabling grounded insights](../guides/GROUNDED_INSIGHTS_GUIDE.md#enabling-grounded-insights). Acceptance **`full/`** configs use **`summary_bullets`** for GI; use **`config/manual/`** or pytest E2E for stub or other modes.
- **`kg_extraction_source: provider`** with **ML-only** summary provider — graph extraction
  is a no-op; pipeline falls back to **summary bullets** when available. Details:
  [KG guide extraction table](../guides/KNOWLEDGE_GRAPH_GUIDE.md#enabling-kg) and [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md).
- **GIL** needs **`generate_metadata`** (and summaries if you use bullet-backed modes).  
  Config reference: [Grounded Insights (GIL)](../api/CONFIGURATION.md#grounded-insights-gil).

---

## 2. Run the pipeline (2–3 episodes)

1. Pick a config from [Ready-made configs](#ready-made-configs-npr-feeds-and-openai-first).
2. Run **`python -m podcast_scraper.cli --config <path>`** (or `make test-acceptance` as in
   that section).
3. Set **`OUT`** to that YAML’s **`output_dir`** value. Artifacts live under
   **`$OUT/metadata/`**.

**Read alongside:** [CLI cost projection / dry-run](../api/CLI.md#cost-projection-in-dry-run-mode) (append `--dry-run` on the CLI if your merge mode allows it); [Pipeline and Workflow Guide](../guides/PIPELINE_AND_WORKFLOW.md).

**Ad-hoc CLI** (if you are not using a preset file):

```bash
python -m podcast_scraper.cli 'https://feeds.npr.org/510289/podcast.xml' \
  --output-dir ./manual-gi-kg-run \
  --max-episodes 3 \
  --generate-metadata \
  --generate-summaries \
  --summary-provider openai \
  --generate-gi \
  --gi-insight-source provider \
  --gi-max-insights 5 \
  --generate-kg \
  --kg-extraction-source provider
```

If you already have transcripts and use **`--skip-existing`**, confirm GI/KG still emit or
re-run with intent (re-processing policy is your choice).

**Custom YAML:** mirror keys from [CONFIGURATION.md](../api/CONFIGURATION.md#knowledge-graph-kg) and the GIL sections; copy patterns from `config/manual/` or `config/acceptance/`.

---

## 3. Locate artifacts

Under your run’s **`output_dir`** (call it **`$OUT`**; e.g.
`.test_outputs/manual/planet_money_openai_gi_kg_summary_bullets` for step **B**):

- **`metadata/<episode_basename>.metadata.json`** — summary bullets, pipeline provenance;
  may include [`grounded_insights`](../guides/GROUNDED_INSIGHTS_GUIDE.md#output-artifact-gijson)
  and [`knowledge_graph`](../guides/KNOWLEDGE_GRAPH_GUIDE.md#output-artifacts) index fields.
- **`metadata/<episode_basename>.gi.json`** — insights, quotes, grounding — see
  [Output artifact: gi.json](../guides/GROUNDED_INSIGHTS_GUIDE.md#output-artifact-gijson).
- **`metadata/<episode_basename>.kg.json`** — entities, topics, edges — see
  [Output artifacts (KG)](../guides/KNOWLEDGE_GRAPH_GUIDE.md#output-artifacts).

Open **one episode** in an editor or `jq` and skim top-level keys and counts.

---

## 4. GI checklist (per episode)

Set **`OUT`** to the same directory as in §2 (parent of `metadata/`). Example for step **B**:

```bash
OUT=.test_outputs/manual/planet_money_openai_gi_kg_summary_bullets
```

**Schema / tooling**

```bash
python -m podcast_scraper.cli gi validate --strict "$OUT/metadata"
```

**Read alongside:** [Schema and Validation (GIL)](../guides/GROUNDED_INSIGHTS_GUIDE.md#schema-and-validation); [RFC-050 explorer / query behavior](../rfc/RFC-050-grounded-insight-layer-use-cases.md).

**Single-episode narrative**

```bash
python -m podcast_scraper.cli gi inspect --output-dir "$OUT" --episode-id '<id>'
```

Pick one **`insight_id`** from the artifact and:

```bash
python -m podcast_scraper.cli gi show-insight --output-dir "$OUT" --id '<insight_id>'
```

(`show-insight` scans **`$OUT`** for the artifact that contains **`insight_id`**; use
**`--episode-path`** to a specific `*.gi.json` if you prefer.)

**Corpus-level (your 2–3 episodes)**

```bash
python -m podcast_scraper.cli gi explore --output-dir "$OUT"
```

**Corpus export (symmetric with `kg export`)**

```bash
python -m podcast_scraper.cli gi export --output-dir "$OUT" --format ndjson --out gi.ndjson
python -m podcast_scraper.cli gi export --output-dir "$OUT" --format merged --out gi-bundle.json
```

Optional **`gi query`** probe (fixed patterns from [RFC-050 / guide](../guides/GROUNDED_INSIGHTS_GUIDE.md#cli), not open-ended NL):

```bash
python -m podcast_scraper.cli gi query --output-dir "$OUT" \
  --question 'What insights about inflation?'
```

**CLI details:** [Grounded insights (`gi`) subcommands](../api/CLI.md#grounded-insights-gi-subcommands); longer option list in [Grounded Insights Guide — CLI](../guides/GROUNDED_INSIGHTS_GUIDE.md#cli).

**Manual judgment**

- [ ] Insight texts are **not** stub placeholders.
- [ ] Mix of **`grounded: true`** and **`grounded: false`** is believable (all false or all
  true is worth noting). See [Troubleshooting — Ungrounded insights](../guides/GROUNDED_INSIGHTS_GUIDE.md#troubleshooting).
- [ ] For grounded rows: open the transcript (or segments) and confirm **quote text** is
  **verbatim** and **on-topic** for the insight.
- [ ] If you use **`gi explore --topic`**, remember current pipeline may not emit Topic
  nodes; substring match on insight text still applies — [Topic nodes and `gi explore --topic`](../guides/GROUNDED_INSIGHTS_GUIDE.md#topic-nodes-about-edges-and-gi-explore-topic).

---

## 5. KG checklist (whole run + per episode)

Use the same **`OUT`** as in §4.

**Schema**

```bash
python -m podcast_scraper.cli kg validate --strict "$OUT/metadata"
```

**Read alongside:** [kg.schema.json](../architecture/kg/kg.schema.json); [KG ontology](../architecture/kg/ontology.md).

**One episode summary**

```bash
python -m podcast_scraper.cli kg inspect --output-dir "$OUT" --episode-id '<id>'
```

**Cross-episode roll-ups (needs ≥2 episodes with KG)**

```bash
python -m podcast_scraper.cli kg entities --output-dir "$OUT"
python -m podcast_scraper.cli kg topics --output-dir "$OUT"
python -m podcast_scraper.cli kg export --output-dir "$OUT" --format ndjson
```

**CLI details:** [Knowledge Graph (`kg`) subcommands](../api/CLI.md#knowledge-graph-kg-subcommands); [KG CLI table](../guides/KNOWLEDGE_GRAPH_GUIDE.md#cli-kg-namespace); use cases in [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md).

**Manual judgment**

- [ ] **Entities** include hosts/guests where the pipeline detected them; names look sane.
- [ ] **Topics** align with summary bullets or LLM extraction (depending on mode).
- [ ] **Edges** look typed and consistent with [KG ontology](../architecture/kg/ontology.md) (no obvious
  garbage labels on a known episode).
- [ ] **`metadata.json`** contains **`knowledge_graph`** provenance (path, counts) when KG
  ran — [Output artifacts](../guides/KNOWLEDGE_GRAPH_GUIDE.md#output-artifacts).

**Future / adjacent:** corpus storage in a DB is separate — [PRD-018](../prd/PRD-018-database-projection-gil-kg.md) / [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md).

---

## 6. Regression notes (what to write down)

After the pass, capture **3–5 bullets** in your own notes or a ticket:

- Feed URL, date, CLI flags (or config path).
- Provider + `gi_insight_source` + `kg_extraction_source`.
- One **good** and one **weak** example (episode id + what was wrong).
- Any CLI friction (missing flag, confusing error, slow step).

That gives you a reproducible manual baseline next time GI/KG or providers change.

---

## 7. Optional Makefile hooks

- **`make validate-gi-schema [ARTIFACTS_DIR=path]`** — validates `*.gi.json` against
  [gi.schema.json](../architecture/gi/gi.schema.json) (see Makefile `help`). Example:
  `ARTIFACTS_DIR=.test_outputs/manual/planet_money_openai_gi_kg_summary_bullets/metadata`
- **`make validate-kg-schema [ARTIFACTS_DIR=path]`** — validates against
  [kg.schema.json](../architecture/kg/kg.schema.json).
- **`make gil-quality-metrics [DIR=path]`** — PRD-017-oriented metrics over `.gi.json`
  (see Makefile `help`).
- **`make kg-quality-metrics [DIR=path]`** — PRD-019-oriented metrics over `.kg.json`.
- **`make quality-metrics-ci`** — GIL + KG `--enforce` on `tests/fixtures/gil_kg_ci_enforce`
  (matches the **GIL and KG quality metrics** step in the GitHub Actions `test-unit` job;
  also invoked via **`make ci-fast`** locally).

Use these if you want parity with CI without remembering script paths.

**Related automated testing (background reading):** [TESTING_STRATEGY.md — GIL and KG CI quality gates](../architecture/TESTING_STRATEGY.md#gil-and-kg-ci-quality-gates). **Operators enabling both layers:** [CONFIGURATION.md — Knowledge Graph (KG)](../api/CONFIGURATION.md#knowledge-graph-kg) (shallow v1 notes link to GIL + KG recorded decisions).
