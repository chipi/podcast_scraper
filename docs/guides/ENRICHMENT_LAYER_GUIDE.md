# Enrichment Layer Guide (RFC-088)

The enrichment layer is the fourth artefact tier on top of GIL / KG /
bridge. It runs typed, opt-in enrichers (deterministic / embedding /
ML / LLM) that produce their own envelopes under `enrichments/` and
surface through HTTP routes, MCP tools, JSONL events, and the viewer
Configuration popup + Dashboard.

This guide is the operator / developer reference. The HTTP + MCP +
JSONL surface itself is documented in
[Enrichment Layer API](../api/ENRICHMENT_LAYER_API.md).

## Quick start

### Hands-free mode — run after every pipeline run (RFC-088 chunk 9)

Set `enrichment.enabled: true` in `viewer_operator.yaml`. The
pipeline's finalize step then spawns
`python -m podcast_scraper.cli enrich` as a **detached background
subprocess** after every successful pipeline run. Three properties:

- The pipeline returns its own count/summary immediately — wall-clock
  unaffected.
- The enrichment subprocess writes its own `run.jsonl` + envelopes +
  registers in the shared jobs registry, so the viewer's
  **Dashboard → Pipeline runs** strip surfaces it alongside pipeline
  jobs (badge `[enrich]`).
- Detached: SIGINT to the parent pipeline doesn't kill enrichment.
  Output streams to `<corpus>/.viewer/enrichment_pipeline_spawn.log`
  — `tail -f` it if you want.

Failures inside the spawn (PATH, output_dir permissions, etc.) log a
WARNING and the pipeline returns normally.

### Run an enrichment pass

```bash
python -m podcast_scraper.cli enrich \
  --output-dir path/to/corpus \
  --profile cloud_balanced
```

Or from the viewer: **Configuration → Enrichment tab → Run enrichment
now**. The same path drives `POST /api/jobs/enrichment` which spawns
the CLI in a subprocess and tracks it through the shared jobs
registry alongside pipeline runs.

### Inspect a run

- **Viewer Dashboard** — `PipelineJobHistoryStrip` shows pipeline AND
  enrichment jobs side by side; use the Pipeline / Enrichment kind
  filter to focus.
- **HTTP** — `GET /api/enrichment/status` (live), `GET /api/enrichment/run-summary`
  (last completed), `GET /api/enrichment/events?limit=50` (JSONL tail).
- **MCP** — `enrichment_run_status` / `enrichment_recent_runs` /
  `enrichment_recent_events`. From a remote agent, `prod_correlate(run_id)`
  joins enrichment events with the pipeline trace.

### Recover an auto-disabled enricher

Three equivalent paths:

```bash
# CLI
python -m podcast_scraper.cli enrich \
  --output-dir corpus --re-enable topic_consensus \
  --re-enable-reason "transient HF outage"
```

```text
# Viewer
Configuration → Enrichment → Re-enable button (per row, only visible
when auto_disabled).
```

```text
# MCP (remote agent)
enrichment_re_enable(enricher_id="topic_consensus",
                     reason="transient HF outage")
```

All three emit the same `enrichment.health.re_enabled` event so the
JSONL audit trail is complete regardless of which surface the
operator used.

## Architecture at a glance

```text
                    ┌─────────────────────────────────────────────────┐
                    │  CONFIG SOURCES (Shape B, RFC-088 v2)            │
                    │                                                 │
                    │  config/profiles/<name>.yaml      ← base set    │
                    │           +                                     │
                    │  <corpus>/viewer_operator.yaml    ← override    │
                    │           (deep-merged per key)                 │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
              ┌────────────────────────────────────────────────────┐
              │  RESOLUTION                                        │
              │  enricher_set_for_profile(profile)                 │
              │  + build_enricher_set_from_yaml(operator yaml)     │
              │  + apply_cli_overrides(--only / --skip / ...)      │
              │  → EnricherSet { enabled_enrichers, per_enricher_  │
              │                  config, opt_in_flags }            │
              └─────────────┬──────────────────────────────────────┘
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌─────────────────┐                  ┌────────────────────────────┐
│  EnricherRegistry│                 │  --with-ml ?               │
│  (deterministic) │                 │   register_ml_enrichers()  │
│                  │                 │   ↳ provider_types registry│
│ + topic_cooccurrence       …       │   ↳ instantiate provider   │
│ + grounding_rate                   │   ↳ build enricher with    │
│ + … (6 deterministic)              │      provider + knobs      │
└─────────┬────────┘                 └──────────────┬─────────────┘
          │                                         │
          └────────────────┬────────────────────────┘
                           ▼
           ┌──────────────────────────────────┐
           │  EnrichmentExecutor              │
           │  (tier-aware retry / circuit /   │
           │   auto-disable / heartbeat /     │
           │   cost cap per-enricher + run)   │
           └──────────────┬───────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │  Envelopes on disk + JSONL events│
           │  + run_summary + status + health │
           └──────────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │  Read surfaces                   │
           │   GET /api/corpus/enrichments/*  │
           │   GET /api/enrichment/{status,   │
           │     health,metrics,events,       │
           │     run-summary,config,...}      │
           │   Viewer Configuration tab       │
           │   MCP tools (prod_correlate)     │
           └──────────────────────────────────┘
```

## The shipped enrichers

| Tier | id | Scope | Reads | Writes | Ships? |
| ---- | -- | ----- | ----- | ------ | ------ |
| deterministic | `topic_cooccurrence_corpus` | corpus | `.kg.json` | `enrichments/topic_cooccurrence_corpus.json` | ✅ |
| deterministic | `topic_theme_clusters` | corpus | `.kg.json` | `enrichments/topic_theme_clusters.json` | ✅ |
| deterministic | `temporal_velocity` | corpus | `.kg.json` | `enrichments/temporal_velocity.json` | ✅ |
| deterministic | `grounding_rate` | corpus | `.gi.json` | `enrichments/grounding_rate.json` | ✅ |
| deterministic | `guest_coappearance` | corpus | `.gi.json` | `enrichments/guest_coappearance.json` | ✅ |
| deterministic | `insight_density` | episode | `.gi.json`, `.metadata.json` | `metadata/enrichments/{stem}.insight_density.json` | ✅ |
| deterministic | `insight_sentiment` | episode | `.gi.json` | `metadata/enrichments/{stem}.insight_sentiment.json` | ✅ (VADER — timeline colour) |
| embedding | `topic_similarity` | corpus | `.kg.json` | `enrichments/topic_similarity.json` | ✅ (needs a provider) |
| ml | `topic_consensus` | corpus | `.gi.json` | `enrichments/topic_consensus.json` | ✅ promoted (precision 0.91, ADR-108 composite) |
| query | `query_topic_relatedness` | per-request | `enrichments/topic_similarity.json` | annotates search hits | ✅ |

> **Note (ADR-108):** the one gated ML enricher is `topic_consensus` — the reimagining of the retired
> 0%-precision `nli_contradiction`, a **composite** (embedding cosine + low NLI contradiction) that
> cleared its eval and is admitted. Its sibling stance enricher (`stance_disagreement`/`stance_timeline`)
> was **retired**: per-person / per-topic stance over time is a read-time CIL query
> (`conversation-arc` / `position-arc`) coloured by the deterministic `insight_sentiment` (VADER), not a
> gated ML enricher. The 9 above (+ the query enricher) are the real set in `enrichment/enrichers/`.

### The accuracy gate — why membership is data-driven

The one gated ML enricher (`topic_consensus`) is wired and registered, and its profile membership is
decided by a data-driven accuracy gate (`enrichment/eval/admission.py` → `profile_sets._admit`), not a
hand-toggle. It declares an `accuracy_gate` (precision ≥ 0.5) on its manifest; until an eval records a
passing precision under `data/eval/enrichment/<id>/gate_metrics.json`, the gate excludes it from the
registry → profiles → UI. **`topic_consensus` cleared it** — precision 0.91 on prod-v2 (ADR-108
composite) → admitted to the cloud / dgx / dev / local profiles. A candidate with no passing eval
auto-promotes with no code edit once one is recorded. The live multi-speaker surface users also see is
**perspectives** (#1146), a CIL query over the GI, **not** an enricher, so it sidesteps the gate. `GET
/api/enrichment/config/admission` reports each enricher's promote/gate decision and reason.

The chunk-7 profile matrix (see `enrichment/profile_sets.py`) decides the CANDIDATE
set per profile; the accuracy gate then filters the ML tier (above):

| Profile | Enricher candidate set |
| ------- | ------------ |
| `test_default`, `eval_default`, `preprod_local_whisper` | (none — CI isolation) |
| `airgapped_thin` | deterministic only |
| `airgapped` | deterministic + `topic_similarity` |
| `cloud_thin`, `cloud_balanced`, `cloud_quality` | deterministic + `topic_similarity` + ML candidate `topic_consensus` (admitted — precision 0.91) |
| `dev`, `local`, `local_dgx_*`, `prod_dgx_*`, `cloud_with_dgx_primary` | same full candidate set (gate applies) |
| unknown profile | (none — conservative default) |

Membership is **data-driven, not a hand-maintained list**: profiles list CANDIDATES; `_admit()`
runs each through its manifest `accuracy_gate` + the recorded `data/eval` metric, so an ML
candidate ships only once it clears precision ≥ 0.5. There is no `config/profiles/prod.yaml` —
the production profiles are `prod_dgx_*`.

CLI flags layer on top: `--profile <name>` (sets the base set per the
matrix above) / `--enrichers <id,id>` (alias for `--only`) /
`--no-enrichers` / `--opt-in <id,id>` / `--skip <id,id>` / `--only <id,id>` /
`--with-ml` (registers ML / embedding / NLI enrichers from their
provider blocks — see [Configuration](#configuration) below).

## Per-enricher reference

Each shipped enricher's algorithm, inputs, output shape, and tunable
knobs. Knob keys map 1:1 to ``enrichers.<id>.<knob>:`` in the YAML
and to form fields in the viewer Configuration → Enrichment editor.

### `topic_theme_clusters` (deterministic, corpus scope)

Groups Topics **discussed together** (co-occurrence lift + greedy average-linkage) into
*themes* — e.g. {shadow fleet, oil prices, sanctions}. Complements the *semantic*
`topic_clusters` (which groups topics that *mean* the same thing); uses the `thc:` graph
compound-node prefix vs the semantic `tc:`. **Reads:** `.kg.json` (Topic nodes per episode).
**Writes:** `enrichments/topic_theme_clusters.json`. **Output:** mirrors `topic_clusters.json`
(`clusters[].members[]`) but tags each `cluster_type="theme"`. **Knobs:** `min_pair` (min
co-occurring episodes to form an edge), `merge_threshold` (linkage cutoff).

### `topic_cooccurrence_corpus` (deterministic, corpus scope)

Aggregates ``topic_cooccurrence`` across every episode bundle:
counts shared episodes per Topic pair, sorts descending. **Reads:**
`.kg.json` (all bundles). **Writes:** `enrichments/topic_cooccurrence_corpus.json`.
**Output:** `{ pairs: [{topic_a_id, topic_b_id, topic_a_label, topic_b_label, episode_count}], episode_count }`.
**Knobs:** none today.

### `temporal_velocity` (deterministic, corpus scope)

Per-Topic monthly mention counts over a trailing window, plus EWMA
trend and a “last completed month / 6-month average” velocity
signal. The "last month" is the most recent month with corpus-wide
activity (handles stale / partial current-month data — see
[real-corpus validation findings](../wip/RFC-088-real-corpus-validation-findings.md#bug-2--temporal_velocityvelocity_last_over_6mo-is-always-00--fixed)).
**Reads:** `.kg.json` (Episode publish_date + Topic nodes).
**Writes:** `enrichments/temporal_velocity.json`. **Output:**
`{ window_months: [YYYY-MM, ...], now, alpha, effective_last_month, topics: [{topic_id, topic_label, monthly_counts, ewma, velocity_last_over_6mo, total}] }`.
**Knobs:**

- ``alpha`` (float, 0 < α ≤ 1, default 0.5) — EWMA smoothing.
- ``window_months`` (int, 1–36, default 12) — trailing window size.

### `grounding_rate` (deterministic, corpus scope)

Per-Person ratio of grounded Insights they support across the
corpus. **Reads:** `.gi.json` (Person / Insight / Quote / SPOKEN_BY /
SUPPORTED_BY). **Writes:** `enrichments/grounding_rate.json`.
**Output:** `{ persons: [{person_id, person_name, total_insights, grounded_insights, rate}], episode_count }`,
sorted by `rate` then `total_insights`. **Knobs:** none today.
Unresolved diarization placeholders (``SPEAKER_NN`` /
``person:speaker-NN``) are filtered out before aggregation.

### `guest_coappearance` (deterministic, corpus scope)

Person pairs by shared episodes. **Reads:** `.gi.json` (Person +
SPOKEN_BY). **Writes:** `enrichments/guest_coappearance.json`.
**Output:** `{ pairs: [{person_a_id, person_b_id, person_a_name, person_b_name, episode_count}], episode_count }`.
**Knobs:** none today. Same SPEAKER_NN filter as `grounding_rate`.

### `insight_density` (deterministic, episode scope)

Insight count per (early / mid / late) third of the episode duration,
based on supporting Quote start times. Falls back to even-thirds-
by-count when timing is missing. **Reads:** `.gi.json` (Insight,
Quote with `start_s`/`start_seconds`/`start`/`timestamp_start_ms`,
SUPPORTED_BY) + `.metadata.json` (`duration_seconds` top-level or
nested under `episode.`). **Writes:**
`metadata/enrichments/{stem}.insight_density.json`. **Output:**
`{ episode_id, duration_seconds, has_timing, counts: {early, mid, late, unknown}, total_insights, insight_segments }`.
**Knobs:** none today.

### `topic_similarity` (embedding, corpus scope)

Per-Topic Top-K cosine-similar neighbours via the injected
``EmbeddingProvider``. **Reads:** `.kg.json` (Topic nodes).
**Writes:** `enrichments/topic_similarity.json`. **Output:**
`{ topics: [{topic_id, topic_label, top_k, neighbours: [{topic_id, topic_label, similarity}]}], top_k, topic_count, missing_topic_ids }`.
**Knobs:**

- ``top_k`` (int, 1–100, default **7**) — neighbours per topic. Retuned 10→7 (#1105) after
  the accuracy eval showed recall@10 already saturated (99%) and a smaller K gives a cleaner
  "related topics" surface (80/80 @7).

**Status: validated + shipping.** Prod-v2 eval (24-topic Opus silver, #1105): recall@10 **99%**,
precision@10 71%. Unlike the ML enrichers it declares **no** `accuracy_gate`, so it is admitted
unconditionally (numbers measured locally, not persisted as a gate metric).

**Provider requirement:** `EmbeddingProvider`. Set
`enrichers.topic_similarity.provider.type` to one of the registered
types (`sentence_transformer_local`, `fake_for_test`, ...) — see
[Provider-type registry](#provider-type-registry).

### `topic_consensus` (ml, corpus scope)

The ADR-108 reimagining of `nli_contradiction`: detect cross-Person **corroboration** per Topic —
"what the corpus agrees on". Real-corpus eval showed *symmetric NLI entailment* has ~0 recall (genuine
agreement is phrased differently), so the emit rule is a **composite** over the injected
``ConsensusScorer``: **embedding cosine ≥ `cos_threshold`** (the shared-question gate — same
proposition) **AND NLI contradiction ≤ `contra_threshold`** (the direction gate — they don't disagree,
which filters similar-but-opposite pairs). No LLM (MiniLM + DeBERTa, both CPU-local). **Reads:**
`.gi.json` (Insight / Person / SPOKEN_BY / ABOUT). **Writes:** `enrichments/topic_consensus.json`.
**Output:** `{ consensus: [{topic_id, person_a_id, person_a_name, person_b_id, person_b_name, insight_a_id, insight_a_text, insight_b_id, insight_b_text, consensus_score, cosine, contradiction}], cos_threshold, contra_threshold, pairs_scored, model_id, model_version }` (`consensus_score` = `cosine`).
**Knobs:**

- ``cos_threshold`` (float, 0–1, default 0.70) — min embedding cosine (shared-question gate).
- ``contra_threshold`` (float, 0–1, default 0.5) — max NLI contradiction, either direction (direction gate).

**Provider requirement:** `ConsensusScorer`. Set
`enrichers.topic_consensus.provider.type` to `consensus_local` (MiniLM + DeBERTa, requires `[ml]`
extra) or `fixed_consensus` (CI-safe test fixture).
**Status: PROMOTED** — measured **precision 0.91** on prod-v2 (curated 28-pair gold), recorded in
`data/eval/enrichment/topic_consensus/gate_metrics.json`, so the `accuracy_gate(precision ≥ 0.5)`
auto-admits it into the cloud / dgx / dev / local profiles. Re-score with
`scripts/eval/score/enrichment_topic_consensus.py`.

### `insight_sentiment` (deterministic, episode scope)

The colour layer for the conversation-timeline surfaces. Scores every Insight's text with **VADER** (a
pure-Python lexicon analyzer that bundles its own lexicon — no model download, no network) → a
`compound` in [−1, +1] + a `negative` / `neutral` / `positive` label (VADER's ±0.05 thresholds).
**Reads:** `.gi.json`. **Writes:** `metadata/enrichments/{stem}.insight_sentiment.json`.
**Output:** `{ episode_id, counts: {negative, neutral, positive}, total_insights, insights: [{insight_id, compound, label}] }`.
Deterministic tier → **no accuracy gate** (unlike a stance *score*, sentiment is a decoration — a
neutral factual insight is a fine grey). The CIL timeline queries (`position-arc`, `topic-timeline`,
`conversation-arc`) join it by `insight_id` to tint each insight.

> **Retired: `stance_timeline` / `stance_disagreement`.** An earlier ADR-108 cut tried to *auto-score*
> a −1..+1 stance per (person, topic) via NLI-entailment vs "{topic} is good/bad" anchors. Real-corpus
> eval showed the stance signal is ~0 on factual insights, so the score was the wrong frame. Per-person
> / per-topic stance **over time** is now a **read-time CIL query** — `GET /api/persons/{id}/positions`
> (per-`(person, topic)` arc), `GET /api/topics/{id}/timeline` (all speakers), and `GET
> /api/topics/{id}/conversation-arc` (aggregate-first weekly volume × sentiment) — coloured by
> `insight_sentiment`. No gated ML enricher, no LLM.

### `query_topic_relatedness` (query enricher, per-request)

Decorates search hits with `topic_similarity` Top-K when the
corpus has an `enrichments/topic_similarity.json` envelope. Not a
batch enricher; the search route opt-in calls it inline.

## Configuration

The ``enrichment:`` block in ``viewer_operator.yaml`` (or any
``--config <yaml>``) uses **Shape B**: per-enricher block under
``enrichers.<id>:``. Presence of the block is the enable; add
``enabled: false`` to opt out without removing the block.

```yaml
enrichment:
  enabled: true                    # master switch (Tier 1)
  max_total_cost_usd_per_run: 5.0  # optional run-wide cost cap
  enrichers:
    temporal_velocity:             # block present = enabled (Tier 2)
      alpha: 0.7
      window_months: 6
    topic_similarity:
      top_k: 10
      provider:                    # ML enrichers declare a provider
        type: sentence_transformer_local
        model: all-MiniLM-L6-v2
    topic_consensus:
      threshold: 0.6
      provider:
        type: deberta_local
    grounding_rate: {}             # empty block = enabled, no knobs
    insight_density:
      enabled: false               # explicit opt-out (preserves block)
```

Resolution model:

1. Profile YAML (under ``config/profiles/<name>.yaml``) provides the
   BASE enrichment block.
2. ``viewer_operator.yaml``'s enrichment block deep-merges on top
   (operator wins per key).
3. CLI flags layer on top of the resolved set
   (``--only`` / ``--skip`` / ``--opt-in`` / ``--no-enrichers``).
4. ``--with-ml`` registers ML enrichers from their `provider:`
   blocks via the [provider-type registry](#provider-type-registry).

### Migrating from the older shape

The Shape B reader still accepts the older explicit ``enabled: true``
form on every block (the implicit-default rule treats them as
equivalent), so an existing ``viewer_operator.yaml`` with:

```yaml
enrichment:
  enrichers:
    temporal_velocity:
      enabled: true
      alpha: 0.5
```

behaves the same as a Shape B block:

```yaml
enrichment:
  enrichers:
    temporal_velocity:
      alpha: 0.5
```

The viewer's editor writes the canonical Shape B form on next Save.
Operators who'd been relying on the legacy ``enabled_enrichers: [list]``
top-level key need to migrate to per-block; the legacy form is no
longer read by either the CLI or the server. The 12 shipped profile
YAMLs were migrated in the same change; the drift test
(``test_yaml_enrichment_block_matches_python_matrix``) accepts both
shapes for back-compat in operator overrides.

### Editing in the viewer

The viewer's **Configuration → Enrichment tab → Configuration**
section is a UI editor for the operator-side block. Every form
field is generated from ``GET /api/enrichment/config/schema``, so
adding a knob to a manifest's ``config_schema`` makes it editable
in the UI without any viewer code change. Save writes via
``PUT /api/enrichment/config`` (atomic; preserves unrelated YAML
keys like ``profile``, ``feeds``, etc.).

## Provider-type registry

ML / embedding / NLI enrichers declare a
``ProviderRequirement(protocol="...", description=...)`` on their
manifest. At runtime, the provider is constructed by name from the
process-scoped registry in
``src/podcast_scraper/enrichment/provider_types/``. Each shipped type
lives under ``provider_types/<protocol>/<name>.py`` and registers
itself at import time.

Shipped types:

| name | protocol | params | notes |
| ---- | -------- | ------ | ----- |
| `fake_for_test` | EmbeddingProvider | `dim` (int 4–1024, default 32) | Deterministic hash embedder. CI-safe. |
| `sentence_transformer_local` | EmbeddingProvider | `model` (string, required); `device` ("cpu"/"cuda"/"mps", optional) | sentence-transformers local model. Requires `[ml]` / `[search]` extras. Lazy-loaded. |
| `fixed_scripted` | NliScorer | `default_contradiction`/`default_neutral`/`default_entailment` (floats 0–1, defaults 0.05/0.85/0.10) | Fixed-score NLI. CI-safe. |
| `deberta_local` | NliScorer | `model` (string, default `cross-encoder/nli-deberta-v3-small`) | Lazy-loaded; requires `[ml]` extra. |

The PUT `/api/enrichment/config` route validates these params (unknown
type names, missing required params like `model` for
`sentence_transformer_local`, typo'd knob names like `alphaa`, and
`provider:` blocks on deterministic enrichers — all rejected at write
time, not at runtime).

The viewer's per-row Provider dropdown is populated from
``GET /api/enrichment/provider-types`` so adding a new type
automatically shows up in the UI.

## Writing a new enricher

### 1. Pick the right tier

| Tier | Use when | Retry / backoff |
| ---- | -------- | --------------- |
| Deterministic | Pure-Python algorithm over core artefacts (KG/GI/bridge). No external models. | 0 retries; auto-disable at 5 consecutive failed runs |
| Embedding | Backend is a vector store / embedder. Local CPU or remote. | 3 retries, 30s max backoff; circuit at 5 |
| ML | Local ML model (e.g. DeBERTa NLI). CPU / MPS. | 2 retries, 60s max backoff; circuit at 3; auto-disable at 2 |
| LLM | Remote LLM provider. Opt-in. | 5 retries, 120s max backoff; circuit at 3; auto-disable at 2 |

### 2. Implement the protocol

```python
from podcast_scraper.enrichment.protocol import (
    EnricherManifest, EnricherResult, EnricherScope, EnricherTier,
    EpisodeArtifactBundle, ProviderRequirement, RunContext,
    STATUS_OK, sync_enricher,
)


def _compute(bundle, corpus_root, all_bundles, config, ctx):
    # Sync body; @sync_enricher wraps it. Read knobs from `config`.
    threshold = float(config.get("threshold", 0.5))
    ...
    return {"my_result_key": [...], "threshold": threshold}


_enrich_async = sync_enricher(_compute)


class MyEnricher:
    manifest = EnricherManifest(
        id="my_enricher",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="my_enricher.json",
        description="What this enricher does.",
        expected_duration_s=30,
        # NEW (RFC-088 v2): declare your tunable knobs so the UI's
        # Configuration → Enrichment editor generates form fields
        # automatically — no per-enricher viewer code.
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                    "description": "Cutoff for emitting a record.",
                },
            },
        },
        # Only set if your enricher takes an injected provider /
        # scorer (EmbeddingProvider / NliScorer / ...). Omit for
        # deterministic enrichers.
        # provider_requirement=ProviderRequirement(
        #     protocol="EmbeddingProvider",
        #     description="Embedding source for the topic vectors.",
        # ),
    )

    async def enrich(self, *, bundle, corpus_root, all_bundles, config, ctx):
        return await _enrich_async(bundle, corpus_root, all_bundles, config, ctx)
```

**Contract notes:**

- The function returns a `dict` (the envelope `data`) on success, or
  raises a domain exception (`BadInputError`, `DependencyAccessError`,
  `ScorerTimeoutError`, `ModelLoadError`) on failure. The
  `@sync_enricher` decorator wraps the dict in `EnricherResult(status=ok, data=...)`
  and converts exceptions to `EnricherResult(status=failed, ...)`.
- Backend exceptions BUBBLE so the executor's retry classifier
  applies the tier policy. Don't try/except them inside the body.
- Long bodies must check `ctx.cancel_event.is_set()` between batches
  and bail with `STATUS_CANCELLED`.
- The envelope filename is `manifest.writes`. Episode scope lands
  under `metadata/enrichments/{stem}.{writes}`; corpus scope under
  `enrichments/{writes}`.

### 3. Register

For deterministic enrichers, add to
`src/podcast_scraper/enrichment/enrichers/__init__.py`'s
`register_deterministic_enrichers()` helper + `ALL_DETERMINISTIC_ENRICHER_IDS`.

For embedding / ML / LLM enrichers (which take an injected provider /
scorer), add a builder to ``src/podcast_scraper/enrichment/ml_wiring.py``'s
``_ML_ENRICHER_BUILDERS`` map. The builder signature is
``(provider_instance, knobs: dict) -> Enricher`` — ``knobs`` is the
per-enricher config dict with ``provider`` / ``enabled`` / ``opt_in``
already stripped, so a builder only sees the actual tunable knobs:

```python
def _build_my_enricher(provider: Any, knobs: dict[str, Any]) -> MyEnricher:
    threshold = float(knobs.get("threshold", 0.5))
    return MyEnricher(provider=provider, threshold=threshold)

_ML_ENRICHER_BUILDERS["my_enricher"] = _build_my_enricher
```

When the CLI sees ``--with-ml`` or the workflow auto-passes it,
``register_ml_enrichers()`` walks the active EnricherSet, looks up
the matching builder, instantiates the provider via the
[provider-type registry](#provider-type-registry), and registers the
enricher.

### 4. Add it to the profile matrix

Edit `src/podcast_scraper/enrichment/profile_sets.py` to decide which
profiles get the new enricher by default. The drift test
(`test_every_real_profile_yaml_has_a_matrix_decision`) ensures every
profile YAML still produces an EnricherSet.

### 5. Test it

- **Unit**: `tests/unit/enrichment/test_<your_enricher>.py` —
  synthetic 1-3 episode fixtures, asserts numerics + envelope shape.
- **Integration**: `tests/integration/enrichment/test_<your>_executor_smoke.py` —
  end-to-end via the EnrichmentExecutor, verify the envelope on disk,
  metrics ok, and JSONL audit events.
- **Resilience**: drive failure scenarios via the chunk-1
  `MockEmbeddingProvider` / `MockNliScorer` / a `ScriptedEnricher`
  (under `tests/fixtures/enrichment/mock_scorers.py`) — no paid
  remote LLM calls in CI (project rule documented in `AGENTS.md`'s
  "What 'no LLM in CI' actually means" section). Heavy local model
  downloads (DeBERTa, sentence-transformers checkpoints) gate behind
  the ``ml_models`` pytest marker.

### 6. Add an eval row (optional)

The scoring scaffolding lives under `scripts/eval/score/enrichment_<tier>.py`.
Drop a gold fixture under `data/eval/enrichment/<id>/gold/`; the
script computes the metric (exact-match for deterministic, recall@K
for embedding, P/R/F1 + Brier for NLI). Run direct-Python — no Make
wrapper for the scoring script itself (per REPLAN-O6).

## Writing a new scorer protocol

Real-model implementations of a scorer protocol live under
`src/podcast_scraper/enrichment/scorers/`. The contract is the
`@runtime_checkable` Protocol in
`enrichment/scorers/protocol.py`:

```python
from podcast_scraper.enrichment.scorers.protocol import EmbeddingProvider

class MyEmbeddingBackend:
    async def topic_vector(self, topic_id: str) -> list[float] | None:
        ...

assert isinstance(MyEmbeddingBackend(), EmbeddingProvider)  # smoke
```

Mirror the shape of the existing
`TopicEmbeddingProvider` / `DeBERTaNliScorer` for testability —
expose the actual backend call as an injected callable so tests can
swap in a `HashEmbedder` / `FixedNliScorer` without loading model
weights.

## Writing a new provider type

Provider types let operators pick a backend by string name in the
YAML (e.g. ``provider.type: sentence_transformer_local``) without
touching Python. Add a new type by:

1. **Create the module** under
   ``src/podcast_scraper/enrichment/provider_types/<protocol>/<name>.py``
   (the existing protocols are ``embedding/`` and ``nli/``; add a new
   subdirectory if you're introducing a new protocol).

2. **Register at import time** with a JSON-Schema fragment for the
   params + a factory:

   ```python
   from podcast_scraper.enrichment.provider_types.registry import (
       register_provider_type,
   )

   def _make_my_provider(params: dict) -> MyProviderImpl:
       return MyProviderImpl(
           model=params["model"],
           device=params.get("device", "cpu"),
       )

   register_provider_type(
       name="my_provider_local",
       protocol="EmbeddingProvider",
       description="One-line UI label.",
       params_schema={
           "type": "object",
           "additionalProperties": False,
           "required": ["model"],
           "properties": {
               "model": {"type": "string", "description": "Model id."},
               "device": {"type": "string", "enum": ["cpu", "cuda"]},
           },
       },
       factory=_make_my_provider,
   )
   ```

3. **Add the import** to the protocol subpackage's ``__init__.py``
   (e.g. ``provider_types/embedding/__init__.py``) so registration
   fires on package import.

The viewer's provider dropdown is fed by
``GET /api/enrichment/provider-types`` — it picks up the new type
automatically on next page load. The form fields below the dropdown
are generated from the ``params_schema`` you declared, so adding new
params is also zero viewer code.

**CI-safety:** any provider type that pulls in heavy ML extras
(sentence-transformers, transformers, torch, ...) MUST import them
**lazily** inside the factory or instance method. Module-top
imports break ``.[dev]``-only CI installs.

## Interpreting health + JSONL events

`<corpus>/.viewer/enrichment_health.json` carries per-enricher
cross-run state:

```json
{
  "topic_similarity": {
    "consecutive_failures": 0,
    "auto_disabled": false,
    "circuit_state": "closed",
    "last_status": "ok",
    "last_run_at": "2026-06-27T11:30:00Z"
  }
}
```

The 11 JSONL event types prefixed `enrichment.` (`run.started`,
`run.completed`, `run.skipped`, `enricher.started`,
`enricher.completed`, `enricher.retry`, `enricher.circuit_opened`,
`enricher.auto_disabled`, `enricher.cancelled`,
`enricher.stall_warning`, `health.re_enabled`) are appended to
`<corpus>/enrichments/run.jsonl`. Every payload carries the
`RunContext` correlation envelope (`run_id`, `enricher_id`,
`enricher_version`, `tier`, `attempt`, `job_id`), so `prod_correlate(run_id)`
joins them with Langfuse traces / Loki logs / Sentry errors / the
pipeline run.

## Operator runbook

### Auto-disabled enricher fires

1. Open Configuration → Enrichment in the viewer. The auto-disabled
   row shows `auto_disabled: yes` with the failure reason.
2. Check `GET /api/enrichment/events?enricher_id=<id>&limit=20` for
   the failure events that led to the cross-run threshold.
3. If transient (HF rate-limit, network blip), click **Re-enable**
   or run `python -m podcast_scraper.cli enrich --re-enable <id>
   --re-enable-reason "transient ..."`.
4. If persistent, investigate via `enrichment_recent_events` +
   Langfuse + Loki via `prod_correlate(<run_id>)`.

### Cost cap fires

Per-enricher: that enricher reports `status=quarantined` with reason
`cost_cap_exceeded`. Other enrichers continue. The cost cap is
declared in the manifest (`max_cost_usd_per_run`) or per-enricher
config (operator YAML).

Run-wide: when total run cost exceeds
`enrichment.max_total_cost_usd_per_run`, subsequent enrichers in
the queue are marked `skipped`. The run's status flips to `failed`
unless `enrichment.fail_on_run_cost_cap: false`.

### Adding a new profile

1. Add the YAML under `config/profiles/<name>.yaml`.
2. Add a branch to `enricher_set_for_profile()` in
   `src/podcast_scraper/enrichment/profile_sets.py`. The drift test
   fails if you skip this step.
3. If the profile uses an LLM-tier enricher, populate `opt_in_flags`
   appropriately (matches `manifest.requires_opt_in=True`).

## References

- [Enrichment Layer API](../api/ENRICHMENT_LAYER_API.md) — HTTP +
  MCP + JSONL reference (this guide's companion)
- [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — the
  architectural spec
- [ADR-104](../adr/ADR-104-enrichment-layer-boundary-vs-kg-direct.md)
  — boundary vs RFC-097 KG-direct connectivity
- Implementation:
  - `src/podcast_scraper/enrichment/` — framework
  - `src/podcast_scraper/enrichment/enrichers/` — concrete enrichers
  - `src/podcast_scraper/enrichment/scorers/` — backend scorers
  - `src/podcast_scraper/server/routes/enrichment.py` — operator-facing routes
  - `src/podcast_scraper/server/routes/corpus_enrichments.py` — user-facing envelope reads
  - `src/podcast_obs/sources/enrichment.py` — MCP source
