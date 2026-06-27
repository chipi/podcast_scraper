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

### Run an enrichment pass

```bash
python -m podcast_scraper.enrichment.cli \
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
python -m podcast_scraper.enrichment.cli \
  --output-dir corpus --re-enable nli_contradiction \
  --re-enable-reason "transient HF outage"
```

```text
# Viewer
Configuration → Enrichment → Re-enable button (per row, only visible
when auto_disabled).
```

```text
# MCP (remote agent)
enrichment_re_enable(enricher_id="nli_contradiction",
                     reason="transient HF outage")
```

All three emit the same `enrichment.health.re_enabled` event so the
JSONL audit trail is complete regardless of which surface the
operator used.

## The shipped enrichers

| Tier | id | Scope | Reads | Writes |
| ---- | -- | ----- | ----- | ------ |
| deterministic | `topic_cooccurrence` | episode | `.kg.json` | `metadata/enrichments/{stem}.topic_cooccurrence.json` |
| deterministic | `topic_cooccurrence_corpus` | corpus | `.kg.json` | `enrichments/topic_cooccurrence_corpus.json` |
| deterministic | `temporal_velocity` | corpus | `.kg.json` | `enrichments/temporal_velocity.json` |
| deterministic | `grounding_rate` | corpus | `.gi.json` | `enrichments/grounding_rate.json` |
| deterministic | `guest_coappearance` | corpus | `.gi.json` | `enrichments/guest_coappearance.json` |
| deterministic | `insight_density` | episode | `.gi.json`, `.metadata.json` | `metadata/enrichments/{stem}.insight_density.json` |
| embedding | `topic_similarity` | corpus | `.kg.json` | `enrichments/topic_similarity.json` |
| ml | `nli_contradiction` | corpus | `.gi.json` | `enrichments/nli_contradiction.json` |
| query | `query_topic_relatedness` | per-request | `enrichments/topic_similarity.json` | annotates search hits |

The chunk-7 profile matrix (see `enrichment/profile_sets.py`) decides
which set runs by default per profile:

| Profile | Enricher set |
| ------- | ------------ |
| `test_default`, `eval_default`, `preprod_local_whisper` | (none — CI isolation) |
| `airgapped_thin` | deterministic only |
| `airgapped` | deterministic + `topic_similarity` |
| `cloud_thin`, `cloud_balanced`, `cloud_quality` | deterministic + `topic_similarity` + `nli_contradiction` |
| `dev`, `prod`, `local`, `local_dgx_*`, `prod_dgx_*`, `cloud_with_dgx_primary` | full set |
| unknown profile | (none — conservative default) |

CLI flags layer on top: `--enrichers <id,id>` / `--no-enrichers` /
`--opt-in <id,id>` / `--skip <id,id>`.

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
    EpisodeArtifactBundle, RunContext, STATUS_OK, sync_enricher,
)


def _compute(bundle, corpus_root, all_bundles, config, ctx):
    # Sync body; @sync_enricher wraps it.
    ...
    return {"my_result_key": ...}


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

For embedding / ML / LLM enrichers (which take an injected scorer),
the operator constructs the enricher and registers it explicitly —
the executor consumes the registry as-is.

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
  (under `tests/fixtures/enrichment/mock_scorers.py`) — no real
  models in CI ([[feedback_no_llm_in_ci]]).

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
   or run `python -m podcast_scraper.enrichment.cli --re-enable <id>
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
