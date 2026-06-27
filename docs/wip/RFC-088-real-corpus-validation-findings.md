# RFC-088 real-corpus validation findings — prod-v2 (2026-06-28)

Item #8 from the post-chunks-0-9 audit. Ran the deterministic enrichers
end-to-end against the on-disk prod-v2 corpus at
`.test_outputs/manual/prod-v2/corpus` (209 episodes, 12 feeds, 99
latest-run-per-feed bundles after dedup).

## Setup

```
.venv/bin/python -m podcast_scraper.enrichment.cli \
  --output-dir .test_outputs/manual/prod-v2/corpus \
  --enrichers grounding_rate,guest_coappearance,insight_density,\
              temporal_velocity,topic_cooccurrence,topic_cooccurrence_corpus \
  --log-level INFO
```

Run summary (after the wiring fix in this branch):

| enricher                  | runs_ok | runs_failed | duration_ms |
|---------------------------|---------|-------------|-------------|
| grounding_rate            | 1       | 0           | 49          |
| guest_coappearance        | 1       | 0           | 57          |
| insight_density           | 99      | 0           | 2           |
| temporal_velocity         | 1       | 0           | 71          |
| topic_cooccurrence        | 99      | 0           | 0           |
| topic_cooccurrence_corpus | 1       | 0           | 58          |

Total wall: 249ms across 99 episode bundles.

## Bug 1 — CLI was a no-op on every corpus  → FIXED in this branch

`src/podcast_scraper/enrichment/cli.py` shipped two unfinished wires:

1. `registry = EnricherRegistry()` was never populated. The CLI module
   docstring claimed "chunk 2+ enrichers register at import time" — they
   don't; `enrichers/__init__.py` only exposes the
   `register_deterministic_enrichers(registry)` helper, never
   auto-called. So the registry was always empty, and every
   `EnricherSet.enabled_enrichers` entry got
   `"EnricherSet enables 'X' but it is not registered; skipping"`.
2. `episode_bundles=[]` with a `# chunk 1 ships no episode-scoping;
   sub-6 wires` TODO that was never closed. The workflow's
   `_maybe_spawn_enrichment_after_pipeline` invokes the same CLI, so
   pipeline-post-run enrichment was also a no-op.

The combination meant `podcast enrich --output-dir <corpus>` reported
`status=ok duration_ms=0` and wrote an empty `run_summary.json`.

Fix on this branch:

- `paths.discover_episode_bundles(corpus_root)` walks the corpus via the
  existing `search.corpus_scope.discover_metadata_files` (latest-run-
  per-feed dedup), builds `EpisodeArtifactBundle` per `*.metadata.json`,
  resolves sibling `.gi.json` / `.kg.json` / `.bridge.json` (None when
  absent), and pulls `episode_id` from `episode.guid` / `episode_id` /
  `guid` with a `stem` fallback.
- CLI calls `register_deterministic_enrichers(registry)` and feeds
  `discover_episode_bundles(corpus_root)` to `executor.run(...)`.
- `apply_cli_overrides` treats `only=[...]` against an empty base as
  force-include (mirrors the help-text claim "force-include (alias for
  --only)" — previously the empty base would filter `--enrichers a,b,c`
  to nothing).

Optional enrichers `topic_similarity` and `nli_contradiction` still
need provider / scorer wiring at the call site (chunk-7 left those for
profile-derived callers). They are NOT auto-registered.

## Bug 2 — `temporal_velocity.velocity_last_over_6mo` is always 0.0

Active topics have populated EWMA curves but a 0 velocity:

```json
{
  "topic_id": "topic:artificial-intelligence",
  "topic_label": "artificial intelligence",
  "total": 26,
  "ewma": {"...": "...", "2026-04": 11.31, "2026-05": 6.66, "2026-06": 3.33},
  "velocity_last_over_6mo": 0.0
}
```

Every top-by-total topic (`artificial intelligence`, `ai development`,
`supply chain`, `ai agents`, `tech industry`, `ai ethics`, `ai safety`,
`geopolitical risk`) reports velocity = 0.0 despite clearly non-flat
monthly counts. Suspect `scripts/`-level computation issue or stale
reference window. Worth: re-derive `velocity_last_over_6mo` from
`monthly_counts` in a unit test against this corpus's known-volatile
topics and assert non-zero.

Not fixed in this branch — surfaces as a follow-up. Scope: read
`enrichers/temporal_velocity.py` and reproduce against `monthly_counts =
{"2026-04": 4, "2026-05": 3, "2026-06": 1}` to see if velocity should
be non-zero.

## Bug 3 — `guest_coappearance` is polluted by `SPEAKER_NN` placeholders

The top 5 corpus-scope pairs by `episode_count` are all unresolved
diarization placeholders:

```
SPEAKER_03 <-> SPEAKER_05  eps=4
SPEAKER_00 <-> SPEAKER_01  eps=3
SPEAKER_00 <-> SPEAKER_03  eps=3
SPEAKER_00 <-> SPEAKER_05  eps=3
SPEAKER_02 <-> SPEAKER_03  eps=3
```

These IDs are episode-local diarization labels (`SPEAKER_00` in episode
A is unrelated to `SPEAKER_00` in episode B). They're getting
namespaced as Persons and then co-counted across episodes — so the
top-of-leaderboard is a meaningless cross-episode coincidence of label
indices.

Two possible fixes (not in scope for this branch):

1. Upstream: don't promote `SPEAKER_NN` labels to Person IDs at all when
   the NER / speaker-link pass hasn't resolved them.
2. Enricher-side: `guest_coappearance` filters out any pair where either
   id matches `^SPEAKER_\d+$` (or the canonical "unresolved guest"
   prefix). Simple but treats the symptom.

Worth doing — the leaderboard is the headline view in the consumer
Insights surface, and surfacing `SPEAKER_03 <-> SPEAKER_05` 4 times is
worse than surfacing nothing.

## Bug 4 — episode metadata has `duration_seconds=0` / `has_timing=False`

Sample `insight_density` envelopes from prod-v2 all report:

```json
{"duration_seconds": 0.0, "has_timing": false, ...}
```

That means the upstream metadata path isn't carrying episode duration
(or carrying it under a different field the enricher doesn't read).
Without duration, `insight_density` falls back to even-thirds-by-count
segmentation, which is the documented degradation path — but it means
the metric is "1/3 of insights are 'early'" rather than the more useful
"insights are concentrated in the first 12 minutes".

Lower priority than 2 and 3 — the enricher's degradation behaviour is
explicit and the output is still structurally usable. Worth a one-line
check: does any pipeline-produced metadata.json in prod-v2 carry a
non-zero `duration_seconds`? If not, the upstream metadata writer is the
bug, not the enricher.

## Notes for the next chunk

- Episode-scope envelopes write to the *latest* run's
  `metadata/enrichments/` per feed. Re-running enrichment after a new
  pipeline run will not back-fill older runs' envelopes — that's by
  design (envelopes are per-run, not per-corpus), but it's worth
  spelling out in the orchestration doc so operators don't expect
  global re-enrichment.
- The bundle discovery helper now lives next to the path helpers
  (`enrichment.paths.discover_episode_bundles`). The
  workflow-orchestration `_maybe_spawn_enrichment_after_pipeline` helper
  doesn't need to change — it calls the same CLI, which now wires
  bundles correctly.
- Per-enricher `records_written: 0` in the run summary is a
  bookkeeping miss (envelopes ARE written — verified by file size and
  field counts). The executor's stats-accumulator path doesn't update
  the counter for write-through paths. Follow-up: trace
  `EnrichmentRunResult.per_enricher[*].records_written` to see why it
  reports zero when the envelope clearly carries N records.

## Bugs 2-4 — open follow-ups (not in this branch)

| # | Title                                                  | Severity | Surface                         |
|---|--------------------------------------------------------|----------|---------------------------------|
| 2 | temporal_velocity.velocity_last_over_6mo is always 0   | Med      | enricher math                   |
| 3 | guest_coappearance dominated by SPEAKER_NN placeholders | High     | upstream Person resolution      |
| 4 | episode metadata has duration_seconds=0                | Med      | upstream metadata writer        |
| 5 | per_enricher.records_written always 0 in run summary   | Low      | executor stats accumulator      |
