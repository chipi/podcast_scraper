# RFC-088 real-corpus validation findings — prod-v2 (2026-06-28)

Item #8 from the post-chunks-0-9 audit. Ran the deterministic enrichers
end-to-end against the on-disk prod-v2 corpus at
`.test_outputs/manual/prod-v2/corpus` (209 episodes, 12 feeds, 99
latest-run-per-feed bundles after dedup).

## Setup

```
.venv/bin/python -m podcast_scraper.cli enrich \
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

## Bug 2 — `temporal_velocity.velocity_last_over_6mo` is always 0.0 → FIXED

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

**Fix:** `_velocity()` now accepts a `last_idx` parameter; a new
`_effective_last_idx()` helper walks back from the window end to the
most recent month with ANY topic activity across the corpus. The
envelope also surfaces `effective_last_month` so callers can tell when
"now" lags the data. Prod-v2 result: `effective_last_month: 2026-05`,
top topics (`financial history`, `infrastructure investment`,
`media influence`, `ai boom`, `ai bubbles`, ...) now report
non-zero velocities.

## Bug 3 — `guest_coappearance` is polluted by `SPEAKER_NN` placeholders → FIXED

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

**Fix:** Took the enricher-side path (option 2). Added a shared
`is_unresolved_speaker_placeholder(person_id, name)` helper in
`enrichers/_loaders.py` that matches
`(person:)?speaker[_-]?\d+` (case-insensitive) on either the id or
the display name. Both `guest_coappearance` and `grounding_rate` skip
placeholders before aggregation. Prod-v2 result: top pairs now read
`Jay Powell <-> Katie Martin` (real co-appearance), top grounding
persons are all real people. 65 of 247 Person nodes in the corpus
matched the placeholder pattern; all are now filtered from
corpus-scope aggregates without touching the underlying
`.gi.json` files.

## Bug 4 — episode metadata has `duration_seconds=0` / `has_timing=False` → FIXED

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

**Investigation result:** Two enricher-side bugs, not one upstream bug.

1. `metadata.json` stores duration under `episode.duration_seconds`,
   not the top-level key the enricher was reading.
2. Real `.gi.json` Quote nodes carry timing as `timestamp_start_ms`
   (millisecond integer), not the `start_s` / `start_seconds` / `start`
   keys the chunk-1 contract documented.

**Fix:** New `episode_duration_seconds(meta)` helper in `_loaders.py`
accepts both metadata shapes (top-level wins, falls back to
`episode.`). `_quote_start_s()` now also accepts
`timestamp_start_ms` (ms → s). Prod-v2 result: episodes from the
2026-06-13 pipeline run now correctly report `duration=1128.0
has_timing=True` with proper early/mid/late segmentation. Older
2026-05-05 runs still report `duration=0` — those .gi.json files
predate the metadata writer including duration and are not touched
by this fix.

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
  the counter for write-through paths. See Bug 5 below.

## Bug 5 — `per_enricher.records_written` always 0 in run summary → FIXED

The 6 deterministic enrichers all return raw `dict`s through
`@sync_enricher`, which wrapped them as `EnricherResult(status="ok",
data=..., records_written=0)` — leaving the dataclass default. The
executor's metrics accumulator faithfully recorded the 0 it was
given. None of the six bothered to count their own records, so every
`run_summary.json` had a per-enricher `records_written: 0` line.

**Fix:** `sync_enricher` now computes
`records_written = max(len(v) for v in dict.values() if
isinstance(v, list), default=0)`. Every deterministic enricher has
exactly one primary list-of-records value at top-level (`pairs`,
`persons`, `topics`, `insight_segments`), so the heuristic is
correct for all six without per-enricher annotation. Prod-v2
result: records_written reports 125 / 87 / 991 / 833 / 4455 / 4428
across the six enrichers respectively, matching the on-disk envelope
record counts.

## Status

All five bugs surfaced by the prod-v2 validation are closed in the
chain of commits on this branch (Bug 1 closed the CLI no-op; Bugs 2–5
close the enricher-side gaps surfaced by running it).

## Follow-up — optional-enricher wiring (deferred; design open)

Not a bug; a real design question worth thinking about between PRs.

**Current state.** Profile YAMLs (`cloud_thin`, `cloud_balanced`,
`cloud_quality`, `airgapped`, `prod_dgx_full_with_fallback`,
`local_dgx_full`) list `topic_similarity` and `nli_contradiction` in
`enabled_enrichers`. The CLI registry only auto-loads the six
deterministic enrichers (those have no constructor injections). When
the executor walks the EnricherSet it logs a WARNING and skips the
two — now with an actionable hint pointing at the wiring requirement
(`registry.py: _PROVIDER_WIRING_HINT`).

**Why not auto-wire in the CLI.** The optional enrichers need
constructor injections that genuinely vary by deployment:
- `TopicSimilarityEnricher(provider: EmbeddingProvider)` — airgapped
  uses a sentence-transformers checkpoint, cloud uses an external
  embeddings API, CI uses `FakeEmbeddingProvider(dim=32)`.
- `NliContradictionEnricher(scorer: NliScorer)` — `FixedNliScorer`
  for CI, `DeBERTaNliScorer` (lazy-loads ~80MB) for real runs.

NOT a CI-vs-real concern. The codebase already imports
`sentence_transformers` lazily inside functions in `gi/`, `kg/`,
`evaluation/`, etc.; default CI (`.[dev]`) doesn't install it, nightly
(`.[dev,ml,llm,search]`) does. Registering an `NliContradictionEnricher`
with a `DeBERTaNliScorer()` doesn't actually pull `sentence_transformers`
into the import graph — the lazy `import` inside `score()` only
triggers when the model is asked to score. So "the CLI can't import the
scorer" is not the blocker. The blocker is: the CLI doesn't know which
provider to construct, because that's a profile / runtime-config
concern, not a CLI-flag concern.

**Option 4 sketch (architectural; canonical long-term path).**
Move the optional-enricher wiring to
`workflow.orchestration._maybe_spawn_enrichment_after_pipeline` —
the workflow already has the resolved pipeline config, which knows:
- whether `vector_search` is enabled → which embedding model is
  loaded → reuse the same provider for `topic_similarity`
- whether the profile has an LLM tier with cost ceiling → use
  `DeBERTaNliScorer` when on, skip `nli_contradiction` when not.

The CLI stays deterministic-only (the operator dev-iteration entry).
The workflow path is the production entry that brings ML.

This matches the RFC-088 chunk 9 split between Mode A (CLI / dev) and
Mode B (workflow / production) — Mode B's existing scaffold is the
right place to add this wiring without breaking the CLI's "no ML
dependencies" contract.

Defer to a chunk-10-style follow-up PR. Today's WARNING-with-hint
closes the honesty gap so operators running `--profile cloud_thin`
through the CLI know exactly why those two are silent.
