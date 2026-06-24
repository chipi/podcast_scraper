# ADR-102: `_retro_audit` marker for in-place artifact mutation

- **Status**: Accepted
- **Date**: 2026-06-24
- **Authors**: Marko Dragoljevic, Claude (Opus 4.7)
- **Related ADRs**: [ADR-101](ADR-101-drop-legacy-kg-gi-shape.md) (strict
  KG v2.0 / GI v3.0 schemas which routine in-place migrations now
  satisfy).
- **Related RFC**: [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md)
  (the v3.0 ontology whose post-pass enrichment surfaced this need).

## Context

Two distinct surfaces have, over the last 30 days, mutated already-published
artifacts in place:

1. **Eval-run retroactive fingerprint backfill** (`data/eval/runs/<run_id>/`,
   2026-06-22) — chunk-7 rerun used new prompt and config fields that
   weren't in the original fingerprint. Rather than reprocess each run,
   the rerun stamped each affected `payload.json` with a `_retro_audit`
   entry recording the marker, the rerun timestamp, and the
   `unknown_fields` list. Documented in
   `docs/guides/eval-reports/CHUNK7_RETRO_FINGERPRINT_AUDIT_2026_06_22.md`.

2. **#1076 NER post-pass retro sweep** (`.test_outputs/manual/prod-v2/corpus`,
   2026-06-24) — the spaCy NER pass was switched on for two airgapped
   profiles. To project the lift onto an existing corpus without
   re-running 99 episodes through the pipeline, `enrich-edges
   --use-ner --retro-audit` mutated each affected `.gi.json` in place
   and stamped `_retro_audit` with the marker, applied-at, and per-edge
   counters.

The pattern works. It satisfies the operator-set constraint
([[feedback_never_mutate_historical_artifacts]]) that historical
artifacts are frozen-once-written; mutation is allowed only as an
explicit extraordinary measure with an audit trail. Without an ADR, the
two surfaces are using slightly different shapes (eval uses
`unknown_fields`, retro-sweep uses `edges_added`) and the trigger
conditions live in scattered docs.

The cost of not codifying: every future in-place mutation invents its
own shape, future readers can't tell whether a missing marker means
"never mutated" or "mutated by a non-audit-aware caller," and the
existing migration scripts (`scripts/migrate_*.py`) silently bypass the
pattern when they sweep through legacy corpora.

## Decision

**`_retro_audit` is the canonical marker for in-place mutation of any
artifact that was already produced + written to its source-of-truth
location.** Greenfield writes (a fresh pipeline run, a brand-new eval
run, a first-time-emitted KG/GI artifact) do not stamp.

### When required

A caller MUST stamp `_retro_audit` when it:

- Modifies an artifact at the same path it was originally written to,
- The artifact is in a directory the project treats as a published or
  frozen-once-written source-of-truth (`data/eval/runs/*`, prod
  corpora, viewer-validation corpora), AND
- The mutation changes the artifact's observable content (edges added,
  fields backfilled, schema bumped).

A caller MAY stamp `_retro_audit` for less-load-bearing mutations (a
local dev experiment on a personal corpus) but is not required to.

### Field shape

`_retro_audit` is a **list of dict entries**. List-valued so repeat
mutations stack chronologically:

```json
{
  "_retro_audit": [
    {
      "marker": "#1076-ner-2026-06-24",
      "applied_at": "2026-06-24T11:02:25+00:00",
      "scope": { "use_ner": true },
      "changes": { "mentions": 2, "has_episode": 0, "spoken_by": 0 }
    }
  ]
}
```

Required keys on each entry:

- `marker` (string) — issue-prefixed, date-suffixed: `"#<issue>-<short
  slug>-<YYYY-MM-DD>"`. Marker doubles as the summary-file basename
  (with `#` and `/` stripped: `_retro_audit_1076-ner-2026-06-24.json`).
- `applied_at` (ISO-8601 UTC string) — `datetime.now(timezone.utc)
  .isoformat(timespec="seconds")`.

Recommended keys (use what fits the mutation):

- `scope` — what the mutation was scoped to. For #1076: the flags
  that were live during the rewrite (`use_ner: true`).
- `changes` — per-artifact counters of what changed. Free-shape,
  chosen by the caller.
- `unknown_fields` — for fingerprint-style mutations, the list of
  cfg/prompt fields that were new at rerun time.

Sites are free to add other keys (e.g. `rerun_id`, `validator_version`)
as long as the required keys are present.

### Summary file

Callers that mutate more than one artifact in a single pass SHOULD
also write a single `<root>/_retro_audit_<sanitized-marker>.json`
listing per-artifact changes and grand totals. The summary makes the
sweep auditable without reading every mutated file.

Required summary shape:

```json
{
  "marker": "#1076-ner-2026-06-24",
  "applied_at": "...",
  "scope": { "use_ner": true },
  "root": "<absolute path>",
  "totals": { "mentions": 46, "has_episode": 0, "spoken_by": 0 },
  "per_episode": [
    { "gi_path": "feeds/.../episode.gi.json",
      "marker": "...", "applied_at": "...",
      "changes": { "mentions": 2, ... } }
  ]
}
```

### Migration scripts

`scripts/migrate_*.py` currently rewrite legacy schema in place
without stamping `_retro_audit`. This is **grandfathered** because:

1. The migration is a one-shot per artifact (idempotent on v3 input);
   the audit need is fundamentally different from an enrichment sweep.
2. Stamping every migrated artifact would balloon the in-tree
   migration scripts when most callers only migrate scratch corpora.
3. The migration target shape (`schema_version`) already records that
   migration happened; no second marker is needed.

If a future in-place migration changes observable content beyond what
`schema_version` advertises, that migration MUST start stamping.

## Consequences

**Positive:**

- Two existing surfaces converge on one shape.
- Future authors copy from this ADR instead of inventing.
- The `[[feedback_never_mutate_historical_artifacts]]` rule has a
  concrete escape hatch the operator can audit.

**Negative:**

- A new top-level key in artifact files that downstream consumers
  must tolerate. Validators currently accept extras
  (`_minimal_validate` checks required keys only), so this is no-op
  for now.
- Forgetting to stamp during a retro-sweep silently violates the
  pattern; CI can't detect "should have stamped but didn't."

**Neutral:**

- Migration scripts intentionally don't stamp. If the operator later
  decides migration audit is needed, that's a follow-up.

## Validation

This ADR codifies existing behavior; nothing to validate beyond:

- `scripts/dev/revert_gi_v3_to_v2.py` reverts both content and
  `_retro_audit` correctly (verified 2026-06-24 on
  `.test_outputs/manual/prod-v2/corpus` in the AI-ML-improvements
  worktree).
- `enrich-edges --use-ner --retro-audit` produces a well-formed
  marker + summary (verified 2026-06-24 on the FUTURE worktree's
  corpus copy).

## Implementation references

- `src/podcast_scraper/search/cli_handlers.py:1505-1623` — the
  enrich-edges retro-audit hook (reference implementation).
- `docs/guides/eval-reports/CHUNK7_RETRO_FINGERPRINT_AUDIT_2026_06_22.md`
  — earlier surface using `unknown_fields` instead of `changes`.
- `scripts/dev/revert_gi_v3_to_v2.py` — reverse path; strips the
  marker as part of reverting content.
