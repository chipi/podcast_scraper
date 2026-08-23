# ADR-132: Per-episode processing manifest — schema, stage ownership, and versioning

- **Status**: Proposed
- **Date**: 2026-07-27
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-109](../rfc/RFC-109-per-episode-observability-manifest.md) (the operating
  model this schema serves)
- **Related ADRs**: [ADR-133](ADR-133-metadata-vs-manifest-source-of-truth.md) (metadata vs manifest
  SoT), [ADR-131](ADR-131-speech-normalized-coverage-gate.md) (the `.asr.json` seed)

## Context & Problem Statement

RFC-109 calls for a per-episode observability record that is complete, versioned, and queryable, and
whose fields are written by the stages that own them. This ADR fixes the **schema** and the two
conventions that keep it honest: **stage ownership** and **method versioning**. Getting the schema
right once matters — it is a contract every stage writes and (later) a dashboard reads, and
re-migrating 100k episodes is expensive.

## Decision

### The artifact

A per-episode `<transcript-base>.manifest.json` sidecar (generalizing the ADR-131 `.asr.json`),
plus a flattened row appended to a corpus-level `manifest.jsonl` ledger (RFC-109 §4). The manifest
is the **source of truth for how the episode was processed** (ADR-133).

### Schema (v1)

```jsonc
{
  "schema_version": 1,
  "episode_id": "…", "feed_id": "…", "run_id": "…",
  "pipeline_version": "<git-sha-or-release>",   // whole-pipeline provenance
  "generated_at": "<iso8601>",                  // stamped after the run (not in-script — see note)
  "stages": {
    "asr": {
      "ran": true, "method": "tailnet_dgx_whisper",
      "model": "deepdml/…-turbo-ct2",           // ACTUAL model, incl. a failover
      "method_version": "asr-gate-1",
      "duration_s": 3.1, "cost_usd": 0.0,
      "metrics": { "speech_coverage": 0.935 },
      "failover": null,                         // or the ADR-131 speech_coverage_failover breadcrumb
      "warnings": []
    },
    "diarization": {
      "ran": true, "method": "tailnet_dgx", "model": "pyannote/…-community-1",
      "method_version": "diar-1",
      "metrics": { "num_speakers": 4, "speech_seconds": 1553, "unattributed_talk_share": 0.05 },
      "warnings": []
    },
    "naming": {
      "ran": true, "method_version": "naming-3",   // bumped at ADR-130 + audit 2a/3
      "metrics": {
        "hosts_detected": 2, "hosts_named": 2,
        "guests_detected": 3, "guests_named": 2,
        "self_intro": 2, "llm_resolved": 1, "snapped": 1, "canonicalized": 2,
        "host_anchor": ["Kevin Roose", "Casey Newton"]
      },
      "flags": []
    },
    "summary": { "ran": true, "method": "gemini", "model": "gemini-2.5-flash-lite",
                 "method_version": "sum-mega-bundled-1", "cost_usd": 0.0011, "metrics": { … } },
    "gi":  { "ran": true, "method_version": "gi-2", "cost_usd": 0.0031,
             "metrics": { "insight_count": 12, "gate_dropped": 7 } },
    "kg":  { "ran": true, "method_version": "kg-2",
             "metrics": { "node_count": 41, "edge_count": 88 } }
  },
  "quality_flags": ["asr_failover", "unnamed_dominant_voice"],  // rework candidates, corpus-queryable
  "cost_usd_total": 0.0042
}
```

Every stage block shares a small common shape (`ran`, `method`, `method_version`, `duration_s`,
`cost_usd`, `metrics`, `warnings`) so the ledger flattens uniformly; stage-specific signals live under
`metrics`. Absent stages are omitted (not `ran: false`) unless a stage was deliberately skipped.

### Convention 1 — stage ownership (the honesty rule)

**Each stage writes its own block from its own result, never from `cfg`.** ASR writes `stages.asr`
(including the actual model on a failover); the roster writes `stages.naming` from the resolved
roster; GI writes `stages.gi`. A manifest field that no stage owns is not added — it is the
`whisper_model`-from-config rot the whole design exists to prevent. A thin manifest accumulator
(passed down the pipeline, or a per-episode collector) merges the blocks; it does not invent them.

### Convention 2 — **layered** versioning (the reprocess key)

A single "pipeline version" cannot answer the two different questions we have — *"exactly what code
ran?"* and *"which episodes need re-running after I change this one stage?"*. So versioning is
**three layers**, each recorded on every manifest:

1. **`git_sha` (+ `dirty`) — the ground truth.** Captured at **runtime** from the running tree (with
   a `dirty` flag when there are uncommitted changes, because then the SHA does *not* fully describe
   the code). Automatic, always present, precise. It is the **backstop**: if a semantic version below
   is forgotten, two SHAs can still be diffed to see what changed. But it is opaque (a SHA says
   nothing about *what* changed) and noisy (it moves on every commit, including a README), so it is
   not the query key.
2. **`pipeline_version` — the composition.** A short semantic version of the **stage graph**: which
   stages run and in what order. It is bumped when the pipeline is **re-wired**, not when a stage's
   internals change — moving the ASR gate downstream of diarization (ADR-131) is exactly this, and no
   single stage's `method_version` captures it. Derivable as a hash of the ordered stage list so it
   changes automatically on a re-wire, with a human alias.
3. **per-stage `method_version` — the logic, and the query key.** A short string each stage bumps
   **when its own logic changes**, not when config changes (naming: `naming-3` after ADR-130 + 2a/3;
   ASR gate: a version per metric change).

**Why all three, and why the per-stage one is the query key.** Reprocessing is *targeted*: after
improving naming you want the episodes whose **naming** is stale, not every episode a noisy `git_sha`
or a monolithic `pipeline_version` would sweep in (their GI/summary were fine). So the reprocess
query is `SELECT episode_id FROM ledger WHERE stages.naming.method_version < 'naming-3'` — precise,
minimal re-run. The `pipeline_version` answers the *shape* question (did a re-wire change results
even though every stage's logic was unchanged?), and the `git_sha` is the exact-code backstop when a
version was mis-bumped. The semantic version strings live in one registry module so a bump is one
greppable edit.

### Convention 3 — per-stage cost (cloud-provider aware)

Every stage block carries `cost_usd` sourced from **that stage's own billing**, so the manifest is
correct whether a stage runs locally (free) or on a paid cloud provider:

- **ASR** — `call_metrics.estimated_cost`, populated by `apply_estimated_cost_if_missing` (pricing
  YAML, per audio-minute). Local DGX/whisper → 0/None; cloud OpenAI/Deepgram → real USD.
- **Diarization** — a cloud diarizer sets `DiarizationResult.cost_usd`; otherwise it is estimated
  centrally in `apply_diarization_to_result` via the same pricing layer (`capability="diarization"`).
  Local pyannote/DGX/MOSS → **0.0** (see the `0.0` vs `null` rule below).
- **Naming** — **not** free by definition. `cloud_balanced` sets `speaker_detector_provider:
  litellm`, so voice resolution is a real LLM call; `EpisodeCostProbe` captures this episode's
  share via `record_llm_speaker_detection_call`. A purely deterministic naming pass costs
  **0.0**.
- **Summary** — `summary_call_metrics.estimated_cost`.
- **GI / KG** — captured per-episode by an `EpisodeCostProbe` that wraps the shared
  `pipeline_metrics` around each episode's build. GI/KG cost is recorded by the LLM providers (all of
  gemini/deepseek/grok/anthropic/mistral funnel through `record_llm_gi_call` / `record_llm_kg_call` /
  `record_llm_gi_evidence_stage_call`) onto **run-level** accumulators shared across parallel
  episodes; the probe forwards everything to the real object (run totals stay correct) while
  isolating **this** episode's cost. Provider-agnostic and non-racy under parallelism.

`cost_usd_total` is the roll-up of the present stage blocks' `cost_usd`.

**`0.0` vs `null` — they are different facts** (corrected 2026-08-15, #1657 acceptance):

| Value | Meaning |
| --- | --- |
| `0.0` | The stage ran and its cost is **known to be zero** — a local engine, no invoice. |
| `null` | **Nobody measured it.** The key is absent from the block. |

This was previously specified the other way round for diarization ("None … not a fabricated
zero") while the code emitted `0.0`, so the document and the implementation disagreed *and* the
implementation disagreed with itself: a locally-diarized episode recorded
`diarization.cost_usd: 0.0` next to `naming.cost_usd: null`, though both ran locally and both
were free.

The corrected rule prefers the measured zero, because it carries information — and it keeps the
remaining `null` meaningful. A fabricated zero on an *uninstrumented* stage is how a roll-up
silently under-reports, which is precisely what `null` must go on protecting.
`measured_or_unmeasured()` in `workflow/processing_manifest.py` is the single implementation;
every stage goes through it.

### `quality_flags` — the rework signal

A flat, corpus-queryable list of the weak-signal conditions each stage emits: `asr_speech_coverage_low`,
`asr_failover`, `unnamed_dominant_voice`, `guest_in_title_not_placed`, `empty_host_anchor`,
`gi_all_gated`. The vocabulary is closed (registered in one place) so the ledger can `GROUP BY` them.

## Amendment 2026-08-15 — `stage_ledger` and `input_fingerprint` (#1647, #1649)

Two additive fields landed with epic #1657. Both are recorded here because "which fields does
a sidecar carry" must be answerable from this document rather than by reading a corpus.

**`processing.stage_ledger`** — per-stage outcome, defined in
[ADR-151](ADR-151-stage-outcomes-over-stage-timings.md). Shape:

```json
"stage_ledger": {
  "speaker_detection": {
    "outcome": "ran",
    "reason": null,
    "detail": {"published_media_bytes": 95900000, "limit_bytes": 26214400,
               "limit_applies_to": "uploaded_audio_after_preprocessing"},
    "duration_seconds": 1.6
  }
}
```

`outcome` ∈ `ran | skipped | failed | degraded`. `reason` is a stable slug so a report can
`GROUP BY` it, matching the closed-vocabulary rule the `qa_flags` list already follows.
`stage_timings` is retained unchanged: it answers "how long", and it never answered "did it
happen" — which is exactly how #1646 stayed invisible across 72 % of the corpus.

**`input_fingerprint`** on the enrichment envelope (not this manifest) — a content hash of an
episode's GI/KG, used as the enrichment staleness key (#1649). Recorded here only so the
cross-artifact picture is in one place; the envelope owns its own shape.

**`schema_version` deliberately unchanged at `1.0.0`.** Both additions are additive optional
fields, and the invariant below says the version moves only on a breaking migration. The cost
of that choice is explicit: a reader cannot tell "this corpus predates the ledger" from the
version alone and must probe for the field — which is what
`scripts/tools/corpus_quality_report.py` does, counting ledger-less episodes as *unknown*
rather than assuming they ran. If that probing becomes load-bearing for more consumers, bump
to `1.1.0` and let readers gate on the version instead; it is a one-line change and this
paragraph is the reason it was not made now.

## Invariants

- **Additive, versioned evolution.** New fields are added under `metrics`; the top-level shape and
  `schema_version` change only on a breaking migration. Readers tolerate unknown `metrics` keys.
- **No PII beyond what the transcript already contains.** The manifest records names the corpus
  already holds; it introduces no new sensitive data.
- **Determinism caveat.** `generated_at` and any wall-clock/pipeline-version stamping is applied at
  the pipeline layer, never inside a pure stage helper (the `Date.now()`/reproducibility rule).

## Consequences

- The `.asr.json` sidecar becomes `stages.asr` of the manifest; ADR-131's provenance is subsumed.
- Cost moves from a per-run JSONL join to a per-episode `cost_usd` roll-up (the JSONL stays as the
  fine-grained event log).
- The ledger enables the RFC-109 morning-after workflow without bespoke scripts.

## Non-Goals

- Not a metrics/tracing backend — a per-episode post-hoc record, not live telemetry.
- Not a gate — the manifest measures; only ADR-131's ASR failover acts on a metric.
