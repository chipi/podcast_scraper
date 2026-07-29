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
  Local pyannote/DGX/MOSS have no pricing entry → **None** (a truthful "no billed cost", not a
  fabricated zero).
- **Naming** — local heuristic → no cost.
- **Summary** — `summary_call_metrics.estimated_cost`.
- **GI / KG** — captured per-episode by an `EpisodeCostProbe` that wraps the shared
  `pipeline_metrics` around each episode's build. GI/KG cost is recorded by the LLM providers (all of
  gemini/deepseek/grok/anthropic/mistral funnel through `record_llm_gi_call` / `record_llm_kg_call` /
  `record_llm_gi_evidence_stage_call`) onto **run-level** accumulators shared across parallel
  episodes; the probe forwards everything to the real object (run totals stay correct) while
  isolating **this** episode's cost. Provider-agnostic and non-racy under parallelism.

`cost_usd_total` is the roll-up of the present stage blocks' `cost_usd`.

### `quality_flags` — the rework signal

A flat, corpus-queryable list of the weak-signal conditions each stage emits: `asr_speech_coverage_low`,
`asr_failover`, `unnamed_dominant_voice`, `guest_in_title_not_placed`, `empty_host_anchor`,
`gi_all_gated`. The vocabulary is closed (registered in one place) so the ledger can `GROUP BY` them.

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
