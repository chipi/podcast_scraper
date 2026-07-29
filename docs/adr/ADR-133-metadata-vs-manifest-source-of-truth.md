# ADR-133: `metadata.json` vs the processing manifest — purpose split and source of truth

- **Status**: Proposed
- **Date**: 2026-07-27
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-109](../rfc/RFC-109-per-episode-observability-manifest.md)
- **Related ADRs**: [ADR-132](ADR-132-per-episode-processing-manifest-schema.md) (manifest schema)

## Context & Problem Statement

RFC-109 adds a per-episode processing manifest. `metadata.json` already exists and already carries
some of what the manifest wants (`processing.config_snapshot`, `processing.stage_timings`,
`content.whisper_model`, `content.detected_hosts/guests`). Two records with overlapping fields is a
recipe for drift and "which one is right?" — exactly what the operator flagged. This ADR draws the
line: **what each file is for, which field's source of truth lives where, and how we migrate without
breaking readers.**

## Decision

### Two records, two audiences

- **`metadata.json` = the PRODUCT record.** What the episode *is*, for a **consumer** (the viewer,
  search, an agent, a downstream app): feed/episode metadata, the transcript path, the summary, the
  entities, and the **resolved** hosts/guests/speakers as a *result*. It answers "what's in this
  episode."
- **`<base>.manifest.json` = the PROCESS record.** How the episode was *produced*, for an
  **operator/analyst**: per-stage provenance, quality metrics, method versions, cost, rework flags
  (ADR-132). It answers "how well did we do, with what, and should we redo it."

A consumer never needs the manifest; an operator introspecting quality never needs to parse the
transcript payload. The split is by **audience and question**, not by convenience.

### The dividing line (source of truth per field)

| field / block | today | source of truth after this ADR |
| --- | --- | --- |
| feed / episode metadata (title, guid, duration, links) | `metadata` | **`metadata`** (product) |
| transcript path, media, `transcript_source` | `metadata.content` | **`metadata`** (product) |
| resolved `detected_hosts` / `detected_guests` / `speakers` | `metadata.content` | **`metadata`** (the *result* is product) |
| summary / GI / KG artifacts + their counts | `metadata.{summary,gi,kg}` | **`metadata`** (product); the manifest keeps only the *quality metrics* (counts, gate drops) |
| **actual ASR model + speech coverage + failover** | `.asr.json` / nowhere | **manifest** (`stages.asr`) |
| **`config_snapshot`** | `metadata.processing` | **manifest** (provenance) |
| **`stage_timings`** | `metadata.processing` | **manifest** (`stages.*.duration_s`) |
| **`run_id` / `pipeline_version` / `method_version`** | partial in `metadata.processing` | **manifest** (provenance) |
| **cost per stage / total** | per-run cost JSONL only | **manifest** (roll-up); JSONL stays the event log |
| naming quality (detected-vs-named, snaps, method) | `.speakers.diagnostics.json` (deep) | **manifest** (`stages.naming` summary); diagnostics stays the deep per-voice detail |
| `qa_flags` / `expectations` | `metadata.content` | **manifest** (`quality_flags`); migrate + deprecate the metadata copy |

Rule of thumb: **a *result a consumer uses* → `metadata`; a *fact about the processing* →
manifest.** Where a signal has a deep form and a summary form (naming diagnostics, cost events), the
**deep artifact stays** (`.speakers.diagnostics.json`, cost JSONL) and the **manifest holds the
queryable summary** — the manifest is the index, the sidecars are the detail.

### Migration (staged, non-breaking)

1. **Add, don't move (write-both).** Introduce the manifest; the migrating fields
   (`config_snapshot`, `stage_timings`, actual model) are written to **both** the manifest (SoT) and
   their old `metadata` location for one release, so no reader breaks.
2. **Back-reference.** `metadata.processing` gains a `manifest_path` pointer, and its provenance
   fields are marked deprecated in the schema docstring (still populated).
3. **Stop writing the duplicates** on a `schema_version` boundary, once readers (viewer, search, any
   downstream) are confirmed off the deprecated `metadata.processing` provenance fields — grep +
   fix referrers first (the doc-vs-code discipline).
4. `metadata.processing` shrinks to a thin `{ run_id, manifest_path, schema_version }` link; all
   provenance/quality lives in the manifest.

No episode is rewritten just to migrate; the split takes effect as episodes are (re)processed, and a
one-off backfill can populate manifests for the existing corpus if/when a query needs them.

## Consequences

- **One answer per question.** "Which model produced this?" → manifest. "What's the summary?" →
  metadata. No field is authoritative in two places once migration completes.
- **The viewer/search keep reading `metadata`** unchanged for product data; only tooling that wants
  provenance/quality reads the manifest — a clean, additive split.
- **`.speakers.diagnostics.json` and the cost JSONL keep their role** as the deep detail the manifest
  summarizes; nothing is deleted, duplication is removed at the *summary* level.

## Non-Goals

- Not merging the two files — they have different audiences and lifecycles.
- Not a big-bang rewrite of the existing corpus's metadata — migration is by (re)processing +
  optional backfill.
