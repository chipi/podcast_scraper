# RFC-109 — Per-episode observability manifest: measure in production, introspect, evolve

**Status:** Draft
**Date:** 2026-07-27
**Tracking issue:** [#1337](https://github.com/chipi/podcast_scraper/issues/1337)
**Depends on / relates to:** [ADR-132](../adr/ADR-132-per-episode-processing-manifest-schema.md)
(manifest schema + versioning), [ADR-133](../adr/ADR-133-metadata-vs-manifest-source-of-truth.md)
(metadata.json vs manifest source of truth), [ADR-123](../adr/ADR-123-quality-gate-transcription-failover.md)
/ [ADR-131](../adr/ADR-131-speech-normalized-coverage-gate.md) (the ASR gate whose provenance seeded
this), the prod-v2.x reprocess arc (#1190).

## Context — the operating model this enables

The whole prod-v1→v2→v3 journey found its defects **by hand**: we noticed turbo mangled host names,
that a coverage gate fired on the wrong metric, that `feed.description` was never wired, that a guest
was over-split into two spellings. Each was a real bug, but each was found by an operator eyeballing
a corpus and a code trace — not by a metric that said "these 6 episodes are weak, here's why."

The intent of this RFC is to invert that. We are **not chasing per-episode perfection.** We are
building the **visibility to run at scale (100–500 episodes/night), review the next morning, find the
weak episodes and the systemic gaps, evolve a specific stage, and reprocess only the affected
subset** — while the system keeps running. The failover gate is the proof case: with per-episode ASR
quality on record from day one, we would have *discovered* the need for it from the data after 1000
episodes, instead of by manual analysis. Observability turns "we happened to notice" into "the
metric told us."

Concretely, the questions we want answerable the next morning, without writing a bespoke script each
time:

- Which episodes have low ASR speech coverage? Which actually failed over, to which model?
- Where did diarization leave a dominant voice unattributed? Where did naming detect a guest in the
  title but fail to place it on a voice?
- What did each episode cost, by stage and provider? Which feeds are expensive?
- Which episodes were produced by an **older version** of a stage's logic (e.g. naming before
  ADR-130), so we can reprocess exactly those after we ship an improvement?

## What exists today (and why it is not enough)

Per-episode signals already exist — but **scattered across 4+ artifacts and a log**, and none of them
is a queryable, complete, versioned record:

- `metadata.json` `content` block: `detected_hosts` / `detected_guests` / `speakers` / `qa_flags`.
- `metadata.json` `processing` block: `config_snapshot`, `stage_timings`, `run_id`.
- `.speakers.diagnostics.json`: the roster's per-voice reasoning + `unattributed_talk_share` + alarms.
- `.asr.json` (ADR-131): the actual ASR model + speech coverage + failover breadcrumb.
- a per-run `cost` JSONL: one `llm_cost` event per LLM call.

The gaps that make this un-runnable as an operating model:

1. **Actual ≠ recorded.** `content.whisper_model` is the *configured* model, not the one that ran (a
   failover is invisible there — only `.asr.json` knows). Any field written from config rather than
   from the stage's result will rot the same way.
2. **No consolidated quality view.** Coverage, unattributed talk, host/guest *detected-vs-named*,
   naming-method breakdown, GI gate drops, cost — each lives in a different file. "Which episodes are
   weak and why" needs a custom join every time.
3. **No method/logic versioning.** Nothing records *which version* of a stage produced the output,
   so "reprocess every episode named before naming-vX" is not expressible.
4. **No aggregation surface.** Everything is per-episode JSON in run dirs; correlation across 90 (or
   100k) episodes means grepping.

## Proposal

### 1. A per-episode **processing manifest** — the observability contract

One canonical, versioned, queryable record per episode describing **how it was produced**, written
by the stages themselves. Schema + versioning are specified in **ADR-132**. Three parts:

- **Provenance** — pipeline version, and per stage `{ran, skipped, method, method_version, model,
  duration_s, cost_usd, warnings}`, with the **actual** model/config, not the configured one.
- **Quality metrics** — ASR (speech coverage, failover), diarization (speakers,
  `unattributed_talk_share`), naming (hosts/guests detected vs named, snaps, canonicalizations, the
  method that named each voice), GI/KG (counts, gate drops), cost totals.
- **Flags** — the "candidate for rework" signals: low coverage, an unnamed dominant voice, a guest
  named in the title but never placed, a failover, an empty host anchor.

### 2. Each stage **owns and writes its own reality**

The single most important rule, and the one that keeps the manifest honest: **the stage that does
the work writes its manifest block from its own result** — ASR writes the ASR block (including a
failover), diarization writes the diarization block, the roster writes the naming block. No block is
ever written from `cfg`. A field nobody owns does not exist. This is the discipline that stops the
`whisper_model`-rot failure mode from recurring.

### 3. Versioning is non-negotiable, from day one

Versioning is **layered** (ADR-132 §Convention 2), because "which code ran?" and "which episodes
need re-running after I changed one stage?" are different questions:

- **`git_sha` (+ dirty)** — captured at runtime, the exact code. The precise ground-truth backstop,
  but opaque and noisy, so not the query key.
- **`pipeline_version`** — a semantic version of the **stage composition** (which stages, in what
  order); bumped on a re-wire like moving the ASR gate downstream of diarization (ADR-131), which no
  single stage's version captures.
- per-stage **`method_version`** — bumped when a stage's **logic** changes (naming → a new version
  after ADR-130 + 2a/3; the ASR gate → a version per metric change). This is the **query key**: it is
  what makes "ship an improvement, then reprocess *exactly* the episodes produced by the old logic" a
  targeted query rather than an over-selecting sweep or a guess.

### 4. An append-only **corpus ledger** for query

Every episode's manifest is also flattened into one row appended to a corpus-level
`manifest.jsonl` (or a small DuckDB/SQLite built from it). An overnight run of 500 episodes produces
a ledger you can query the next morning — bottom-5%-by-coverage, cost-by-feed,
episodes-below-naming-vX — with DuckDB/pandas, and later a dashboard. This is where the operating
model actually lives.

### 5. Relationship to `metadata.json` — no duplication, clear source of truth

Specified in **ADR-133**. In short: `metadata.json` is the **product** record (what the episode *is*
— transcript, summary, entities, the resolved hosts/guests a consumer reads); the **manifest** is
the **process/observability** record (how it was produced — provenance, quality, versions, cost, an
operator/analyst reads). Provenance that lives in `metadata.processing` today
(`config_snapshot`, `stage_timings`) migrates to the manifest as the SoT, with `metadata` keeping a
back-reference; product fields stay in `metadata`. The migration is staged so nothing breaks.

## Phased rollout

1. **Manifest sidecar.** Generalize the ADR-131 `.asr.json` into a per-episode `.manifest.json`,
   stage-owned, versioned. (Seed already exists.)
2. **Ledger.** Append a flattened row per episode into a corpus `manifest.jsonl`; a tiny query helper
   (DuckDB) for the morning-after workflow.
3. **Metadata SoT migration.** Move provenance out of `metadata.processing` per ADR-133; deprecate
   the duplicated fields on a version boundary.
4. **Dashboard / query surface.** Expose the ledger to the operator viewer / an observability tool
   for correlation. (Separate RFC when we get there.)

## Non-Goals

- Not a real-time metrics/tracing system (Prometheus/OTel) — this is a **per-episode, post-hoc**
  record for introspection and targeted reprocessing, not live SRE telemetry.
- Not a replacement for `metadata.json` — the two are complementary (ADR-133).
- Not chasing per-episode correctness — the manifest measures quality; it does not gate on it (the
  ASR failover is the one place a metric drives an action, and that stays owned by ADR-131).

## Why now (v2.4 / v2.5)

We are about to change GI/KG (v2.4) and swap gemini for a DGX LLM (v2.5). Both are exactly the kind
of change where "ship it, measure the delta on the next night's run, keep or revert" beats "reason
about it in advance." Standing this up now means v2.4/v2.5 are practiced *with* the observability
loop, not retrofitted after.
