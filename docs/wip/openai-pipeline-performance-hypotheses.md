# OpenAI / API-heavy pipeline — performance hypotheses (WIP)

**Status:** Working notes — not normative. Tie-break with fresh `metrics.json`, frozen profiles, and
`*.stage_truth.json` after RFC-064 capture.

**Context:** WIP OpenAI-oriented runs (e.g. `data/profiles/v2.6-wip-openai.yaml` class of configs)
often show **large aggregate stage walls** versus **modest end-to-end wall time** because many
stages run in **parallel** or overlap in worker pools. Hypotheses below are **testable**; numbers
come from real captures, not this doc.

**Related:** [GitHub Issue #477](https://github.com/chipi/podcast_scraper/issues/477) — bundling /
consolidating LLM calls (cleaning + summary + bullets, etc.) to reduce cost and latency. That
issue is the **engineering experiment**; this note is **how to interpret profiles** so runtime
stage data lines up with token/latency eval. See also RFC-060 (cleaning cost/quality alignment
with #477).

---

## H1 — Cleaning and summarization dominate *sequential* cost per episode

**Hypothesis:** For OpenAI transcript cleaning + chat summarization, **LLM-bound wall** in
`transcript_cleaning` and `summarization` (per-episode or aggregate) accounts for a large share of
**work** even when overall CPU% looks low (network/API wait).

**How to test:** Compare `avg_cleaning_seconds`, `avg_summarize_seconds`, and per-episode maps
(`cleaning_time_by_episode`, `summarize_time_by_episode`) in `metrics.json` / `stage_truth` excerpt
vs `transcription` when Whisper is off or cached.

**If true:** Biggest wins are **fewer round-trips** (batching, prompt size), **cheaper/faster
models** where quality allows, and **caching** of cleaning/summary inputs.

---

## H2 — Aggregate stage times sum to far more than `run_duration_seconds` (parallelism)

**Hypothesis:** `sum(stage wall estimates)` ≫ `totals.wall_time_s` / `run_duration_seconds` because
stages are **not** serialized; the profile’s proportional split is a **cost accounting** view, not
a critical path.

**How to test:** Read `parallelism_hint_ratio` in `*.stage_truth.json` and compare to workflow
concurrency settings (episode workers, stage overlap).

**If true:** Optimize **critical path** (often feed fetch + last stragglers + index flush), not only
the largest single-stage aggregate.

---

## H3 — Low CPU% on GI/KG/API stages reflects I/O wait, not “free” hardware

**Hypothesis:** Stages that call remote APIs or block on I/O show **low average CPU%** in psutil
samples while still consuming **wall time**; machine is not idle — threads wait on network.

**How to test:** Correlate `io_and_waiting_wall_seconds` / wait buckets with those stages; inspect
concurrency.

**If true:** Improving **latency** (region, connection reuse, smaller payloads) beats local CPU
tuning.

---

## H4 — Cache hits collapse transcription and preprocessing in repeat runs

**Hypothesis:** Second run on the same corpus shows **near-zero** transcription and download time if
cache keys hit; profiles without cache look dominated by ML/API stages.

**How to test:** Diff two freezes (same config) with cold vs warm cache; compare
`transcribe_count`-derived wall vs `vector_index_seconds`.

**If true:** Report **cold vs warm** explicitly in performance reports; avoid comparing mixed modes.

---

## H5 — `vector_indexing` RSS is easy to misread for short, MPS-heavy windows

**Hypothesis:** Very short index windows may fall between psutil samples; **MPS** further skews RSS.
Stage truth uses **fallbacks** (see Performance Profile Guide) — treat extreme values as **bounds**,
not exact attribution.

**How to test:** Tighten `--sample-interval`, compare `resource_by_stage_psutil` across captures on
the same machine.

---

## H6 — Speaker detection / NER is rarely the top cost when LLM cleaning is on

**Hypothesis:** With aggressive LLM cleaning, **`speaker_detection`** wall is smaller than cleaning
+ summary in typical episode lengths.

**How to test:** Rank `wall_seconds_by_stage` from `stage_truth` for representative configs.

**If false:** Profile-specific; invest in NER batching or model swap for that corpus.

---

## Next steps (engineering)

1. Keep **one** reference OpenAI WIP freeze + `stage_truth` per release line for regression diffs.
2. When optimizing, pick **one** lever (e.g. cleaning model, `max_episodes`, concurrency) per
   experiment so `profile-diff` stays interpretable.
3. Revisit this list after RFC-066 / compare UI automates run-to-run stage tables.
4. Bundled clean+summary experiment plan and configs: [issue-477-llm-bundle-experiment-plan.md](issue-477-llm-bundle-experiment-plan.md).
