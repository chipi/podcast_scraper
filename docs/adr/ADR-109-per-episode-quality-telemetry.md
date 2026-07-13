# ADR-109: Per-episode quality telemetry — make silent GI failures loud

**Status:** Proposed
**Date:** 2026-07-13
**Deciders:** Marko
**Related:** ADR-053 (grounding contract), #1182 (zero quotes on unstructured transcripts)

## Context

Nine defects were found in a single session. Every one produced plausible output and reported
success:

| defect | what the pipeline reported |
| --- | --- |
| insight clamp (`gi_max_insights: 12` dead in 26 profiles) | success, 10 insights/episode |
| 50k transcript cut (6 providers) | success, insights "front-loaded" |
| value gate judge 404ing -> gate inert | success, 44.3 insights/episode |
| stub fallback on token overrun | success, 1 insight |
| ollama endpoint ignored -> ran on the laptop | success, "DGX" numbers |
| blob transcripts -> ZERO quotes, every provider | success, 0 grounded insights |
| insight temperature hardcoded -> not reproducible | success, grounding 79.8% or 94.5% |

They were caught only because a human was staring at 3-20 episodes by hand.

**That does not scale, and the next run is 100 episodes.** A corpus with a third of its evidence
missing would look identical to a healthy one: GI quality is aggregated **per run**, not per
episode, so thirty broken episodes hide behind a healthy-looking average. This is exactly how
prod-v2 accumulated 125 single-line-blob transcripts that nobody noticed for two months.

## Decision

Emit **per-episode** GI quality metrics, surface them in logs and Prometheus, and end every run
with a quality summary that makes a bad corpus impossible to ship by accident.

### 1. Per-episode quality fields (`EpisodeMetrics`)

- `gi_insights_emitted` — before the value gate
- `gi_insights_dropped_by_gate`
- `gi_insights_grounded`
- `gi_quotes_total`
- `gi_grounding_rate_pct`
- `gi_quotes_per_insight`

### 2. Failure flags — one per known silent mode

Each flag exists because that failure actually happened and was invisible:

- `stub_fallback` — extraction died; the episode carries 1 placeholder insight
- `zero_quotes` — no evidence at all (#1182: the blob-transcript failure)
- `gate_failed_open` — the value gate errored and kept everything (a broken gate and a permissive
  gate produce identical artifacts; only this flag distinguishes them)
- `truncation_salvaged` — the model overran its token budget and we kept a partial list
- `insights_at_ceiling` — the count equals `gi_max_insights`, i.e. the prompt is not constraining
  it and we are truncating

### 3. One-line per-episode quality log

```text
[quality] ep=0007 insights=24 (gate dropped 11) grounded=20 (83.3%) quotes=41 q/i=1.71  OK
[quality] ep=0008 insights=1  (gate dropped 0)  grounded=0  (0.0%)  quotes=0  q/i=0.00  STUB_FALLBACK ZERO_QUOTES
```

Visible in a real run without opening a JSON file.

### 4. Prometheus gauges

`prometheus_client` is already an optional dependency and wired for guardrail telemetry; the
pipeline's GI metrics simply are not exported. Add gauges/counters for the fields above, labelled
by feed. This is the Grafana / o11y hook.

### 5. Run-level quality summary — the closed loop

```text
GI quality: 100 episodes
  below ADR-053 contract (<80% grounded): 12
  zero evidence:                           3
  stub fallback:                           1
  gate failed open:                        0
  insights at ceiling:                    41   <- the prompt is not constraining the count
```

A run that produces this line cannot silently ship a broken corpus.

## Consequences

**Positive**

- Every silent failure found this session becomes loud, per episode.
- A 100-episode run self-reports its own quality; no manual inspection needed.
- Grafana can trend grounding rate per feed over time — regressions surface as they happen rather
  than months later in an eval.

**Negative / costs**

- `EpisodeMetrics` grows; the GI stage must thread metrics through paths that currently pass
  `pipeline_metrics=None` (notably the eval harness).
- Prometheus export is optional-dependency-gated, so behaviour must degrade cleanly when it is
  absent.

**Explicitly NOT in scope**

- Failing the run on a quality breach. Report first, decide thresholds from real data. A hard gate
  before we know the natural distribution would just be another magic number.

## Validation

The telemetry must **catch the failures we already know about** before it is trusted on 100
episodes:

1. Run 10-20 episodes with the telemetry on; confirm the summary matches the eval numbers.
2. Feed it a known-bad episode (a single-line blob transcript) and confirm `zero_quotes` fires.
3. Point the value gate at a bad model and confirm `gate_failed_open` fires (this is the exact
   404 failure that let a full 10-episode run complete ungated).
4. Only then run the 100-episode reprocess.

## Alternatives considered

**Rely on the eval harness.** Rejected: evals run on pinned datasets after the fact. Production
runs are where the silent failures actually happen, and they are the runs nobody inspects.

**Aggregate-only metrics with alert thresholds.** Rejected: an average hides the failure. The 8
zero-evidence episodes in the generalization set were invisible in the mean and obvious per episode.
