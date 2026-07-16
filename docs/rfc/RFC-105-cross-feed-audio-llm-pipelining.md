# RFC-105: Cross-feed audio↔LLM pipelining

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core Pipeline, Ops
- **Related RFCs**:
  - `docs/rfc/RFC-063-multi-feed-corpus-append-resume.md` (multi-feed corpus contract)
- **Related Issues**:
  - [#1180](https://github.com/chipi/podcast_scraper/issues/1180) — parallelism observability metrics that would prove this design out
- **Related Guides**:
  - `docs/guides/PIPELINE_AND_WORKFLOW.md` § Concurrency swim-lane
  - `docs/architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md` § B.6a Per-feed concurrency model

## Abstract

Within one `run_pipeline(feed)` the pipeline overlaps audio work (Whisper /
diarization) with LLM work (metadata / summary / GI / KG) via a queue handoff
between two dedicated threads. Between feeds, however, the pipeline serialises:
`corpus_operations.py` runs `run_pipeline(feed_1)` to completion before
starting `run_pipeline(feed_2)`. On a multi-feed corpus, that means feed-N's
LLM slot idles while feed-(N+1) is still downloading + transcribing audio — a
gap the #1180 overlap-coefficient metrics can now measure but the code cannot
close.

This RFC sketches a shared-queue model across feeds so feed-(N+1)'s audio
starts as soon as feed-N's LLM tail is thin enough. Keeps the current
per-feed contract intact where it can, calls out where it must break.

## Problem Statement

The current per-feed pipeline shape (see PIPELINE_AND_WORKFLOW.md swim-lane):

```
Feed 1:  Main → Transcription → ProcessingProcessor →  END
Feed 2:                                                 → Main → Transcription → Processing → END
Feed 3:                                                                          → Main → ...
```

Between-feed idle: feed-N's ProcessingProcessor drains while nobody prepares
audio for feed-(N+1). On a healthy in-feed run, `processing_overlap_ratio`
sits near 0.6-0.8; the between-feed dip drags the corpus-wide effective
overlap materially lower — especially on API-based summary providers where
processing tail is dominated by rate limits, not compute.

## Goals

- **Feed-(N+1) audio starts overlapping feed-N LLM tail.** The gap between
  `run_pipeline(feed_N)` return and `run_pipeline(feed_N+1)` audio start
  collapses to ~zero on healthy runs.
- **Per-feed metrics accounting still works.** Each feed's summary JSON
  reports its own `processing_overlap_ratio` / `processing_thread_busy_ratio`
  covering its own episodes; cross-feed contribution shows in a new
  corpus-wide roll-up.
- **No behaviour change on single-feed runs.** A pipeline invoked on one
  feed should be indistinguishable from today.
- **Failures do not cross feeds.** A crash processing feed-N does not
  corrupt or block feed-(N+1)'s in-flight state.

## Non-Goals

- **Not increasing per-feed parallelism.** `transcription_parallelism` and
  `processing_parallelism` stay per-feed knobs. Cross-feed pipelining is a
  separate axis.
- **Not sharing ML model loading across feeds.** Model warmup already
  amortises across a run via the `multi_feed_ml_cleanup_deferred` path
  (`orchestration.py:659`). Cross-feed queuing extends that idea to
  work-in-flight, not to models.
- **Not changing the corpus write contract.** Each episode still gets its
  own final artifact directory; no cross-feed write locking beyond the
  existing single-corpus manifests.
- **Not implementing multi-worker services or Redis / RQ.** The
  `PLATFORM_ARCHITECTURE_BLUEPRINT.md` "Part B" service split is a much
  larger move; this RFC is strictly about within-process cross-feed
  queueing.

## Proposed Design

### High level

Lift the `TranscriptionProcessor` and `ProcessingProcessor` threads out of
per-feed scope and up to `corpus_operations.py` scope. `run_pipeline(feed)`
becomes a producer that pushes into shared queues; the two long-running
threads consume across feeds.

```
Corpus run
  ├── SharedTranscriptionQueue
  ├── SharedProcessingQueue
  ├── TranscriptionProcessor thread  (persists across all feeds)
  ├── ProcessingProcessor thread     (persists across all feeds)
  └── for feed in feeds:
        run_pipeline(feed) → downloads + enqueues audio jobs
        (returns as soon as its audio is enqueued, not when its LLM tail drains)
```

The shared workers keep pulling from the queues until every enqueuer signals
"no more work" and both queues drain.

### Queues

- **Transcription queue.** Bounded (`transcription_queue_size`) — same
  backpressure story as today, but now capacity is shared across feeds.
  Jobs carry `feed_id` so per-feed metrics accounting can still attribute
  work.
- **Processing queue.** Same shape, `ProcessingJob` extended with
  `feed_id`.
- **Poison-pill terminator.** When a feed finishes downloading, it enqueues
  a per-feed sentinel; the workers count sentinels to know when all
  producers are done.

### Metrics accounting

- Each `ProcessingJob` and transcription job records its own `enqueued_at`
  timestamp and `feed_id`. Thread-active intervals record `feed_id` too.
- Per-feed summary JSON filters intervals to that feed's jobs before
  computing `processing_overlap_ratio` / `processing_thread_busy_ratio`.
  Numbers stay meaningful per feed.
- A new **corpus-wide roll-up** (`corpus_summary.json`) reports the
  cross-feed overlap: fraction of total wall-time during which BOTH the
  transcription and processing thread had at least one active feed. That's
  the number that tells us whether the design pays off.
- The safety-net counter (`safety_net_processed_episodes_count`) rolls up
  across feeds — a per-feed warning still fires locally.

### Error handling

- **Feed-level fail-fast** stays local. A feed hitting `fail_fast=True`
  stops enqueuing its own jobs but does not affect other feeds' in-flight
  work.
- **Worker-level failures** — an unhandled exception in the shared workers
  kills the corpus run. Same behaviour as today but with a bigger blast
  radius. Mitigation: wrap the worker main loops in a top-level catch that
  logs and marks the corpus run failed but drains the current job before
  exiting.
- **Interim checkpoints** (see `_InterimCheckpointManager`) still trigger
  per feed — the shared workers' state has no additional persistence.

### Deferred cleanup

The existing multi-feed ML-cleanup deferred flag (
`begin_multi_feed_ml_batch` / `end_multi_feed_ml_batch`) already knows how
to hold model unloads until the whole corpus is done. Cross-feed pipelining
piggybacks on that path — no model gets unloaded mid-run.

## Risks

1. **Metrics attribution complexity.** Per-feed ratios need per-feed
   interval filtering. Get this wrong and the "did cross-feed pipelining
   help?" question becomes unanswerable. Mitigation: unit tests over
   the filtering math using the same interval helpers from #1180.
2. **Cross-feed error blast radius.** A crash in shared workers kills all
   feeds. Mitigation: hard exception boundary; mark corpus run failed but
   drain gracefully.
3. **Downstream write conflicts.** `corpus_manifest.json` and index
   incremental updates already handle multi-feed appends
   (`append_resume.py`). Cross-feed WRITE parallelism to the same corpus
   manifest is new — needs a mutex or a queue.
4. **Cost-cap enforcement.** `enforce_cost_soft_cap` currently runs
   per-stage per-feed. Cross-feed pipelining means one feed's LLM cost
   spike could hit the cap while others are still transcribing — needs a
   corpus-wide cap check + per-feed rollback semantics.
5. **Live monitor semantics.** `.pipeline_status.json` currently reflects
   one feed's stage. With shared workers there is no single "stage" — the
   monitor needs a new schema, per-feed rows or an aggregate.

## Open Questions

- Does the deferred model-cleanup rely on the LAST feed knowing it's last?
  If so we need an explicit "corpus done" signal separate from queue
  drain.
- Should we ship this behind a flag (`cross_feed_pipelining`) that
  defaults False for one release cycle to shake out issues? Recommended.
- What's the right cap on `transcription_parallelism` shared across feeds
  vs summed across feeds? Prior default 1 was chosen because Whisper
  local can't safely go higher — that constraint doesn't change.
- Cost-cap semantics — is a corpus-wide cap even coherent, or should each
  feed still enforce its own cap independently?

## Rollout Plan

1. **Measure baseline.** Land #1180 (done). Run a real multi-feed corpus,
   record per-feed `processing_overlap_ratio` and estimate the between-feed
   gap from `run_duration_seconds` minus (per-feed active windows).
2. **Prototype behind a flag.** `cross_feed_pipelining: bool = False` in
   `Config`. When True, `corpus_operations.py` uses shared queues.
3. **Shadow test.** Run the same corpus with and without the flag; compare
   corpus-wide wall-time + confirm per-feed metrics are unchanged for the
   flag=False path.
4. **Bake in a real deploy for one sprint.** Watch the corpus-wide overlap
   metric.
5. **Flip the default.** Only after (4) shows a clear win and no
   regressions.

## Alternatives Considered

- **Multi-worker services (Compose split).** Real answer for large-scale
  multi-tenant. Massively over-scoped for our current single-process
  pipeline. Deferred to the "Part B" work in
  `PLATFORM_ARCHITECTURE_BLUEPRINT.md`.
- **Naive `asyncio` rewrite.** Would blur the thread boundaries the current
  metrics are built on; unclear win over shared thread-safe queues; large
  refactor. Not recommended.
- **Do nothing.** Acceptable if the between-feed gap turns out small in
  practice — which is what step 1 of the rollout answers before we commit
  to (2)+.

## Success Metrics

- **Corpus-wide overlap ratio > 0.5** on a representative 5-feed run with
  the flag on. Baseline measurement needed first.
- **Per-feed `processing_overlap_ratio` unchanged** on flag-off runs (proves
  we didn't regress the per-feed path).
- **`safety_net_processed_episodes_count` remains 0** across a corpus run
  with the flag on (proves cross-feed handoff didn't drop episodes).
