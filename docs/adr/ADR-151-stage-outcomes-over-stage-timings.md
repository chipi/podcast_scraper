# ADR-151 — Record stage OUTCOMES, not just stage timings

- **Status:** Accepted
- **Date:** 2026-08-15
- **Issue:** [#1647](https://github.com/chipi/podcast_scraper/issues/1647)
- **Epic:** [#1657 — corpus integrity](https://github.com/chipi/podcast_scraper/issues/1657)
- **Supersedes nothing.** Extends the per-episode metadata sidecar (Issue #379 stage timings).

## Context

The pipeline recorded per-episode stage **durations** in
`processing.stage_timings` — `download_media_time`, `transcribe_time`,
`extract_names_time`, `cleaning_time`, `summarize_time`. Every one of them is
`Optional[float]`, and every absence renders as `null`.

`null` is ambiguous across three very different facts:

| What happened | What was recorded |
| --- | --- |
| The stage was deliberately skipped | `null` |
| The stage raised and the error was swallowed | `null` |
| The stage was never configured for this run | `null` |
| The stage ran and took no measurable time | `null` |

That ambiguity was not theoretical. In [#1646](https://github.com/chipi/podcast_scraper/issues/1646),
speaker detection was skipped for every episode whose audio exceeded 25 MB.
The skip path returned before `record_extract_names_time` was called, so it
produced no timing, no log line, and no error. Measured on the live corpus:

```text
488 / 678 episodes (72.0 %)      speaker detection skipped
2,112 / 8,952 insights (23.6 %)  unsurfaceable as a result
82 episodes (12.1 %)             lost every insight they had
```

Every other signal stayed green throughout, because every other signal measured
artifact **presence**: `/api/corpus/coverage` reported `with_gi=678, with_kg=678,
with_neither=0` on that same corpus. Coverage is not correctness, and a corpus
can be fully covered and substantially unusable at the same time.

## Decision

**Every stage records an explicit outcome. `null` stops being a legal way to say
"nothing happened".**

`ProcessingMetadata` gains `stage_ledger: Dict[str, StageOutcome]`:

```json
"speaker_detection": {
  "outcome": "ran",
  "reason": null,
  "detail": {"published_media_bytes": 95900000, "limit_bytes": 26214400,
             "limit_applies_to": "uploaded_audio_after_preprocessing",
             "preprocessing_enabled": true,
             "transcription_provider": "deepgram", "has_transcript_urls": false}
}
```

**Why the size key is `published_media_bytes` and not `media_bytes`.** The probe HEADs the
publisher's URL, but the 25 MB cap applies to the file that is **uploaded** — and audio
preprocessing runs in between, cutting ~90 % (measured: 91.5 MB → 9.1 MB). A bare
`media_bytes` invited exactly the misreading that a large published file meant a rejected
upload. It never did; the uploaded files were nowhere near the cap. `limit_applies_to`
records that explicitly so a future reader cannot repeat the inference.

Four outcomes, chosen so the distinctions that matter survive:

- **`ran`** — completed.
- **`skipped`** — deliberately not run.
- **`failed`** — it *ran* and produced nothing.
- **`degraded`** — produced output through a fallback path.

`failed` versus `skipped` is the distinction that carries the most weight.
"Ran and found nobody" and "never ran" are different facts about an episode, and
collapsing them is what made #1646 unreadable from the outside.

`reason` is a **stable machine-readable slug**, not prose, so a report can group
by it — `412 skipped: media_over_size_limit_no_transcript_urls` is a bug report,
where `412 skipped` is merely a number. `detail` carries the deciding inputs so
the decision is auditable without logs that may no longer exist.

`stage_timings` is retained unchanged. It answers "how long", which is still a
useful question; it simply never answered "did it happen".

## Consequences

**Positive.**

- A skipped stage is identifiable from the metadata sidecar alone, with its
  reason, without reading logs.
- Attribution becomes measurable per episode and corpus-wide, alongside the
  existing coverage figures.
- The run-scale quality report (`scripts/tools/corpus_quality_report.py`) can
  answer *"I ran 1 / 10 / 50 / 5000 episodes — how did it go?"* at any scale,
  with a `NOT MEASURED` section of equal prominence, because a report that lists
  only what it checked lets silence read as health.
- `unattributed_alarm` can finally fire on the case it was built for. The
  "Pattern B" rule in `providers/ml/diarization/roster.py` correctly excludes
  `unidentified` talk from the defect share — "nobody in the episode says who
  they are" is not our failure, and counting it fired the alarm on narrated desks
  like Planet Money for doing nothing wrong. That reasoning holds *only if
  detection actually looked*. With the stage skipped, every voice degrades to
  `unidentified`, the defect share collapses to `0.0`, and the alarm read `false`
  on episodes that had lost 100 % of their insights. The alarm now also trips when
  nothing was named **and** detection never ran; the threshold and basis still come
  from the labeling profile ([ADR-140](ADR-140-versioned-labeling-profiles.md)).

  *Note:* the code comments cite "ADR-139" for Pattern B, but `ADR-139` in this
  tree is the text-normalization contract. The citation looks stale; it is left
  alone here rather than rewritten on a guess, and is worth confirming separately.

**Negative / accepted costs.**

- One more field per episode in the sidecar. Bounded: outcomes are counts and
  slugs, never name lists.
- Episodes processed before this ADR carry no ledger. Reports count them as
  *unknown* rather than assuming they ran — assuming is the original defect.
- Every new stage must remember to record an outcome. Mitigated by making the
  recorder a small closure in each stage function, so the recording sits next to
  the `return` it describes rather than in a distant wrapper.

## Alternatives considered

**Infer the outcome from the timing.** Rejected: this is exactly the inference
that failed. `null` cannot be disambiguated after the fact, which is why the
corpus needed a 678-episode re-scan to establish what had happened.

**Log it and parse logs later.** Rejected: logs are rotated, sampled, and absent
from the artifact an operator actually inspects. The `.viewer/` spawn log for
enrichment was already unreadable through the API when it was needed
([#1653](https://github.com/chipi/podcast_scraper/issues/1653)).

**Fail hard on a skipped stage.** Rejected as too blunt. Some skips are correct
(`dry_run`, `auto_speakers_disabled`). The defect was never that stages get
skipped — it was that skipping was indistinguishable from success.

## Verification

`tests/unit/podcast_scraper/quality/test_report_surfaces_1646.py` replays all
**678 real pre-fix episodes** from `data/baselines/corpus-integrity-2026-08-14.json`
through the quality report and asserts a reader can see the damage — `skipped=488`,
`6840/8952 = 0.764`, 82 episodes fully zeroed — using only what the report prints.

It asserts nothing about the size gate. If the report is later simplified and that
test still passes, the report works; if it fails, the blindness has returned,
whatever the gate is doing.
