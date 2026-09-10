# Backfill every feed to one year of coverage

**Status:** Queued — starts after Batch C class-A scout.
**Decided:** 2026-09-10 (operator).
**Measured:** 2026-09-10 against the live prod corpus (34 feeds, 1,694 episodes).

## The goal

Every feed in the corpus should span **at least 6 months**, ideally **1 to 1.5 years**.
Two years is fine. The purpose is a long enough timeline to judge how the enrichment
capabilities actually perform, which a 3-month window cannot support.

**Working target for this piece of work: 365 days per feed.** That is the bare minimum,
not the ideal.

## Sequence

1. Batch B phase 3 — finishing 2026-09-10 (Peter Attia, In Moscow's Shadows, TRIP: Leading).
2. **Batch C class-A scout** — 20 feeds, 1 episode each (§5g phase 1).
3. **This backfill** — the next substantial piece of work after that.

## Cost

**1,452 episodes** to bring every short feed to >=365 days.

- Disk: irrelevant. At the measured ~0.28 GB per 100 episodes (durable layer only; audio
  offloads to cold storage), 1,452 episodes is roughly **4 GB**.
- Time: **the real constraint**. At the measured ~7x realtime and ~50 min mean episode
  length, this is about **170 GPU-hours — roughly 7 days** of continuous pipeline.

The operator accepted the 7-day cost explicitly.

## Current state — 16 of 34 feeds already clear a year

18 feeds are short. **9 are below the 6-month floor.**

### Below 6 months (the laggards)

| feed | set | eps | span | median gap | needs |
| --- | --- | ---: | ---: | ---: | ---: |
| Odd Lots | A | 41 | 80d | 2d | +143 |
| The Daily | pre | 78 | 94d | 1d | +271 |
| The a16z Show | A | 83 | 111d | 1d | +254 |
| The Journal. | pre | 75 | 111d | 1d | +254 |
| ChinaTalk | A | 41 | 121d | 3d | +82 |
| Empire: World History | A | 42 | 127d | 3d | +80 |
| The Rest Is History | A | 42 | 133d | 3d | +78 |
| Past Present Future | B | 40 | 135d | 3d | +77 |
| Latent Space | pre | 41 | 174d | 4d | +48 |

### Between 6 months and a year

| feed | set | eps | span | median gap | needs |
| --- | --- | ---: | ---: | ---: | ---: |
| Unhedged | pre | 72 | 243d | 2d | +61 |
| Planet Money | pre | 74 | 248d | 3d | +39 |
| Macro Musings | B | 40 | 273d | 7d | +14 |
| In Moscow's Shadows | B | 38 | 280d | 7d | +13 |
| EconTalk | A | 41 | 280d | 7d | +13 |
| Lenny's Podcast | A | 56 | 299d | 7d | +10 |
| Complex Systems | B | 40 | 315d | 7d | +8 |
| Sinica Podcast | A | 40 | 323d | 7d | +6 |
| The Peter Attia Drive | B | 40 | 364d | 7d | +1 |

### Already at or above a year (no action)

Dwarkesh 403d, Invest Like the Best 406d, Hard Fork 413d, Conversations with Tyler 413d,
MLST 424d, In Our Time 445d, Capitalisn't 455d, NVIDIA AI Podcast 506d, Explaining Brazil
508d, No Priors 511d, The Pragmatic Engineer 518d, Ground Truths 539d, Ideas of India 546d,
The Long Run 637d, Latin America in Focus 679d, The Flip 1028d.

## Tiers — cheapest first

| tier | feeds | episodes | effect |
| --- | --- | ---: | --- |
| 1 | Peter Attia +1, Sinica +6, Complex Systems +8, Lenny's +10, EconTalk +13, In Moscow's Shadows +13, Macro Musings +14 | **65** | 7 feeds cross the year line |
| 2 | Planet Money +39, Latent Space +48, Unhedged +61 | **148** | 3 more; 26 of 34 at >=1yr |
| 3 | Past Present Future +77, Rest Is History +78, Empire +80, ChinaTalk +82 | **317** | 30 of 34 at >=1yr |
| 4 | Odd Lots +143, a16z +254, Journal +254, Daily +271 | **922** | all 34 |

Tier 1 alone is under 8 hours of pipeline time.

## Batch sizing — the 4-hour processing-loop cap

`_run_parallel_processing_loop` is bounded by `DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS`
(14400s = 4h), overridable per-config with `processing_loop_budget_seconds` (`0` disables).
When it trips, in-flight episodes are ABANDONED — not marked complete — and the feed lands
short of target. `skip_existing` keeps a re-run idempotent, so the fix is a follow-up batch.

Observed 2026-09-10: TRIP: Leading was asked for 39 episodes and tripped the cap at 4h01m
with 19 complete. Peter Attia was asked for 30 and took 5.9h without tripping (the bound
covers the processing loop, not the whole run).

**Rule: `episodes <= 14400 * throughput / mean_audio_seconds`.**

Measured on the two ASR-path feeds today — both ~5.1-5.35x realtime, ~63-65 min mean episode:

| throughput | 30min eps | 45min | 60min | 75min | 90min |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5x | 40 | 26 | **20** | 16 | 13 |
| 6x | 48 | 32 | 24 | 19 | 16 |
| 8x | 64 | 42 | 32 | 25 | 21 |
| 10x | 80 | 53 | 40 | 32 | 26 |

**For ASR-path feeds at ~1h/episode, cap a run at 18-20 episodes.** Feeds that serve
publisher transcripts are far cheaper and effectively exempt: In Moscow's Shadows did 38
episodes in 1h41m because it skips ASR entirely (`audio_sec=null, transcribe_sec=null`) —
check for that before sizing, it changes the answer by ~5x.

**Consequence for this backfill.** The 1,452-episode total is NOT one continuous ~170-hour
run. At ~20 episodes per run it is roughly **75 sequential runs**, each under the 4h cap,
on a single-writer queue. Tier 1 (65 episodes) is ~4 runs. Plan the nightly around them:
the nightly fires at 03:00Z on the same queue and will wait behind whatever is running.

## The concern that was raised, and the decision

**Four feeds are 922 of the 1,452 episodes — 64% of the total cost**: The Daily, The a16z
Show, The Journal., Odd Lots. These are exactly the feeds `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`
flagged in §5f as sitting below the bar now applied to new feeds (The Daily / The Journal /
Planet Money as news-cadence, the a16z feed as thesis marketing) and in the open-problems list
as carrying dated back catalogues that should be capped at the newest N rather than deepened.

This was put to the operator. **The decision is to backfill all of them to a year anyway**,
accepting the ~7-day cost. Recorded here so the trade-off is not rediscovered as a surprise.

## How the numbers were derived

Per feed: pull every episode's `publish_date` from `GET /api/corpus/episodes?feed_id=...`,
take the span between oldest and newest, and the **median gap** between consecutive
episodes. Episodes needed = `ceil((365 - span) / median_gap)`.

## NOT verified

- **Cadence is measured inside the already-ingested window.** Extending backwards assumes
  the show published at the same rate historically. A show that changed frequency will
  drift from these estimates — most likely the news-cadence feeds, which are also the
  most expensive.
- **Back-catalogue depth is unchecked for most feeds.** The onboarding doc records ample
  depth for Odd Lots (1193), In Our Time (1102), EconTalk (1062), Rest Is History (714),
  a16z (657), ChinaTalk (555). Latent Space, Unhedged, Hard Fork, Planet Money, The Journal
  and The Daily have no item count I have verified — a feed that does not expose enough
  history simply cannot reach the target.
- **The Rest Is Politics: Leading is excluded** — it was at 1 episode when this was
  measured and was ingesting to 40. Re-measure it before starting.
- **No per-episode cost estimate.** All ingest so far has run on the DGX at
  `estimated_cost=0.0000`; this assumes that continues.
- The 7x realtime figure is measured across two feeds (Peter Attia 5.35x, TRIP: Leading
  ~10.3x). Feed-to-feed variance is large and the 170-hour total inherits it.

## Operational note

Backfilling older episodes requires `episode_selection=unprocessed` **per request**.
Setting it in the corpus-global operator YAML is TRAP 2 in
`HANDOVER-2026-09-07-ready-to-scale.md`: it applies to the nightly too and turns it into a
back-catalogue crawler.
