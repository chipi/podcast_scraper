# Handoff — beta-user shows: add to prod + 1-episode smoke (2026-10-01)

**For:** an agent with prod access.
**Task:** add the 26 feeds below to prod, run a **1-episode smoke** on each, assess each
episode, and report back one row per show. Do **not** deepen any show past 1 episode —
depth is decided after the smoke results are reviewed.

## Where these came from

Three beta users sent screenshots of their Spotify libraries — 74 unique shows. Each was
resolved via the iTunes search API to its RSS feed, the feed was fetched and parsed, and the
same gates as `config/corpus-expansion.feeds.yaml` were applied:

| gate | threshold |
|---|---|
| language | English only |
| freshness | newest item <= 270 days old (measured 2026-10-01) |
| depth | >= 30 items |
| duration ceiling | median of newest 12 `<itunes:duration>` <= 120 min |
| not in corpus | deduped against repo docs (corpus as of 2026-09-10) |
| prior editorial cuts | news roundups, vendor/content marketing, narrative/anecdote shows, audio articles |

26 passed everything. The 48 rejections and their reasons are at the bottom — do not add them.

## The 26 feeds

Items / median / newest are as measured 2026-10-01.

| # | Show | Items | Median | Newest | RSS |
|---:|---|---:|---:|---|---|
| 1 | Freakonomics Radio | 939 | 51m | 2026-09-25 | `https://feeds.simplecast.com/Y8lFbOT4` |
| 2 | People I (Mostly) Admire | 249 | 45m | 2026-09-26 | `https://feeds.simplecast.com/rP60Wf24` |
| 3 | The Gray Area with Sean Illing | 801 | 50m | 2026-09-28 | `https://feeds.megaphone.fm/theezrakleinshow` |
| 4 | Explain It to Me | 825 | 30m | 2026-09-27 | `https://feeds.megaphone.fm/theweeds` |
| 5 | Unexplainable | 313 | 28m | 2026-09-28 | `https://feeds.megaphone.fm/VMP9331026707` |
| 6 | StarTalk with Neil deGrasse Tyson | 1134 | 54m | 2026-09-29 | `https://feeds.simplecast.com/4T39_jAj` |
| 7 | The Rest Is Science | 93 | 49m | 2026-09-30 | `https://feeds.megaphone.fm/GLT6907573392` |
| 8 | Google DeepMind: The Podcast | 51 | 48m | 2026-09-09 | `https://feeds.simplecast.com/JT6pbPkg` |
| 9 | Why This Universe? | 116 | 39m | 2026-08-24 | `https://rss.buzzsprout.com/1162613.rss` |
| 10 | Curious Cases (BBC Radio 4) | 166 | 29m | 2026-09-29 | `https://podcasts.files.bbci.co.uk/b07dx75g.rss` |
| 11 | Azeem Azhar's Exponential View | 229 | 39m | 2026-06-04 | `https://feeds.simplecast.com/e_GRxR9a` |
| 12 | The Every Podcast | 130 | 48m | 2026-09-30 | `https://rss2.flightcast.com/vh1lt5f2oeh4a5hn5jglgpzg.xml` |
| 13 | AI and Design | 36 | 48m | 2026-09-30 | `https://api.riverside.fm/hosting/8P3kTQz9.rss` |
| 14 | AI 4 UX with John Whalen, PhD | 40 | 46m | 2026-06-04 | `https://feed.podbean.com/brilliantexperience/feed.xml` |
| 15 | NN/g UX Podcast | 67 | 38m | 2026-09-16 | `https://anchor.fm/s/2125a62c/podcast/rss` |
| 16 | Design Meets Business | 41 | 63m | 2026-01-14 | `https://feeds.transistor.fm/design-meets-business` |
| 17 | FUTURES Podcast | 99 | 33m | 2026-04-15 | `https://rss.libsyn.com/shows/230621/destinations/1700315.xml` |
| 18 | Huberman Lab | 446 | 82m | 2026-10-01 | `https://feeds.megaphone.fm/hubermanlab` |
| 19 | ZOE Science & Nutrition | 354 | 34m | 2026-10-01 | `https://feeds.megaphone.fm/ZOELIMITED9301524082` |
| 20 | The Knowledge Project | 289 | 68m | 2026-09-29 | `https://feeds.megaphone.fm/FSMI7575968096` |
| 21 | How I Write | 142 | 69m | 2026-09-30 | `https://feeds.megaphone.fm/TFTEE2608650139` |
| 22 | The Curiosity Shop with Brené Brown and Adam Grant | 116 | 63m | 2026-10-01 | `https://feeds.megaphone.fm/daretolead` |
| 23 | Philosophy For Our Times | 570 | 36m | 2026-09-29 | `https://rss.art19.com/philosophy-for-our-times` |
| 24 | The Living Philosophy | 102 | 78m | 2026-08-09 | `https://anchor.fm/s/ee73d900/podcast/rss` |
| 25 | Unbelievable? | 1275 | 71m | 2026-09-30 | `https://pcr-rss.streamguys1.com/the-unbelievable/unbelievable.xml` |
| 26 | How to Touch Grass | 65 | 37m | 2026-08-24 | `https://feeds.megaphone.fm/howto` |

## Before you start — check, don't assume

1. **Dedupe against prod.** The dedupe above was against repo docs dated 2026-09-10, not the
   live prod corpus. Any feed already on prod: skip it and note it in the report.
2. **Pipeline capacity.** The 2026-09-30 handover (`HANDOVER-2026-09-30-outage-fixed-corpus-mop-up.md`)
   describes post-outage mop-up. Confirm nothing from that is still queued/running before adding
   26 jobs' worth of work.
3. **Queue order vs the 72-feed expansion.** `config/corpus-expansion.feeds.yaml` lists 72
   earlier-vetted feeds. If those are not yet smoked on prod, ask the operator whether these 26 go
   before, after, or interleaved — do not decide it yourself.

## Smoke settings

- `max_episodes=1`, `episode_order=newest`, `skip_existing=true`
- **One show at a time** — keeps each result attributable.
- **Run Huberman Lab first or isolated:** median is 82m but individual episodes exceed 2h (one
  seen at 2h36m). Watch cost against the $10/run soft cap.
- **Run Design Meets Business early:** its newest episode is 2026-01-14; it crosses the 270-day
  freshness gate around 2026-10-11.

## Per-episode assessment (§5g Phase 2 of `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`)

| Check | Source | PASS | INVESTIGATE | FAIL |
| --- | --- | --- | --- | --- |
| Job outcome | `GET /api/jobs/{id}` | `succeeded` | — | `failed` / `stale` |
| Artifacts complete | `/api/corpus/coverage` | GI **and** KG present | — | either missing |
| Insight count | index `doc_type=insight` | 6–31 | < 6 | 0 |
| KG nodes | `kg_entity` + `kg_topic` | ~20–29 | < 15 | 0 |
| GI↔KG bridging | `bridge_partition` in episode detail | `both` > 0 | `both == 0` | — |
| Summary substance | `summary_bullets` | specifics: names, numbers, mechanisms | generic/vague | empty |
| Cost | `podcast_pipeline_run_cost_usd_total` delta | ≲ $0.30/ep | > $0.50/ep | approaches $10/run cap |

Also check the transcript for: **ad-read contamination** (Vox, Freakonomics, Huberman, megaphone
feeds are ad-heavy), wrong language, untranscribed music.

**Bucket each show** (§5g Phase 3):

- **DEEPEN** — clears the bar, processed cleanly.
- **PARK** — good show, pipeline handled it badly (ads, diarization, thin insights on rich
  material). Stays at 1; log the pipeline defect. A poor probe is **not** a reason to drop.
- **DROP** — content does not clear the editorial bar. The only content-driven exit.
- **BLOCKED** — structurally not ingestible (dead feed, over ceiling, non-English).

## Known traps

- **Re-purposed feeds.** Three URLs carry an older show's back-catalogue under a new name:
  `theezrakleinshow` → The Gray Area, `theweeds` → Explain It to Me, `daretolead` → The Curiosity
  Shop. A newest-first 1-episode smoke is fine. **Any later backfill must stop at the current
  show's start date** — those dates have not been looked up.
- **Count RSS items by occurrence, not by line** (`grep -o '<item[ >]' | wc -l`) — line counts
  gave two false findings before (§5j).
- **Feed language tags lie.** One rejected show (ALARM, Serbian) declares `<language>en`. If the
  pipeline's language gate reads that tag, it would have let it through. Not in this list — noted
  in case a transcript comes back in the wrong language.

## Report back — one row per show

| # | Show | Job id | Outcome | Insights | KG nodes | `both` | Cost | Ads? | Bucket | Note |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|

Plus: any feed skipped (already on prod / unreachable), and anything you could NOT verify.

## NOT verified by the vetting pass

- Prod corpus dedupe (repo docs only, as of 2026-09-10).
- Audio enclosures — only RSS metadata was read; no audio was fetched.
- Show identity — matched by name + publisher; two initial matches were wrong and were corrected
  (Economist, The Living Philosophy). The rest were checked by publisher name only.
- Format of the 26 — judged from title, publisher and median length, not transcripts.
- Cross-feed bridging — unmeasurable until episodes are ingested.

## Rejected (48) — do not add

| Reason | Shows |
|---|---|
| Already in corpus | The Rest Is History, Empire: World History, In Our Time, The Journal., Lenny's Podcast |
| Earlier decision | The Rest Is Politics (main feed swapped for *Leading*, §5f), Acquired (236m median, §5h) |
| Non-English | Σκληρές αλήθειες, Archaeostoryteller (Greek); ALARM, Agelast podcast (Serbian); Imposturas Filosóficas, #Provocast (Portuguese) |
| Over 120m median | Small Town Murder (131m), The Joe Rogan Experience (154m) |
| Stale > 270d | The Infinite Monkey Cage (280d), Hear This Idea (333d), Brave UX (337d), The Video Archives Podcast (379d), Simplifying Complexity (408d), The Ricky Gervais Show (496d), How We Scaled It (701d), Spark & Fire (742d), GOSSIPMONGERS (1685d), The Habitat (2183d), Stephen Fry's 7 Deadly Sins (2396d) |
| < 30 items | Grounded with Louis Theroux (23) |
| No single feed | Economist Podcasts (Spotify umbrella; nearest is The Intelligence, daily news) |
| Audio articles / readings | Science, Spoken (WIRED, 11m), Aeon Magazine (narrated essays, 23m), TED Talks Daily (16m), The Daily Stoic (12m) |
| News roundup | Today, Explained, Global News Podcast, Politics Weekly America |
| Vendor / content marketing | Compiler (Red Hat), This New Way (Fellow.ai), Product Impact Podcast, Thinkers & Ideas (BCG) |
| Narrative / entertainment | Darknet Diaries, Myths and Legends, The Dollop, No Such Thing As A Fish, The Adam Buxton Podcast, We Can Be Weirdos, The Louis Theroux Podcast, Soul Boom, What Now? with Trevor Noah |
