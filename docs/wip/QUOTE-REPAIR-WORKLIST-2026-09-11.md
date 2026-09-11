# Quote-bundle repair work-list — 2026-09-11

**What:** the 21 episodes that hit `extract_quotes_bundled parse FAILED` during the
2026-09-10/11 Batch B deepen and Batch C class-A scout. IDs are in
`quote-repair-worklist-2026-09-11.txt`, one per line, ready for `--episode-ids`.

**Why this file exists.** The failure is NOT detectable from the artifacts. `prod_dgx_full`
sets `gi_require_grounding: true`, so an insight that loses its quotes is DROPPED rather than
kept in a degraded form — a damaged episode looks exactly like a legitimately sparse one, just
with fewer insights. The only definitive record is the job log, and logs age out. This file
turns that ephemeral signal into a durable one.

`check_corpus_gi_integrity` will NOT find these: a degraded episode still has a valid,
non-placeholder artifact with a plausible insight count, which that gate considers legal.

## How to repair (AFTER the fix is deployed)

```sh
podcast-scraper gi-repair \
  --output-dir /app/output \
  --config <profile> \
  --episode-ids quote-repair-worklist-2026-09-11.txt \
  --force-healthy
```

`--force-healthy` is required: these are not legacy placeholders, they are healthy-looking
artifacts. gi-repair re-derives from the transcript plus existing summary bullets — no ASR, no
RSS fetch, no new run dir. It rewrites `gi.json` IN PLACE, so take a corpus backup first
(`backup-corpus-prod.yml`).

**Repairing before the deploy is pointless** — the same code produces the same failure.

## Measuring what was lost

Record each episode's insight count before and after. A count that rises is a quantified
recovery; one that does not means the bundle failure cost nothing for that episode (the
per-insight staged fallback covered it). That before/after is the only honest measure of the
defect's real cost — nobody has established it yet.

## The 21 episodes

| feed | episode | episode_id |
| --- | --- | --- |
| BRAVE Southeast Asia Tech: Sin | What 500 Harvard Case Studies Actually Do To Your Brain -  | `0d33a2e1-ae08-4d04-ac6e-e086c93c9381` |
| In Moscow's Shadows | In Moscow's Shadows 228: Blood & Soil versus Bread & Butte | `Buzzsprout-18391160` |
| In Moscow's Shadows | In Moscow's Shadows 237: How A 1552 Siege Explains A 2022  | `Buzzsprout-18725041` |
| In Moscow's Shadows | In Moscow's Shadows 243: Who Controls The Story In Russia? | `Buzzsprout-18949513` |
| In Moscow's Shadows | In Moscow's Shadows 251: The Near Abroad Recedes: Armenia  | `Buzzsprout-19304003` |
| In Moscow's Shadows | In Moscow's Shadows 253: The Fall Of Antikvar | `Buzzsprout-19374986` |
| Ottoman History Podcast | Breadwinner Soldiers in the Ottoman Empire | `tag:blogger.com,1999:blog-1793063735579568706.post-9096706244812833516` |
| The Business of Africa | Regional economic integration: The good, the curious and t | `https://iono.fm/e/1716156` |
| The Peter Attia Drive | #391 ‒ Colorectal cancer screening: importance of early sc | `8586d7bc-4230-4409-955d-079c3ce30103` |
| The Peter Attia Drive | #396 ‒ Breast cancer screening: understanding risk, decidi | `13b1a890-2bad-4692-bd84-30520944d067` |
| The Peter Attia Drive | Longevity 101: a foundational guide to Peter's frameworks  | `b1529269-ebc8-4742-9136-dd0ea583638c` |
| The Peter Attia Drive | Special episode: Understanding true happiness and the tool | `d4d26152-52f6-4e88-8167-9e428c174308` |
| The Peter Attia Drive | Transforming education with AI and an individualized, mast | `5348df1b-3a22-4aeb-97ce-4c929a198e29` |
| The Peter Attia Drive | Women's sexual health: desire, arousal, and orgasms, navig | `d0b7401b-9137-449d-9431-67931be178f3` |
| The Rest Is Politics: Leading | How Westminster Breaks Politicians and Why Britain Isn't R | `755e53ae-9245-11f1-87ca-b388ea80ff93` |
| The Rest Is Politics: Leading | Is Peace in the Middle East Closer Than We Think? (A Pales | `ae34e004-9884-11f1-b99e-e34e51c94e8d` |
| The Rest Is Politics: Leading | Is Putin Losing his Grip on Russia? | `8ecbe2ba-6bdb-11f1-99e2-6bca7069ce74` |
| The Rest Is Politics: Leading | Neil Kinnock: A Labour Rebel’s Path To Power (Part 1) | `d2f45ea6-f2e9-11f0-96b3-87666a591bbe` |
| The Rest Is Politics: Leading | President Stubb: Trump’s Unlikely Best Friend | `0e9a5b4a-1950-11f1-84a2-436d7b81518e` |
| The Rest Is Politics: Leading | President of Moldova, Maia Sandu: Holding the Line Between | `486d6a94-ed75-11f0-a57a-1fbb360ba02c` |
| The Rest Is Politics: Leading | President of Ukraine: Volodymyr Zelenskyy | `76a4b584-3406-11f1-9155-8bb829cd29eb` |

## Provenance

Extracted from the prod job logs of the 2026-09-10/11 runs: `f566ceb6` (Peter Attia),
`499d721a` (In Moscow's Shadows), `3c1d28fa` (TRIP: Leading), and three Batch C scout jobs
(`0277b742` The Business of Africa, `f9aa5340` BRAVE SE Asia, `85cbcf45` Ottoman History).
Log episode indices were mapped to titles via the transcript-save / download lines, then
matched to corpus `episode_id`s by fuzzy title match (all 21 scored 0.84-0.97).

22 parse-failure events, 21 distinct episodes (one episode failed twice).

**NOT verified:** that every affected episode is in this list. It covers the jobs from these
two batches only — earlier runs were not scanned, and their logs may already be gone.
