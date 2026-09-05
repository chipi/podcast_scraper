# Corpus expansion — vetted feed registry (rev 2, 2026-09-05)

> **The feed list here is superseded.** `config/corpus-expansion.feeds.yaml` is canonical —
> it carries the operator's cuts (book-tour format out, Monocle trimmed to its two news
> shows) and is 76 feeds, not the 82 tabulated below. This document is kept for the
> reasoning and the corrections log, not the list.
> **Rev 2** supersedes the first cut after three independent reviews. What changed and why is
> in *Corrections* at the bottom — read that before trusting anything you remember from rev 1.

Target: ~5,000 episodes, maximum cross-feed connectivity, strongly international, English only.

## Gates

| gate | threshold | basis |
|---|---|---|
| language | starts `en` | non-English is blocked by the pipeline |
| freshness | newest item <= 270d | a dormant feed cannot grow with the corpus |
| depth | >= 30 items | fewer cannot support a 40-100 episode ingest |
| **duration ceiling** | **median <= 120 min** | **operator rule (§5h). Rev 1 had no ceiling and re-admitted a show §5h disqualified.** |
| ~~duration floor~~ | ~~median >= 18 min~~ | **REMOVED — measured false. See Corrections.** |
| not already ingested | — | deduped against the live corpus |
| not already approved | — | deduped against the pending Batch B |

**82 genuinely-new feeds.**

## Selection rule (this is the part that matters)

Feeds do not bridge because they share a *topic label*. They bridge because they chase the
**same live agenda** — the news cycle forces them onto the same subjects in the same weeks.

Measured on the live corpus (`selfrepeat.py`), self-repeat = of the topics only this show
touches, the share it revisits across more than one of its OWN episodes — a within-show number,
so neighbour count cannot affect it:

| show | topics only it touches | revisits itself | reads as |
|---|---:|---:|---|
| In Our Time | 21 | 10% | anthology — new subject weekly, will never bridge at any corpus size |
| Empire | 33 | 48% | recurring obsessions, no partner in the corpus |
| The Rest Is History | 33 | 61% | recurring obsessions, no partner in the corpus |
| ChinaTalk | 7 | 29% | bridges 92% — shared live agenda |

Empire and The Rest Is History are both history shows, both at 40 episodes, and share **2**
topics with each other. Adding feeds under a shared label does not create overlap.

Consequence for allocation: **climate/energy, geopolitics, conflict, migration, China, Africa,
LatAm and Russia have a shared current agenda and should be deepened. Ocean, religion, space,
Long Waves and Monocle mostly do not — probe them, do not fund them to depth on faith.**

## Accepted feeds by theme

### China (7)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | CN | Round Table China | 400 | 2026-09-04 | 26m | CGTN - Chinese state media |
| B | CN | ChinaPower | 234 | 2026-08-27 | 35m |  |
| A | CN/AF | The China-Global South Podcast | 182 | 2026-09-04 | 47m |  |
| B | CN | Pekingology | 148 | 2026-09-03 | 38m |  |
| B | CN | China Global | 132 | 2026-09-02 | 31m |  |
| A | CN | Biz Talk | 100 | 2026-09-04 | 26m | China Plus/CGTN - Chinese state media |
| A | CN | The Trivium China Podcast | 84 | 2026-09-04 | 49m |  |

### Africa (9)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | AF | Into Africa | 200 | 2026-08-27 | 35m |  |
| B | GLOBAL | In Pursuit of Development | 178 | 2026-06-17 | 53m |  |
| A | ZA | BizNews Radio | 150 | 2026-09-04 | 21m |  |
| A | AF | The Africa Report | 100 | 2026-09-02 | 6m |  |
| A | NG | The Open Africa Podcast | 93 | 2026-07-30 | 42m |  |
| B | AF | Foresight Africa Podcast | 85 | 2026-08-28 | 34m |  |
| A | AF | Africa Tech Summit Podcast | 45 | 2026-09-03 | 42m |  |
| A | AF | The Business of Africa | 33 | 2026-08-19 | 22m |  |
| A | AF | Made In Africa Podcasts | 32 | 2026-09-03 | 44m |  |

### South America / LatAm (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | MX | Mexico Business Now Podcast Expert Contributor Audio Articles | 3745 | 2026-09-04 | 7m |  |
| B | LATAM | The Americas Quarterly Podcast | 206 | 2026-08-27 | 30m |  |
| A | CO | Colombia Calling - The English Voice in Colombia | 100 | 2026-09-01 | 63m |  |

### Russia & Eurasia (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | RU | The Eurasian Knot | 375 | 2026-09-02 | 57m |  |
| A | EE | Talk Eastern Europe | 319 | 2026-09-04 | 50m |  |
| B | RU | The Power Vertical Podcast by Brian Whitmore | 246 | 2026-09-04 | 57m |  |
| A | RU | The Naked Pravda | 189 | 2026-02-24 | 32m | Meduza - Russian exile media; last ep 2026-02-24, may be dormant |

### India / South Asia (2)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | IN | Grand Tamasha | 298 | 2026-09-02 | 44m |  |
| A | IN | Interpreting India | 154 | 2026-08-27 | 43m |  |

### East & Southeast Asia (8)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | SEA | BRAVE Southeast Asia Tech: Singapore, Indonesia, Vietnam, Philippines, Thailand & Malaysia Startups, Founders & Venture Capit | 726 | 2026-09-02 | 35m |  |
| A | SG | Analyse Podcast | 536 | 2026-08-20 | 37m |  |
| A | JP | Disrupting Japan | 266 | 2026-08-17 | 36m |  |
| A | SG | Eco-Business Podcast | 146 | 2026-08-11 | 28m |  |
| A | KR | Korea Deconstructed | 135 | 2026-08-27 | 100m |  |
| A | JP | Nikkei Asia News Roundup with Jada and Brian | 109 | 2026-03-27 | 12m |  |
| B | KR | The Korea Society | 100 | 2026-09-01 | 61m |  |
| B | JP | Japan Memo | 61 | 2026-09-01 | 40m |  |

### Middle East & Turkey (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | ME | Middle East Dossier | 270 | 2026-06-28 | 66m |  |
| A | TR | Turkey Book Talk | 250 | 2026-09-01 | 35m |  |
| A | TR | Ottoman History Podcast | 123 | 2026-08-28 | ? | no itunes:duration in feed; length unmeasured |
| B | ME | Middle East Focus | 100 | 2026-07-30 | 41m |  |

### Geopolitics (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Carnegie Council Podcasts | 757 | 2026-09-02 | 40m |  |
| B | GLOBAL | Ones and Tooze | 273 | 2026-09-04 | 39m |  |
| B | GLOBAL | The President’s Inbox | 100 | 2026-09-02 | 35m |  |
| B | GLOBAL | Geopolitics Decanted with Dmitri Alperovitch | 81 | 2026-08-31 | 53m |  |

### Wars & Conflict (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | War on the Rocks | 342 | 2026-08-26 | 37m |  |
| B | GLOBAL | CONFLICTED | 230 | 2026-09-03 | 62m |  |
| B | GLOBAL | The Red Line | 145 | 2026-06-22 | 79m |  |
| B | GLOBAL | The Geopolitics & Power Podcast | 80 | 2026-02-23 | 63m | en-au; 6mo stale, watch |

### History (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | New Books in History | 500 | 2026-09-03 | 57m |  |
| B | GLOBAL | Throughline | 460 | 2026-09-03 | 49m |  |
| B | GLOBAL | Tides of History | 390 | 2026-08-24 | 44m |  |
| B | GLOBAL | School of War | 342 | 2026-09-04 | 49m | military history, not Russia-specific |

### Socioeconomics (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Macro Musings with David Beckworth | 565 | 2026-08-31 | 57m |  |
| B | GLOBAL | Hidden Forces | 520 | 2026-08-31 | 57m |  |
| B | GLOBAL | Maiden Mother Matriarch with Louise Perry | 284 | 2026-09-02 | 50m | gender/culture, not religion |

### Long Waves (2)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | The Great Simplification with Nate Hagens | 423 | 2026-09-04 | 72m |  |
| B | GLOBAL | Long Now | 331 | 2026-05-20 | 77m |  |

### Climate & Energy (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Volts | 448 | 2026-09-04 | 62m |  |
| B | GLOBAL | Cleaning Up: Leadership in an Age of Climate Change | 303 | 2026-09-02 | 61m |  |
| B | GLOBAL | Catalyst with Shayle Kann | 272 | 2026-09-03 | 41m |  |
| B | GLOBAL | Columbia Energy Exchange | 100 | 2026-09-01 | 54m |  |

### Migration & Demographics (2)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Parsing Immigration Policy | 272 | 2026-09-03 | 37m | Center for Immigration Studies - restrictionist ADVOCACY org, not neutral analysis |
| B | GLOBAL | Migration Policy Institute Podcasts | 150 | 2026-08-31 | 58m |  |

### Religion (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | New Books in Religion | 2758 | 2026-09-03 | 55m |  |
| B | GLOBAL | Religion Unplugged | 252 | 2026-09-01 | 30m |  |
| B | GLOBAL | The Sacred | 247 | 2026-08-19 | 49m |  |
| B | GLOBAL | Interfaith Voices Podcast | 39 | 2026-08-25 | 33m |  |

### Biotech & Science (1)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | The Readout Loud | 416 | 2026-07-30 | 31m |  |

### Space (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Planetary Radio: Space Exploration, Astronomy and Science | 1358 | 2026-09-04 | 36m |  |
| B | GLOBAL | Main Engine Cut Off | 340 | 2026-08-28 | 31m |  |
| B | GLOBAL | Off-Nominal | 260 | 2026-09-04 | 62m |  |

### Ocean (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | How To Protect The Ocean | 1995 | 2026-09-04 | 21m |  |
| B | GLOBAL | The Deep-Sea Podcast | 152 | 2026-08-28 | 45m |  |
| B | GLOBAL | Ocean Science Radio | 114 | 2026-08-31 | 25m |  |

### Urbanism (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Talking Headways: A Streetsblog Podcast | 820 | 2026-09-03 | 44m |  |
| B | GLOBAL | The War on Cars | 251 | 2026-08-25 | 34m |  |
| B | GLOBAL | The Urbanist | 100 | 2026-09-03 | 27m |  |

### Robotics (2)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | The Robot Report Podcast | 264 | 2026-09-04 | 68m |  |
| B | GLOBAL | Robot Talk | 171 | 2026-08-13 | 28m |  |

### Monocle Radio (6)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | The Foreign Desk | 100 | 2026-09-02 | 28m |  |
| B | GLOBAL | The Entrepreneurs | 100 | 2026-09-02 | 28m |  |
| B | GLOBAL | The Stack | 100 | 2026-08-29 | 33m |  |
| B | GLOBAL | Monocle on Design | 100 | 2026-09-01 | 30m |  |
| B | GLOBAL | The Globalist | 100 | 2026-09-04 | 59m |  |
| B | GLOBAL | The Big Interview | 100 | 2026-06-26 | 30m |  |

`A` = in-region editorial ownership. `B` = outlet covering the region from outside.

## Regional quotas (operator target: >= 5 each)

| region | new here | already approved | total |
|---|---:|---:|---:|
| China | 7 | 0  | **7** |
| Africa | 9 | 0  | **9** |
| South America / LatAm | 3 | 2 (Explaining Brazil, Latin America in Focus) | **5** |
| Russia & Eurasia | 4 | 1 (In Moscow's Shadows) | **5** |

All four met.

## Depth

At 60 episodes/feed: **4,829** new episodes.

**Do not ingest at a flat 60.** The decided §5g protocol is probe -> assess -> deepen, and depth
is an output of measurement, not an input. Probe 10-12 episodes per feed (~1,000 episodes,
~20% of budget), measure each feed's realized cross-feed cluster hits against the live corpus,
then allocate the rest proportional to measured bridging. Feeds that measure like Empire stay
at probe depth.

## Cost — corrected

Rev 1 said this was never measured. It was. `FAILURE-MODES-2026-08-31-dgx-100-batch.md` §J gives
per-episode stage medians: GI 820.3s, ASR 487.7s, summary 231.1s, KG 39.4s — **~26 min of
pipeline time per episode** on DGX. Dollar cost is ~0 (self-hosted vLLM). The real currency is
DGX-hours: ~4,800 episodes is on the order of **2,000+ serial DGX-hours**. Concurrency reduces
wall-clock by a factor nobody has measured.

## Blocking prerequisite

**#1972** — theme clustering falls off a cliff at 400 linkage topics and emits ZERO themes above
it. Prod is at 199/400 (50%) at 1,167 episodes. This expansion crosses it. The alarm
(WARN 300 / CRITICAL 360) should ship before the ingest, whatever happens to the algorithm.

## NOT covered / NOT verified

- **No quality gate has run on any of these 82 feeds.** Candidate registry, not a passed one.
- **LatAm is thin and it is not a tooling artifact any more.** Eleven separate search
  formulations for Argentina, Chile, Peru, Mexico and regional analysis resolved to unrelated
  shows (a surfing podcast, a British English-teaching show, a Cuban feed dead since 2018). That
  is a failed search, which is weaker than proof of absence — but it has now failed repeatedly.
- **The Russia Contingency (Kofman) has no public feed** — it resolves to its parent, War on the
  Rocks. Members-only.
- **Publisher concentration is unmeasured in the connectivity metric.** Monocle is 6 feeds of one
  editorial voice; CSIS is 3 under different artist strings. The bridging metric counts them as
  independent perspectives. Cap at 2/publisher, or fix the metric.
- **No viewpoint field exists in the schema.** The state-media and advocacy flags below live only
  in this prose and cannot reach the synthesis features.
- **Format was never a selection axis**, and prior art calls dialogic/oppositional format the
  largest gap in the corpus. Nothing here was chosen for it.
- **Licensing / ToS: not checked for any feed.**
- **Clustering behaviour at 5,000 episodes is unknown.** Nobody has run it.

**Viewpoint flags:**

- **Round Table China** — CGTN - Chinese state media
- **Biz Talk** — China Plus/CGTN - Chinese state media
- **Parsing Immigration Policy** — Center for Immigration Studies - restrictionist ADVOCACY org, not neutral analysis
- **The Naked Pravda** — Meduza - Russian exile media; last ep 2026-02-24, may be dormant

## Corrections to rev 1

| # | what rev 1 said | what is true |
|---|---|---|
| 1 | 18-min duration floor, justified as "a 6-min brief cannot yield 18 KG nodes" | **False, measured.** A 6.1-min In Our Time episode yields ~21 KG identities; 130-min episodes yield no more. Output volume is prompt-bound, not duration-bound. Gate removed; Nikkei Asia, The Africa Report and Mexico Business Now re-admitted. |
| 2 | "no live English-language show out of Mexico cleared the gates" | The **gate** created that gap. Mexico Business Now is back in. |
| 3 | 86 new feeds, 5,069 episodes | 82 new. Four were already ingested (Sinica, Odd Lots, Ground Truths, The Long Run); three were already approved Batch B. Never deduped against the live corpus. |
| 4 | The Seen and the Unseen accepted | §5h disqualified it on a 2-hour operator ceiling on 2026-08-13 and replaced it with Ideas of India. Dropped, and a ceiling gate added. |
| 5 | "53% of clusters are single-feed" | **48%.** The measuring script fetched 1,000 episodes against a 1,167-episode corpus and never followed the pagination cursor. |
| 6 | "per-episode cost never measured" | It was, in our own failure-modes doc. ~26 min/episode. |
| 7 | feed resolution | The verifier took the first search result with no asked-vs-got check. Five were different shows. A similarity guard now runs. |
| 8 | The Readout Loud accepted | §5f cut it as thin news-cadence. Still listed — flagged, needs an editorial call. |
