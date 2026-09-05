# Corpus expansion — vetted feed registry (2026-09-05)

Target: ~5,000 episodes, maximum cross-feed **connectivity**, strongly international,
English-language only. This document is the output of a *publisher-first* sweep followed by a
live verification pass. Every row below was resolved to a real RSS URL and fetched.

## What was actually verified

For each candidate: iTunes Search API resolution -> fetch the RSS -> read `<language>`, count
`<item>`, take newest `pubDate` and median `itunes:duration`. Gates applied mechanically:

| gate | threshold | why |
|---|---|---|
| language | must start `en` | non-English is blocked by the pipeline — hard gate |
| freshness | newest item <= 270d | a dormant feed cannot grow with the corpus |
| depth | >= 30 items | fewer cannot support a 40-100 episode ingest |
| substance | median >= 18 min | **judgment, not measurement** — a 6-min news brief cannot plausibly yield the 18 KG nodes §5i requires. Not empirically calibrated. |

**86 accepted, 23 rejected.** Rejections are listed in full at the bottom —
several were my own earlier suggestions, including two I cited to you as evidence.

## The method, and why it changed

Chart-first search finds what is popular in a market, which in a non-English market is
non-English. Publisher-first search asks instead *which institution in this market publishes in
English* — national English-language papers, think tanks, regional business media. That
register — the internationally-facing professional class — is exactly the analytical tone this
corpus wants. The sweep confirms it: Trivium China and Biz Talk out of Beijing, The Open Africa
Podcast out of Lagos, Colombia Calling out of Bogota, Korea Deconstructed out of Seoul,
Disrupting Japan out of Tokyo, Turkey Book Talk out of Istanbul.

## Accepted feeds by theme

### China (8)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | CN | Sinica Podcast | 558 | 2026-08-18 | 61m |  |
| A | CN | Round Table China | 400 | 2026-09-04 | 26m | CGTN - Chinese state media |
| B | CN | ChinaPower | 234 | 2026-08-27 | 35m |  |
| A | CN/AF | The China-Global South Podcast | 182 | 2026-09-04 | 47m |  |
| B | CN | Pekingology | 148 | 2026-09-03 | 38m |  |
| B | CN | China Global | 132 | 2026-09-02 | 31m |  |
| A | CN | Biz Talk | 100 | 2026-09-04 | 26m | China Plus/CGTN - Chinese state media |
| A | CN | The Trivium China Podcast | 84 | 2026-09-04 | 49m |  |

### Africa (8)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | AF | Into Africa | 200 | 2026-08-27 | 35m |  |
| B | GLOBAL | In Pursuit of Development | 178 | 2026-06-17 | 53m |  |
| A | ZA | BizNews Radio | 150 | 2026-09-04 | 21m |  |
| A | NG | The Open Africa Podcast | 93 | 2026-07-30 | 42m |  |
| B | AF | Foresight Africa Podcast | 85 | 2026-08-28 | 34m |  |
| A | AF | Africa Tech Summit Podcast | 45 | 2026-09-03 | 42m |  |
| A | AF | The Business of Africa | 33 | 2026-08-19 | 22m |  |
| A | AF | Made In Africa Podcasts | 32 | 2026-09-03 | 44m |  |

### South America / LatAm (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | BR | Explaining Brazil | 395 | 2026-09-01 | 22m |  |
| B | LATAM | Latin America in Focus | 241 | 2026-07-30 | 31m |  |
| B | LATAM | The Americas Quarterly Podcast | 206 | 2026-08-27 | 30m |  |
| A | CO | Colombia Calling - The English Voice in Colombia | 100 | 2026-09-01 | 63m |  |

### Russia & Eurasia (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | RU | The Eurasian Knot | 375 | 2026-09-02 | 57m |  |
| A | EE | Talk Eastern Europe | 319 | 2026-09-04 | 50m |  |
| B | RU | The Power Vertical Podcast by Brian Whitmore | 246 | 2026-09-04 | 57m |  |
| A | RU | The Naked Pravda | 189 | 2026-02-24 | 32m | Meduza - Russian exile media; last ep 2026-02-24, may be dormant |

### India / South Asia (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | IN | The Seen and the Unseen - hosted by Amit Varma | 455 | 2026-08-24 | 175m |  |
| B | IN | Grand Tamasha | 298 | 2026-09-02 | 44m |  |
| A | IN | Interpreting India | 154 | 2026-08-27 | 43m |  |

### East & Southeast Asia (7)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| A | SEA | BRAVE Southeast Asia Tech: Singapore, Indonesia, Vietnam, Philippines, Thailand & Malaysia Startups, Founders & Venture Capit | 726 | 2026-09-02 | 35m |  |
| A | SG | Analyse Podcast | 536 | 2026-08-20 | 37m |  |
| A | JP | Disrupting Japan | 266 | 2026-08-17 | 36m |  |
| A | SG | Eco-Business Podcast | 146 | 2026-08-11 | 28m |  |
| A | KR | Korea Deconstructed | 135 | 2026-08-27 | 100m |  |
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

### Socioeconomics (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Odd Lots | 1267 | 2026-09-04 | 43m |  |
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

### Religion (3)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | New Books in Religion | 2758 | 2026-09-03 | 55m |  |
| B | GLOBAL | Religion Unplugged | 252 | 2026-09-01 | 30m |  |
| B | GLOBAL | Interfaith Voices Podcast | 39 | 2026-08-25 | 33m |  |

### Biotech & Science (4)

| class | region | show | items | newest | median | notes |
|---|---|---|---:|---|---:|---|
| B | GLOBAL | Nature Podcast | 924 | 2026-09-04 | 25m |  |
| B | GLOBAL | The Readout Loud | 416 | 2026-07-30 | 31m |  |
| B | GLOBAL | The Long Run with Luke Timmerman | 206 | 2026-08-25 | 65m |  |
| B | GLOBAL | Ground Truths | 95 | 2026-08-30 | 45m |  |

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

## Corpus math

Episodes currently sitting in these 86 feeds: **28,095**. Combined with the ~24 feeds
already ingested, the corpus becomes ~110 feeds.

| depth per feed | new episodes |
|---|---:|
| <= 40 | 3,424 |
| <= 60 | 5,069 |
| <= 100 | 8,228 |

**60 per feed lands on 5,069 — your ~5,000 target, near exactly.** That is the recommended
depth: deep enough that a show's recurring themes repeat and cluster, shallow enough that no
single feed dominates the topic graph.

## NOT covered / NOT verified

This section is deliberately as detailed as the one above.

**No quality gate has been run on any of these 86 feeds.** Item counts and dates are feed
metadata; they say nothing about whether an episode clears §5i (`both >= 8`, `kg_nodes >= 18`,
`gi_only <= 2`). This is a *candidate* registry, not a passed one. The §5g protocol — probe 1,
assess, deepen — still has to run on every feed.

**Regional targets not met:**

- **South America: 4 feeds, 2 Class A** — short of your >= 5. Explaining Brazil (BR)
  and Colombia Calling (CO) are genuinely in-region; the other two are US think tanks covering
  the region. I found no live English-language show out of Argentina, Chile, Peru or Mexico that
  cleared the gates. Mexico Business Now exists but runs 7-minute briefs.
- **Russia & Eurasia: 4 feeds** — short of >= 5 after I rethemed School of War to
  history where it belongs. The Naked Pravda (Meduza, the one true in-region voice) last
  published 2026-02-24 and may be dormant.
- **No coverage at all**: Vietnam, Indonesia (only via the regional BRAVE SEA feed), Egypt, the
  Gulf, Pakistan, Bangladesh, Ethiopia, Ghana, Central Asia.

**Thin themes:** Migration (2, and one is advocacy), Long Waves (2), Robotics (2), Ocean (3, and
mostly conservation advocacy rather than research-grade marine science).

**Viewpoint flags — these are not neutral analysis and should be labelled in the corpus:**

- **Round Table China** — CGTN - Chinese state media
- **Biz Talk** — China Plus/CGTN - Chinese state media
- **Parsing Immigration Policy** — Center for Immigration Studies - restrictionist ADVOCACY org, not neutral analysis
- **The Naked Pravda** — Meduza - Russian exile media; last ep 2026-02-24, may be dormant

**Never measured:** per-episode ingest cost. I have never recorded it, so I cannot tell you
what 5,000 episodes costs in DGX hours or tokens. That should be measured on the next batch
before committing to the full expansion.

**Not checked:** licensing / ToS for any of these feeds.

## Rejected, with reasons

| reason | resolved to | originally asked for |
|---|---|---|
| REJECT short-form 12min | Nikkei Asia News Roundup with Jada and Brian | Nikkei Asia podcast |
| REJECT short-form 6min | The Africa Report | The Africa Report podcast |
| REJECT short-form 7min | Mexico Business Now Podcast Expert Contributor Audio Articles | Mexico Business Now podcast |
| REJECT stale 1115d | The Robot Brains Podcast | The Robot Brains Podcast Pieter Abbeel |
| REJECT stale 1159d | The Religious Studies Project | The Religious Studies Project |
| REJECT stale 1954d | Africa Unconstrained: Perspectives on the Continent’s New Era | The Continent podcast Africa |
| REJECT stale 2261d | TechChats on Business, Tech and Marketing in Asia | Analyse Asia |
| REJECT stale 2356d | Kenyan Wallstreet News Brief Podcast | The Kenyan Wall Street Podcast |
| REJECT stale 2380d | Latin America Reports: The Podcast | Latin America Report podcast |
| REJECT stale 2691d | Displaced | Displaced podcast refugees |
| REJECT stale 312d | Africa Daily | Africa Daily BBC |
| REJECT stale 3417d | African Tech Conversations | African Tech Roundup |
| REJECT stale 359d | Chinese Whispers | Chinese Whispers Spectator |
| REJECT stale 3706d | Refugee Studies Centre | Refugee Studies Centre Oxford podcast |
| REJECT stale 441d | Afrobility: Africa Tech and Business | Afrobility |
| REJECT stale 450d | Ideas Untrapped | Ideas Untrapped Tobi Lawson |
| REJECT stale 463d | Brazil Unfiltered | Brazil Unfiltered |
| REJECT stale 470d | Babel: Translating the Middle East | Babel Middle East CSIS |
| REJECT stale 639d | COMPLEXITY | Complexity Santa Fe Institute |
| REJECT stale 774d | Deep Dive from The Japan Times | Deep Dive from The Japan Times |
| REJECT stale 821d | Asia Matters Podcast | Asia Matters podcast |
| REJECT thin 22 items | Fall of Civilizations Podcast | Fall of Civilizations Podcast |
| REJECT thin 27 items | The Migration Podcast | The Migration Podcast |

Three of these deserve calling out because I previously presented them to you as evidence:

- **Deep Dive from The Japan Times** — I cited this as proof that English-language shows exist in
  non-English markets. It last published **2024-07-22**, 774 days ago. The claim was right; this
  particular example was dead. Disrupting Japan and Japan Memo carry Japan instead.
- **Ideas Untrapped** (Tobi Lawson, Nigeria) — the strongest analytical African economics show I
  found, and dormant since 2025-06-11.
- **The Kenyan Wall Street Podcast** — 3 items, last 2020. Kenya has no accepted feed.

