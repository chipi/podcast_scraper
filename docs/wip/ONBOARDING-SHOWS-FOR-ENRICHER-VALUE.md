# Onboarding more shows/episodes to unlock enricher value

Living notes (started 2026-07-06, consolidated 2026-08-12).

**Goal:** grow the **eval** corpus with more real shows so the enrichers we built produce
*visible value*. This is about **real content for eval**, distinct from deterministic **test**
fixtures.

> **Note (2026-08-12):** this doc previously pointed at
> `docs/wip/CORPUS-EVOLUTION-FOR-COMPLEX-ENRICHERS.md` for the test-fixture side. **That file
> does not exist** anywhere in the repo — it was either never written or deleted, and the
> reference here was the only mention of it. The nearest live documents are
> [ENRICHER-HARDENING-ROADMAP.md](ENRICHER-HARDENING-ROADMAP.md) and
> [CONTENT_EVOLUTION_BLUEPRINT.md](../architecture/CONTENT_EVOLUTION_BLUEPRINT.md) — but note
> the latter is about pluggable content types and transports (podcast/news/social,
> RSS/scrape/filesystem), **not** feed selection. Neither replaces the missing doc.

> **Why this matters:** `topic_perspectives` (#1146) is live and disagreement (#1144) is
> *scale-gated*. Both get richer strictly with more and better real content — perspectives
> deepen per topic; disagreement and prediction-tracking only *appear* at scale and time span.
> More shows is the lever.

---

## 1. Current corpus — verified live, 2026-08-12

Pulled from `GET /api/corpus/feeds?path=/app/output` on the prod box mid-batch, so counts
move. **9 feeds**, not the ~10 previously recorded here.

| Show | Episodes | RSS | Cluster |
| --- | --- | --- | --- |
| No Priors | 27 | `feeds.megaphone.fm/nopriors` | AI / tech |
| Planet Money | 28 | `feeds.npr.org/510289/podcast.xml` | Economics |
| The Journal. | 26 | `video-api.wsj.com/podcast/rss/wsj/the-journal` | Business news |
| NVIDIA AI Podcast | 26 | `feeds.megaphone.fm/nvidiaaipodcast` | AI / tech |
| The Daily | 21 | `feeds.simplecast.com/54nAGcIl` | General news |
| Invest Like the Best | 19 | `feeds.megaphone.fm/investlikethebest` | Investing |
| Hard Fork | 17 | `feeds.simplecast.com/l2i9YnTd` | AI / tech |
| Unhedged | 17 | `feeds.acast.com/public/shows/6478a825654260001190a7cb` | Markets |
| Latent Space | 13 | `rss.flightcast.com/vgnxzgiwwzwke85ym53fjnzu.xml` | AI engineering |

**Correction to earlier notes:** this doc previously listed **Odd Lots** as part of prod-v2.
It is **not** in the corpus — **Unhedged** is. Odd Lots remains a strong candidate (see §5).

### Cluster distribution

| Cluster | Shows | Episodes (approx) |
| --- | --- | --- |
| AI / tech | 4 — No Priors, NVIDIA AI, Hard Fork, Latent Space | ~83 |
| Markets / investing / econ | 3 — Planet Money, Invest Like the Best, Unhedged | ~64 |
| General & business news | 2 — The Journal, The Daily | ~47 |

---

## 2. Value model — what each enricher wants from new content

| Enricher / feature | What richer content unlocks | Selection signal |
| --- | --- | --- |
| `topic_perspectives` (#1146) | More distinct speakers per topic → deeper perspective cards; more topics clear the dashboard's ≥2-speaker bar | **Overlap:** new shows covering topics existing shows already cover |
| disagreement / prediction (#1144) | The signal ~absent today: cross-person opposition + "who called it" | **Debate / panel / dialogue** shows; recurring contested topics **over time** |
| `guest_coappearance` | Real co-appearance edges | **Multi-guest** episodes (2+ named guests) |
| `temporal_velocity` | Meaningful "heating up" trends | Episodes **spread across months** |
| `topic_similarity` (#1105) | Denser, more reliable neighbour clusters | Broad but **thematically clustered** coverage |

### The selection rule

**The highest-leverage single lever is topic OVERLAP across shows.** Every cross-person
enricher — perspectives, disagreement, co-appearance — needs multiple speakers on the *same*
topic. Shows that re-cover existing topics compound value across all of them at once.
Disconnected niche shows do not.

---

## 3. Gap analysis of the current 9

1. **No dialogic or debate format at all.** Every show is interview, monologue, or
   co-host explainer. #1144 needs *opposition* — two people who actually disagree on a
   recorded topic. Nothing in the corpus reliably produces it. **This is the largest gap and
   the one that unlocks a currently-dark enricher.**
2. **Thin multi-guest coverage.** Most shows are host + one guest, which produces few
   `guest_coappearance` edges.
3. **Narrow time span.** The corpus skews to recent months (publish histogram is dominated by
   2026-05 → 2026-08). `temporal_velocity` and prediction-tracking both need a longer baseline.
4. **AI/tech is the densest cluster** — good news, since it's where overlap already exists and
   new AI shows pay off immediately across perspectives + similarity.

---

## 4. Onboarding mechanics

- Corpus = pipeline output over a set of RSS feeds (`feeds.spec.yaml` in the corpus root).
- Onboard = add feed(s) to the spec → run the pipeline → enrichers run per the profile.
- Reprocessing existing transcripts is cheaper than re-scraping (transcripts are ours to keep;
  audio is bridge-only) — relevant when re-running enrichers over a grown corpus.
- Via the operator API, a feed is added with `POST /api/jobs` using `feed=<rss>`,
  `skip_existing=true`, `max_episodes=N`, `episode_order=newest`, `episode_offset=M`.

---

## 5. Candidate expansion set (target: +10 to +20 feeds)

Ranked by **enricher payoff**, not novelty. Grouped by what each group unlocks.

> **RSS URLs are deliberately omitted.** I have not resolved or verified feed URLs for any
> candidate below, and inventing them would produce jobs that fail at fetch. Resolve and
> verify each before adding to `feeds.spec.yaml`.

### Tier 1 — closes the disagreement gap (#1144)

The only group that unlocks a currently-dark enricher. Highest value per feed.

| Show | Why | Format |
| --- | --- | --- |
| **Open to Debate** (fmr. Intelligence Squared US) | Formal, motion-based debate. Two sides, same topic, explicit opposition. The single best structural fit for #1144. | Debate |
| **Intelligence Squared** (UK) | Same format, different topic mix and speaker pool. | Debate |
| **The Argument** (NYT) | Explicitly built around disagreement between recurring voices. | Debate |
| **Machine Learning Street Talk** | Genuinely contested AI takes, multi-host, guests pushed back on. Overlaps the densest existing cluster *and* is dialogic — rare combination. | Panel / dialogic |
| **The Compound and Friends** | Multi-guest markets panel; hosts disagree on record. Overlaps the finance cluster. | Panel |

### Tier 2 — deepens existing topic clusters (perspectives + similarity)

New voices on topics the corpus already covers. Immediate `topic_perspectives` payoff.

| Show | Cluster it deepens | Note |
| --- | --- | --- |
| **Dwarkesh Podcast** | AI / tech | Long-form, high-signal, unusually deep guests |
| **Odd Lots** (Bloomberg) | Markets | Two hosts + guest; was mis-recorded as already present |
| **Conversations with Tyler** | Cross-cluster | Contrarian, spans econ / tech / policy — bridges clusters |
| **a16z Podcast** | AI / tech + markets | Frequently multi-guest |
| **The TWIML AI Podcast** | AI / tech | Practitioner-level, complements NVIDIA AI |
| **Masters in Business** (Ritholtz) | Investing | Long-running; deepens Invest Like the Best overlap |
| **The Cognitive Revolution** | AI / tech | High cadence — useful for velocity |

### Tier 3 — adds the time dimension (velocity + prediction tracking)

| Show | Why |
| --- | --- |
| **EconTalk** | Runs since 2006. Deep back-catalog on recurring contested econ topics — the cheapest route to a real timeline. |
| **Planet Money back-catalog** | Already onboarded; pull *older* episodes rather than a new feed. |
| **The Journal / The Daily back-catalog** | Same — extend existing feeds backward instead of widening. |

### Tier 4 — diversity and quality, weaker overlap

Add only after Tiers 1–3; they broaden the corpus but compound less.

| Show | Cluster |
| --- | --- |
| **Sean Carroll's Mindscape** | Science / philosophy |
| **The Ezra Klein Show** | Policy / ideas |
| **Search Engine** | Narrative / general |
| **Acquired** | Business history — very long episodes, cost note below |
| **Lex Fridman** | Cross-cluster — very long episodes, cost note below |

---

## 5b. Geographic axis — global coverage (decided 2026-08-12)

**Decision:** expand geographically, **English-language only** for now. Native-language feeds
are blocked (see §5c) and tracked separately.

### Why this is not in tension with the overlap rule

The instinct is that geographic spread means less topic overlap, and therefore less enricher
payoff. The opposite is true if shows are chosen correctly. A Nigerian, Brazilian, or Indian
show discussing **AI regulation**, **dollar strength**, or **chip export controls** is the
*same topic* with a maximally different vantage point — which is exactly what
`topic_perspectives` wants, and the nearest thing to genuine opposition short of a formal
debate format.

So the selection rule sharpens to: **same topics, different continents.**

This also corrects the corpus's real bias. All 9 current feeds are US-produced and heavily
NY/SF financial-tech press. Every "perspective" the corpus can currently surface is drawn
from one media culture.

### Produced *in-region* vs produced *about-region*

A distinction worth enforcing, because it determines whether we get real perspective diversity
or the same editorial lens pointed elsewhere:

| Class | Meaning | Perspective value |
| --- | --- | --- |
| **A — in-region** | Editorially owned and produced in the region | **High** — genuinely different priors |
| **B — about-region** | Western outlet covering the region (BBC World Service, FT, Economist) | Moderate — better sourcing, same editorial lens |

**Prefer Class A.** Class B is a useful supplement and much easier to source, but a corpus
built only from Class B would produce the *appearance* of global coverage while still
reflecting a single viewpoint — the exact failure mode this axis exists to avoid.

### Candidates by region — all English-language

> **Unverified.** Names below come from domain knowledge. No RSS URL resolved, no check that
> the show is still active, no licensing review. Treat as a research list, not a work order.

**Asia**

| Show | Class | Overlap with existing clusters |
| --- | --- | --- |
| **ChinaTalk** | A/B | China tech + AI policy, chip controls — strong overlap with the AI cluster |
| **Sinica Podcast** | A | China politics/society, long-running |
| **The Seen and the Unseen** (India) | A | Long-form econ/policy; unusually deep |
| **Grand Tamasha** (India) | B | Indian politics |
| **Analyse Asia** (Singapore) | A | SE Asia tech + business |
| **China Global South Podcast** | A | China–Africa/LatAm relations — bridges two regions |

**Africa**

| Show | Class | Overlap |
| --- | --- | --- |
| **The Flip** | A | African tech/startups — direct overlap with the AI/tech + investing clusters |
| **Afrobility** | A | African business/tech deep dives |
| **Africa Daily** (BBC WS) | B | Broad daily coverage |
| **TechCabal** (Nigeria) | A | Nigerian tech ecosystem |

**South America**

| Show | Class | Overlap |
| --- | --- | --- |
| **Explaining Brazil** (The Brazilian Report) | A | Brazilian politics + economy — overlaps markets cluster |
| **Crossing Borders** | A | LatAm startups/VC — overlaps investing cluster |
| **Latin America in Focus** (AS/COA) | B | Regional politics/economics |

**Russia / Eurasia**

| Show | Class | Overlap |
| --- | --- | --- |
| **The Naked Pravda** (Meduza) | A* | Russian politics from Russian journalists |
| **The Eurasian Knot** | B | Russia/Eurasia scholarship |
| **Talk Eastern Europe** | A/B | Regional politics |

**\* Caveat worth stating plainly:** independent Russian-language media operates largely in
exile (Meduza is based in Riga). "In-region" is doing loose work here, and English-language
Russian media skews heavily toward the emigre/opposition perspective. A corpus using only
these sources will not represent mainstream domestic Russian discourse. That may be
acceptable, or even desirable — but it should be a **conscious** choice, not an artifact of
what happens to be available in English. This is the region where the English-only constraint
costs the most.

---

## 5c. Native-language support — blocked, scoped

**Status: BLOCKED.** Cannot be done through the operator API today.

Evidence:

- `config.py:1167` — `language: str = Field(default=DEFAULT_LANGUAGE, alias="language")`.
  A **single global scalar** on `Config`, not a per-feed field.
- `config.py:550` — documented as "Language code for transcription (e.g. `en`, `fr`, `de`)".
- `docs/api/HTTP_API.md` — `POST /api/jobs` exposes **no `language` parameter**.
- The prod box runs every job from one `viewer_operator.yaml` (visible in job argv).

So a non-English feed would transcribe under whatever global language the box is configured
for. What it would take:

1. Per-feed `language` on the feed spec, plumbed through `run_pipeline`.
2. A `language` parameter on `POST /api/jobs`.
3. **Open question, not investigated:** whether the enrichers and `topic_similarity` link
   concepts *across* languages. If topic clustering is embedding-based on English text, a
   Russian-language episode may form its own island and contribute **nothing** to
   cross-speaker perspective signal — which would defeat the entire purpose. **This should be
   answered before any of the above work is scheduled**, since it may make native-language
   ingestion worthless for enricher value regardless of pipeline support.

---

## 5d. Domain axis — coverage breadth (decided 2026-08-13)

**Operator decision:** the corpus must also span **biotech, culture, history, and
politics/daily news** — not just geography. "I need not only biodiversity. I need also
diversity on topics and coverage."

### Name the tension honestly

This is a **second objective**, not a refinement of the first.

| Objective | Wants | Optimal shape |
| --- | --- | --- |
| **Enricher value** (§2) | topic **overlap** | many shows, few topics |
| **Coverage breadth** (§5d) | topic **spread** | many topics, few shows each |

For a fixed slot budget they compete directly. A single biotech show is an **island**: no other
speaker in the corpus discusses CRISPR pricing, so it produces zero perspective cards, zero
co-appearance edges, and zero disagreement pairs. It adds searchable content and nothing else.

### The resolution: add domains in clusters, never singletons

**Minimum two shows per new domain, ideally three.** Three biotech shows covering the same
month's developments give cross-speaker perspectives *within* biotech — spread across domains,
overlap inside each. This is the only way to satisfy both objectives at once, and it means the
unit of expansion is a **domain cluster**, not a feed.

Corollary: adding four domains at 3 shows each consumes ~12 slots on its own. The domain axis
is therefore not a garnish on the geographic plan — it is comparable in size, and §6 is
re-cut accordingly.

### Operator's named picks — tech (all US, Class A)

Requested explicitly. All three deepen the existing AI/tech cluster, so they carry overlap
value as well as being wanted.

| Show | Note |
| --- | --- |
| **a16z Podcast** (Andreessen Horowitz) | frequently multi-guest → co-appearance edges; overlaps AI + investing clusters |
| **Lenny's Podcast** (Lenny's Newsletter) | product/growth; guests are operators rather than investors — a different speaker population on the same topics |
| **The Pragmatic Engineer** (Gergely Orosz) | engineering practice; complements Latent Space's AI-engineering angle |

### Candidates by domain

> **Unverified**, same as §5 and §5b — no RSS resolved, no activity check, no licensing review.

**Biotech / life sciences**

| Show | Note |
| --- | --- |
| **Ground Truths** (Eric Topol) | medicine × AI — bridges into the existing AI cluster rather than islanding |
| **Bio Eats World** (a16z bio) | biotech venture; shares the a16z house with the pick above |
| **The Long Run** (Timmerman Report) | biotech industry, long-form interviews |
| **The Readout LOUD** (STAT) | biotech news cadence — the "daily" of this domain |

**History**

| Show | Note |
| --- | --- |
| **The Rest Is History** | two hosts who argue on record — see the double-win below |
| **Empire** (Dalrymple & Anand) | two-host dialogic; colonial history, strong non-US framing |
| **Fall of Civilizations** | long-form narrative, single voice |
| **History Extra** (BBC) | interview cadence, broad coverage |

**Politics / daily news**

| Show | Note |
| --- | --- |
| **The Rest Is Politics** (Campbell & Stewart) | explicitly built on two people who disagree — the closest thing to a debate format outside §5 Tier 1 |
| **Today in Focus** (Guardian) | daily; non-US editorial lens on shared stories with The Daily |
| **FT News Briefing** | daily markets/politics; overlaps Unhedged and The Journal directly |
| **Pod Save America** | US partisan perspective — a genuinely different prior from the existing press-institution shows |

**Culture**

| Show | Note |
| --- | --- |
| **The New Yorker Radio Hour** | culture + politics interviews |
| **The Rest Is Entertainment** | two-host, media/culture industry mechanics |
| **Switched on Pop** | music analysis; two hosts, highly consistent format |
| **Articles of Interest** | design/fashion narrative |

### The double-win worth prioritising

Several history/politics candidates are **two-host shows whose hosts openly disagree** — *The
Rest Is History*, *The Rest Is Politics*, *Empire*, *The Rest Is Entertainment*. These satisfy
the domain axis **and** the §3 disagreement gap simultaneously, which no other candidate group
does.

That materially weakens my earlier sequencing argument. I recommended geographic-before-debate
because debate's payoff is deferred behind a dark enricher (#1144). But these shows are not
*only* debate feeds — they carry domain coverage that pays off immediately regardless of
whether #1144 ever lights up. **The "The Rest Is…" family and Empire should therefore be in the
first batch**, not deferred with the formal-debate group.

---

## 5e. VERIFIED feed registry (2026-08-13)

Every candidate below was resolved through the **iTunes Search API** (authoritative `feedUrl`)
and then **fetched and parsed** — HTTP status, channel title, `<item>` count, newest `pubDate`.
This supersedes the "unverified" warnings in §5, §5b and §5d for the shows listed here.

> Item counts use `content.count("<item>")`, **not** `grep -c`, which counts lines and
> catastrophically undercounts minified feeds — the error that produced a false "dead feed"
> report on 2026-08-12 (withdrawn B2).

### Verification killed two candidates and corrected four names

| Candidate | Finding |
| --- | --- |
| **Crossing Borders** | **DEAD** — newest episode `01 Aug 2023`, three years stale. Was in my Batch A geographic group. |
| **Overheard at National Geographic** | **DEAD** — newest `11 Jul 2023`. Was my NatGeo/adventure pick. |
| a16z Podcast | renamed → **The a16z Show** |
| Bio Eats World | renamed → **Raising Health** |
| Analyse Asia | renamed → **Analyse Podcast** |
| The Rest Is Politics | first resolution returned the **US spinoff**; the main UK show is a different feed |

Two of fifteen Batch-A picks were dead on arrival, and one would have ingested the wrong show.
That is the argument for resolving before planning, not after.

### Batch A — 15 feeds

Everything alive, currently publishing, and paying off without depending on a dark enricher.

| # | Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- | --- |
| 1 | **The a16z Show** | tech/VC | 657 | 2026-08-12 | `https://feeds.simplecast.com/JGE3yC0V` |
| 2 | **Lenny's Podcast** | tech/product | 356 | 2026-08-09 | `https://api.substack.com/feed/podcast/10845.rss` |
| 3 | **The Pragmatic Engineer** | tech/eng | 71 | 2026-08-12 | `https://api.substack.com/feed/podcast/458709.rss` |
| 4 | **The Rest Is History** | history · dialogic · UK | 714 | 2026-08-12 | `https://feeds.megaphone.fm/GLT4787413333` |
| 5 | **The Rest Is Politics** | politics · dialogic · UK | 597 | 2026-08-12 | `https://feeds.megaphone.fm/GLT9190936013` |
| 6 | **Empire: World History** | history · dialogic · UK | 399 | 2026-08-12 | `https://feeds.megaphone.fm/empirepodcast` |
| 7 | **ChinaTalk** | geo: China · tech policy | 555 | 2026-08-10 | `https://feeds.megaphone.fm/CHTAL4990341033` |
| 8 | **Explaining Brazil** | geo: Brazil · politics/econ | 393 | 2026-08-10 | `https://rss.beehiiv.com/podcasts/019fb9ea-ace8-7470-a622-111dd3c715f1.xml` |
| 9 | **The Seen and the Unseen** | geo: India · econ/policy | 117 | 2026-08-10 | `https://rss.libsyn.com/shows/91647/destinations/458496.xml` |
| 10 | **Ground Truths** (Topol) | biotech · medicine×AI | 93 | 2026-08-04 | `https://api.substack.com/feed/podcast/587835/s/119690.rss` |
| 11 | **The Readout Loud** (STAT) | biotech news | 416 | 2026-07-30 | `https://feeds.megaphone.fm/thereadoutloud` |
| 12 | **Odd Lots** (Bloomberg) | finance · dialogic | 1193 | 2026-08-10 | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss` |
| 13 | **EconTalk** | economics · deep back-catalog | 1062 | 2026-08-10 | `https://feeds.simplecast.com/wgl4xEgL` |
| 14 | **The Explorers Podcast** | adventure/exploration | 265 | 2026-08-11 | `https://feeds.megaphone.fm/ADL4434397541` |
| 15 | **Outside Podcast** | adventure/outdoors | 471 | 2026-08-12 | `https://feeds.megaphone.fm/POM5001301518` |

**Composition:** 6 of 15 are non-US-produced. Worth noting because it broadens the §5b
argument — geographic diversity is not only a Global South question. The three Goalhanger
shows are British, so they simultaneously serve the history/politics domains, the dialogic
format gap, and the US-monoculture correction. That triple duty is why they earn Batch A slots
over additional Global South feeds.

**Domain clusters honoured:** biotech 2, adventure 2, finance/economy 2 — none is a singleton
island. Finance and economy additionally land on **existing** overlap (Unhedged, Invest Like
the Best, Planet Money, The Journal), so they are the highest-overlap additions in the batch.

### Batch B — 10 feeds

| # | Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- | --- |
| 1 | **Open to Debate** | formal debate | 472 | 2026-08-07 | `https://feeds.megaphone.fm/PNP1207584390` |
| 2 | **Intelligence Squared** | formal debate · UK | 938 | 2026-08-11 | `https://feeds.megaphone.fm/NSR6363847171` |
| 3 | **Machine Learning Street Talk** | AI · contested | 258 | 2026-08-10 | `https://anchor.fm/s/1e4a0eac/podcast/rss` |
| 4 | **The New Yorker Radio Hour** | culture | 1055 | 2026-08-11 | `https://feeds.simplecast.com/TRuO_SRo` |
| 5 | **Switched on Pop** | culture/music | 548 | 2026-08-11 | `https://feeds.megaphone.fm/switchedonpop` |
| 6 | **The Rest Is Entertainment** | culture · dialogic · UK | 296 | 2026-08-12 | `https://feeds.megaphone.fm/GLT2052042801` |
| 7 | **Sinica Podcast** | geo: China · society | 557 | 2026-08-10 | `https://rss.art19.com/sinica` |
| 8 | **The Flip** | geo: Africa · tech | 101 | 2026-07-30 | `https://anchor.fm/s/114238e48/podcast/rss` |
| 9 | **The Long Run** (Timmerman) | biotech industry | 205 | 2026-08-11 | `https://feeds.soundcloud.com/users/soundcloud:users:317770704/sounds.rss` |
| 10 | **Dwarkesh Podcast** | cluster deepening | 136 | 2026-08-11 | `https://apple.dwarkesh-podcast.workers.dev/feed.rss` |

### Verified bench — resolved and live, not yet slotted

Available for Batch C or as substitutes.

| Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- |
| Nature Podcast | science | 917 | 2026-08-12 | `https://feeds.acast.com/public/shows/0185cea5-9e3b-4b82-a887-26f91f92765f` |
| Ideas of India | geo: India | 170 | 2026-08-12 | `https://rss.libsyn.com/shows/288629/destinations/2249435.xml` |
| Masters in Business | finance | 797 | 2026-08-12 | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/4e4cd910-40a1-4619-a5f3-ae2b0012ffff/...` |
| The Compound and Friends | finance · panel | 596 | 2026-08-11 | `https://feeds.megaphone.fm/TCP4771071679` |
| Analyse Podcast | geo: SE Asia | 535 | 2026-08-06 | `https://anchor.fm/s/10a88303c/podcast/rss` |
| Latin America in Focus | geo: LatAm | 241 | 2026-07-30 | `https://feeds.simplecast.com/_DUdLkxj` |
| Tooth &amp; Claw | nature/adventure | 209 | 2026-08-03 | `https://feeds.megaphone.fm/QCD9705695442` |
| Raising Health (ex-Bio Eats World) | biotech | 202 | 2026-07-06 | `https://feeds.simplecast.com/BXDamaKF` |
| Armchair Explorer | adventure | 207 | 2026-07-06 | `https://feeds.captivate.fm/armchairexplorer/` |

### Still NOT verified for any of these

Licensing / bridge constraints, episode-length cost profile, and whether the show's topics
*actually* overlap the corpus (that needs `topic_similarity` output, not my reading of a show
description). Ingesting is safe; the overlap claim is still an assumption.

---

## 6. Recommended shape of the next batch

**Three** axes now compete for slots, not two: the **format axis** (§5 — debate/panel, unlocks
#1144), the **geographic axis** (§5b — global perspective), and the **domain axis** (§5d —
biotech / culture / history / politics). All three are operator-endorsed.

**The slot budget no longer fits.** With a minimum of 2–3 shows per new domain (§5d), four new
domains alone consume ~10–12 slots. Adding that to 6–8 geographic and 4–5 format lands at
20–25 feeds, above the original +10 to +20. Two honest options: raise the ceiling, or run this
as **two batches**. The cut below assumes two batches.

### Batch A — ~14 feeds, everything that pays off immediately

| Group | Slots | Picks |
| --- | --- | --- |
| **Operator picks — tech** (§5d) | 3 | a16z Podcast, Lenny's Podcast, The Pragmatic Engineer |
| **Dialogic domain shows** (§5d) | 3 | The Rest Is History, The Rest Is Politics, Empire — the double-win: domain coverage *and* recorded disagreement |
| **Geographic Class A** (§5b) | 5 | ChinaTalk, The Flip, Explaining Brazil, The Seen and the Unseen, Crossing Borders |
| **Biotech cluster** (§5d) | 3 | Ground Truths, Bio Eats World, The Readout LOUD — three so the domain is not an island |

Everything in Batch A either deepens an existing cluster, corrects the US-only bias, or lands a
domain as a viable cluster. Nothing here depends on a dark enricher.

### Batch B — ~10 feeds, after Batch A is measured

| Group | Slots | Picks |
| --- | --- | --- |
| **Formal debate** (§5 Tier 1) | 3–4 | Open to Debate, Intelligence Squared, The Argument, Machine Learning Street Talk |
| **Culture cluster** (§5d) | 3 | New Yorker Radio Hour, The Rest Is Entertainment, Switched on Pop |
| **Remaining geographic** (§5b) | 2–3 | Analyse Asia, Sinica, Afrobility, China Global South |
| **Cluster deepening** (§5) | 2 | Odd Lots, Dwarkesh |
| **Back-catalog** (§5 Tier 3) | 0 new | Extend existing feeds backward — cheapest temporal signal per unit |

**Revised sequencing.** My earlier recommendation was geographic-before-debate, on the grounds
that debate's payoff sits behind a scale-gated dark enricher. §5d changes that in one respect:
the **two-host dialogic shows** (The Rest Is…, Empire) carry domain coverage that pays off
immediately *regardless* of whether #1144 ever lights up, so they belong in Batch A. **Formal**
debate shows — whose only value is the dark enricher — stay in Batch B. The distinction is
whether the feed is useful if #1144 never ships.

**Before ingesting any of them**, two cheap prerequisites:

1. **Resolve and verify every RSS URL.** Nothing in §5 or §5b has a verified feed.
2. **Run the actual topic clusters** over the current corpus so "overlap" is measured rather
   than assumed. The cluster column in §1 is my reading of show descriptions, not
   `topic_similarity` output.

---

## 7. Open decisions — none of these are settled

- **Target corpus size.** v2 → 500? 1000 episodes? And at what point does #1144's disagreement
  signal become *measurable*? Nobody has defined the scale gate numerically.
- **Curated vs broad ingest.** Hand-pick for topic overlap, or ingest widely and let the
  enrichers find density? The value model argues for curation; cost may argue otherwise.
- **Licensing / bridge constraints** on new feeds — audio is never rehosted; transcripts and
  derivatives are ours. Needs a per-feed check, especially for the debate shows.
- **Cost.** Transcription + ML enrichment per episode. Budget the growth step before
  committing — note that Acquired and Lex Fridman episodes run 3–4 hours and cost several
  times a typical 40-minute episode.
- **Episodes per new feed.** The current batch uses 10/feed. For Tier-1 debate shows, more
  episodes per feed may matter more than more feeds, since opposition signal needs recurring
  topics.

---

## 8. Verification status of this document

- **Verified:** the 9-feed table in §1 — pulled live from the corpus API on 2026-08-12.
- **Verified:** the missing `CORPUS-EVOLUTION-FOR-COMPLEX-ENRICHERS.md` reference — searched
  the whole repo, one hit, which was this doc's own pointer.
- **NOT verified:** every candidate show in §5. Names are from domain knowledge, not from
  checking a feed. No RSS URL has been resolved, no licensing checked, no episode-length or
  cost estimate measured, and no check made for whether a show is still active.
- **NOT verified:** the cluster assignments in §1 are my reading of each show's description
  field, not output from `topic_similarity`. Running the actual topic clusters over the corpus
  would give a real overlap map and should precede final selection.
