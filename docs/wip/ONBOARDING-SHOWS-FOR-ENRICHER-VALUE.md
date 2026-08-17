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

## 5f. FINAL LIST after editorial review (2026-08-13) — supersedes §5e

An independent content review was run against the operator's stated bar: *boutique, best of the
best; a mediocre show actively degrades the product because low-signal episodes inject noise
into every downstream summary, insight and graph edge.* Calibration standard: the operator's
three tech picks.

**All feeds below verified live** (iTunes resolve → fetch → parse). Ingest at **10 episodes
each** as a probe; depth is earned, not assumed.

### What the review changed

| Verdict | Shows |
| --- | --- |
| **Cut — failed the bar** | The Explorers Podcast, Outside Podcast (narrative/experiential; mines as anecdote, not insight), The Readout Loud (thin news-cadence) |
| **Cut — press-tour / repackaged discourse** | Intelligence Squared (book-tour vehicle at scale), Open to Debate (performed positions, not discovered insight), New Yorker Radio Hour (house-organ promotion), Masters in Business, The Compound and Friends, Raising Health (VC content marketing) |
| **Cut — dead or unusable on verification** | Crossing Borders (2023), Overheard at National Geographic (2023), **Afrobility (Jun 2025, 14 months dormant)**, **The Studies Show (5 items — wrong match or unusable feed)** |
| **Promoted to A** | Dwarkesh Podcast, Sinica, The Long Run |
| **Added — the category I had missed** | Conversations with Tyler, In Our Time, Acquired |
| **Swapped** | The Rest Is Politics main feed → its *Leading* sub-feed (long-form interviews with heads of state are durable; weekly news commentary is not) |

**The review's central correction:** the list under-weighted *preparation-heavy interview
shows* — the highest insight-per-hour sources that exist. Conversations with Tyler, Dwarkesh,
In Our Time and Acquired were either missing or mis-slotted.

**Adventure was dropped as a domain, not filled.** The genre's ceiling sits below the bar —
survival stories and gear talk mine as anecdote. The "National Geographic" intent is served
instead through science/natural-world (In Our Time's science episodes, Nature Podcast) and
exploration-as-history (Fall of Civilizations). Filling a domain badly is worse than not having
it — the rule from §5d, applied to itself.

### BATCH A — 15 feeds, probe at 10 episodes each

| # | Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- | --- |
| 1 | The a16z Show | tech/VC *(operator)* | 657 | 08-12 | `https://feeds.simplecast.com/JGE3yC0V` |
| 2 | Lenny's Podcast | tech/product *(operator)* | 356 | 08-09 | `https://api.substack.com/feed/podcast/10845.rss` |
| 3 | The Pragmatic Engineer | tech/eng *(operator)* | 71 | 08-12 | `https://api.substack.com/feed/podcast/458709.rss` |
| 4 | **Conversations with Tyler** | ideas/econ — *the biggest omission* | 297 | 08-12 | `https://rss.libsyn.com/shows/137081/destinations/850607.xml` |
| 5 | **Dwarkesh Podcast** | AI/ideas *(promoted from B)* | 136 | 08-11 | `https://apple.dwarkesh-podcast.workers.dev/feed.rss` |
| 6 | **In Our Time** (BBC) | history/science/philosophy | 1102 | 08-07 | `https://podcasts.files.bbci.co.uk/b006qykl.rss` |
| 7 | The Rest Is History | history, dialogic, UK | 714 | 08-12 | `https://feeds.megaphone.fm/GLT4787413333` |
| 8 | Empire: World History | history, dialogic, UK | 399 | 08-12 | `https://feeds.megaphone.fm/empirepodcast` |
| 9 | ChinaTalk | geo: China, tech policy | 555 | 08-10 | `https://feeds.megaphone.fm/CHTAL4990341033` |
| 10 | **Sinica Podcast** | geo: China *(promoted — China pair)* | 557 | 08-10 | `https://rss.art19.com/sinica` |
| 11 | **Ideas of India** *(replaces The Seen and the Unseen — see §5h)* | geo: India, econ/policy | 170 | 08-12 | `https://rss.libsyn.com/shows/288629/destinations/2249435.xml` |
| 12 | Odd Lots | finance, dialogic | 1193 | 08-10 | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss` |
| 13 | EconTalk | economics | 1062 | 08-10 | `https://feeds.simplecast.com/wgl4xEgL` |
| 14 | Ground Truths (Topol) | biotech, medicine×AI | 93 | 08-04 | `https://api.substack.com/feed/podcast/587835/s/119690.rss` |
| 15 | **The Long Run** (Timmerman) | biotech *(promoted — replaces Readout Loud)* | 205 | 08-11 | `https://feeds.soundcloud.com/users/soundcloud:users:317770704/sounds.rss` |

### BATCH B — 10 feeds

| # | Show | Domain | Items | Newest | RSS |
| --- | --- | --- | --- | --- | --- |
| 1 | **Capitalisn't** *(replaces Acquired — see §5h)* | political economy | 248 | 08-06 | `https://feeds.simplecast.com/XytPkydI` |
| 2 | The Peter Attia Drive | medicine/longevity | 451 | 08-10 | `https://rss.libsyn.com/shows/121729/destinations/713489.xml` |
| 3 | In Moscow's Shadows (Galeotti) | geo: Russia — *the one that clears* | 280 | 08-09 | `https://rss.buzzsprout.com/1026985.rss` |
| 4 | **The Rest Is Politics: Leading** *(the durable half of that franchise)* | politics, long-form interviews | 207 | — | `https://feeds.megaphone.fm/GLT9029505120` |
| 5 | Explaining Brazil | geo: Brazil | 393 | 08-10 | `https://rss.beehiiv.com/podcasts/019fb9ea-ace8-7470-a622-111dd3c715f1.xml` |
| 6 | Latin America in Focus | geo: LatAm *(LatAm pair)* | 241 | 07-30 | `https://feeds.simplecast.com/_DUdLkxj` |
| 7 | Macro Musings | monetary economics | 562 | 08-10 | `https://rss.libsyn.com/shows/138806/destinations/865793.xml` |
| 8 | Complex Systems (patio11) | financial infrastructure | 101 | 08-13 | `https://feeds.transistor.fm/complex-systems-with-patrick-mckenzie-patio11` |
| 9 | Past Present Future (Runciman) | history of ideas | 331 | 08-12 | `https://feeds.megaphone.fm/ARML2708405200` |
| 10 | Machine Learning Street Talk | AI, contested | 258 | 08-10 | `https://anchor.fm/s/1e4a0eac/podcast/rss` |

### Bench — verified live, unslotted

Capitalisn't (248, 08-06) · 80,000 Hours (347, 08-11) · Nature Podcast (917, 08-12) · Switched on Pop (548, 08-11) · The Rest Is Entertainment (296, 08-12) · The Flip (101, 07-30) · Fall of Civilizations (22, 07-31 — tiny but wholly durable) · The Rest Is Politics: *Leading* (207, resolved not yet fetch-verified) · The Life Scientific (356, **05-26 — BBC hiatus, re-check before use**)

### Open problems in this list

1. **Africa is now a singleton.** Afrobility was the intended pair for The Flip and is 14 months
   dormant. Africa currently has one viable feed, violating the §5d clustering rule. Needs
   either another verified African show or a conscious decision to accept the singleton.
2. **Russia is a singleton by design** — In Moscow's Shadows is the only show that clears. The
   review's recommendation is to *not* force a second mediocre Russia show, and instead treat
   geopolitics (ChinaTalk, Sinica, Explaining Brazil, Moscow's Shadows) as one domain where
   cross-speaker comparison happens at the regional-analysis level.
3. **Backfill caps needed on high-volume archives.** Odd Lots (1193), In Our Time (1102),
   EconTalk (1062) and The a16z Show (657) all carry large back-catalogs whose older material is
   dated. The a16z feed additionally carries a decade of thesis marketing. Cap the probe at the
   newest N rather than ingesting depth.
4. **The bar applies retroactively to the existing corpus.** The review flags The Daily, Planet
   Money and The Journal as news-cadence and the NVIDIA AI Podcast as vendor marketing — four of
   the nine current feeds sit below the standard now being applied to new ones. Not acted on;
   recorded because silence would imply they passed.

---

## 5g. Onboarding protocol — smoke → assess → deepen (decided 2026-08-13)

**Operator decision.** The existing nine feeds are **not** re-litigated — they stay, for
reasons outside the enricher-value model. §5f's retroactive flag is recorded but **not
actionable**.

New shows are onboarded by a three-phase loop, one show at a time. The ordering matters:
**one episode is a far cheaper failure detector than ten**, and several Batch A shows are
cost outliers where a bad first result would otherwise be discovered ten episodes deep.

### Phase 1 — smoke, 1 episode per show

`max_episodes=1`, `episode_order=newest`, `skip_existing=true`, all 15 Batch A shows.
Roughly 15 episodes total — minutes each, negligible spend — and it exercises the
transcription path, the enrichers, and the cost profile on real content from each publisher.

### Phase 2 — assess, before any depth

Per episode, from the API (no SSH needed):

| Check | Source | PASS | INVESTIGATE | FAIL |
| --- | --- | --- | --- | --- |
| Job outcome | `GET /api/jobs/{id}` | `succeeded` | — | `failed` / `stale` |
| Artifacts complete | `/api/corpus/coverage` | GI **and** KG present | — | either missing |
| Insight count | index `doc_type=insight` | 6–31 (batch-1 band) | < 6 | 0 |
| KG nodes | `kg_entity` + `kg_topic` | ~20–29 | < 15 | 0 |
| GI↔KG bridging | `bridge_partition` in episode detail | `both` > 0 | `both == 0` | — |
| Summary substance | `summary_bullets` | specifics: names, numbers, mechanisms | generic/vague | empty |
| Cost | `podcast_pipeline_run_cost_usd_total` delta | ≲ $0.30/ep | > $0.50/ep | approaches the $10/run cap |

**"Catastrophic" — stop the show, move on, record why:** job fails twice, zero insights,
missing artifacts, or a transcript that is obviously wrong (untranscribed music, wrong
language, ad-read soup).

**"Weak" — a judgement call, not an auto-stop:** in-band but thin. Record and decide; a show
can be genuinely good and simply produce fewer, denser insights.

### Phase 3 — the decision is depth, and it has TWO independent axes

**Operator framing (2026-08-13), and it corrects an earlier error in this protocol.** A poor
probe result is **not** a reason to drop a show. Onboarding is deliberately a pattern-finding
exercise: *"how far can I get with what I have, and how much do I need to invest to up the
game."* Shows are partly test material for the pipeline, not only content.

The earlier version of this section said "catastrophic → drop the feed." **That was wrong.** It
conflated two things that fail independently:

| | Show clears the editorial bar | Show does not |
| --- | --- | --- |
| **Pipeline handles it well** | **DEEPEN** — 10, then 20/50 as warranted | **DROP** — the only legitimate reason to drop |
| **Pipeline handles it badly** | **PARK at 1** — log the pipeline defect, revisit after the fix | DROP (editorial), and still log the defect |

So the single decision per show is **continue ingesting, or park until the pipeline improves** —
never "this show is bad because our cleaning stage is bad."

**Bucket definitions:**

- **DEEPEN** — content clears the bar and processing is clean. Go to 10; revisit for 20/50.
- **PARK** — content clears the bar, processing does not (ad contamination, poor diarization,
  thin insights on rich material). Stay at 1. **Record the defect in the rollout-followups
  homework**, not here. Revisit when that defect is fixed.
- **DROP** — content does not clear the editorial bar. The *only* content-driven exit.
- **BLOCKED** — structurally not ingestible: over the 2-hour ceiling (§5h), non-English (§5c),
  dead feed. Not a quality judgement at all.

**Why this matters beyond bookkeeping:** parking a show because the cleaning stage mangles ad
reads and dropping it as "low quality" produce the same corpus today, but opposite outcomes in
three months. The first leaves a queue of good shows waiting on a known fix; the second
silently discards them and hides the fix's value.

Then loop to the next show. **One show at a time** — the pipeline runs one job anyway, and
serial execution keeps each result attributable.

### The probe is also a pipeline capability assessment

Each show exercises a different weakness: heavy ad load (a16z, Lenny's, Pragmatic Engineer),
long-form endurance (Dwarkesh, Ideas of India), non-US accents and names (ChinaTalk, Sinica,
Ideas of India, Explaining Brazil), multi-speaker dialogic formats (The Rest Is History, Empire,
Odd Lots), and dense technical vocabulary (Ground Truths, The Long Run, Complex Systems).

Findings therefore land in **two** places, and confusing them loses information:

- **This document** — content verdicts, buckets, depth decisions.
- **`INCREMENTAL-ROLLOUT-FOLLOWUPS-2026-08-11.md`** — pipeline defects the probe exposes.
  Those are reusable across every feed, including the nine existing ones.

### Ad-read contamination — an explicit check (added 2026-08-13)

**Operator concern, and it is the sharpest first test available.** The a16z Show, Lenny's
Podcast and The Pragmatic Engineer are all heavy on sponsor reads, pre-roll, mid-roll and long
branded intros. Those are not neutral: sponsor copy is fluent, confident, specific-sounding
prose, which is exactly the shape the summariser and insight extractor reward. Left uncleaned
it becomes summary bullets and grounded "insights" about a project-management SaaS.

The pipeline has a cleaning stage (`llm_cleaning_cost_usd` is a tracked per-stage cost), so the
probe answers empirically whether that stage strips ad content on real ad-heavy feeds.

**Add to the Phase-2 assessment, per episode:**

| Check | PASS | FAIL |
| --- | --- | --- |
| Sponsor copy in `summary_bullets` | none | any bullet describing a sponsor/product plug |
| Sponsor entities in KG | none | brand nodes that are advertisers, not subjects |
| Insights derived from ad reads | none | any |
| Host boilerplate ("subscribe", "rate us", "our sponsor") | absent | present in summary or insights |

**A failure here is not a reason to drop the show** — it is a reason to fix the cleaning stage,
because it would affect every ad-supported feed in the corpus including existing ones. Record
it as a pipeline defect, not a content verdict.

### Cost and duration outliers to smoke FIRST

These are exactly why the 1-episode phase exists. Long episodes cost several times a typical
40-minute one, and the per-run soft cap is **$10** (§ cost analysis in the rollout followups):

### Probe group 1 — the five to smoke first (decided 2026-08-13)

Chosen to stress the two things most likely to go wrong, not to be representative:

| # | Show | Median | Why first |
| --- | --- | --- | --- |
| 1 | **The a16z Show** | 49 min | **heavy ad load** — sponsor reads, branded intros |
| 2 | **Lenny's Podcast** | 92 min | **heavy ad load** + long |
| 3 | **The Pragmatic Engineer** | 85 min | **heavy ad load** + longest tail (max 171 min) |
| 4 | **Dwarkesh Podcast** | 87 min | longest median retained (max 157 min) |
| 5 | **Ideas of India** | 93 min | longest median in the batch |

Shows 1–3 are the ad-contamination test; 3–5 are the duration/cost test; the overlap is
deliberate. If the cleaning stage handles a16z and Lenny's, it will handle almost anything else
on the list.

Note the two shows originally slated as duration outliers — The Seen and the Unseen (271 min)
and Acquired (238 min) — were **disqualified entirely** by the 2-hour ceiling (§5h), so they
are not probed at all.

Remaining backfill caution for later phases: **In Our Time** (1102 episodes), **Odd Lots**
(1193), **EconTalk** (1062) and **The a16z Show** (657) carry large, partly-dated archives —
probe newest only and cap depth rather than ingesting the back-catalog.

### What gets recorded, and where

Per show, appended to this document as the loop runs: episode count ingested, insight and KG
counts, modelled cost, the verdict, and — when a show is dropped — **the specific reason**.
A dropped show with no recorded reason will be re-proposed by someone in six months.

---

## 5h. Episode-duration ceiling — 2 hours (decided 2026-08-13)

**Operator rule:** maximum acceptable episode length is about **two hours**. This is why Lex
Fridman and similar shows were never on the list.

Rationale beyond cost: a 4-hour episode is one row in the corpus regardless of length, so it
dilutes per-episode signal density, dominates transcription time and spend, and is the item
most likely to trip the **$10 per-run soft cap on its own**.

Measured from `<itunes:duration>` over the newest 12 episodes of every candidate — median
matters more than max, so one long special doesn't disqualify a show and a consistently-long
show can't hide behind one short episode.

### Disqualified

| Show | Median | Max | Was |
| --- | --- | --- | --- |
| **The Seen and the Unseen** | **271 min (4.5 h)** | 342 min | Batch A #11 |
| **Acquired** | **238 min (4.0 h)** | 291 min | Batch B #1 |
| **Fall of Civilizations** | **204 min (3.4 h)** | 329 min | bench |

Note both Batch-A/B casualties were *strong* editorial picks — Acquired was one of the
reviewer's four "omissions I'd call errors". The duration rule overrides editorial merit,
which is the point of having it as a hard rule rather than a preference.

### Replacements

| Slot | Out | In | Why |
| --- | --- | --- | --- |
| A #11 | The Seen and the Unseen | **Ideas of India** (median 93 min) | promoted from B; keeps India covered within the ceiling |
| B #1 | Acquired | **Capitalisn't** (median 49 min) | political economy, strengthens the economics cluster |
| B #4 | Ideas of India *(promoted)* | **The Rest Is Politics: Leading** (median 69 min) | the durable half of that franchise — long-form interviews rather than weekly commentary |

### Full measurements (newest 12 episodes, minutes)

| Show | Median | Max | Verdict |
| --- | --- | --- | --- |
| The Flip | 12 | 16 | OK |
| Nature Podcast | 17 | 32 | OK |
| Explaining Brazil | 31 | 52 | OK |
| Latin America in Focus | 34 | 37 | OK |
| Switched on Pop / The Rest Is Entertainment | 37 | 50–58 | OK |
| In Moscow's Shadows | 44 | 61 | OK |
| Empire | 46 | 57 | OK |
| The a16z Show · Capitalisn't | 49 | 65–81 | OK |
| Odd Lots | 50 | 67 | OK |
| In Our Time · Ground Truths | 54 | 58–64 | OK |
| Conversations with Tyler | 55 | 61 | OK |
| Macro Musings · Past Present Future | 58 | 68–72 | OK |
| The Peter Attia Drive | 59 | 145 | occasional >2 h |
| ChinaTalk | 68 | 120 | occasional >2 h |
| The Long Run · The Rest Is Politics: Leading | 69 | 78 | OK |
| EconTalk | 70 | 89 | OK |
| The Rest Is History | 72 | 82 | OK |
| Sinica | 76 | 114 | OK |
| 80,000 Hours | 78 | 168 | occasional >2 h |
| Machine Learning Street Talk | 79 | 113 | OK |
| The Pragmatic Engineer | 85 | 171 | occasional >2 h |
| Dwarkesh Podcast | 87 | 157 | occasional >2 h |
| Lenny's Podcast | 92 | 99 | OK |
| Ideas of India | 93 | 114 | OK |

### Unresolved: the ceiling can only be applied at feed level

Five retained shows publish the **occasional** episode over two hours — Peter Attia (max 145),
ChinaTalk (120), 80,000 Hours (168), The Pragmatic Engineer (171), Dwarkesh (157). Their
medians are well inside the limit, so they stay.

But **`POST /api/jobs` has no duration filter**, so a long episode inside an otherwise-short
feed will be ingested. The rule is enforceable when *choosing feeds*, not when selecting
episodes. If per-episode enforcement is wanted, that is a pipeline feature request (skip or
flag episodes whose `itunes:duration` exceeds a configured maximum) — not something the
current API can express.

---

## 5i. Threshold calibration against the existing nine (2026-08-13)

Read-only exercise proposed by the operator: apply the §5g buckets to the **existing** feeds
first, to calibrate the thresholds before judging new shows by them. No ingestion, no cost.
Sample: **4 episodes per feed, 36 total.**

Content verdicts are moot here — those nine stay regardless — so this measures the **pipeline
axis** only.

### Results

| Feed | bullets/ep | KG nodes/ep | GI↔KG linked | unlinked GI | ad markers |
| --- | --- | --- | --- | --- | --- |
| Invest Like the Best | 11.8 | 25.0 | 16.8 | 0.0 | 0/4 |
| NVIDIA AI Podcast | 8.8 | 22.8 | 15.8 | 0.0 | 0/4 |
| Hard Fork | 8.2 | 25.2 | **20.8** | 0.2 | 0/4 |
| The Daily | 8.2 | 22.5 | **11.8** | 0.0 | 0/4 |
| No Priors | 8.0 | 24.5 | 15.5 | 0.2 | 0/4 |
| The Journal. | 8.0 | 21.5 | 13.8 | 0.5 | 0/4 |
| Unhedged | 8.0 | 23.2 | 15.5 | 0.0 | 0/4 |
| Latent Space | 8.0 | 25.0 | 18.0 | 0.0 | 0/4 |
| Planet Money | 7.2 | 22.8 | 16.5 | 0.5 | 0/4 |

### Finding 1 — ad cleaning works; **0 of 36** episodes contaminated

No sponsor copy, promo codes, or host boilerplate reached any summary — including on the
ad-heavy feeds (Hard Fork, No Priors, The Journal). The cleaning stage handles ad-supported
content.

**Consequence for the probe:** the a16z / Lenny's / Pragmatic Engineer group becomes a
*confirmation* test rather than a discovery test. Still worth running — those three are heavier
on ads than anything here — but the prior is now "this works" rather than "unknown".

### Finding 2 — the §5g thresholds were decoration

Observed spread is far tighter than the bands I inherited from batch-1:

| Metric | Observed | My §5g band | Verdict |
| --- | --- | --- | --- |
| KG nodes/ep | 21.5 – 25.2 | 20–29 | passes everything |
| GI↔KG linked | 11.8 – 20.8 | `> 0` | passes everything |
| bullets/ep | 7.2 – 11.8 | (unstated) | — |

A gate nothing can fail is not a gate.

### Finding 3 — a prediction of mine, refuted

I predicted format-dependence: that **The Daily** (~25-min news) would produce markedly thinner
output than **Latent Space** (~90-min technical long-form), and therefore that thresholds would
need to be per-format.

**Wrong.** The Daily 8.2 bullets / 22.5 KG; Latent Space 8.0 / 25.0. Near-identical.

The likely mechanism matters more than the miss: **summary bullet count is a property of the
prompt, not of the episode.** The summariser targets a roughly fixed count regardless of input
length or richness. So bullets/ep measures the pipeline's configuration, not the content — and
grading new shows on it would have been grading a constant.

### Revised, evidence-based thresholds

Replaces the §5g bands. Derived from the observed distribution, not inherited.

| Signal | INVESTIGATE below/above | Rationale |
| --- | --- | --- |
| KG nodes/ep | **< 18** | observed floor 21.5; 18 is a real outlier |
| GI↔KG linked (`both`) | **< 8** | observed floor 11.8 — **the most discriminating signal**, ~2× spread |
| unlinked GI (`gi_only`) | **> 2** | observed max 0.5; a rise means GI is producing content the graph cannot corroborate |
| any ad marker in summary | **any** | observed 0/36 — a single hit is a **pipeline defect**, not a content verdict |
| bullets/ep | *not a gate* | pipeline constant; record for interest only |

**`bridge_partition.both` is promoted to the primary quality signal.** It has the widest real
spread (11.8 → 20.8) and it measures something meaningful: how much extracted insight is
corroborated by graph structure.

### Caveats

- **36 episodes, 4 per feed.** Sufficient to show the bands are too wide and that ads are clean;
  **not** sufficient for confident per-feed rankings. Hard Fork's 20.8 vs The Daily's 11.8 is
  suggestive, not established.
- Sampling took the **newest** episodes per feed, so this reflects current pipeline behaviour,
  not the whole corpus (which spans several pipeline versions).
- Ad detection is **string-matching** on summary text. It cannot detect a sponsor segment that
  was summarised into neutral prose without trigger phrases. A clean result is good evidence,
  not proof.

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
