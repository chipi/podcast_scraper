# RFC-103: Momentum Layer — Read-Time Trending Across Saveable Entities

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: Consumer app (Home / discovery / library), enrichment layer, ranking
- **Related PRDs**:
  - `docs/prd/PRD-042-home.md` (Home / Learning Hub — Trending surfaces)
  - `docs/prd/PRD-041-consolidation.md` (saved insights / resurfacing)
- **Related RFCs**:
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` (enrichers produce the durable data)
- **Related UX specs**:
  - `docs/uxs/UXS-012-consumer-home.md`
- **Related Documents**:
  - `src/podcast_scraper/enrichment/enrichers/temporal_velocity.py`
  - `src/podcast_scraper/server/app_discover_view.py` (ranking consumer)
  - `src/podcast_scraper/server/routes/app_discover.py`, `app_user_state.py` (telemetry / saves)
  - `web/learning-player/src/components/TrendingTopics.vue`

## Abstract

"Trending" should apply to **everything a user can save or that feeds recommendations** — topics,
semantic clusters (`tc:`), storylines (`thc:`), people, episodes, shows, and insights — not just bare
topics. This RFC defines a **Momentum Layer**: one primitive — an **EWMA over a per-entity weekly
event series, anchored to `today`, computed at read** — applied uniformly across all those entities,
along **two axes** (velocity = *rising*, volume = *most-active*) and **two event sources** (content =
corpus mentions/appearances; engagement = saves/plays/opens).

It also fixes the layering. Today `temporal_velocity` bakes a `now`-anchored, monthly, topic-only
velocity into the enricher, read raw in two places. Instead: **enrichers produce durable data; a
dedicated momentum capability derives + serves it** to its two consumers — **UI surfaces** (via a
dedicated `GET /api/app/trending`) and **recommendations** (the discover ranker) — so "what's hot"
means one thing everywhere and is always relative to today.

**Architecture Alignment:** Preserves the RFC-088 split — enrichers do the expensive, durable
aggregation on ingest; the momentum capability does the cheap, time-relative derivation per read.

## Problem Statement

The current trending signal has four coupled problems:

- **Stale.** `velocity_last_over_6mo` is frozen at the enricher's run-time `now` (`_now_utc`). As the
  calendar advances the anchor ages; keeping it current needs a re-run cron whose only job is
  re-anchoring time.
- **Wrong timescale.** It is *last **month** ÷ trailing **6 months***, monthly-bucketed — a slow
  seasonal signal, not "hot now" for a fast feed.
- **Topic-only.** Users save and rank **episodes, insights, shows, people** too; trending should
  cover every saveable entity and feed recommendations, not just the topic rail.
- **No owning layer.** Velocity is produced *and* half-consumed in the enricher, then read raw by the
  Trending rail and the discover ranker independently — no single capability owns "what's hot" for
  both UI and recommendations.

**Use Cases:**

1. **Trending across kinds** — "heating up" topics, storylines, shows, episodes, and insights, each
   as of *today*, from one signal.
2. **Recommendations** — the discover feed boosts episodes whose *content* is trending and the things
   a user's *followed* entities are trending, using the same momentum the UI shows.
3. **Deterministic e2e** — the app's Playwright suite sees a stable momentum signal regardless of the
   date the tests run.

## Goals

1. **One primitive, every saveable entity**: topics / `tc:` / `thc:` / people / episodes / shows /
   insights all trend via the same EWMA momentum.
2. **Two axes, two event sources**: velocity + volume; content + engagement.
3. **Today-relative, read-time**: momentum reflects `today` on every read; no cron just to re-anchor.
4. **Clean layering**: enrichers produce durable data; a momentum capability derives + serves it.
5. **One vocabulary for UI + recommendations + operator**: a dedicated `GET /api/app/trending`
   (consumer, per-user + corpus scope), a `GET /api/corpus/trending` (operator Dashboard global view
   across every kind), and a programmatic `momentum(entity)` the ranker calls — one capability.
6. **Fully config-driven**: half-lives, blend weights (global + per-kind), heating-up threshold, and
   the engagement floor are all config, no hardcoded constants.
7. **Deterministic fixtures**: a pinned reference `now` + seeded events make momentum byte-stable.
8. **Backward compatible**: keep `velocity_last_over_6mo` as a fallback during migration.

## Constraints & Assumptions

**Constraints:**

- No daily cron required for correctness (freshness comes from read-time anchoring).
- Read-time derivation is cheap: O(entities × weeks) arithmetic, memoized per `(corpus, week,
  params)`.
- Deterministic in CI/e2e (pinned `now` + seeded engagement; no wall-clock dependence).
- Engagement aggregation exposes **counts only** (no per-user data in the corpus-wide signal).

**Assumptions:**

- Weekly granularity is the floor (fine for a fast feed, smooth vs daily noise; daily is a later
  option over the same series).
- Per-topic/person weekly counts over multi-year history are small (one int per entity per active
  week); engagement counts are similarly small.

## Design & Implementation

### 1. Layering — the spine

```text
Enrichers (RFC-088, on ingest)   →  durable weekly series (facts, no `now`)
        │  content: per-topic / per-person mention counts
        │  engagement: per-entity save/play/open counts (aggregated from telemetry)
        ▼
Momentum capability (server module, per read)
        │  EWMA momentum + volume, anchored to configurable `now`
        │  aggregation (groups + point-in-time entities), content⊕engagement blend, cache
        ├──────────────►  Consumer UI:  GET /api/app/trending          (per-user + corpus scope)
        ├──────────────►  Operator UI:  GET /api/corpus/trending       (Dashboard global view)
        └──────────────►  Recommendations:  momentum(entity)           (programmatic, the ranker)
```

- **Enrichers only produce data.** `temporal_velocity` (content) emits per-topic/person weekly
  counts; a small engagement aggregator rolls telemetry (impressions/clicks/playback/favorites) into
  per-entity weekly counts. Neither embeds `now` or a window.
- **The momentum capability owns derivation + serving.** One module computes momentum from the series
  against `now`, aggregates, blends, caches, and exposes both the endpoint and `momentum(entity)`.
- **Two consumers, one source of "hot":** UI surfaces and the discover ranker.

### 2. Event model — content ⊕ engagement per entity

Each entity's weekly **event series** is defined from two sources:

| Entity | Content events (corpus) | Engagement events (telemetry) |
| --- | --- | --- |
| topic / `tc:` / `thc:` / person | mentions / appearances per week | follows + card opens |
| episode | Σ its topics' + people's mentions | plays, saves, queue-adds, discover clicks |
| show | Σ its episodes' content series | subscribes + plays of its episodes |
| insight | its topic's mentions | saves + opens |

Recurring entities (topics/people) have a native content series; **point-in-time** entities
(episodes/insights/shows) derive their content series by **aggregating the topics/people they
contain** — so everything rolls up from the same per-topic/person atom.

### 3. The primitive — EWMA momentum, read-time

```python
def ewma_alpha(half_life_weeks: float) -> float:
    return 1.0 - 0.5 ** (1.0 / half_life_weeks)

def momentum(series: list[int], fast_hl=3.0, slow_hl=12.0) -> tuple[float, float]:
    fast = ewma(series, ewma_alpha(fast_hl))[-1]   # recent level  → volume axis
    slow = ewma(series, ewma_alpha(slow_hl))[-1]   # baseline level
    velocity = round(fast / slow, 4) if slow > 0 else 0.0   # >1 rising, <1 cooling
    return velocity, fast                                   # (rising, recent volume)
```

- `series` is zero-filled **up to the reference week derived from `now`**, so a silent entity's fast
  EWMA decays below its slow → it cools automatically as days pass ("changes with any given day").
- **Fast half-life ~3 weeks, slow ~12 weeks** (config). Same formula for every entity + source.
- **Velocity** (rising) and **volume** (recent level) are the two axes the UI already plots.

### 4. Aggregation

```python
def series_for(entity_id, weekly_by_topic_or_person, contained, members):
    if entity_id.startswith(("topic:", "person:")):
        return weekly_by_topic_or_person.get(entity_id, {})
    if entity_id.startswith(("tc:", "thc:")):            # cluster / storyline
        return sum_weekly(series_for(m, ...) for m in members(entity_id))
    return sum_weekly(series_for(c, ...) for c in contained(entity_id))  # episode/show/insight
```

Deriving (not storing) group/contained series keeps momentum consistent when clusters are
re-derived or new episodes arrive — no recompute-on-recluster coupling.

### 5. Blend — content ⊕ engagement

Per-entity momentum blends the two sources' momenta with configurable weights:

```text
score(entity) = w_content · momentum(content_series) + w_engagement · momentum(engagement_series)
```

Weights are per-kind config (start content-heavy where engagement is sparse — a small-audience app
has thin engagement early; content is dense from day one). Each source keeps its own velocity +
volume; the blend is exposed alongside the components so a surface can show either.

### 6. Corpus-wide vs per-user

- **Content momentum** is **corpus-wide** — one "what's hot" for everyone.
- **Engagement momentum** defaults to a **corpus-wide aggregate** (counts only; a configurable
  min-count floor before an entity is shown, to avoid single-user identifiability), with a **per-user
  scope** (`scope=mine` — reuses the existing "your corpus" lens) for "*your* recent momentum",
  **shipped in v1**. The per-user scope has no min-count floor (it is the user's own data).

### 7. Dedicated endpoint — `GET /api/app/trending`

```text
GET /api/app/trending?kind=topic|cluster|storyline|person|episode|show|insight
                      &scope=corpus|mine &horizon=week|month|quarter &limit=N
→ [{ id, kind, label, velocity, volume, heating_up, series, components:{content,engagement} }]
```

- Owned by the momentum capability (derivation + blend + cache live in one place); memoized per
  `(corpus, as_of_week, params)`.
- **Why not extend `GET /api/app/corpus/enrichment`:** that endpoint is a thin envelope reader;
  trending needs derivation, blending, per-kind aggregation, scope, and caching — a distinct
  capability, not an envelope passthrough.
- **Operator global view:** the same capability backs an operator surface — a corpus-scoped route
  (`GET /api/corpus/trending`) feeding a **Dashboard "Trending (global)" panel** in the operator
  viewer, showing momentum across **every supported kind** in one place (the operator's bird's-eye
  "what's hot corpus-wide"), **with a per-kind sparkline** (the weekly `series`) per row so the
  operator sees each entity's trajectory, not just its current momentum. Consumer app and operator
  viewer read one capability; only the route prefix (`/api/app` vs `/api/corpus`) and default scope
  differ.

### 8. Recommendations consumer

The discover ranker's `SIGNAL_TREND_VELOCITY` (`app_discover_view.py`
`_topic_velocities` → `_trend_boost`) generalizes to **`momentum(entity)`**: boost an episode by the
momentum of its content and by the momentum of the user's followed entities (`topic:`/`tc:`/`thc:`/
`person:`). Same capability, programmatic call — the feed and the rail agree on "hot."

### 9. Fixture determinism

`app-validation-corpus/v3` provides content weekly series (per-topic/person counts) **and** seeded
engagement events (playback/favorites/discover-clicks in the committed per-user state or a seed);
the e2e webServer pins `APP_TRENDING_NOW` to the corpus's latest week. Momentum for every kind is
then deterministic — **no episode re-dating**, audio-safe (only enrichment/seed JSON changes).

### 10. Configuration

Everything tunable lives under a `momentum` config block — **global defaults with per-kind
overrides**, no hardcoded constants. Proposed defaults (content-weighted for content-native entities,
balanced for consumption objects where engagement matters more; all overridable):

```yaml
momentum:
  ewma:
    fast_half_life_weeks: 3        # recent level
    slow_half_life_weeks: 12       # baseline level
  heating_up:
    velocity_threshold: 1.5        # τ — velocity ≥ τ ⇒ "rising"
    min_total: 3                   # sample-noise floor
  blend:                           # score = w_content·content + w_engagement·engagement
    default: { content: 0.70, engagement: 0.30 }
    per_kind:
      topic:     { content: 0.85, engagement: 0.15 }
      cluster:   { content: 0.85, engagement: 0.15 }
      storyline: { content: 0.85, engagement: 0.15 }
      person:    { content: 0.80, engagement: 0.20 }
      episode:   { content: 0.50, engagement: 0.50 }
      show:      { content: 0.60, engagement: 0.40 }
      insight:   { content: 0.60, engagement: 0.40 }
  engagement:
    min_events_corpus: 5           # corpus-wide identifiability floor (per-user scope: none)
  horizons:                        # UI toggle presets → (fast_hl, slow_hl) in weeks
    week:    { fast: 1, slow: 4 }
    month:   { fast: 3, slow: 12 }
    quarter: { fast: 8, slow: 26 }
```

Rationale for the defaults: content is dense from day one and engagement is sparse in a
small-audience app, so content leads globally (0.70) and dominates for content-native kinds
(topics/clusters/storylines/people ≈ 0.85); episodes/shows/insights are consumption objects where
"what people are on" matters more, so engagement rises toward parity (episodes 0.50/0.50). These are
starting points to A/B, not commitments — hence config.

## Key Decisions

1. **Dedicated `GET /api/app/trending`, not an extension of the enrichment endpoint** — the momentum
   capability owns derivation + blend + cache; the enrichment endpoint stays a thin reader.
2. **Enrichers produce data; the capability derives + serves** — clean RFC-088 layering; the enricher
   emits `now`-free series, all time-relative math is read-time.
3. **Content ⊕ engagement, both in v1, blended with per-kind weights** — content gives a dense
   day-one signal; engagement adds "what people are on"; the blend degrades gracefully when
   engagement is sparse.
4. **One atom + aggregation for every entity** — per-topic/person weekly counts; clusters, storylines,
   episodes, shows, insights are all aggregations.
5. **EWMA (fast/slow), read-time, anchored to `now`** — no hard window, no re-anchor cron; ~3wk/~12wk
   default half-lives.

## Alternatives Considered

1. **Extend `GET /api/app/corpus/enrichment`** — Pros: fewer routes. Cons: fuses thin envelope reads
   with heavy derivation/blend/cache; two responsibilities in one handler. **Rejected.**
2. **Pre-baked velocity + daily cron** — Pros: minimal code. Cons: up-to-a-day stale; a cron whose
   only job is re-anchoring time; fixtures still need a pinned bake. **Rejected.**
3. **Content-momentum only (defer engagement)** — Pros: no telemetry work. Cons: misses "what people
   are on." **Not chosen** — engagement is in v1, blended, so it can start at low weight.
4. **Store per-entity momentum in the enricher** — Cons: `now`-dependent (non-deterministic to
   commit) and recompute-on-recluster. **Rejected** in favour of stored atoms + read-time derivation.

## Testing Strategy

**Test Coverage:**

- **Unit (momentum)**: `ewma_alpha`/`momentum` — flat → ~1.0; recent spike → rising; decay as the
  reference week advances with no new events; volume vs velocity.
- **Unit (aggregation + blend)**: cluster/storyline/episode/show/insight series = Σ members/contained;
  blend weights combine content + engagement as specified.
- **Integration (endpoint)**: `GET /api/app/trending` per `kind`/`scope`; `APP_TRENDING_NOW` override
  makes `heating_up` deterministic; memoization key.
- **Integration (ranking)**: the discover ranker boosts the same entities the endpoint marks hot under
  a pinned `now` (one source of "hot" across rail + feed).
- **E2E**: with `APP_TRENDING_NOW` pinned, trending surfaces render for topics, storylines, shows,
  episodes, insights.

**Test Data:** `app-validation-corpus/v3` content weekly series + seeded engagement; deterministic.

## Migration Path

1. **Phase 1**: enricher emits per-topic/person **weekly** content series (additive; monthly kept as
   fallback).
2. **Phase 2**: engagement aggregator rolls telemetry (impressions/clicks/playback/favorites) into
   per-entity weekly counts.
3. **Phase 3**: momentum capability + `GET /api/app/trending` (consumer, `scope=corpus|mine`) +
   `GET /api/corpus/trending` (operator); discover ranker switches to `momentum(entity)`; all fall
   back to `velocity_last_over_6mo` if series absent.
4. **Phase 4a**: consumer **player** trending surfaces per entity
   (topics/storylines/shows/episodes/insights) — the richer surface, built first.
5. **Phase 4b**: operator **Dashboard global view** (all kinds, per-kind sparklines).
6. **Phase 5**: deprecate `velocity_last_over_6mo`.

## Resolved Decisions (were open questions)

1. **Blend weights** — global default `0.70/0.30` with per-kind overrides, all in config (§10). ✓
2. **Engagement min-count floor** — configurable, default `min_events_corpus: 5` (§10); no floor for
   `scope=mine`. ✓
3. **Per-user `scope=mine`** — shipped in **v1** (§6). ✓
4. **UI surfaces** — consumer player **everywhere needed** (topic rail + storylines exist;
   shows/episodes/insights added) **and** an operator **Dashboard global view** across every
   supported kind (§7). ✓

5. **Build order** — **consumer player first** (the richer, more complex surface), **then operator
   (admin)**. Operator global view **includes per-kind sparklines** (the weekly `series` per row). ✓
6. **Week numbering** — ISO-8601 weeks end-to-end for `as_of_week`. ✓

## Open Questions

- Consumer surface *ordering* within the player phase (topic rail + storylines exist; which of
  shows/episodes/insights lands next) — decided as we build.

## References

- **Related RFC**: `docs/rfc/RFC-088-enrichment-layer-architecture.md`
- **Related PRD**: `docs/prd/PRD-042-home.md`, `docs/prd/PRD-041-consolidation.md`
- **Source Code**: `src/podcast_scraper/enrichment/enrichers/temporal_velocity.py`,
  `src/podcast_scraper/server/app_discover_view.py`,
  `web/learning-player/src/components/TrendingTopics.vue`
