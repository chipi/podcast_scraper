# Enrichment visual-inspection & test plan (prod-v2, 2026-07-03)

Manual test plan for where RFC-088 enrichment data surfaces across the two UIs,
on the enriched **prod-v2** corpus (209 episodes). Refreshed after the unified
node-view work (Signals tab), the N7 contradiction-text change, and the N8
timeline chart. **Supersedes** the 2026-07-01 prod-pilot version of this file
(that one predates the unified `NodeDetail` node view and had `nli_contradiction`
/ `guest_coappearance` legitimately empty on 6 episodes — both are now populated).

## Setup

- API serving prod-v2: `--output-dir .test_outputs/manual/prod-v2/corpus` on `:8000` (via `make serve`).
- **Viewer** (`:5173`, `web/gi-kg-viewer`): open with the absolute corpus path; sign in as **admin**
  (ada-admin) so the operator Enrichment panel is reachable. To re-establish after a reload: List → Graph → close the Corpus-artifacts modal.
- **Player** (`:5174`, `app/`): sign in as any dev user.

## The enricher roster (master reference)

9 enrichers. Scope/tier read from the manifests; counts are prod-v2 reality on disk today.

| # | Enricher | Scope | Tier | Signal (one line) | prod-v2 count | Viewer surface | Player surface |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `grounding_rate` | corpus | deterministic | Per-person grounded-insight ratio | 125 persons | Signals tab (person) | — |
| 2 | `guest_coappearance` | corpus | deterministic | Person-pairs sharing episodes | 87 pairs | Signals tab (person) | — |
| 3 | `insight_density` | **episode** | deterministic | Insights per early/mid/late third | 209 eps | Episode → Enrichment tab | — (scrubber ticks use `confidence`, not this) |
| 4 | `nli_contradiction` | corpus | **ML** (NliScorer) | Cross-person opposing claims per topic | **660** (v1.1.0, w/ texts) | Signals tab (person) **+** graph EnrichmentEdgesPanel | — |
| 5 | `temporal_velocity` | corpus | deterministic | Topic mentions/mo + EWMA + velocity ratio | 833 topics | Signals tab (topic) | — |
| 6 | `topic_cooccurrence` | **episode** | deterministic | Per-episode topic pairs (`count`=1) | — | **orphaned** (removed from UI by design) | — |
| 7 | `topic_cooccurrence_corpus` | corpus | deterministic | Corpus topic pairs + lift/PMI | 4 428 pairs | Signals tab (topic) | — |
| 8 | `topic_similarity` | corpus | **embedding** | Top-K cosine-similar topics | 833 topics | graph EnrichmentEdgesPanel | — |
| 9 | `topic_theme_clusters` | corpus | deterministic | THEME clusters (co-discussed topics, lift) | 6 clusters | **Details** tab (topic) | ✅ Knowledge panel / entity card |

**Headline:** Viewer surfaces **8 / 9** (all but the intentionally-orphaned episode-scope `topic_cooccurrence`).
Player surfaces **1 / 9** (`topic_theme_clusters`, and only via the `/entities` route, not the enrichment endpoints).

## What changed since the prod-pilot plan (the delta to re-test)

1. **Unified node view.** Person / Topic / Entity / Podcast all render in ONE `NodeDetail` rail with tabs
   **Details · Timeline · Positions · Signals · Neighbourhood** (was separate `TopicEntityView` / `PersonLandingView` rails).
   Enrichment "signals" moved into the **Signals** tab (`node-detail-rail-tab-enrichment` → `NodeEnrichmentSection.vue`).
2. **N7 — contradiction texts.** `nli_contradiction` v1.1.0 now persists `insight_a_text` / `insight_b_text`;
   the Signals Contradictions row shows the **two opposing claims** (attributed, oriented to the focused person),
   not just who/what-topic. prod-v2 envelope regenerated (660 records, all with texts).
3. **N8 — topic Timeline dot chart.** Not an enricher, but new: the topic Timeline tab now has a dot time-series
   above the Episodes/Mentions toggle.
4. **Contradiction topic label** is now title-cased and click-throughs to the topic's Key voices.

---

## Part A — Viewer: inspect each surface

For each: **go here → expect this (prod-v2) → judge these UX questions.**

### A1. Topic node → **Signals** tab (`NodeEnrichmentSection.vue`)
Focus any topic (graph node or search) → open the Signals rail tab.
- **temporal_velocity** — `node-enrichment-velocity`: a velocity badge (e.g. `2.15×`) + 12-mo mention total; emerald >1.5×, rose <0.5×.
  - Judge: is `×` legible without a label? Any "accelerating/declining" word?
- **topic_cooccurrence_corpus** — `node-enrichment-cooccurrence-lift`: clickable related-topic chips sorted by lift/PMI (≥2 eps, lift>1), capped 8.
  - Judge: are chips ordered by strength? Clickable → focus topic? On prod-v2 (4 428 pairs) does the lift ranking surface non-obvious pairs?

### A2. Topic node → **Details** tab (`NodeDetail.vue`)
- **topic_theme_clusters** — `node-detail-theme-cluster`: teal "Theme · {label}" header + member topics with lift-to-cluster badges (distinct from the blue semantic "Cluster" section).
  - Judge: is "Theme" vs "Cluster" distinction clear? 6 clusters on 209 eps — do memberships look right?

### A3. Person node → **Signals** tab (`NodeEnrichmentSection.vue`)
- **grounding_rate** — `node-enrichment-grounding`: `% · grounded/total insights` badge.
  - Judge: is "87% · 13/15" clear? Does 100%-for-many read as suspicious?
- **guest_coappearance** — `node-enrichment-coappearance`: clickable person chips (avatar + name + episode count), top 8 by count.
  - Judge: right people? Chip → focus person?
- **nli_contradiction** — `node-enrichment-contradictions` + **`node-enrichment-contradiction-claims`** (N7): counterpart + topic, then the **two claims** ("This person: …" / "Counterpart: …"), top 8.
  - Judge (**new this session**): are both claims readable and correctly oriented (focused person's claim first)? Is the pair a *genuine* contradiction or NLI noise? (Good tester: `person:jack-clark`, 120 pairs.)

### A4. Episode → **Enrichment** tab (`EpisodeEnrichmentSection.vue`)
Open an episode subject (Library/Digest/Graph) → Enrichment tab.
- **insight_density** — `episode-enrichment-density`: three bars (early / mid / late) of insight counts.
  - Judge: readable? Actionable ("where the substance is")?

### A5. Graph canvas → **EnrichmentEdgesPanel** (`EnrichmentEdgesPanel.vue`, above the canvas)
- **topic_similarity** — `enrichment-edges-similarity`: topic→topic rows with cosine score; focused topic's top-K, else corpus top-N.
  - Judge: **discoverability** — is the panel findable? Do neighbours look right? Is the score threshold sensible?
- **nli_contradiction** — `enrichment-edges-contradictions`: compact `A ⚡ B on topic` rows + score, top 10 by score.
  - Judge: does this graph-wide view complement the person-scoped Signals view, or duplicate it confusingly?

### A6. Sources dialog → **Enrichment** tab (`EnrichmentPanel.vue`) — operator only
- Health / metrics / events + config editor (control/monitoring, no derived data).
  - Judge: all 9 producers healthy after the runs? `records_written` correct? Config editor provider prefill working?

### Cross-surface viewer questions to log
- Are Signals sections labelled/explained or raw numbers?
- Missing-envelope degradation: do surfaces silently hide (best-effort) rather than error?
- Is there any single "this corpus's enrichment at a glance", or is it scattered across Signals / Details / Edges panel?

---

## Part B — Player: the gap map (1 / 9 today)

The player consumes **only `topic_theme_clusters`**, and via the `/episodes/{slug}/entities` route
(`theme_cluster_id` / `_label` / `_size` on Topic → `KnowledgePanel.vue` / `EntityCardBody.vue`), **not** the
enrichment endpoints. Two dedicated player endpoints exist but are **unused by the UI**:
`GET /api/app/episodes/{slug}/enrichment` and `GET /api/app/corpus/enrichment` (`routes/app_enrichment.py`).

> Watch-out: the player scrubber's insight-density ticks come from `Insight.confidence`
> (`app/src/player/insightMarkers.ts`), **not** the `insight_density` enricher — don't mistake that for wiring.

Where each enricher *would* add consumer value (candidates, not built):

| Enricher | Player surface it belongs on | Consumer value |
| --- | --- | --- |
| temporal_velocity | Topic card; Home "trending" | "This topic is heating up" |
| topic_similarity | Topic card → related topics (scored) | "Explore adjacent topics" |
| grounding_rate | Person/speaker card | Speaker credibility signal |
| guest_coappearance | Person card → related people | "Who they appear with" |
| nli_contradiction | Insight / search result badge | "Experts disagree here" (+ now the actual claims, N7) |
| insight_density | Player transcript sidebar | Where the substance is (early/mid/late) |
| topic_cooccurrence_corpus | Topic card | "Often discussed alongside" |

Part B is a product decision: which of these to wire into the player first (and whether the two unused
`/api/app/*/enrichment` endpoints get adopted or retired).

---

## prod-v2 expectations & the co-occurrence A-vs-B check

- All 9 producers ran; nothing is legitimately empty now (unlike prod-pilot). `nli_contradiction` = 660 with texts,
  `guest_coappearance` = 87 — both were empty on the 6-episode pilot.
- **Co-occurrence A vs B** (`topic_cooccurrence_corpus` v1.1.0 exposes both): the Signals chips rank by **lift/PMI**
  (distinctive pairs), not raw count. On 209 episodes (4 428 pairs) the lift ranking should visibly diverge from a
  raw-count ranking — this is the "does B earn its keep at scale" check the pilot couldn't run.

## Orphan / decision (still open)
- **Episode-scope `topic_cooccurrence`** is orphaned: the corpus aggregator recomputes from the KGs directly and the
  per-episode UI chips were removed by design ("trivial on one episode"). Decide: leave (harmless), disable, or rewire.

## Deliverables from this pass
1. Per-viewer-surface UX notes (clarity, discoverability, empty/degraded states).
2. Player gap confirmed + prioritized shortlist of which enrichers to surface there first.
3. Any real bugs (vs. expected-empty) → repro + fold into the current issue/branch.
