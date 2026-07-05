# Handover — Theme clusters (co-occurrence) feature

**Date:** 2026-07-01
**Branch:** `feat/consumer-remember` (NOT pushed)
**Decision (locked with operator):** two coexisting cluster types —
**Theme** (co-occurrence, "discussed together") + **Similar** (semantic
embedding, "meaning alike"). Rename the shipped "Theme ·" lead-in (which is
actually the *semantic* cluster) to "Similar ·". Theming must be consistent
across player pills, wide rows, AND graph cluster nodes — one token, not
panel-local styling.

---

## DONE + committed (foundation, all tested)

| Commit | What |
|---|---|
| `6efaddaf` | viewer: Enrichment subject-card tab + lift-ranked co-occurrence (topic card now shows co-occurrence "above chance" only; A/frequency row dropped) |
| `f94113a7` | **enricher**: `topic_cooccurrence_corpus` v1.1.0 emits `lift`/`pmi`/per-topic counts; **new `topic_theme_clusters` enricher** (co-occurrence lift → average-linkage → clusters, `thc:` graph prefix, `cluster_type="theme"`); registered + wired into all 12 deterministic profiles |
| `ce9ed333` | **server**: `GET /api/corpus/theme-clusters` serving `enrichments/topic_theme_clusters.json` (+ 3 integration tests) |
| `3b0c3ede` | drift: stale "six enrichers" → "seven" |

Tests green: enrichment unit **448 passed**; cluster routes **7 passed**;
viewer typecheck 0 + enrichment vitests **90 passed**.

Producer verified end-to-end on prod-pilot (3 eps → 0 clusters, correct: all
pairs co-occur once, below `min_pair=2`).

---

## VALUE TEST — PASSED (2026-07-01, my-manual-run-10, 84 eps)

Ran `topic_theme_clusters` on `.test_outputs/manual/my-manual-run-10` (84 eps,
689 topics) → **11 theme clusters, genuinely meaningful:**
- **[10] energy storyline**: crude oil, LNG, nuclear, renewables, Strait of
  Hormuz, geopolitics, decarbonization, supply chain.
- **[6] AI-and-jobs**: ai development, future of work, job displacement, labor
  market, market disruption.
- **[6] energy security**: china economy, middle east, oil prices, strategic
  reserves, geopolitical risk.
- 2-pairs mostly sensible: fed+monetary policy, interest rates+market volatility,
  private markets+VC, commodity markets+global trade.

**Verdict: co-occurrence clustering works — real storylines, not noise.** Building
the graph/timeline surfaces is justified. Use `my-manual-run-10` as the build +
eyeball corpus (prod-pilot too small; prod-v2 discovers 0 bundles — layout issue).
`make enrich CORPUS=.test_outputs/manual/my-manual-run-10 ONLY=topic_theme_clusters CORPUS_ONLY=1`.

## GRAPH CONSTRAINT — one parent per node (design fork, operator call)

Cytoscape compound nodes are a **tree**: a Topic node can have exactly ONE
`parent`. `applyTopicClustersOverlay` sets `node.parent = tc:id`. So **theme
(`thc:`) and semantic (`tc:`) compound nodes cannot both wrap the same topic at
once.** "Two coexisting cluster types" works for *pills* (a chip carries two
rings) but NOT for graph compound nesting. Options:
1. **Toggle** — graph shows semantic OR theme compound overlay (recommended;
   default semantic = current behaviour; add a "cluster by: Similar | Theme"
   control). Reversible, low-risk.
2. Theme clusters as **hulls/regions** (not compound parents) — coexist with
   semantic compounds, but needs a hull renderer (cytoscape plugin) — bigger.
3. Theme as edge-colour halos only — weakest.
Recommend (1). The timeline (`TopicTimelineDialog` cluster mode) rides on this:
once `thc:` nodes exist, open the dialog with the theme members' topic ids.

## BLOCKED on validation data — read before building the frontend

Theme clusters are **empty on every usable local corpus**:
- **prod-pilot** = 3 episodes → 0 clusters (too small; correct behaviour).
- **prod-v2** (209 KGs) → enrichment CLI **discovers 0 bundles** (its on-disk
  layout doesn't match `discover_episode_bundles`; needs debugging or a
  manifest/run-dir fix — separate task).

**Before building the consumer UI, get non-empty theme clusters to build AND
eyeball against** (this is also the deferred *value test* — are co-occurrence
clusters actually meaningful themes, or junk?):
1. Find/enrich a corpus that BOTH discovers bundles AND has ~30+ episodes with
   recurring topics (try `prod-10`, `my-manual-run-10`, or fix prod-v2
   discovery). `make enrich CORPUS=<root> ONLY=topic_theme_clusters CORPUS_ONLY=1`.
2. Inspect `enrichments/topic_theme_clusters.json` top clusters. **If the
   clusters read as real storylines → build the frontend. If junk → stop;
   co-occurrence clustering isn't worth the UI.**

The frontend was deliberately NOT built blind: it touches shipped UXS-013
consumer UX and can't be visually validated without this data.

---

## REMAINING — consumer wiring (execute-ready)

### 1. Server attach (theme cluster → episode topics)
The player's topic pills carry `cluster_id/cluster_label/cluster_size`
**server-attached** via `consumer_topic_cluster_map(root)` (semantic). Add a
sibling for themes:
- New `consumer_theme_cluster_map(corpus_root)` → `topic_id → {theme_cluster_id,
  theme_cluster_label, theme_cluster_size}` reading
  `enrichments/topic_theme_clusters.json`. Mirror `consumer_topic_cluster_map`
  in `src/podcast_scraper/search/topic_clusters.py` (or a new
  `search/theme_clusters.py`); the `_load` path is `enrichments/…`, not `search/…`.
- Attach `theme_cluster_*` onto the topic payload at the SAME call sites as the
  semantic map:
  - `server/app_relational_view.py:120, :171`
  - `server/app_discover_view.py:92`
  - `server/routes/app_episodes.py:286`

### 2. Player types + render
- `app/src/services/types.ts` (topic type ~L162-164 and ~L328-330): add
  `theme_cluster_id/label/size`.
- `app/src/components/KnowledgePanel.vue`: it computes `dominantClusterId` from
  `cluster_id` and renders the "Theme ·" lead-in (`t('kp.theme', …)`, L299-300)
  + dominant `ring-topic` (L311). Add a parallel `themeDominantCluster` from
  `theme_cluster_id`; render the **Theme** lead-in from it, and **rename the
  existing semantic lead-in to "Similar ·"**. Theme chips get the theme token.
- `app/src/components/EntityCardBody.vue`: same rename + theme section.

### 3. i18n (`app/src/i18n/locales/en.json`)
- `kp.theme` currently means the *semantic* cluster → reassign: `kp.theme` = the
  co-occurrence THEME; add `kp.similar` for the semantic cluster. Update all
  locale files + the KnowledgePanel/EntityCard tests that assert these strings.

### 4. Design token — player DONE; viewer graph is a big subsystem
- Player token `--lp-theme` + tailwind `theme` colour: DONE (316af454).
- **Viewer graph `thc:` compound nodes — LARGE, deferred.** The semantic `tc:`
  clusters are a deep Cytoscape subsystem, not just a style rule. To mirror for
  `thc:` you must touch:
  - `web/gi-kg-viewer/src/stores/artifacts.ts` — memoized fetch of
    `/api/corpus/topic-clusters` (~L101-141), building `tc:` compound nodes +
    `ensureTopicClusterCompoundVisible` / catalog member-episode loading
    (~L525-730). Need a parallel `theme-clusters` fetch + `thc:` compound build.
  - `web/gi-kg-viewer/src/stores/graphNavigation.ts` — `tc:` collapse
    state (`toggleTopicClusterCanvasCollapsed`, ~L35-51) → `thc:` analog.
  - `App.vue` (~L155) — cy-id routing opens NodeDetail for `tc:`/compound ids.
  - Cytoscape stylesheet — add a `thc:`/`cluster_type=theme` compound style
    using the theme colour (find the `tc:` compound style rule).
  - This is ~a feature on its own; do it with real data + operator, after the
    value test. The `--lp-theme` token / a viewer CSS var is the only cheap part.
- Operator's rule [[feedback_consumer_ux_consistency]]: define the theme
  style once (shared token), not per-element.

### 5. Docs
- UXS-013 + RFC-102 update: document the two cluster types + the "Theme/Similar"
  naming (and that "Theme ·" was reassigned). `make docs` (strict) before push.

---

## Cluster-member timeline for theme clusters (operator Q, 2026-07-01)
Semantic clusters have a member timeline ("when/how a topic landed in the
cluster"). **Do the same for theme clusters — and it's a better fit:**
- Semantic membership is ~timeless (topics that *mean* alike don't change).
- A **theme is a storyline** — it emerges, peaks, fades. So "when was this theme
  live" is genuinely informative and maps onto `temporal_velocity`.
- **Data already exists:** each theme-cluster member carries `episode_ids`
  (emitted by `topic_theme_clusters`). Map those → episode `publish_date` → a
  per-member / per-theme timeline. No new producer work; just the consumer view.
- Build it alongside the semantic timeline (reuse that component), keyed on
  `theme_cluster_id`. Deferred with the rest of the consumer surfaces.

## Player theme rendering — DONE (2026-07-01, commits 316af454, 52b44d50)
- `--lp-theme` token + tailwind `theme` colour; "Theme→Similar" rename.
- Server: `AppTopic.theme_cluster_*` + `consumer_theme_cluster_map` +
  `episode_entities` attaches both cluster identities.
- Player `KnowledgePanel`: dominant-theme "Theme ·" lead-in + **theme-member
  pills ringed `ring-theme`**. Full-stack tested with seeded theme data.
- STILL TODO: EntityCardBody theme section; entity/relational-view attach
  (`app_relational_view.py:120/171`); **graph `thc:` node theming (viewer)**;
  member timeline (above); the value-test on real data.

## Also parked (unchanged)
- Orphaned episode-scope `topic_cooccurrence` enricher (no consumer; aggregator
  recomputes from KGs). Leave / disable — operator's call.
- **#1140** insight_density → player skip-line/guide (weight viz). NOT started —
  same "large player-UX, validate-with-operator" reasoning; `insight_density`
  data does exist (prod-pilot), so it's more buildable than theme rendering.
  **Operator brainstorm (2026-07-01, refine when we get there):**
  - **Insight markers on the transcript/player timeline** — dots/ticks at the
    timestamps where insights occur.
  - **Colour-coded** — different colours per insight (by type? grounded-vs-not?
    weight?). Decide the encoding when we design it.
  - **Reserve a colour for the CURRENT USER's saved insights** (ties into the
    consumer-remember/save feature — a user's own saved moments get a distinct
    marker on the timeline).
  - These markers **fold into a graphical read of insight *density* over the
    timeline** — the skip-guide "where the substance is" (early/mid/late from
    `insight_density`, but continuous along the scrubber, not just 3 buckets).
  - Visual is open — brainstorm the exact look together before building.

## Pre-push checklist (when operator returns)
`git fetch origin main && git rebase origin/main`; `make docs` (md changed);
viewer `npm run build` + `make ci-ui-full` (testid/chip surfaces);
`make ci-fast` last; PR body `Closes #…` / `Part of #…`; ready-not-draft.
