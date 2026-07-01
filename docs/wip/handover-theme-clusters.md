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

### 4. Design token (define ONCE, apply everywhere)
- Player: add `--lp-theme` alongside `--lp-topic`/`--lp-person` (UXS-011 tokens,
  `app/src/theme/theme.ts` + CSS). Apply to theme chips + wide-row lead-in.
- Viewer: matching token; apply to graph **`thc:` compound nodes** (mirror the
  `tc:` compound-node styling — find it in the viewer graph/cytoscape layer)
  and consume `GET /api/corpus/theme-clusters`.
- Operator's rule [[feedback_consumer_ux_consistency]]: define the theme
  style once (a shared class/token like `.lp-*`), not per-element.

### 5. Docs
- UXS-013 + RFC-102 update: document the two cluster types + the "Theme/Similar"
  naming (and that "Theme ·" was reassigned). `make docs` (strict) before push.

---

## Also parked (unchanged)
- Orphaned episode-scope `topic_cooccurrence` enricher (no consumer; aggregator
  recomputes from KGs). Leave / disable — operator's call.
- **#1140** insight_density → player skip-line/guide (weight viz). NOT started —
  same "large player-UX, validate-with-operator" reasoning; `insight_density`
  data does exist (prod-pilot), so it's more buildable than theme rendering.

## Pre-push checklist (when operator returns)
`git fetch origin main && git rebase origin/main`; `make docs` (md changed);
viewer `npm run build` + `make ci-ui-full` (testid/chip surfaces);
`make ci-fast` last; PR body `Closes #…` / `Part of #…`; ready-not-draft.
