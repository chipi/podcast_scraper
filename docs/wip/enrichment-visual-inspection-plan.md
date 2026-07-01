# Enrichment visual-inspection plan (prod-pilot, 2026-07-01)

Manual visual inspection of where RFC-088 enrichment data surfaces in the two UIs, on the
enriched **prod-pilot** corpus (6 episodes). Goal: confirm what renders, judge the UX, and log gaps.

## Setup

- API serving prod-pilot: `--output-dir .test_outputs/manual/prod-pilot` on `:8000`.
- **Viewer** (`:5173`): open with the corpus path (absolute — the relative form doesn't resolve):
  `http://localhost:5173/?path=/Users/markodragoljevic/Projects/podcast_scraper-ai-ml-improvements/.test_outputs/manual/prod-pilot`
  Sign in as **ada-admin** (admin) so the operator surfaces (Enrichment panel) are reachable.
- **Player** (`:5174`): sign in as any dev user.
- **Prod-pilot enrichment reality** (what to expect on disk / from the API):
  - Non-empty: `temporal_velocity` (30), `topic_cooccurrence(_corpus)` (135), `insight_density` (27),
    `grounding_rate` (3), `topic_similarity` (30 topics w/ neighbours).
  - **Empty**: `nli_contradiction` (0 — no cross-person shared-topic insights in 6 eps) and
    `guest_coappearance` (0). Their surfaces will be legitimately empty — that's expected, and a good
    empty-state test, NOT a bug.

## Headline finding (already established)

| | Viewer (`gi-kg-viewer`) | Player (consumer app) |
|---|---|---|
| Enrichment **data** surfaced | **8 / 8** enrichers | **0 / 8** — API exists, UI never calls it |
| Read path | `/api/corpus/enrichments/*` | `/api/app/*/enrichment` (served, unused) |

So Part A is "inspect + judge the viewer surfaces"; Part B is "confirm the player gap + decide where each enricher should land."

---

## Part A — Viewer: inspect each surface

For each: **go here → expect this → judge these UX questions.**

### A1. Topic rail — click a topic (graph node or search result) → right SubjectRail (`TopicEntityView.vue`)
- **temporal_velocity**: an "Enrichment signals" section — velocity ratio (last month / 6-mo avg) + 12-mo total.
  - Judge: is the ratio legible without explanation? Is "1.3×" meaningful to a user? Any label (accelerating/declining)?
- **topic_cooccurrence_corpus**: co-occurring topic chips.
  - Judge: are chips clickable → focus that topic? Too many/too few? Ordered by strength?

### A2. Person rail — click a person → right SubjectRail (`PersonLandingView.vue`)
- **grounding_rate**: a grounding % badge + insight counts (e.g. Paul Tudor Jones 23/23 = 100%).
  - Judge: is "100% grounded" clear? Does 100%-for-everyone (small corpus) look suspicious/unhelpful?
- **guest_coappearance**: co-guest chips — **expected empty on prod-pilot**. Judge the empty state (hidden vs "none").
- **nli_contradiction**: contradiction rows — **expected empty**. Judge empty state.

### A3. Episode detail — click an episode → right rail (`EpisodeDetailPanel` → `EpisodeEnrichmentSection.vue`)
- **insight_density**: early/mid/late bar chart.
  - Judge: is the 3-bucket chart readable? Does it tell you anything actionable?
- **topic_cooccurrence** (episode-scope): topic-pair chips.
  - Judge: distinct from the corpus-scope co-occurrence? Confusing to have both?

### A4. Graph → EnrichmentEdgesPanel (right of the canvas)
- **topic_similarity**: similar-topic edges (topic-focused, or corpus top-N) with scores.
  - Judge: **discoverability** — is this panel findable, or hidden? Do the neighbours look right
    (spot-check: "act of kindness" → "philanthropic efforts" 0.49)? Is 0.49 "similar enough" to show?
- **nli_contradiction**: cross-person contradictions — **expected empty**. Judge empty state.

### A5. Sources dialog → Enrichment tab (`EnrichmentPanel.vue`)
- Run **health / metrics / events** + the **config editor** (control/monitoring only — no derived data).
  - Judge: does health show all 8 enrichers as healthy after our runs? Does `records_written` now read
    correctly for `topic_similarity` (was the bug we fixed)? Is the config editor's provider prefill working?

### Viewer UX questions to log across all surfaces
- Are "Enrichment signals" sections labelled/explained, or raw numbers?
- Missing-envelope degradation: do surfaces silently hide when an enricher didn't run?
- Is there any single place to see "this corpus's enrichment at a glance", or is it scattered across rails?

---

## Part B — Player: the gap (what SHOULD surface, and where)

The player renders none of it. This is the value-gap map — where each enricher would add consumer value
(candidates, not built):

| Enricher | Player surface it belongs on | Consumer value |
|---|---|---|
| temporal_velocity | Topic entity card; Home "trending" | "This topic is heating up" |
| topic_similarity | Topic card → related topics (scored) | "Explore adjacent topics" |
| grounding_rate | Person entity card | Speaker credibility signal |
| guest_coappearance | Person card → related people | "Who they appear with" |
| nli_contradiction | Insights / search results | "Experts disagree here" badge |
| insight_density | Player transcript sidebar | Where the substance is (early/mid/late) |
| topic_cooccurrence(_corpus) | Topic card; Player topics | "Often discussed alongside" |

Inspection for Part B is confirmation + product decision: which of these are worth wiring into the
player (ties back to the "player value gap" question, and to whether P3's consumer enrichment surface
should be completed in the UI or was intentionally API-only for now).

## Deliverables from the inspection

1. Per-viewer-surface UX notes (clarity, discoverability, empty states).
2. Confirm/deny the player gap (should be confirmed) + a prioritized shortlist of which enrichers to
   surface in the player first.
3. Any real bugs (vs. expected-empty) → file with a repro.

## Follow-up: co-occurrence A vs B — CHECK WHEN WE SCALE UP CORPUS SIZE

We built two rankings into `topic_cooccurrence_corpus` (v1.1.0, +`lift`/`pmi`/per-topic
episode counts) and both show on the Topic node's Enrichment tab:
- **A** — "Co-occurs with · most frequent" (raw `episode_count`).
- **B** — "Distinctive pairs · above chance (PMI/lift)" — gated to pairs with ≥2 episodes
  and `lift > 1`.

**What to check as the corpus grows (3 → 10 → 30 → 100 → 209 eps):** at what N does the
**B row start to populate and visibly diverge from A**? B needs volume; on small corpora it
is correctly empty.

- **prod-pilot (3 eps): B is EMPTY by design** — every pair is `count=1`, `lift=3.0` flat.
  Not a bug. Do NOT judge B here.
- **Next real test: prod-v2 (~209 eps)** — expect B to fill in and separate from A. Command:
  `make enrich CORPUS=.test_outputs/manual/prod-v2 ONLY=topic_cooccurrence_corpus CORPUS_ONLY=1`
  then load prod-v2 → topic node → Enrichment tab → compare the A and B rows.

**Gotcha to remember:** default `make enrich` (no `--only`) did **not** run
`topic_cooccurrence_corpus` on prod-pilot — it's not in that corpus's stored active-set, so
we had to force it with `--only`. Confirm the active-set/config for each corpus we test so it
runs by default (else the envelope silently stays stale and B never appears).

**Open decision (parked):** the episode-scope `topic_cooccurrence` enricher is now
**orphaned** — the corpus aggregator recomputes pairs from the KGs directly (`load_kg`), it
does NOT read the per-episode envelope, and the per-episode UI chips were removed. Decide:
leave it (harmless dead output), disable it, or rewire the aggregator to consume it
(true bottom-up).
