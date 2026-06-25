# UXS-013: Knowledge clusters & entity cards (consumer)

- **Status**: Draft (cluster-first panel — Implemented, Epic 3.1)
- **PRD**: `docs/prd/PRD-043-knowledge-layer.md`
- **RFC**: `docs/rfc/RFC-102-knowledge-clusters-entity-cards.md`
- **Inherits**: UXS-011 (Editorial Bold tokens, `--lp-*`) and UXS-012 (Home).

---

## Scope

The knowledge-navigation UX of Epic 3: cluster-first topics in the Insights panel (3.1, shipped),
and the person/topic **entity cards** + entity search results (3.2–3.4, design). Mobile-first;
WCAG 2.1 AA; i18n (no hard-coded strings).

## Cluster-first "Topics & People" (3.1 — shipped)

Within the Insights panel's compact, expandable **Topics & People** row:

- **Order:** the **dominant cluster** (most of this episode's topics, ≥2) leads; its chips get a
  1px **`ring-topic`** outline to stand out. Other clustered topics follow (larger intra-episode
  groups first); singleton topics trail; **people** chips (`text-person`) come after topics.
- **Theme lead-in:** a small `text-topic` line beside the section header — **"Theme · {cluster}"**
  — names the dominant cluster (hidden when there is none).
- **Affordance:** tapping any chip still navigates to corpus search for that term (Epic 2 behaviour);
  the dominant ring is a *visual* cue, not a new control. Collapsed at 6 chips; **+N …** expands.
- **Degrade:** no `topic_clusters.json` → no rings, no theme line, flat order (today's behaviour).

**Tokens:** topic `--lp-topic`, person `--lp-person`, chip bg `--lp-overlay`, ring `--lp-topic`.
**A11y:** the ring is supplementary to label text + colour (not the only signal); chips keep their
`aria-label` ("Search your library for {term}"); contrast per UXS-011.

## Entity cards (3.2 / 3.3 — design)

- **Person card** (sheet on mobile, side panel/popover on desktop): avatar/initial, name, role,
  "appears in" episode list (artwork + title), related people/topics chips, and a "Search the
  library for {name}" action. Mirrors the viewer's `PersonLandingView` content, re-skinned to
  `--lp-*`.
- **Topic card:** the cluster (`cluster_label`) as a header, sibling-topic chips, and "episodes
  about this" list. Opening either replaces the Epic-2 chip→search default; the search action stays
  available inside the card.
- **Open/close:** tap to open; mobile = bottom sheet with backdrop (same pattern as the Knowledge
  sheet); desktop = inline panel. Focus trap + ESC close + restore focus (a11y).

## Entities in search (3.4 — design)

When a query matches a person/topic, an **entity card** sits **above** the grouped passage results
(UXS-012 search), linking into the person/topic card. Clearly distinct from passage cards (kicker
"Person"/"Topic", entity styling), never blocking the passages below.

## Open questions (parked for operator)

- Person-card data source: dedicated consumer endpoint vs thin proxy of the viewer relational API.
- Entity-match threshold for surfacing a card in search (exact vs fuzzy).
