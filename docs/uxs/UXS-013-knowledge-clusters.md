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
- **Affordance:** tapping a chip opens its **entity card** (3.2/3.3 — shipped; the Epic-2
  chip→search default now lives as an explicit action inside the card). The dominant ring is a
  *visual* cue, not a new control. Collapsed at 6 chips; **+N …** expands.
- **Degrade:** no `topic_clusters.json` → no rings, no theme line, flat order (today's behaviour).

**Tokens:** topic `--lp-topic`, person `--lp-person`, chip bg `--lp-overlay`, ring `--lp-topic`.
**A11y:** the ring is supplementary to label text + colour (not the only signal); chips keep their
`aria-label` ("Open {term}"); contrast per UXS-011.

## Entity cards (3.2 person · 3.3 topic — shipped)

One `EntityCard` overlay serves both (sheet on mobile, centred panel on desktop):

- **Person card:** a "Person" kicker + name, an "In {n} episodes" list (artwork + title), related
  people/topics chips, and a "Search the library for {name}" action. No avatar/role/bio — the
  consumer scope is lean. Data: KG co-occurrence via `GET /api/app/persons/{id}`.
- **Topic card:** a "Topic" kicker + label, the cluster **"Theme · {cluster}"** line, sibling-theme
  chips ("More in this theme"), a "Discussed in {n} episodes" list, and related people. Data:
  `GET /api/app/topics/{id}`.
- **Re-entrant:** tapping a related person/topic chip walks to that entity in place, with a back
  (‹) control; the search action lives inside the card, not on chip-tap.
- **Open/close:** tap to open; mobile = bottom sheet with backdrop; desktop = centred panel. Modal
  a11y: `role="dialog"`/`aria-modal`, focus trap, initial focus + restore-on-close; dismiss via
  ESC, backdrop, or the ✕ control.

## Entities in search (3.4 — shipped)

When a query exact/near-exact-matches a person/topic, an **entity card** sits **above** the grouped
passage results (UXS-012 search): a kicker ("Person"/"Topic") + name + a "View ›" affordance,
tapping which opens the §Entity-cards overlay. Distinct from passage cards, never blocking the
passages below; the "no grounded passages" line is suppressed when an entity matched (we *did* find
something). Resolved in parallel with the passage search, so a miss never delays results.

## Personalized discovery (3.5 — shipped)

- **First-Home card:** a dismissible "Personalize your Home" card (signed-in only; remembered via
  `localStorage`) opens the **interests picker**.
- **Picker:** a modal over the corpus's **top-12 clusters** (toggle chips; pre-selected from saved
  interests; Save / Not now). Modal a11y matches the entity card (focus trap, ESC/backdrop).
- **Effect:** the Home "What's new" feed re-ranks by interest affinity **only** when the deployment
  flag is on; by default (and signed-out) it is recency — visually identical to today.
- **Tokens:** selected chip = `--lp-accent`; unselected = `--lp-overlay` / `--lp-topic`.

## Decisions (operator, 2026-06-25)

- Person **and** topic cards use **dedicated** `/api/app/persons|topics/{id}` endpoints (KG
  co-occurrence), not a proxy of the viewer relational API — effort over coupling.
- Entities-in-search (3.4) surfaces a card only on **exact/near-exact** entity-name match
  (consumer first), reusing the 3.2/3.3 cards.
