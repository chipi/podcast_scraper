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
- **Picker:** a modal over the corpus's **top-12 clusters** (toggle chips; preselected from saved
  interests; Save / Not now). Modal a11y matches the entity card (focus trap, ESC/backdrop).
- **Effect:** the Home "What's new" feed re-ranks by interest affinity **only** when the deployment
  flag is on; by default (and signed-out) it is recency — visually identical to today.
- **Tokens:** selected chip = `--lp-accent`; unselected = `--lp-overlay` / `--lp-topic`.

## Knowledge bands on the episode, show and topic pages (documented 2026-09-03)

Three bands that render the knowledge layer where the listener already is, rather than sending them
to a separate "graph" surface. All three shipped with automation and no spec; all three follow the
same rule as the rails: **absent intelligence omits cleanly** — a band with no signal is not
rendered, never rendered empty.

### `EntitySignals` — why this entity matters here

The signal strip on an entity card: how much of the corpus this person/topic touches, and where.
Counts are stated, never implied by bar length alone, because a bar with no axis reads as a
precision the underlying data does not have.

### `PodcastSignalsBand` — what sets THIS show apart

On a show page. Topics split deliberately into two groups: **distinctive** topics carry `lift`
above the corpus base rate — the ones that make this show itself — and the remainder are listed
plainly. The split exists because an alphabetical tiebreak once let a show's signature topic lose
to wallpaper every show covers.

> The momentum bubble cloud was **removed** from this band: its `velocity` is corpus-wide, not
> show-scoped, so it sized topics by a number that did not answer the band's own question.

### `TopicPerspectives` — who disagrees, and where

On a topic page: the positions different people take on one topic, grounded in episodes. This is
the surface that distinguishes a knowledge layer from a tag cloud, so it is held to the grounding
rule strictly — every position carries its source, and an ungrounded claim is not shown at all.

### `TopicConversationArc` — how a topic moved over time

A bar series of a topic's presence across the corpus timeline. Same restraint as the trend sparks:
shape over precision.

### `ShowActivityChart` — a show's publishing rhythm

Episodes over time on a show page. It answers "is this alive?" — a question a listener asks before
following, and one an episode list buried in dates answers badly.

### `EpisodeDensity` / player insight band

Where the insights sit inside an episode, as `early` / `mid` / `late` segments with ticks. It gives
a listener a reason to scrub somewhere specific rather than sampling blindly. `density-peak` is a
caption element, **not** a fourth segment — a distinction worth stating because it has been read as
one.

## Decisions (operator, 2026-06-25)

- Person **and** topic cards use **dedicated** `/api/app/persons|topics/{id}` endpoints (KG
  co-occurrence), not a proxy of the viewer relational API — effort over coupling.
- Entities-in-search (3.4) surfaces a card only on **exact/near-exact** entity-name match
  (consumer first), reusing the 3.2/3.3 cards.
