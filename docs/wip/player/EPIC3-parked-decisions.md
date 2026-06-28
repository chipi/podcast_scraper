# Epic 3 — parked items (need an operator decision)

> **DECIDED (2026-06-25, operator):** **3.2** → new dedicated consumer endpoint (a). **3.3** →
> **also a new dedicated, specialized endpoint** (operator overrode the "reuse search" lean — effort
> is not the problem; KG-grounded episodes, not search relevance), tap opens the card. **3.4** →
> exact/near-exact match, consumer first. **3.5** → first-Home interests card, top-12 clusters,
> flag-gated ranking, recency default. All dedicated `/api/app/*` endpoints reuse the existing PURE
> builders (`cil_queries.person_profile`/`topic_timeline`, `relational_queries.*`,
> `consumer_topic_cluster_map`) — no proxying of operator routes.


Autonomous session of 2026-06-25 completed the **low-assumption** slices — Retro (#1100), **3.1**
cluster API + cluster-first panel (#1092), **3.6** char-level quote highlight (#1099). The
remaining four each hinge on a design decision I deliberately did **not** guess (per "no crazy
assumptions — park if questions"). Each is unblocked the moment its question is answered.

## 3.2 — Person profile card (#1095)

Tap a person → a profile card. **Decision: data source.**

- **(a)** New lean consumer endpoint `GET /api/app/persons/{id}` projecting KG + relational data to
  a small card shape (keeps the consumer self-contained, no operator coupling).
- **(b)** Thin proxy of the viewer's existing relational API (RFC-094 `positions_of` / `who_said` /
  `cross_show_synthesis`) — less new code, but couples the consumer to operator surfaces.
- **(c)** Reuse #1089's `useRelationalCache` / `subject` store directly in the consumer app.

**Also:** which fields on the card — role + "appears in" episodes + related people/topics, or also a
bio/summary? My lean recommendation: **(a)** + role/episodes/related (no bio).

## 3.3 — Topic card (#1096)

Tap a topic → a card (cluster theme + sibling topics + episodes about it). **Two decisions:**

1. **Corpus-wide siblings + "episodes about this topic" source.** Today the API only exposes
   topic→cluster (per episode). Options: (a) new `GET /api/app/topics/{id}` (cluster members +
   episodes, scanning KG/index), or (b) reuse `GET /api/app/search?q={label}` (already grouped by
   episode) for the episodes part + expose cluster members for siblings.
2. **Chip-tap behavior.** Opening a card on chip tap **changes the Epic-2 behavior** (chip → corpus
   search). Card-with-a-search-action-inside, or keep search as the primary and add the card via a
   secondary affordance?

My lean recommendation: **(b)** + tapping a topic opens the card (search action lives inside it).

## 3.4 — Entities as search results (#1097)

Surface a person/topic **entity card above passage results** when a query matches an entity.
**Decision: match threshold** — exact/near-exact entity-name match only (safe, few false positives)
vs fuzzy (more cards, more noise)? And **which surface first** — consumer, viewer, or both together?
My lean recommendation: **exact/near-exact**, consumer first (reuse 3.2/3.3 cards).

## 3.5 — Personalized discovery (#1098)

Interests at sign-up (pick topic **clusters**) → digest × cluster-affinity ranking of Home. **Three
decisions:**

1. **Onboarding placement** — a step in the sign-up flow, or a dismissible "set your interests" card
   on first Home visit?
2. **Cluster picker** — present top-N clusters by `member_count` (N = ?); free-text search too?
3. **Ranking formula** — `score = w1·digest_significance + w2·interest_cluster_affinity`; the weights
   + whether to gate behind a feature flag until tuned.

This is the largest slice (onboarding UX + per-user profile field + ranking) and most benefits from
your input before building.

---

**Status:** branch `feat/knowledge-clusters` off the merged main; commits for retro/3.1/3.6 are
ready (NOT pushed — another PR is landing, awaiting your go). `make ci-fast` validated locally.
