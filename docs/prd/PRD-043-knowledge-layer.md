# PRD-043: Knowledge Layer — Topic Clusters, Entity Cards & Personalized Discovery

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.8 (Epic 3)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Depends on**: PRD-039 (Player / Knowledge panel), PRD-042 (Home), RFC-090 (hybrid search),
  the corpus **topic-clustering** subsystem (`search/topic_clusters.py` → `search/topic_clusters.json`),
  and the viewer's relational/subject work shipped in #1089 (`PersonLandingView`, `useRelationalCache`).
- **Related UX spec**: `docs/uxs/UXS-013-knowledge-clusters.md` (3.1 first; entity cards follow)
- **Related RFC**: `docs/rfc/RFC-102-knowledge-clusters-entity-cards.md`
- **Related issue**: Epic 3.1 — topic-cluster understanding (GH issue TBD)

---

## Summary

Epic 2 made the consumer app a player that *overlays* grounded intelligence. The **Knowledge
Layer** makes that intelligence **first-class and navigable**: entities (topics, people) become
things you can *open*, that are *clustered* into themes, that show up *in search*, and that
*personalize* what the app surfaces. The transcript↔insight bridge (Epic 2) was step one — a
quote you can tap. This epic turns the whole knowledge graph into a browsable, personal surface.

It deliberately **merges what were proposed as Epic 3 (knowledge) and Epic 4 (personalization)**:
personalization is not a separate feature — it is *powered by the same cluster layer*. Picking
"interests" = picking topic **clusters**; ranking "What's new for you" = topic-cluster affinity.
The cluster API is the shared spine.

## Background & Context

- The corpus already computes **topic clusters** across episodes (`topic_clusters.json`, 86
  clusters in prod-v2 at threshold 0.75) and the operator viewer exposes `/api/corpus/topic-clusters`.
  The **consumer** app does not surface any of it: the Knowledge panel shows a flat, unordered
  Topics list and a People list whose only action (post-Epic-2) is a library search.
- #1089 shipped the viewer's **person profile** surface (`PersonLandingView` + relational cache +
  subject store) — a rich "who is this, where do they appear, who are they connected to" card.
  The consumer app has no equivalent; tapping a person just searches.
- Home's "What's new" / "Recommended" are recency-only; the **corpus digest** (significance +
  topic bands) and per-user signal are unused for ranking.
- Operator play-test (2026-06-24/25) repeatedly asked for: cluster-first topic ordering with the
  dominant cluster standing out; a person profile card on click; a topic card focused on related
  topics; entities as search-result items; digest-driven discovery; and interests at registration.

## Goals

- **G1** — Surface the corpus topic-cluster structure in the consumer app so topics read as
  *themes*, not a flat bag; the episode's dominant cluster leads and stands out.
- **G2** — Make people and topics **openable** as profile/topic **cards** (bio/role, episodes,
  related entities) — reusing the viewer's relational work, adapted to the consumer surface.
- **G3** — Surface high-confidence entity matches as **knowledge-panel-style results** in search.
- **G4** — **Personalize discovery**: let the user pick interest *clusters* at sign-up and rank
  Home's What's new / Recommended by digest significance × topic-cluster affinity.
- **G5** — No request-time LLM, *bridge-never-rehost* unchanged; all derived from existing
  corpus artifacts (KG, GI, digest, topic_clusters.json) + per-user state.

## Non-Goals

- Re-clustering or changing the clustering algorithm (consume `topic_clusters.json` as-is).
- Cross-user / social features, collaborative filtering across users (single-user corpus for now).
- New scraping/discovery of external podcasts (PRD-037 / #1069 territory).
- A graph-visualization surface in the consumer app (that's the operator viewer's job).

## Personas

- **The returning learner**: listens regularly, wants the app to lead with *their* themes and to
  pull the thread on a person or topic across everything they've heard.
- **The new user**: at sign-up, tells the app what they care about so Home is relevant on day one.

## User Stories

- As a listener, when I open an episode's Insights, I see its topics **grouped by theme** with the
  main theme first, so I grasp what it's *about* at a glance.
- As a listener, tapping **a person** opens a card: who they are, which episodes they appear in,
  and related people/topics — not just a search.
- As a listener, tapping **a topic** opens a card: the cluster it belongs to, sibling topics, and
  episodes about it.
- As a searcher, when my query *is* a person or topic, I get an **entity card** at the top, then
  the passage results.
- As a new user, I pick a few **interest areas** at sign-up and Home's "What's new for you"
  reflects them.

## Functional Requirements

### FR1: Topic-cluster understanding (3.1 — foundation)

- **FR1.1**: `GET /api/app/episodes/{slug}/entities` returns, per topic, its cluster identity:
  `cluster_id` (`graph_compound_parent_id`), `cluster_label` (canonical), and `cluster_size`
  (cross-corpus member count). `null`/absent when a topic is a singleton.
- **FR1.2**: The Knowledge panel groups the episode's topics by `cluster_id`, orders clusters by
  **intra-episode** membership (most topics in this episode first), and visually distinguishes the
  dominant cluster. Singletons trail. People remain a separate, adjacent affordance.
- **FR1.3**: Degrades cleanly when `topic_clusters.json` is absent (flat topic list, today's behaviour).

### FR2: Entity cards (3.2 person · 3.3 topic)

- **FR2.1**: A **person card** (sheet on mobile / panel on desktop) shows the name, the episodes the
  person appears in, and related people/topics — from **KG co-occurrence across the corpus** via a
  dedicated `GET /api/app/persons/{id}` (NOT a proxy of the operator relational API; the `/api/app`
  boundary stays clean). No biography (consumer scope is lean).
- **FR2.2**: A **topic card** shows the topic's cluster, sibling topics in that cluster, and the
  episodes about the topic — via a dedicated `GET /api/app/topics/{id}` (KG-grounded).
- **FR2.3**: Tapping a person/topic chip opens its card (replacing the Epic-2 "→ library search"
  default); the card offers an explicit "search the library for this" action. Re-entrant (related
  chips walk to that entity with a back stack); modal a11y (focus trap, ESC/backdrop close).

### FR3: Entities in search (3.4)

- **FR3.1**: When a query exact/near-exact-matches a person or topic name, the **consumer** search
  surfaces an **entity card** above the passage results (viewer parity deferred). Resolution is a
  dedicated `GET /api/app/entities/search` (kept off the shared search response).
- **FR3.2**: The entity card opens the person/topic card (FR2); tapping an episode opens the player.

### FR4: Personalized discovery (3.5)

- **FR4.1**: A first-Home **dismissible** "set your interests" card (signed-in only) opens a picker
  over the corpus's **top-12 clusters** (`GET /api/app/clusters`); the chosen cluster ids are saved
  as **per-user files** (`GET/PUT /api/app/interests`) — no new persistence layer, no sign-up step.
- **FR4.2**: The Home feed (`GET /api/app/discover`) ranks by **significance × interest-cluster
  affinity** when personalization is **enabled** (env `APP_PERSONALIZED_RANKING`, default off) AND
  the signed-in user has interests; otherwise **recency** — the unchanged default. The score is
  provisional and flag-gated until tuned.

### FR5: Char-level quote highlighting (3.6 — Epic-2 polish)

- **FR5.1**: Upgrade the transcript grounded-quote highlight from segment-level to the exact quoted
  substring (char offsets), guarded against transcript-version drift; no-op when offsets don't align.

## Success Metrics

- Topics in the panel are clustered + dominant-cluster-led for ≥95% of episodes that have a cluster
  artifact; flat fallback otherwise.
- Person/topic cards render from existing artifacts with **zero** request-time LLM calls.
- Personalized Home measurably re-orders vs recency for a user with set interests (deterministic test).

## Dependencies

- `search/topic_clusters.py` / `topic_clusters.json` (cluster source); `/api/corpus/topic-clusters`
  (operator parallel).
- #1089 relational/subject layer (`useRelationalCache`, subject store) for entity cards.
- Corpus digest (`corpus_digest.py`) for FR4 ranking.
- Per-user state store (profile) — **no new persistence layer** beyond the existing per-user files.

## Constraints & Assumptions

**Constraints:**

- No request-time LLM (D6). Bridge audio, never rehost. No new persistence layer.
- Reuse the viewer's relational data/endpoints where possible rather than re-deriving.

**Assumptions:**

- `topic_clusters.json` is present for processed corpora (it is, for prod-v2); cluster ids are stable
  enough to key UI grouping per request.

## Design Considerations

### Cluster ordering: cross-corpus size vs intra-episode dominance

- **Option A** — order an episode's topics by each cluster's cross-corpus `member_count`.
- **Option B (chosen)** — order by **intra-episode** cluster membership (how many of *this* episode's
  topics fall in the cluster); `member_count` is a secondary tiebreak / "prevalence" cue. Better
  reflects what the episode is actually about.

### Entity cards: port vs re-build

- Reuse #1089's relational data and (where shaped right) endpoints; adapt the *presentation* to the
  consumer design system (UXS-011 tokens), don't fork the data layer.

## Integration with Epic 2 (Player / Home / Search)

- **Knowledge panel** (PRD-039): topics become cluster-grouped; chips open entity cards.
- **Home** (PRD-042): What's new / Recommended ranking gains the digest × interest signal.
- **Search** (RFC-099 §Home): entity cards ride above the existing grouped passage results.
