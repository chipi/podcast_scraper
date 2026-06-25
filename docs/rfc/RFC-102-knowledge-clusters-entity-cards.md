# RFC-102: Knowledge Clusters, Entity Cards & Personalized Discovery

- **Status**: Draft (§1 Implemented — Epic 3.1, #1092)
- **Authors**: Marko
- **PRD**: `docs/prd/PRD-043-knowledge-layer.md`
- **UX spec**: `docs/uxs/UXS-013-knowledge-clusters.md`
- **Depends on**: RFC-098/099 (consumer client), `search/topic_clusters.py` (clustering), the
  viewer relational/subject layer (#1089: `PersonLandingView`, `useRelationalCache`).
- **Epic**: GitHub #1093 (children #1092, #1095–#1100)

---

## Abstract

Technical design for Epic 3 — making the consumer app's knowledge **navigable**: topic clusters
(§1), person/topic entity cards (§2/§3), entities surfaced in search (§4), and personalized
discovery (§5). All read-only over existing corpus artifacts; **no request-time LLM**, bridge
audio / never rehost, no new persistence layer. §1 is implemented; §2–§5 are slice designs.

## Problem

The corpus computes topic clusters (`topic_clusters.json`) and the viewer has a rich person
profile surface (#1089), but the consumer app shows a flat topic list and tapping an entity only
runs a library search. The knowledge is present but not navigable.

## Design

### §1 Topic clusters + cluster-first panel (3.1 — implemented, #1092)

**Data.** `search/topic_clusters.py :: consumer_topic_cluster_map(corpus_root)` → `{topic_id:
{cluster_id, cluster_label, cluster_size}}`, built from the same `topic_clusters.json` the search
layer/operator viewer use. `cluster_id` = the cluster's `graph_compound_parent_id`; `cluster_size`
= cross-corpus member count. Singletons are absent (→ no cluster fields). Path-safe load shared
with `load_topic_cluster_enrichment_map` via `_load_topic_clusters_payload`.

**API.** `GET /api/app/episodes/{slug}/entities` enriches each `AppTopic` with `cluster_id`,
`cluster_label`, `cluster_size` (defaults `null`/`null`/`0`). No-op (flat list) when the artifact
is absent — today's behaviour preserved.

**UI (Insights panel).** Topics are ordered **cluster-first**: the **dominant** cluster (the one
with the most of *this episode's* topics, ≥2; tie → larger `cluster_size`) leads and its chips get
a `ring-topic` standout; other clustered topics follow (larger intra-episode groups first);
singletons trail; People remain an adjacent affordance in the same row. A "Theme · {cluster}"
lead-in names the dominant cluster. Stable sort preserves original order within a rank.

**Why intra-episode dominance (not cross-corpus size):** it reflects what *this* episode is about;
`cluster_size` is only a tiebreak / prevalence cue (PRD-043 Design Considerations).

### §2 Person profile card (3.2, #1095)

Tap a person → a card (sheet mobile / panel desktop): name, episodes they appear in, related
people/topics. Data from a **dedicated** `GET /api/app/persons/{id}` — KG co-occurrence over the
corpus (`app_relational_view.build_person_card` reuses `entities_from_kg`), NOT the operator
relational API / `subject` store (the `/api/app` boundary stays clean; effort over coupling). No
biography. Replaces the Epic-2 person-chip→search default; the `EntityCard` keeps an explicit
"search the library" action, is re-entrant (related chips walk with a back stack), and is a proper
modal (focus trap, ESC/backdrop close, restore focus). UXS-011 tokens.

### §3 Topic card (3.3, #1096)

Tap a topic → the same `EntityCard`: its cluster (`cluster_label`), sibling topics in the cluster
(from `topic_clusters.json` members via `consumer_cluster_siblings`), episodes about it, and related
people. Data from a **dedicated** `GET /api/app/topics/{id}` (KG-grounded; not a reuse of search).
Builds on §1's exposed cluster identity.

### §4 Entities as search results (3.4, #1097)

When a query exact/near-exact-matches a person/topic name, the search response carries an **entity
hit** that the client renders as a card above passage results (consumer + viewer). Links into
§2/§3 cards.

### §5 Personalized discovery (3.5, #1098)

Sign-up interests = chosen topic **clusters** (from the corpus's top clusters), stored on the
per-user profile (existing per-user files). Home "What's new"/"Recommended" rank by **corpus-digest
significance × interest-cluster affinity**; fall back to recency when absent.

## Testing

- §1: unit (`consumer_topic_cluster_map` shape), integration (`/entities` attaches cluster fields;
  flat fallback without artifact), frontend unit (cluster-first ordering + dominant standout +
  theme). **Shipped with 3.1.**
- §2–§5: each slice lands its own test pyramid + critical pass per the loop.

## Phasing

§1 (clusters) is the shared spine and ships first; §2/§3 (cards) and §5 (personalization) consume
it; §4 builds on §2/§3. Char-level highlighting (PRD-043 FR5 / 3.6) is independent.
