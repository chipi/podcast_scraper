# Epic 3 proposal — what's next for the consumer Learning Platform

Written 2026-06-25, after the Epic-2 play-test + the `feat/consumer-app` integration onto
post-#1089 main. Informed by the full play-test feedback set (see `EPIC2-playtest-backlog.md`)
and what main now provides (#1089 shipped the viewer's **PersonLandingView** profile card,
GI v3 relational edges, `useRelationalCache`, and `search/topic_clusters.json`).

## The through-line the operator kept pointing at

Across the play-test the requests converged on one product idea: **make knowledge first-class**.
Not just "play + transcript", but entities (people, topics) you can *open*, that are *clustered*,
that show up *in search*, and that *personalize* what you see. The transcript↔insight bridge we
shipped is the first step; the operator then asked for: cluster-first topics, a person profile
card on click, a topic card, entities as search results, digest-powered discovery, and interests
at registration. These are facets of the same layer.

## Recommended: **Epic 3 = "Knowledge layer"** (entity cards + clusters), then personalization, then o11y

### E3.1 — Topic clusters API + cluster-first UI  *(foundation; data ready)*
- Expose per-topic cluster on `/api/app/episodes/{slug}/entities` by joining
  `search/topic_clusters.json` via `load_topic_cluster_enrichment_map(root)` →
  `{topic_id: {graph_compound_parent_id, canonical_label}}` (+ member_count for size).
- `AppTopic` gains `cluster_id`, `cluster_label`, `cluster_size`.
- Knowledge panel: group the episode's topics by cluster, lead with the dominant intra-episode
  cluster, make it stand out. (Operator's "cluster first + stand out".)

### E3.2 — Entity profile cards in the consumer app  *(port + adapt the viewer's work)*
- **Person card**: main shipped `PersonLandingView` + `useRelationalCache` + `subject` store in
  the viewer (#1089). Adapt to the consumer surface — tap a person → a profile card (bio/role +
  episodes they appear in + related people) instead of a raw search.
- **Topic card**: sibling card focused on related topics (the cluster) + episodes about it.
- New consumer endpoints (or reuse viewer's relational API) for person/topic subjects.

### E3.3 — Entities as search results
- When a query strongly matches a person/topic, surface an **entity card** at the top of search
  results (knowledge-panel style), in the consumer app *and* the viewer. Reuses E3.1/E3.2.

### E3.4 — Char-level quote highlighting *(quick refinement of the Epic-2 headline)*
- Upgrade transcript highlight from segment-level to the exact quoted substring (char offsets),
  guarded against transcript-version drift.

## Then: Epic 4 — Personalized discovery
- **Digest powers What's new / Recommended** — replace "newest" with digest significance +
  topic-affinity ranking (corpus digest already exists).
- **Interests at registration** — pick topic clusters at sign-up (from `topic_clusters.json`),
  store on profile, personalize discovery. Depends on E3.1.

## Then: Epic 5 — Observability (RFC-099 §8)
- UX analytics parity with the viewer, Sentry, Grafana. Mechanical; gated on a stable surface
  (now true). Operator flagged this explicitly.

## Sequencing rationale
E3.1 is the cheap, high-leverage foundation (data + seam already exist). E3.2/E3.3 turn the
operator's "profile card / topic card / entities in search" into one coherent feature and reuse
#1089's viewer investment. E3.4 is a fast polish. Personalization (Epic 4) needs E3.1's clusters
+ a registration step, so it follows. Observability (Epic 5) is independent and can slot in
whenever a stabilization window opens.

Each sub-epic still runs the operator's loop: GH issue → PRD/UXS → RFC → implement w/ full test
pyramid → critical pass → API docs.
