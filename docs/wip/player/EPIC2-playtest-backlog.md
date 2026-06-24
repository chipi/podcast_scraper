# Epic 2 — Play-test feedback backlog (2026-06-24)

Captures the full operator feedback set from the live play-test session. Top section =
already shipped + verified this session (on `feat/consumer-app`, uncommitted). Lower sections =
remaining backlog, grouped into coherent workstreams with sequencing notes.

## ✅ Shipped + verified this session (uncommitted on `feat/consumer-app`)

- **Editorial Home** — "What's new" → no-scroll ranked (01 hero + numbered rows); removed the
  duplicate Featured block; Recommended → no-scroll grid.
- **Episode card** — clean one-line lede + ✦ insights popover (full summary on hover/tap);
  fixed the root data bug (`topics` was full summary-bullet sentences). Insights icon
  hover-reveals on library rows.
- **Search** — grouped by episode, kind labels (Insight/Transcript/Topic), real timestamps
  only (no fake 0:00), episode artwork thumbnails, shorter clamped passages.
- **Auth** — local sign-in fixed (dev proxy `changeOrigin:false`); sign-up + full loop.
- **Naming** — panel title aligned to "Insights" (matches dock button + cards).
- **"More like this"** — tiny episode artwork per row.
- **Topic/Person chips** → explore the term across the library (search). *(Superseded by B2/B3.)*
- **Show page** — logo + description header.
- **Queue** — episode artwork + full titles (no ellipsis).
- **Transcript ↔ Insights (headline)** — grounded quotes highlighted in the transcript
  (green ● + underline), tap → opens the panel and centre-scrolls to the claim; grounded ●
  indicator distinguishes sourced claims from ungrounded ones.
- **Show name clickable** (Player, Queue) → show page. **Topics + People** merged into one
  compact, expandable row.
- **Decision recorded** — transcript/derived artifacts are ours to host/transform; audio is
  bridge-only (see `project_transcript_vs_audio_hosting` memory).

## A — Transcript fidelity
- **A1. Sync drift 5–10s.** Cause: acast **dynamic ad insertion** — live stream has ads not in
  our transcribed copy; transcript leads audio by the pre-roll, grows with mid-rolls. Sync code
  is correct (absolute time, raw segments). Options: (a) manual sync-offset nudge [pragmatic,
  ships now], (b) serve our transcribed audio copy = perfect sync but is "rehosting" [operator's
  line], (c) playback-time alignment [large]. **Needs operator decision.**
- **A2. Char-level quote highlight** (currently segment-level). Refinement.

## B — Knowledge entity cards + clusters  *(one coherent feature; needs PRD/UXS)*
- **B1. Topic-cluster API.** Seam ready: `search/topic_clusters.json` +
  `load_topic_cluster_enrichment_map(root)` → `{topic_id: {cluster_id, canonical_label}}` (86
  clusters in prod-v2). Expose per-topic cluster on `/entities`; panel leads with the dominant
  intra-episode cluster + makes it stand out.
- **B2. Person profile card** in the player on person-click (port the viewer's new profile card,
  added in latest main) — instead of going straight to search.
- **B3. Topic card** — related-topics / cluster-focused card (viewer + consumer).
- **B4. Entities as search results** — person/topic profile cards as first-class result items
  (viewer + consumer), knowledge-panel style.

## C — Personalized discovery  *(needs PRD/UXS; depends on B1 + digest)*
- **C1. Digest powers What's new / Recommended.** Today "What's new" = newest episodes only;
  it should use the corpus digest ranking (significance, not just recency). Recommended should
  use digest topic-affinity.
- **C2. Interests at registration.** During sign-up, user picks topic clusters of interest
  (from `topic_clusters.json`); store on profile; personalize What's new / Recommended.

## D — Desktop polish
- **D1.** App "feels weird" on desktop (mobile-first retained, but must look great wide).
  Investigate player 3-column + max-width/whitespace across views.

## E — Observability  *(RFC-099 §8; operator gated to post-home-stabilization — ~now)*
- **E1.** UX analytics parity with the viewer. **E2.** Sentry. **E3.** Grafana.

## Suggested sequencing
1. Commit shipped work (2 commits: UX batch, transcript-intelligence batch).
2. **D1 desktop polish** (no design, makes shipped work look right).
3. **A1 sync decision** (operator picks approach; nudge ships fast).
4. **B (knowledge cards + clusters)** — PRD/UXS, then B1 → B2/B3/B4.
5. **C (personalization)** — PRD/UXS; builds on B1 + digest + registration step.
6. **E (observability)** — mechanical; home is built.
