# RFC-114: Personal Corpus — a Unified Definition Across Surfaces

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8), advisor (Fable 5)
- **Related epic**: [#1470](https://github.com/chipi/podcast_scraper/issues/1470)
- **Stakeholders**: Core team, Consumer App, Search/Retrieval, MCP
- **Related PRDs**:
  - `docs/prd/PRD-041-consolidation.md` (shipped the read-time personal projection; this generalizes it)
  - `docs/prd/PRD-046-delivery-and-curation.md` (collections/highlights/favorites that feed the corpus)
- **Related RFCs**:
  - `docs/rfc/RFC-101-personal-knowledge-corpus.md` (the shipped `user_episode_set`; recall stays pinned to it)
  - `docs/rfc/RFC-112-remote-mcp-transport-and-auth.md` (its personal-scope toggle consumes **Phase 1**)
  - `docs/rfc/RFC-113-graph-aware-pkm-export.md` (its incremental cursor consumes **Phase 1**'s revision counter)

## Implementation status (2026-08-06)

Built on `feat/next-arc-rfcs` (unpushed). **Phase 1 + Phase 2 shipped.**

- **Phase 1** — faceted membership: `experienced` = heard ∪ captured (highlights, notes,
  saved-**insights**) and **excludes whole-episode favorites**, which are the separate `saved` facet
  (operator-confirmed correction — a favorite is save-for-later, not engagement). Reconcile-on-read
  **revision counter + change log** (adds + tombstones) at `corpus_log.json`; `GET /api/app/corpus`,
  `/corpus/episodes?facet=`, `/corpus/changes?since=`.
- **Phase 2** — weighted **strength** model (`w_h=0.4/w_c=0.3/w_f=0.1/w_r=0.2`, caps C=5/R=3) +
  `/api/app/corpus/ranked`.
- **Consumers of the facet change**: `scope=mine` search/relational/consolidation (intended), plus
  `app_auto_picks` + digest — a favorited-but-unplayed episode can now resurface (flows from the
  save-for-later correction; flagged for operator confirmation).
- **Review-hardening (fable-5)**: membership computed **inside** the reconcile lock (no phantom
  tombstone/re-add under concurrent web+shell reconciles); `changes_since(since=0)` now reports
  `truncated` correctly against a trimmed log; the multiuser isolation test updated to the new
  `experienced` semantics (a capture, not a favorite, puts an episode in recall).

## Abstract

"Personal corpus" is the platform's core thesis (PRD-035): listening compounds into a growing,
connected, grounded body of knowledge that is *the user's*. Today the ingredients are **scattered**
across surfaces with slightly different semantics. PRD-041 shipped one read-time projection
(`user_episode_set` = heard ∪ captured) for recall/resurfacing, but there is no single agreed
**definition** every surface (Home, Library, recall, digest, MCP, export) shares. This RFC defines it
in **two phases**: **Phase 1 (keystone)** — faceted membership (`experienced` vs `saved`), a per-user
**revision counter + change log**, and one `/api/app/corpus` API; **Phase 2** — a weighted **strength**
model for ranking. Phase 1 is the dependency RFC-112 (personal scope) and RFC-113 (incremental export)
both wait on; Phase 2 trails independently. Crucially, this **does not change shipped recall
semantics**: recall stays pinned to `experienced`.

## Problem Statement

The signals exist but aren't unified, and folding them naively **breaks a shipped contract**:
- **Playback history** (PRD-039) — what I played, ≥30% = "heard".
- **Highlights / notes / saved insights** (PRD-040) — what I captured.
- **Favorites** (episodes, insights) — what I saved for later (**not** necessarily heard).
- **Collections** (PRD-046 #1417) — sets of **highlights** (⇒ their episodes are already *captured*).
- **Interest profile** (RFC-101 §6) — inferred person/topic frequencies.

`user_episode_set` folds heard ∪ captured for recall, and RFC-101 is explicit that recall must "cite
the user's own experience." A **favorited-but-never-played** episode is "saved for later", not
"learned" — so it must **not** silently enter recall. Every surface also re-derives its own notion of
"mine", and downstream consumers (MCP `scope=mine`, the export, a future "your brain" view) need one
authoritative, change-trackable definition.

## Goals

1. **Faceted membership** consumed by every surface: `experienced` (heard ∪ captured) vs `saved`
   (favorited) — never conflated.
2. **A per-user revision counter + change log** (the keystone primitive) so consumers can ask "what
   changed in my corpus since X", including **deletions** (tombstones).
3. **One API** (`/api/app/corpus`) surfacing membership + the graph projection; existing `scope=mine`
   flags re-point to it.
4. **Phase 2 strength** — a transparent weighted per-episode score for ranking (no ML).
5. **Read-time, no rebuilt per-user index** (PRD-041 D-decision), backward compatible.
6. Be the **stable dependency** RFC-112 + RFC-113 point at.

## Constraints & Assumptions

- **Recall stays on `experienced`** — recall/connections read `experienced` only. This *corrects*
  today's `user_episode_set` by dropping episode-favorites (§1.1); it does not otherwise change what
  those surfaces return.
- **No per-user index rebuild** — read-time projection over shared artifacts + the per-user overlay.
- **Privacy** — strictly the user's own signals; nothing crosses users.

## Design & Implementation

### Phase 1 — membership, revision counter, API (keystone; ships first)

**1.1 Faceted membership.**
- `experienced(user)` = heard (≥threshold) ∪ highlights ∪ notes ∪ **saved-insights** (a bookmarked
  grounded insight = engagement with that episode's content). Collections add nothing (they hold
  highlights ⇒ already captured).
- `saved(user)` = episodes the user **favorited as a whole episode** but has not experienced —
  "save for later", not "learned". A **distinct** facet.
- `personal_corpus(user)` = `{ experienced, saved }`. **Consumers choose the facet:** recall +
  cross-episode connections use `experienced` **only**; Library, export, and the digest may show
  `saved` too, **visibly labelled** as saved-not-heard.
- **Deliberate correction (operator-confirmed 2026-08-05, NOT "no change"):** today's
  `user_episode_set` wrongly folds **episode-favorites** into the captured set, so a
  favorited-but-never-played episode currently appears in recall ("what have I learned about X").
  This RFC **removes** episode-favorites from `experienced` → recall returns fewer results for users
  who favorited episodes they never heard. That is an intended fix of a pre-existing conflation, not
  a no-op rename. Saved-*insights* stay in `experienced` (content engagement); episode-*favorites*
  move to `saved`. A regression test pins the corrected recall set.

**1.2 Revision counter + change log (the missing primitive).**
- A per-user monotonic `corpus_revision` integer, incremented on every personal-signal write:
  highlight/note add/delete, favorite add/remove, collection membership change, and a
  **playback-threshold crossing** (an episode passing ≥30% flips it into `experienced`).
- A bounded per-user **change log**: `{ revision, kind: added|removed, facet, episode_slug|entity_id }`
  entries, so a consumer polling `since=<rev>` gets both additions **and tombstones** (removals) — the
  thing flat "changed-after-timestamp" cannot express. This is what makes RFC-113's incremental export
  and any cached projection correct.

**1.3 API.**
- `GET /api/app/corpus` → `{ revision, experienced_count, saved_count, top_entities }`.
- `GET /api/app/corpus/episodes?facet=experienced|saved` and `/entities` → the projected views.
- `GET /api/app/corpus/changes?since=<rev>` → the change-log delta (additions + tombstones + new
  revision). Consumed by RFC-113.
- Existing `scope=mine` (search, relational, resurfacing) re-points to `experienced` — same set they
  use today, now from one definition.

**1.4 Entity projection.** Aggregate `experienced` episodes' KG entities/topics (RFC-072) into a
per-user entity view (person/topic → episodes, first/last heard). The graph recall traverses + the
export serializes.

### Phase 2 — strength model (trails; ranking only, no consumer blocks on it) — **implemented**

- A transparent per-episode **strength** score, `[0,1]`, from the present signals (`app_corpus_strength`).
  **Formula:** `strength = w_h·heard_fraction + w_c·min(captures,C)/C + w_f·favorited + w_r·min(relistens,R)/R`,
  weights summing to 1, clamped, **monotonic** in each signal. **v1 weights/caps** (`Weights`, tunable):
  `w_h=0.4, w_c=0.3, w_f=0.1, w_r=0.2`; `C=5` captures, `R=3` relistens. Signals: `heard_fraction` from
  playback/duration, `captures` = highlights + episode-notes, `favorited` = episode-favorite,
  `relistens` = `max(0, opens−1)` from the listen log. Comparable **within** a user, not across users
  (v1). No ML.
- **Surfaced** at `GET /api/app/corpus/ranked` (experienced episodes, strongest first). **Not yet
  adopted** by recall/digest ordering — that's incremental consumer wiring, a follow.
- Recency decay + negative signals (dismissed/skipped) remain **open questions**, out of v1.
- Nothing downstream **blocks** on Phase 2 — MCP scope + export need membership (Phase 1) only.

### Surface consumption

Home ordering, Library, recall (`experienced`), digest (`app_digest_personal`), MCP `scope=mine`
(RFC-112, `experienced`), and the PKM export (RFC-113, over the change log) all read this one
definition — so "my corpus" is identical everywhere, and "what I've learned" never includes what I
only saved.

## Key Decisions

1. **Faceted `experienced` vs `saved`** — recall stays `experienced`-only; favorites are a separate,
   labelled facet. (Preserves the shipped RFC-101 recall contract — the load-bearing decision here.)
2. **Revision counter + change log is a first-class primitive** — deletions are expressible; it's the
   dependency of incremental export and any future materialized projection.
3. **Two phases** — membership+counter first (keystone), strength trails (blocks nobody).
4. **Generalize `user_episode_set`** (→ `experienced`) rather than replace; collections are not a
   membership signal.

## Alternatives Considered

- **Union of all signals into one flat "mine" set (the earlier draft).** Rejected — it silently put
  favorited-but-unheard episodes into recall, breaking RFC-101's "cite your own experience" contract.
- **Keep per-surface `scope=mine` derivations.** Rejected — drift is the disease this RFC treats.
- **Timestamp cursor instead of a revision counter.** Rejected — can't express deletions, breaks on
  same-second writes, and retroactive membership joins (a favorite joining an old episode) have no
  "changed-after" artifact.
- **Materialize a per-user corpus index.** Deferred (PRD-041 cost/D6); the revision counter is the
  invalidation key if/when scale forces it (Open Questions).

## Testing Strategy

- Unit: `experienced` = heard ∪ highlights ∪ notes ∪ saved-insights, and **excludes episode-favorites**
  (regression pins the corrected recall set); `saved` = episode-favorites − experienced; revision
  counter increments on each signal write; change log emits tombstones on delete; playback-threshold
  crossing flips facet + bumps revision.
- Integration: recall/connections return identically before/after (frozen semantics); `since=<rev>`
  delta = adds + removes; multi-user isolation (A's corpus never includes B's signals).

## Rollout & Monitoring

- Ship Phase 1 with `experienced` == `user_episode_set` so **nothing regresses**; re-point `scope=mine`
  incrementally; add `saved` surfaces behind the existing personalization posture. Phase 2 strength
  ships later behind a flag.
- Monitor: corpus-size distribution, change-log volume, recall parity vs the old flag.

## Open Questions

- Phase 2 strength weights + whether strength is ever cross-user comparable.
- Recency decay; do negative signals (dismissed/skipped) subtract?
- Change-log retention bound + when to materialize the projection (restated PRD-041 OQ-1).

## References

- PRD-041 / RFC-101 (`user_episode_set` → `experienced`), PRD-039 (playback), PRD-040 (capture),
  PRD-046 (favorites/collections), RFC-072 (entities), RFC-112 (MCP scope), RFC-113 (export cursor).
