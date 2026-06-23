# PRD-035: Learning Platform (parent)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7
- **Child PRDs**:
  - PRD-036 — Foundation / Identity (minimal multi-user)
  - PRD-037 — Discovery
  - PRD-038 — Catalog
  - PRD-039 — Player
  - PRD-040 — Capture (highlights + notes)
  - PRD-041 — Consolidation (personal knowledge corpus)
- **Related RFCs** (existing intelligence layer this platform consumes):
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` — GIL insights + grounded quotes
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` — KG entities/edges
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — canonical identity
  - `docs/rfc/RFC-090-hybrid-corpus-search.md` — hybrid semantic + keyword retrieval
  - `docs/rfc/RFC-097-unified-kg-gi-ontology-v2.md` — v2 ontology
- **Related analysis**:
  - `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` — server-side multi-user lift + sequencing

> **Note on numbering**: the player drafts in `docs/wip/player/` (PRD-027 discovery,
> PRD-029 catalog, PRD-030 player) reuse numbers already taken by live viewer PRDs in
> `docs/prd/`. This platform initiative takes a clean block, PRD-035–041. Those drafts
> are superseded by the child PRDs above and will be folded in, not promoted as-is.

---

## Summary

The Learning Platform is an end-user layer on top of the existing podcast_scraper
pipeline. At its simplest it is a **podcast player** — Spotify-grade, familiar, polished.
What makes it different is everything around the playback: the player streams audio from
each podcast's original host while overlaying the structured intelligence the pipeline
already produces (grounded insights, quotes, entities, cross-episode knowledge), and lets
the listener **capture and consolidate** what they hear so that — over weeks and months —
their personal knowledge corpus compounds.

The thesis: **just listening does not make knowledge stick.** Today's podcast apps optimise
for consumption. This platform optimises for *retention* — listening is the input, a growing,
connected, grounded personal knowledge corpus is the output. The player is the surface; the
consolidation loop is the moat. Crucially, the expensive half of that moat (grounded insights,
the knowledge graph, hybrid search, cross-show synthesis) already exists — this platform makes
it tangible and personal.

---

## Background & Context

- **The pipeline already produces the hard part.** Per episode we emit transcripts with
  segment-level timing and speaker IDs, grounded insights (each claim tied to a verbatim quote
  with a millisecond timestamp and speaker), a knowledge graph (people, orgs, podcasts, topics with
  canonical identity), hybrid semantic search, and cross-episode relational traversals. This is
  exposed today through a read-only HTTP API and an operator-facing viewer (`web/gi-kg-viewer`).
  No competing podcast app has this substrate.
- **What's missing is the consumer surface and the personal layer.** The viewer is an operator
  tool over a single shared corpus, with no user model, no playback, and no capture. There is no
  way for a listener to *use* the intelligence while actually listening, and nothing accumulates
  per person.
- **Why now.** The intelligence layer reached maturity in the v2.x line (RFC-049/055/072/090/097).
  The marginal cost of a consumer layer is now mostly UX + a thin per-user backend — not new ML.
- **Relationship to the viewer.** The viewer (RFC-062) remains the operator/debug surface over the
  raw corpus. The Learning Platform is a separate consumer product that consumes the same read API
  and artifacts. They share data contracts, not UI.

---

## Goals

- Deliver a Spotify-grade player as the familiar floor: browse, queue, play, resume, fast and
  polished on mobile.
- Make the pipeline's intelligence tangible *during* listening — transcript-synced playback,
  tap-to-seek, inline grounded insights and entities.
- Let listeners **capture** what matters (Kindle-style highlights + personal notes) with zero friction.
- **Consolidate** captures into a personal, grounded, cross-episode knowledge corpus that grows over
  time and resurfaces for reflection — the core differentiator.
- Stand up a **minimal multi-user foundation** (OAuth identity + per-user state) without forking the
  shared corpus per user.
- Bake in accessibility (a11y) and internationalisation (UI i18n) from the first line of code.

## Non-Goals

- **Not** a rehosting service. The platform never serves third-party podcast audio; it streams from
  the original enclosure URL (see Principle 4).
- **Not** a replacement for the operator viewer (RFC-062) — that remains the debug/operator surface.
- **Not** social: no comments, sharing, follows, or "what others are listening to" in v2.7.
- **Not** content translation (translating transcripts) — that is a future *pipeline* capability,
  distinct from UI i18n which *is* in scope. Voice control is north-star, not v2.7.
- **Not** per-user corpus tenancy. Episode artifacts are shared across all users by design; only the
  personal overlay is per-user. This deliberately sidesteps the "Large" multi-tenancy lift identified
  in `MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md`.
- **Not** organisations, roles, or sharing/permission models.

---

## Principles (the spine — every child PRD inherits these)

1. **A learning environment shaped like a player.** Spotify-grade playback is the floor, not the
   ceiling. Familiar and polished first; augmentation makes it a place to learn.
2. **The moat is consolidation, and it reuses what we built.** A user's personal corpus is a
   *per-user projection over the existing GIL/KG ontology* — captured highlights and saved insights
   become nodes in *their* graph, grounded in the exact moments they marked. We add a personal layer,
   not a new brain.
3. **Shared artifacts, personal overlay.** Episode-derived artifacts (transcript, insights, KG,
   search index) are global and shared. Everything a user marks, saves, queues, or annotates is
   private to them. This boundary is the foundation of "minimal multi-user."
4. **Bridge, never rehost.** The player streams audio from the original podcast host's enclosure URL.
   The platform stores only *derived* artifacts. Internal audio files remain a pipeline-only data
   source (for reprocessing as code changes) and are never served to end users; they may be dropped.
5. **Multi-user, minimal foundation.** Identity via a single OAuth provider (Google to start) plus
   per-user state. No orgs, roles, sharing, or social.
6. **a11y + i18n from line one; voice is north-star.** Accessibility and UI internationalisation are
   non-functional requirements present from the first commit, not retrofits. Voice control depends on
   the a11y foundation and is deferred.

---

## Personas

- **The active learner** (primary): listens to long-form interview/idea podcasts to *learn*, not just
  pass time. Wants to remember, connect, and revisit what they hear. Frustrated that nothing sticks.
  - Needs: capture moments effortlessly; later recall "what did I learn about X / from this guest."
- **The researcher / knowledge worker**: uses podcasts as a source. Needs grounded, citable, jump-to-
  the-moment evidence and cross-episode synthesis.
  - Needs: grounded Q&A, quote-accurate timestamps, cross-show "who said what about a topic."
- **The casual listener**: wants a great, fast player. Will discover the learning features gradually.
  - Needs: it must "just work" as a player even if they never open the knowledge surfaces.

---

## User Stories (platform-level; detailed stories live in child PRDs)

- *As an active learner, I can highlight a moment while listening so that it joins my personal corpus.*
- *As an active learner, I can ask "what have I learned about X" and get grounded answers drawn from
  episodes I have actually heard.*
- *As a researcher, I can tap any transcript line to jump to that exact moment, and see the insight it
  grounds.*
- *As a casual listener, I can search for a show, queue an episode, and play it with resume — without
  ever touching the intelligence features.*
- *As any user, I can sign in with my Google account and find my library, queue, and highlights exactly
  where I left them, on any device.*

---

## Functional Requirements (platform-level scope map)

Detailed FRs live in the child PRDs. This section fixes the boundaries and the shared contracts.

### FR1: Segments — the scraper→player data contract

- **FR1.1**: The platform consumes a stable `segments.json` per episode as the canonical contract
  between the pipeline and the player (transcript segments with `start`/`end`/`text`/optional `speaker`).
  This contract is owned by this parent PRD; child PRDs and RFCs may not break it without amending here.
- **FR1.2**: All per-episode intelligence (insights, entities, summary) is addressed by episode `slug`
  via the read API; the player never parses raw corpus files directly.

### FR2: Audio bridge

- **FR2.1**: Audio is played client-side from the original enclosure URL resolved from the source feed.
  The platform exposes a *resolution* endpoint, never a *streaming/proxy* endpoint for third-party audio.
- **FR2.2**: Internal pipeline audio (used for transcription/reprocessing) is never exposed through any
  end-user route.

### FR3: Identity & per-user state (→ PRD-036)

- **FR3.1**: A single OAuth provider (Google to start) authenticates users. Sessions gate all per-user
  routes.
- **FR3.2**: Per-user state covers: library subscriptions, playback positions, queue, highlights, notes,
  and topics/interests. Shared corpus artifacts are read-only and identical for all users.

### FR4: Segment surfaces (→ child PRDs)

- **FR4.1 Discovery (PRD-037)**: find shows, add to library, request scrape-on-demand.
- **FR4.2 Catalog (PRD-038)**: browse library episodes with ready/pending status and enriched previews.
- **FR4.3 Player (PRD-039)**: queue + transcript-synced playback + inline knowledge panel.
- **FR4.4 Capture (PRD-040)**: highlights + notes, persisted per user.
- **FR4.5 Consolidation (PRD-041)**: personal knowledge corpus, grounded recall, spaced resurfacing.

---

## Phasing within v2.7

Sequenced so each phase is independently shippable and bisectable (one themed branch per phase).

- **P0 — Foundation (PRD-036)**: OAuth identity, per-user data model, the read API surface + the
  `segments.json` contract, enclosure resolution. Proven by a thin reference player.
- **P1 — Core (PRD-037/038/039)**: Discovery + Catalog + Player with queue. The Spotify-grade floor.
- **P2 — Capture (PRD-040)**: highlights + notes.
- **P3 — Consolidation (PRD-041)**: personal corpus + resurfacing — the differentiator.

P0 de-risks everything downstream by locking the data contract and identity before any rich UI.

---

## Success Metrics

- A new user can sign in, find a show, queue and play an episode with transcript sync and resume —
  end-to-end, on mobile — without reading docs.
- The currently-playing transcript segment is highlighted and in view with no perceptible lag; tapping
  any line seeks within 0.5s of the segment start.
- A highlight is captured in ≤1 interaction and appears in the user's personal corpus.
- "What have I learned about X" returns grounded answers citing only episodes the user has heard,
  with jump-to-moment links.
- The player is fully functional (audio + transcript) for episodes lacking GIL/KG artifacts — no broken
  panels.
- a11y: core listen→capture flow is fully keyboard- and screen-reader-operable (WCAG 2.1 AA target).

---

## Dependencies

- Existing read API + artifacts (RFC-049/055/072/090/097, `docs/api/HTTP_API.md`).
- A new per-user persistence store (introduced in PRD-036; technology chosen at RFC stage).
- OAuth provider registration (Google to start).
- Original-host audio availability per episode (enclosure URLs from source feeds).

## Constraints & Assumptions

**Constraints**

- Bridge-only audio: no third-party audio is stored or served (Principle 4).
- Shared corpus: no per-user forking of episode artifacts (Principle 3).

**Assumptions**

- Users have connectivity to the original podcast hosts for playback.
- The existing read API can serve per-episode artifacts by `slug` at interactive latency (validated in P0).
- Scrape-on-demand latency (minutes) is acceptable and surfaced as progress, not hidden.

---

## Resolved Platform Decisions

These four were open during drafting and are now decided (formerly OQ1–OQ4; child PRDs that cite
"OQ1–OQ4" map 1:1 to D1–D4). The throughline is **simplest thing that works, with a clean growth path** —
no premature infrastructure.

- **D1 — Web-first responsive PWA** (was OQ1). Ship a mobile-friendly, installable PWA; no native wrapper
  in v2.7. Rationale: covers mobile + a11y + i18n with zero app-store friction. Native shell + OS
  media-session is north-star (pairs with PRD-039 background audio).
- **D2 — Per-user state is stored as plain per-user files, reusing what the project already does; no
  persistence-layer work in this phase** (was OQ2). Per-user overlay state (library, playback, queue,
  highlights, notes, interests) is written as per-user files (JSON/JSONL under a per-user directory) with the
  existing helpers (`atomic_write`, `filelock`) — exactly the pattern `jobs.jsonl` already uses. **No new
  abstraction, no repository/interface layer, no schema.** This keeps the Principle 3 boundary physical:
  **shared corpus = artifact files; personal overlay = per-user files.** **A real persistence layer — a
  database and any abstraction/interfaces around it — is a separate, potentially large refactor, explicitly
  out of scope here** and re-assessed only after these requirements are locked. This phase is
  requirements-gathering, not persistence work.
- **D3 — A new consumer app at the repo top level, separate from `web/gi-kg-viewer`** (was OQ3). The
  consumer app lives as its own top-level project (not nested under the viewer), consumes the platform API,
  and reuses extracted UI primitives where sensible; the viewer stays operator-only. Rationale:
  comprehension — don't conflate operator and consumer concerns in one codebase.
- **D4 — Minimal scrape guardrail** (was OQ4). A config-driven per-user rate limit + concurrent-scrape cap
  at the scrape endpoint; global dedup already prevents duplicate work. No billing/quota subsystem now;
  the limiter is swappable for a richer policy later.
- **D5 — A dedicated consumer API surface, independent of the operator/viewer API.** The platform gets its
  own API namespace (e.g. `/api/app/*` or a separately-mounted app), shaped for the consumer client and
  versioned on its own, rather than overloading the existing operator routes (`/api/corpus/*`,
  `/api/relational/*`). It reads the same shared artifacts + per-user persistence underneath, but the contract is
  decoupled so the two surfaces evolve independently. Rationale: simplicity + comprehension + safe
  evolution — the operator API keeps serving the viewer untouched; the consumer API is free to be
  consumer-shaped (slug-addressed, auth-gated, BFF-style). This is the recommended direction; final
  namespace/mount mechanics are settled in the foundation RFC.
- **D6 — No request-time LLM; answer features are retrieval, not generation.** The Player's "ask about this
  episode" and Consolidation's "what have I learned about X" are served by the existing hybrid search layer
  (RFC-090) + relational traversal (RFC-072): they return verbatim grounded insights/quotes/highlights with
  jump-to-moment links, not synthesized prose. No LLM is called at request time, so the server needs no LLM
  provider/credentials and the no-LLM-in-CI concern disappears for these features. A generative summary
  layer on top is a parked future option. Rationale: simpler, cheaper, faster, and more trustworthy —
  nothing to hallucinate.

## Related Work

- `docs/prd/PRD-027-platform-discovery.md`, `PRD-029-platform-catalog.md`,
  `PRD-030-platform-player.md` — superseded drafts, folded into PRD-037/038/039.
- `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` — server-side multi-user lift + sequencing.
- `docs/guides/GIL_KG_CIL_CROSS_LAYER.md`, `docs/api/HTTP_API.md` — intelligence layer + API surface.

## Release Checklist

- [ ] Parent PRD reviewed and approved
- [ ] Child PRDs 036–041 drafted and cross-linked
- [ ] `segments.json` contract frozen and documented
- [ ] RFC(s) created for foundation (identity + per-user store) and the bridge/enclosure model
- [ ] a11y + i18n acceptance criteria defined before P1 UI work
