# graph-v3 harden follow-ups — 2026-07-17

The 2026-07-17 harden pass on `feat/graph-v3` flagged eight items in
"needs decision" plus seven follow-ups. Most landed as fixes on this
branch (HD1–HD9); this file tracks the three that were self-noted as
"untracked" inside other WIP docs, so they're findable by title
without spelunking through sub-sections. No GH issues opened per
operator's standing rule (memory: never open GH issues without
approval).

## FU1 — USERPREFS-1 three deferred items — 2/3 CLOSED

Source: `docs/wip/USERPREFS-1.md`, section *"Not shipped (documented
follow-ups)"*.

1. **Per-pref localStorage key migration for other stores** — **CLOSED
   2026-07-19.** Audit (2026-07-19) confirms every store using its own
   `localStorage` also has a USERPREFS-1 write-through EXCEPT `shell.ts`
   (`ps_corpus_path`, intentionally device-local — filesystem path)
   and — at audit time — `LibraryTab.vue` (view mode). The latter
   landed on this branch as a proper `libraryViewMode` store following
   the `graphLoadMode` pattern. No candidates remain.

2. **Import/export UX** — still deferred. Pure client work: a "download my
   preferences JSON" + "restore from file" pair on the account
   settings page. No server API change needed — GET returns the
   payload; PUT replaces it. The server API is done, the UI isn't.

3. **Conflict resolution for concurrent multi-tab writes on nested
   payloads** — **CLOSED as no-action.**
   `USERPREFS-1-CONFLICT-RESOLUTION.md` surveys the
   design ladder (SSE push → ETag optimistic concurrency → nested-key
   split → CRDT) and lands on "stay on today's design; no ladder step
   justified by observed user pain today". No adopter has landed a
   nested-payload preference that would trigger the ladder. When one
   does, the design note above is the launching pad. Nothing to
   implement here until then.

## FU2 — aggregatedEdges V1 enricher-gate — CLOSED

**Closed 2026-07-18.** `GraphLensesChip.aggregatedEdgesAvailable` in
`web/gi-kg-viewer/src/components/graph/chips/GraphLensesChip.vue` now
does the source-edge probe (any `ABOUT` OR `SPOKEN_BY` edge in
`artifacts.fullArtifact.data.edges`) instead of a per-corpus enricher
artifact, so the lens row hides when no roll-up is possible. Tests
added: `GraphLensesChip.test.ts` covers the 4 gate cases (no artifact
hides; ABOUT-only shows; SPOKEN_BY-only shows; unrelated edge types
hide).

## FU3 — Speaker + Quote shape live-verify

Source: `docs/wip/graph-v3/SUMMARY.md`, section *"Not done (still)"*.

The tier 4-M Speaker (round-diamond) and Quote (round-tag) shapes
are code-only verified — no corpus in the ops set currently emits
either type at scale. When a corpus with Speaker or Quote nodes
lands (candidates: post-diarization prod-v3 after the DGX sweep,
`#1170`; or a Quote-heavy interview corpus), one screenshot each of
the loaded canvas closes this item.

## Not tracked as a GH issue — why

Per operator memory rule
`feedback_never_open_gh_issues.md`: "NEVER create GH issues without
explicit operator approval; fold all surfaced work into the EXISTING
issue/branch in play. Operator alone decides when issues are
created." The 2026-07-17 harden pass surfaced these as WIP-only
notes; this doc consolidates them so they're findable, without
opening issues.
