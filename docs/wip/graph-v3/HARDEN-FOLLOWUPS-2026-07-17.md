# graph-v3 harden follow-ups — 2026-07-17

The 2026-07-17 harden pass on `feat/graph-v3` flagged eight items in
"needs decision" plus seven follow-ups. Most landed as fixes on this
branch (HD1–HD9); this file tracks the three that were self-noted as
"untracked" inside other WIP docs, so they're findable by title
without spelunking through sub-sections. No GH issues opened per
operator's standing rule (memory: never open GH issues without
approval).

## FU1 — USERPREFS-1 three deferred items

Source: `docs/wip/USERPREFS-1.md`, section *"Not shipped (documented
follow-ups)"*.

1. **Per-pref localStorage key migration for other stores.** Theme,
   left/right panel open state, graph hints seen, graph bottom bar
   collapsed, graph theme legend collapsed are already synced. Not
   yet synced: any store that still owns its own localStorage key
   without a USERPREFS-1 write-through. When (if) other stores adopt
   cross-device sync, they follow the `graphLenses` pilot pattern.

2. **Import/export UX.** Pure client work: a "download my
   preferences JSON" + "restore from file" pair on the account
   settings page. No server API change needed — GET returns the
   payload; PUT replaces it. Deferred: the server API is done, the
   UI isn't.

3. **Conflict resolution for concurrent multi-tab writes on nested
   payloads.** `USERPREFS-1-CONFLICT-RESOLUTION.md` surveys the
   design ladder (SSE push → ETag optimistic concurrency → nested-key
   split → CRDT) and lands on "stay on today's design; no ladder step
   justified by observed user pain today". Kept explicit here so
   future PRs adding a new nested-payload preference know to
   re-evaluate before merging.

## FU2 — aggregatedEdges V1 enricher-gate

Source: `docs/wip/graph-v3/SUMMARY.md`, section *"Not done (still)"*.

The `ABOUT_AGG` / `SPOKE_IN_AGG` edges are a runtime roll-up inside
`toCytoElements`, not a per-corpus enricher artifact. That's why the
existing enricher-gated pattern (`artifacts.themeClustersDoc === null`
→ hide the row) doesn't apply. If we want to hide the lens row when
the roll-up would be empty, the check is a small computed over
`fullArtifact` counts (`countAboutAggEdges(fullArtifact) > 0 ||
countSpokeInAggEdges(fullArtifact) > 0`). Left explicit here so a
future PR that adds the computed doesn't accidentally duplicate the
themeClustersDoc-shape check.

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
