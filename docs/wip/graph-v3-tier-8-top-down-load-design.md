# graph-v3 tier 8 — top-down default graph load (design)

**Status:** shipped 2026-07-17 (all six sub-tiers landed on `feat/graph-v3`)
**Owner:** operator (Marko)
**Branch:** `feat/graph-v3` (same as tier 6 + tier 7 stack)
**Depends on:** tier 7 shipped (super-theme rollup lives in
                `topic_theme_clusters` v1.1.0 + hierarchical legend)

## Sub-tier commit map

| # | Commit | Sub-tier |
|---|---|---|
| 8-5 | `c32ac8f1` | load-mode opt-in flag (plumbing) |
| 8-1 | `cfe3a49c` | top-down synthetic slice mount |
| 8-2 | `24ab3fa4` | expand-on-tap for SuperTheme nodes |
| 8-3 | `07f24096` | search reveals hidden |
| 8-4 | `422457e6` | filter re-scope over expanded slice |
| 8-6 | `9e304d39` | ego + cross-episode re-scope |

The `DEFAULT_MODE` constant in `web/gi-kg-viewer/src/stores/graphLoadMode.ts`
still ships as `'everything'` per the design's "flip after 8-2 stabilises"
rule. Flipping it to `'topDown'` is a one-line change once the branch has
been kicked-the-tires in a real corpus.

---

## Intent

Change the default graph mount from "load everything and declutter" to
"load 10 super-themes, expand on demand". The Palantir / InfraNodus
pattern: the user's first impression is a **story map** (5-8 bubbles
you can read), not a hairball. Every current declutter tier (6, and
most of 5) becomes redundant for the default view — they still matter
when the user opts back into "show everything".

## What ships (scope)

| # | Item | Rough size |
|---|------|------------|
| 8-1 | Load model — mount only super-theme nodes + inter-community bridges | M |
| 8-2 | Expand-on-tap — click a super-theme → inject its child clusters + tagged nodes | M |
| 8-3 | Search reveals hidden — search result triggers expand-chain up to the target node | S |
| 8-4 | Filter over hidden slice — types filter reasons over "expanded + hidden" separately | M |
| 8-5 | "Show everything" opt-in — lens chip / kbd shortcut to fall back to full-load view | S |
| 8-6 | Ego / cross-episode expansion — re-scope so incremental loads compose with top-down | M |

Total ~= tier 6 + tier 7 combined.

## What does NOT ship

- Enricher changes. The super-theme data ships in `topic_theme_clusters`
  v1.1.0 already; we just consume it differently on the client.
- A new Cytoscape layout algorithm. fcose still works — the initial
  10 nodes just have inter-community edges (bridge-derived) instead
  of the full mesh.
- Automatic re-collapse. Once the user has expanded a super-theme,
  it stays expanded for the session; collapse is manual (via the
  legend chevron) or via "reset view".

## Data model

The default view is a **synthetic slice** of the display artifact:

```
nodes:
  10 × TopicCluster nodes labelled `super_theme_id` (compound-parent)
  ~5 × bridge Persons/Orgs/Podcasts that span >=2 super-themes
edges:
  aggregated "co-super-theme" edges from `topic_theme_clusters`
    (weight = mean cross-cluster lift, already in the enricher output)
```

Two representations:

- **Full artifact** — always fetched + kept in the artifacts store as
  today. `displayArtifact` gets a new `topDownSlice` mode.
- **Top-down slice** — a new `topDownDisplayArtifact` computed that
  keeps only the synthetic super-theme nodes + edges + whatever the
  user has expanded so far.

## Expand semantics

Click a super-theme node → inject:
1. Every TopicCluster (`thc:…`) under that super-theme, as a compound
   child of the super-theme node.
2. Every Topic tagged with that theme (already via themeClusterId
   propagation from tier T).
3. One hop of Insights / Persons that connect to those Topics.

Re-clicking a super-theme collapses it back. State kept in a small
Pinia store (`graphTopDown`): `expandedSuperThemeIds: Set<string>`.
Persists via USERPREFS-1 for cross-device muscle memory.

## Interactions that need work

### 8-3 Search reveals hidden

Current: search result → focus on the node in the graph. Assumes
the node is loaded.

Top-down: if the target node is under a collapsed super-theme, the
search action must:
1. Compute the target's super-theme id (from the node's
   `themeClusterId` → cluster's `super_theme_id`).
2. Add it to `expandedSuperThemeIds`.
3. Fire the existing focus/pan.
4. If the node is a Person or Insight, walk one edge to find the
   Topic that determines the super-theme.

If the target has no super-theme (singleton Topic, isolated Person,
etc.), the top-down slice never had a home for it — the search
action falls back to auto-triggering "show everything" for one
session.

### 8-4 Filter over hidden slice

Current: types filter operates on the loaded slice. Toggling `Insight`
off hides Insight nodes; toggling it back on shows them.

Top-down: the "loaded slice" is now the expanded slice. Turning on
`Insight` when nothing is expanded is a no-op — there are no Insights
in the synthetic super-theme view. Options:

1. **Filter over hidden**: types filter chips show counts from the
   FULL artifact ("Insight (384)"), and toggling on expands every
   super-theme that has ≥1 Insight. Powerful but noisy.
2. **Filter over expanded**: types filter counts only from the
   currently-expanded slice. Simpler; matches user's mental model
   ("this filter operates on what I'm looking at"). Search-reveals-
   hidden covers the "I know a specific Insight I want to see" case.
3. **Two-tier filter**: chip counts show `expanded / total` (`3 / 384`),
   clicking the chip toggles visibility on the expanded slice, click-
   with-shift auto-expands to reveal all matching nodes.

Recommendation: **option 2** for tier 8-4. Option 3 is a nice
follow-up.

### 8-5 "Show everything" opt-in

New lens chip: `Load-mode: Top-down ▾` → dropdown with
`Top-down (default)` / `Load everything`. Persists via USERPREFS-1.
Kbd shortcut `Shift+E` to toggle in-session.

When the user picks "Load everything", the current tier 6 declutter
still applies (plumbing dots + zoom-gated Insights + Quotes hidden).
So the full-load view isn't a regression — it's the graph the user
had before tier 8.

### 8-6 Ego / cross-episode expansion re-scope

Existing "ego expansion" (double-tap a node to expand its 1-hop
neighbourhood) needs to consider the current mode:

- **Top-down mode**: ego-expand pulls the node's neighbourhood into
  the expanded slice, marks the node as "manually expanded". These
  survive super-theme collapse.
- **Full mode**: no change.

Cross-episode auto-merge (`maybeMergeClusterSiblingEpisodes`) fires
only when the user has explicitly loaded an episode. In top-down
mode, until the user expands a specific cluster, no episode is
loaded → auto-merge stays quiet. Good.

## Tier 6 declutter status in top-down view

Tier 6-1 (plumbing dots), 6-2 (zoom-gated Insights), 6-3 (default-hide
Quotes) all become **no-ops** in top-down mode (no plumbing / no
Insights / no Quotes in the synthetic slice). They kick in the moment
the user expands something OR flips to "Load everything". No change
needed — the existing conditionals just don't match anything.

Tier 6-4 (bridge tooltip) still fires — the ~5 bridge nodes in the
synthetic slice ARE bridges by definition.

## Rollback plan

If top-down causes a regression, the "Load everything" opt-in becomes
the default via a one-line change in `graphLenses.ts` (or a
`localStorage` migration). No data loss — the full artifact is always
in memory.

## Sequencing inside the branch

1. WIP note lands (this file).
2. **8-5 opt-in flag first** — ship the "Load-mode" chip as a no-op
   toggle. Both settings behave identically at this point. Purpose:
   land the plumbing (store field, USERPREFS-1 sync, chip UI, kbd
   shortcut) so subsequent PRs can wire behaviour to it without
   discussion.
3. **8-1 mount** — implement the top-down slice + the initial 10-node
   render. Behind the opt-in.
4. **8-2 expand** — the expand-on-tap semantics + the
   `graphTopDown` store.
5. **8-3 search-reveals-hidden** — search wiring.
6. **8-4 filter re-scope** — option 2 (filter over expanded).
7. **8-6 ego re-scope** — smallest of the six; ships last.

Each step keeps the graph shippable — worst case, the opt-in stays
off and no user-visible behaviour changes.

## Open questions

- Should the initial slice be `super_theme` only, or `super_theme +
  bridge nodes`? Bridges add analytic story ("this podcast bridges
  finance and AI") but also visual noise. Recommendation: include
  bridges when `bridgeRing` lens is on; hide them otherwise. That
  makes tier 7-3 focus-on-click compose naturally.
- What's the interaction between top-down and the theme legend
  focus (tier 7-3)? If the user clicks a super-theme in the legend
  while top-down is active, does that:
  (a) expand the super-theme AND focus (current behaviour of legend
      click assumes graph is loaded), or
  (b) just focus the synthetic super-theme node without expanding?
  Recommendation: **(a)** — expand + focus. Matches "I want to see
  this bit of the story".
- Should the "Load everything" opt-in preserve the user's tier 6
  declutter state, or auto-enable it? Recommendation: preserve
  whatever it was before top-down was enabled. If the user turned
  off zoom-gating before top-down, they meant to.

## Non-goals

- Not writing a proper community-of-communities enricher (would let
  us go one level up above super-themes, giving 2-3 mega-themes).
  Interesting for very large corpora; out of scope until we see
  a corpus large enough to need it.
- Not building a per-super-theme temporal signal (sparkline on the
  super-theme node showing episode cadence). Was considered for
  tier 7-4; parked.
