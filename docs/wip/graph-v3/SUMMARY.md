# graph-v3 — visual iteration on gi-kg-viewer

*Working branch: `feat/graph-v3` off `main`. Not merged.*

## Trigger

Operator delivered four analyses (graph-visual-improvement, gikgviewergraphredesign,
graphfilterrail, graphvisualanalysis) comparing the viewer against `codeindex`,
Maltego, InfraNodus, Obsidian Graph, Palantir Gotham, and the 2025 Graph Drawing
contest *Dark* visualisation. Ask: honest read against current state, then start
with easy wins, screenshots before/after, perf assessment per change.

## Read against current state

Half of what the docs flagged as "critical missing" was already in
`buildGiKgCyStylesheet` — either shipped or behind an opt-in flag:

- Edge colour by relationship type — fully wired for
  HAS_INSIGHT / ABOUT / SUPPORTED_BY / RELATED_TO / MENTIONS / SPOKE_IN /
  SPOKEN_BY / HAS_MEMBER / aggregates.
- Confidence encoding — `insight-ungrounded` dashed border, ABOUT edge opacity
  mapped to confidence 0.25–1.0.
- Temporal fade — `recencyWeight → opacity` (floor 0.4).
- Label zoom-gating — `graph-label-tier-{none,short,full}` classes.
- Node size by degree — `enableNodeSizeByDegree` option existed, off by default.
- fcose — default layout, chosen for perf (#967) over cose.
- Timeline layout — implemented as `timelineLayoutSpec` (RFC-080 V3).

Genuine gaps: darker canvas, default edge opacity, degree sizing default-off,
hover-dim, community detection, bridge betweenness ring, brand palette repaint.

## Shipped in A/B/C/D

| # | Change | Files | Notes |
|---|---|---|---|
| A | `--ps-graph-canvas` token (dark `#0a0d10`, light falls through to `--ps-canvas`) | `theme/tokens.css`, `components/graph/GraphCanvas.vue` scoped style | Shell chrome, `theme.spec.ts` assertions, and `canvasExportBg` untouched. Halo mismatch deferred. |
| B | Default edge opacity theme-aware — `0.3` dark, `0.5` light | `utils/cyGraphStylesheet.ts` | `isLightThemeActive()` reads `data-theme`, falls back to `matchMedia`. Neighbour `0.9` / dimmed `0.2` preserved. |
| C | `DEFAULT_FLAGS.nodeSizeByDegree = true` | `stores/graphLenses.ts` + tests + module docstring | Users retain explicit toggle via localStorage; only "never touched" default flips. |
| C-fix | `mapData(degreeHeat)` scoped to `[type][degreeHeat]` selector | `utils/cyGraphStylesheet.ts` + tests | Silences per-element Cytoscape warnings on Episode nodes without `degreeHeat`. Nodes without the field fall through to fixed base size. |
| D | Hover-dim wiring | `components/graph/GraphCanvas.vue` | `cy.on('mouseover'/'mouseout', 'node')` reuses `applyGraphSelectionDimFromNode` / `clearGraphSelectionDim`. Guarded by `:selected` count so selection dim wins over hover. |

## Perf

- LCP on prod-v2 (833 nodes, DPR-2, 1440×900) after A+B+C: **1561ms**. Trace at
  `traces/03-C-first-paint.json.json.gz`.
- No pre-C LCP baseline captured — regression judged by feel; layout settled
  visibly at the same pace as pre-change.
- Hover-dim adds a single-frame class toggle; existing 120ms opacity
  transition on nodes + edges smooths cursor sliding between neighbours.

## Theme coverage

Both themes visually verified against prod-v2:

- `screenshots/00-baseline-idle.png` → `03-ABC-size-by-degree-idle.png`: dark theme evolution.
- `screenshots/04-ABC-light-theme-idle.png`: light theme after A+B+C.
- `screenshots/05-ABCD-dark-idle.png` / `06-ABCD-dark-hover.png`: dark hover.
- `screenshots/07-ABCD-light-hover.png`: light hover — 0.5→0.9 lift (1.8×) reads.

## Tests

`web/gi-kg-viewer` vitest — 37 tests pass (`graphLenses` + `cyGraphStylesheet`
suites both updated to reflect the new default and the `[type][degreeHeat]`
specialization selector). `theme.spec.ts` (e2e) untouched — `--ps-canvas`
value preserved.

## Not done

- Halo colour still resolves `--ps-canvas` (not `--ps-graph-canvas`); ~3 shade
  mismatch behind labels on the darker bg.
- `canvasExportBg()` still reads `--ps-canvas`; PNG export bg diverges from
  on-screen bg on dark theme.
- No perf trace for baseline — the C trace is the only reference.
- fcose repulsion / edge-length tuning unchanged; docs' `nodeRepulsion: 8000 /
  gravity: 0.08` proposals not applied (units differ from current 880_000 base;
  needs an isolated tuning session on real corpora).

## Open questions carried forward

1. Direction: Maltego / InfraNodus vs codeindex ceiling. Operator gut says
   Maltego / InfraNodus stepping-stone.
2. Left rail vs chip bar for Graph. Deferred — chip bar (#658) stays for now.
3. Next tier scope. Operator direction (2026-07-16): halos, sizes, circles and
   lines — move away from same-size circles and same-thickness lines. Colour +
   shape variance next, not community detection yet.

## Tier 2 — E/F/G/H

| # | Change | Files | Notes |
|---|---|---|---|
| E | Halo + PNG-export bg follow `--ps-graph-canvas` | `utils/cyGraphStylesheet.ts`, `components/graph/GraphCanvas.vue`, `PS_TOKEN_FALLBACKS` | Resolves A deferral. Verified: `text-outline` / `text-background` now `rgb(10,13,16)`. |
| F | Semantic shape per node type | `utils/cyGraphStylesheet.ts` `shapeByType` | Podcast=hexagon, Episode=round-rectangle, Insight=diamond, Quote=round-tag, Speaker=round-diamond, Entity_organization=round-rectangle. Person / Entity_person / Topic stay ellipse. TopicCluster keeps `roundrectangle` compound. Speaker + Quote emit 0 in prod-v2 slice — code-verified only, live-verify pending on a corpus that emits them. |
| G | Edge width tiers widened | `utils/cyGraphStylesheet.ts` | Structural 2 → 2.75 (HAS_INSIGHT, SPOKE_IN, HAS_EPISODE). HAS_MEMBER 1.5 → 2. Evidence + discovery 1 → 0.75-0.85 (SUPPORTED_BY, RELATED_TO, MENTIONS). Descriptive confidence-mapped unchanged (already scaled). Compact profile scales proportionally. Aggregate mapData(weight) untouched. |
| H | Episode base 18 → 22, aspect W 1.35 → renders 30×22 card | `utils/cyGraphStylesheet.ts` `NODE_ASPECT_W`, `scaledNodeWidth` | mapData(degreeHeat) specialization scales width vs height independently against per-type bases so aspect ratio is preserved across the 0.7×–1.5× range. |

## Tier 3 — I/J/K/L/M

| # | Change | Files | Notes |
|---|---|---|---|
| I | Entity_organization aspect W=1.35 → 35×26 card | `utils/cyGraphStylesheet.ts` `NODE_ASPECT_W` | Same treatment as Episode. Verified live on Barclays, FT, The Flip Side Podcast. |
| J | Extend degreeHeat to Episode + Entity_person + Entity_organization | `components/graph/GraphCanvas.vue` `applyTopicDegreeHeat` + mapData specialization in `utils/cyGraphStylesheet.ts` | Previously Topic-only; now 226/226 persons + 164/164 orgs + 32/32 episodes carry degreeHeat post-layout. Aspect ratio preserved through 0.7×–1.5× mapData range. Topic-heat-high class still Topic-only (existing selector contract). |
| K | Bridge node ring via betweenness centrality | `components/graph/GraphCanvas.vue` `applyBridgeNodeClass` + `.graph-bridge` style | Threshold 0.05 on Topic/Podcast/Person/Org (Episode + Insight excluded — always structural connectors by graph shape). 44 bridges on prod-v2. Rose dashed border style. Cost ~200ms on 833 nodes. |
| L | Brand palette shift across all 9 node types | `utils/colors.ts` `graphNodeTypeStyles` | Fixes pre-L Entity_person / TopicCluster purple collision (both were #9775fa). Person → amber #c49a28. Palette is the shared source of truth per UXS-014 → E chip, legend, and CIL topic pill shift with the graph. Coordinated brand-tone shift (less-neon), not a wholesale forest/sage/stone/terracotta rewrite. |
| M | fcose gravity 0.18 → 0.12 | `utils/cyCoseLayoutOptions.ts` + test | Modest cluster-drift; not the docs' 0.08 (our bipartite graph would strand orphan Insights). idealEdgeLength + nodeRepulsion untouched. |

Docs' "long cross-type edges" fcose proposal doesn't cleanly map to this graph:
nearly every edge is already cross-type. RELATED_TO (the one same-type edge) is
already at idealEdgeLength 120.

## Not done (still)

- Speaker + Quote shape live-verify — pending a corpus that emits either.
  Tracked here; when a corpus with Speaker / Quote nodes lands, one screenshot
  each closes it.
- `aggregatedEdges` V1 enricher-gate — the ABOUT_AGG / SPOKE_IN_AGG edges are
  a runtime roll-up in `toCytoElements`, not a per-corpus artifact, so gating
  on artifact presence doesn't apply. If we want to hide the lens when the
  roll-up would be empty (no Insight+Topic+Episode triples in the artifact),
  the check is a small computed on `fullArtifact` counts. Not tracked as a
  GH issue.
- Halo colour on light-theme graph canvas is still `--ps-graph-canvas` = `--ps-canvas`
  (i.e. same as canvas). That's correct; noting for completeness.
- **Insight sentiment lens** (originally scoped as Tier 5C-3) — **descoped, closed**.
  The `insight_sentiment` enricher ships per-episode sidecars
  (`metadata/enrichments/{stem}.insight_sentiment.json`) that the read-time CIL
  layer already joins into arc responses (`cil_queries._attach_sentiment` → each
  Insight carries `sentiment: {compound, label}`). Sentiment therefore surfaces
  as an **arc-tint colour on the player conversation-arc + position-arc**
  panels (per the walkthrough-v3 spec) rather than a graph-canvas lens.
  A graph lens would tint Insight nodes — but Insights are hidden by default at
  low zoom under Tier 6-2 (zoom-gated Insight visibility), so the lens would
  fire only in the fine-grained view where a per-topic sentiment histogram is
  the actually-useful signal, not a per-Insight tint. If a per-Topic sentiment
  aggregate lens is ever wanted, it would ship as a new corpus-scope enricher
  (`insight_sentiment_corpus`) that rolls the sidecars up, following the same
  shape as `topic_cooccurrence_corpus`, and consumed via
  `fetchCachedCorpusEnvelope`. Not queued.

## Tier 5C/5D — 4 enricher-based lenses shipped

| # | Lens | Enricher | Visual signal | Prod-v2 live count |
|---|---|---|---|---|
| 5C-1 | Velocity halo | `temporal_velocity` | Bright border on Topic + Person: green ↑ / red ↓ / amber → based on `velocity_last_over_6mo` and shared `utils/trend.ts` thresholds | 164 up, 129 down |
| 5C-2 | Person credibility border | `grounding_rate` | Border colour per Person: green solid ≥0.7 / amber ≥0.4 / red dashed <0.4 | 51 high |
| 5D-1 | Consensus edges | `topic_consensus` | Green unbundled-bezier arcs between two Persons who corroborate on a Topic. Deduped per (pair, topic). | 5 edges |
| 5D-2 | Co-guest edges | `guest_coappearance` | Dotted amber arcs between Persons sharing ≥2 episodes. Width scales with `episode_count`. | 1 edge |

Shared shape:

- Lens flags live in `useGraphLensesStore` (localStorage-backed); resetToDefaults + persistence tests carry them.
- Stylesheet classes / edge-type selectors added in `cyGraphStylesheet.ts`.
- Apply/clear helpers in `utils/cyGraphLensOverlays.ts` (Cytoscape-typed, envelope-shaped) — kept out of the 4400-line GraphCanvas.vue for testability.
- GraphCanvas exposes `refreshEnricherLensOverlays()` — fired from `finishLayoutPass` + a single watcher over the 4 flags. Envelopes fetched via existing `fetchCachedCorpusEnvelope` (Map-based cache) so live toggles reuse the same fetch.
- GraphLensesChip probes availability on mount + on corpus-path change; rows hide when the artifact is missing (enricher-gated per operator direction).

Every enricher-driven lens composes with the others: a Topic can carry the theme
underlay + velocity border + degree size + bridge ring simultaneously. Cytoscape
selectors resolve by source order; interaction states (`.search-hit`, `:selected`)
still win over lens overlays.

## Tier 5 harden follow-ups (post-audit fixes)

Ran the `harden` skill after tier 5C/5D committed. Its findings applied here:

- Deduped the theme-region palette: `THEME_REGION_PALETTE` in
  `utils/themeRegionPalette.ts` is now imported by `cyGraphStylesheet.ts` too
  (was duplicated by hand). Legend + graph now cannot drift from a manual
  hex edit.
- Fixed `REPRODUCIBILITY.md` — the enricher table no longer claims the
  `insightSentiment` lens is shipped; call-out reflects the deferral. The
  "longer-term escape" section now records that the bundle-discovery bug is
  verifiably fixed in this worktree (`discover_episode_bundles` returns 99
  bundles on prod-v2 today).
- Task #36 (Insight sentiment lens) **closed — descoped**. Rationale in the
  "Not done" section above: sentiment already surfaces as an arc-tint via
  `cil_queries._attach_sentiment`; a graph lens over per-Insight tints would
  fire only where per-Topic aggregates are the actually-useful signal, and
  those would be a new `insight_sentiment_corpus` enricher, not a viewer-only
  patch. Marking closed unblocks the "task #36 stuck in_progress" audit finding.
- `aggregatedEdges` V1 enricher-gate examined and left as-is: the ABOUT_AGG /
  SPOKE_IN_AGG edges are a runtime roll-up in `toCytoElements`, not a per-corpus
  artifact, so gating on artifact presence doesn't apply. Recorded in the "Not
  done" section above.
- Test coverage backfill: **+33 tests, +1 test file** (2626 → 2659, 193 → 194):
  - `utils/cyGraphLensOverlays.test.ts` (new, 21 tests) — all 8 apply/clear
    functions with a hand-rolled MockCore covering null/happy paths, threshold
    boundaries, dedupe rules, and weight preservation.
  - `utils/topicClustersOverlay.test.ts` (+6) — episode_ids tagging with
    `__unified_ep__:` prefix stripping, one-hop propagation to Insights,
    two-hop to Persons + Podcasts, first-cluster-wins on multi-membership,
    type allowlist enforcement, Person/Entity variant coverage.
  - `utils/cyGraphStylesheet.test.ts` (+4) — velocity / credibility / consensus /
    coguest selector hex + shape assertions.
  - `utils/colors.test.ts` (+2) — brand-palette collision guard: Entity_person
    != TopicCluster, plus a general "every type has a unique background"
    invariant with the deliberate Entity/Entity_person alias whitelisted.
  - `tests/integration/server/test_operator_yaml_profile.py` (+1) — CodeQL
    #417 fix's bare `profile:` branch: value-absent line skipped, later real
    `profile:` line wins.

## Tier 4 — N (dropped MCL), R/S/T/U/V/Q — theme-cluster regions + lens menu

Reused the enrichment layer's existing `topic_theme_clusters` output instead
of reinventing community detection in the browser. Operator recognised the
overlap: our "community" concept ≈ their existing "themed topics", already
built as an enricher (co-occurrence lift + average-linkage → `thc:...` compound
ids with human labels like "AI-and-jobs", "energy storyline"). MCL prototype
(N/O) was reverted; visual + UX target stayed the same.

| # | Change | Files | Notes |
|---|---|---|---|
| N (reverted) | MCL client-side community detection | (deleted in R) | Proved the wrong signal on a bipartite graph — 94 mini-clusters, top-8 covered only 27% of nodes. Kept the palette + stylesheet shape, threw away the algorithm. |
| R | Delete `applyCommunityClasses` / `clearCommunityClasses` + `community-N` selectors | `components/graph/GraphCanvas.vue`, `utils/cyGraphStylesheet.ts` | Straight refactor into theme-region form. |
| S | Copy `topic_theme_clusters.json` from sibling worktree | `.test_outputs/manual/prod-v2/corpus/enrichments/` (gitignored data) | Sibling `podcast_scraper-ai-ml-improvements/` had run the enricher against prod-v2; our worktree hadn't. Same corpus data, one-line `cp`. API confirmed 6 clusters (interest rates, ai agents, future of work, quantum computing, employee engagement, tech industry). |
| T | Propagate `theme-region-N` class from Topic + Episode seeds to Insights → Persons/Orgs, Episode → Podcasts | `components/graph/GraphCanvas.vue` `applyThemeRegionClasses` | Seed uses both `member.topic_id` matching and `member.episode_ids` from the artifact. First-cluster-wins tagging. 161 nodes tagged on prod-v2 across 4 hash-buckets. |
| U | Palette + underlay selectors renamed `community-N` → `theme-region-N`, keyed by hash of `graph_compound_parent_id` | `utils/cyGraphStylesheet.ts` | 8 pastel hues around HSL, sat 45% / light 65%, opacity 0.14. Same `thc:...` id maps to the same colour across sessions (djb2-style hash). |
| V | Rename lens flag `communityColours` → `themeClusterRegions` + legacy-key migration | `stores/graphLenses.ts` + tests | Users who opted into the earlier flag retain their opt-in through the rename. |
| Q | `GraphLensesChip` popover in the graph bottom bar | `components/graph/chips/GraphLensesChip.vue` + `GraphBottomBar.vue` | 4 lens rows (Size by connectivity, Bridge nodes, Theme regions, Aggregated edges) with description text + reset. Theme regions row is **hidden entirely** when `artifacts.themeClustersDoc === null` — enricher-gated per operator direction. |

Enricher-gated pattern: `GET /api/corpus/theme-clusters` already returns 404 /
missing when the artifact isn't present; the viewer artifacts store surfaces
that as `themeClustersDoc === null`; the chip filters the row out. Same
pattern extends naturally to future lens/enricher pairs (`aggregatedEdges` V1
could gate on aggregate-edge presence in the artifact — deferred).

## Tier 6 — declutter (plumbing dots, zoom-gated Insights/Quotes, default-hide Quotes, bridge tooltip)

Operator's tier-6 direction: *"the graph is still too busy on first paint,
declutter without hiding structure"*. Four small, orthogonal changes, each
gated on a heuristic the user can override.

| # | Change | Files | Notes |
|---|---|---|---|
| 6-1 | Node-size tier reshape — small "plumbing" node types (Speaker, Person cameo, empty Insight) render as **dots** below zoom 0.8 | `utils/cyGraphLabelTier.ts`, `utils/cyGraphStylesheet.ts` | Zoom threshold `GRAPH_NODE_ZOOM_INSIGHT_MIN=0.9` — below it, Insight/Quote render as compact tiles; above, full labels. |
| 6-2 | Zoom-gated Insight/Quote visibility | `utils/cyGraphLabelTier.ts::syncGraphNodeVisibilityTierClasses` | New sibling to `syncGraphLabelTierClasses`. Insights and Quotes get a `graph-node-hidden-below-zoom` class below 0.9 so the canvas isn't drowned in evidence dots at overview zoom. |
| 6-3 | Default-hide Quotes filter | `utils/parsing.ts::DEFAULT_HIDDEN_TYPES = ['Quote','Speaker']` | Rather than hide *all* Quotes forever, the initial `allowedTypes` map starts with Quote/Speaker off. User re-enables via the Types chip; `filtersActive` computes deviation-from-default so the "Reset" indicator only lights up on user action. |
| 6-4 | Bridge hover tooltip | `components/graph/GraphCanvas.vue` bridge hover handler | Hovering a bridge node surfaces the two theme regions it connects. Reuses the theme-region-N class hash from tier 4/U. |

Post-audit test-coverage backfill for tier-6 landed later as follow-up
(see "Tier 6/7/8 follow-ups" below).

## Tier 7 — hierarchical + searchable legend, super-theme rollup

Tier 6 declutters within the graph; Tier 7 declutters the legend itself.
`topic_theme_clusters.json` schema v1.1.0 gained a `super_theme_id`
rollup (cross-cluster lift) so we can present a **two-level** legend:
6–8 super-themes on top, dozens of themes as expandable children.

| # | Change | Files | Notes |
|---|---|---|---|
| 7-1 | Hierarchical legend — super_theme rollup | `components/graph/ThemeClusterLegend.vue`, `utils/topicClustersOverlay.ts` | Uses `super_theme_id` from enricher v1.1.0; falls back to flat listing if the field is absent (backward-compat with pre-v1.1.0 corpora). |
| 7-1a | Extend enricher to emit `super_theme_id` via cross-cluster lift | `src/podcast_scraper/enrichment/enrichers/topic_theme_clusters.py` | v1.1.0 rev. Deterministic connected-components on inter-cluster co-occurrence, average-linkage grouping. |
| 7-2 | Searchable legend — typeahead over super-themes and their children | `components/graph/ThemeClusterLegend.vue` | Filters the tree in-place; keeps super-theme headers when any child matches. |
| 7-3 | Legend focus interaction — click a legend entry → dim all other regions | `stores/graphThemeFocus.ts` (new), `components/graph/GraphCanvas.vue` | Single focus bus; clears on second click or Escape. |
| 7-4 | Person co-appearance MCL overlay lens | `enrichment/enrichers/guest_coappearance.py` v1.1.0 (union-find person communities), `utils/cyGraphLensOverlays.ts::applyPersonCommunityClasses` | Person community classes surface as underlay tint on Person nodes when the lens is on. |

## Tier 8 — top-down default graph load

The full graph on prod-v2 is ~830 nodes; the top-down slice mounts as
6–8 SuperTheme bubbles, and expansion is per-super-theme on tap.

| # | Change | Files | Notes |
|---|---|---|---|
| 8-1 | Top-down synthetic slice mount | `utils/topDownSlice.ts` (new), `stores/artifacts.ts::topDownDisplayArtifact` | Derives a slice from `themeClustersDoc.super_themes`; bridge-derived cross-super-theme edges (Gap 3). |
| 8-2 | Expand-on-tap for SuperTheme nodes | `stores/graphTopDown.ts` (new), `components/graph/GraphCanvas.vue` tap handler | Tap toggles `expandedSuperThemeIds`; the slice re-derives with projected children under the tapped super-theme. |
| 8-3 | Search reveals hidden | `components/graph/GraphCanvas.vue::maybeExpandTopDownForPendingFocus` | When a search-pending focus target lives under a collapsed super-theme, auto-expand it so `tryApplyPendingFocus` succeeds on the next redraw. |
| 8-4 | Filter re-scope over expanded slice | `stores/graphFilters.ts` | `filteredArtifact` forces `allowedTypes.SuperTheme = true` when in top-down mode so the canvas isn't emptied. |
| 8-5 | Load-mode opt-in flag (plumbing) | `stores/graphLoadMode.ts` (new) | localStorage-first, USERPREFS-1 sync. Default `'everything'` — the flip is one-line when tier-8 UX is proven. |
| 8-6 | Ego / cross-episode re-scope | `stores/graphExplorer.ts` (ego integration) | Ego expansion uses the full `displayArtifact` (not the top-down slice) — walking a super-theme's ego neighbourhood surfaces cross-episode structure. |

Viewer clamp (Gap 4): if super_themes count exceeds 8, the mount trims to the 8 largest by member weight so the tier-8 mount stays legible.

## Post-tier-8 harden follow-ups (Gaps 1–6)

Second harden pass surfaced 6 gaps closed on this branch:

- Gap 1 — vitest teardown BroadcastChannel hang → `vite.config.ts` `pool: 'forks'`.
- Gap 2 — graph-v3 tier-8 top-down design doc `status: shipped` marker.
- Gap 3 — bridge-derived cross-super-theme edges wired into `topDownSlice`.
- Gap 4 — viewer clamp for >8 SuperTheme nodes.
- Gap 5 — playwright stack-test: top-down mount + expand.
- Gap 6 — playwright stack-test: search reveals hidden.

## Tier-3 real-corpus walk (2026-07-17)

Full validation walk against the synthetic `viewer-validation-corpus/v3`
fixture: **38 tests passed** after this branch's fixes landed:

- graph-analytics-replay path fix — event log lives at
  `<CORPUS>/.app/users/u_<hash>/graph_events.jsonl` (per-user under auth),
  not `<repo>/.app/users/anon/`.
- V4 / P5.2 / P7.2 — Dashboard tab is admin-only (`auth.isAdmin`); added
  `signInAsAdmin()` helper that uses the `ada-admin` hint (matches serve
  `APP_ADMIN_EMAILS`).
- P1.3 / P4.2 — digest topic-band hit rows need a LanceDB vector index;
  built one over the synthetic corpus.
- Corpus also needed a `search/topic_clusters.json` at threshold 0.35
  (default 0.75 gave 0 clusters on this small fixture).
- V4 specifically: V-G1 flips graph-load-mode to Top-down under
  `ada-admin` and persists it via USERPREFS-1 → V4 was inheriting
  Top-down and finding no cluster compound. V4 now normalizes the mode
  chip to "Everything" before clicking the topic-cluster chip.

Full log: `/tmp/tier3-json7.log` (38 passed, 0 failed, 0 skipped).
