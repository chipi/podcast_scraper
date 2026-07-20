# UXS-016: Query Workspace (operator viewer)

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) — shared tokens, typography, layout, states
- **Related PRDs**:
  - [PRD-045: Search v3 — Query Workspace](../prd/PRD-045-search-v3-query-workspace.md)
- **Related RFCs**:
  - [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md) — technical design
  - [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) — shipped `activeSearchContext` + `PanelRetrievalStore` (composed with, not replaced)
  - [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — QueryEnricher (Enriched Answer source)
  - [RFC-093](../rfc/RFC-093-litm-context-packs.md) — `build_briefing_pack` (Compare operator source)
  - [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) — hybrid retrieval
  - [RFC-092](../rfc/RFC-092-ml-query-router.md) — intent taxonomy
- **Related ADRs**:
  - [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) — `topic_consensus` (Consensus operator source)
  - [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) — saved-queries persistence
- **Related UX specs**:
  - [UXS-005](UXS-005-semantic-search.md) — retains compact-launcher role after Workspace lands
  - [UXS-008](UXS-008-enriched-search.md) — Enriched Answer surface (heroified here)
- **Shell IA**: [VIEWER_IA.md](VIEWER_IA.md) — Search as 5th main tab

---

## Summary

The **Query Workspace** is the operator viewer's Search main tab (5th tab: Digest · Library · **Search** · Graph · Dashboard). Full-width main-area surface. Hosts the merged Search + Explore query UI, the enriched-answer hero, result-set operators (cluster / on-graph / timeline / compare / consensus), and the saved-queries sidebar (via LeftPanel). Also the target of the shell-wide Cmd-K palette's "Open in Workspace" action and every rail's "Search inside this X" launcher.

Detail per-surface is in [UXS-005](UXS-005-semantic-search.md) (**Revised in §S4-shell:** compact launcher retired; LeftPanel hosts Saved + Recent only on all tabs) and [UXS-008](UXS-008-enriched-search.md) (Enriched Answer visual contract, promoted from Advanced-dialog-gated to hero placement here).

---

## Placement

Full-width main area (`w-full`, no `w-72` constraint). Right subject rail persistence rule from VIEWER_IA.md holds: opening the Search tab does NOT clear the current subject. Subject rail sits to the right of the Workspace as on any other main tab.

**Revised in §S4-shell (2026-07-20):** The LeftPanel is **visible on ALL main tabs** (Digest / Library / Search / Graph / Dashboard). On the Search tab, it hosts the Saved + Recent query sections (collapsible right-edge pattern, §Sidebar below). On other tabs, it remains visible for these sections, supporting the compact-launcher patterns those tabs use — see UXS-005 for the launcher integration. The compact SearchPanel launcher that previously occupied the LeftPanel on non-Search tabs has been retired; all query surfaces now route through the Workspace (Search tab) or the Cmd-K palette.

---

## Component tree

```text
SearchTab.vue                                    [w-full main area]
├─ WorkspaceHeader.vue
│    ├─ QueryField.vue                           input; Preset dropdown; Enter=submit
│    ├─ IntentChip.vue                           search-query-type (RFC-092 label)
│    └─ SearchFilterBar.vue                      Since | Top-k | Doc types | Topic contains | Speaker contains | Min confidence | Grounded only | Enriched | More (feed / embedding model)
├─ EnrichedAnswerHero.vue                        UXS-008 contract; visible when enriched_search_available and enrich_results=true
├─ ResultSetOperatorBar.vue                      Cluster | On graph | Timeline | Compare | Consensus
└─ WorkspaceResults.vue                          scrollable
     ├─ ResultCard.vue                           tier badge (Insight / Transcript / Reference) + compound "+ insight" badge
     ├─ CompoundCard.vue                         when hit has `lifted` (RFC-072 KL1)
     └─ ClusterGroupCard.vue                     when operator === 'cluster'
```

**Revised in §S4-shell:** `WorkspaceSidebar.vue` is now hosted in the LeftPanel (not as a child of SearchTab), supporting the pattern that LeftPanel is visible on all main tabs. See §Sidebar below for details.

---

## Header

`WorkspaceHeader` sits at the top of the Workspace. Rows:

1. **Query row**: `QueryField.vue` (`min-w-0 flex-1`, sm text, Enter = submit, Shift+Enter = newline, IME-safe — same rules as UXS-005). Preset dropdown to the left of the field (`Preset: <label> ▾` when active; `Preset ▾` default). Search button + Clear button to the right.
2. **Intent row**: `IntentChip.vue` when the response carries `query_type` (RFC-092 label — Entity lookup / Raw evidence / Temporal tracking / Cross-show synthesis / Semantic). Muted; transparency-only. `data-testid="search-query-type"` (unchanged from UXS-005).
3. **Filter chip bar**: `SearchFilterBar.vue`. Chips left-to-right: **Since** (`search-chip-since`), **Top-k** (`search-chip-topk`, default 10), **Doc types** (`search-chip-doctypes`), **Topic contains** (`search-chip-topic-contains`, new — from merged Explore), **Speaker contains** (`search-chip-speaker-contains`, new), **Min confidence** (`search-chip-min-confidence`, new), **Grounded only** (`search-chip-grounded-only`, new), **Enriched** (`search-chip-enriched`, when enrichment configured), **More** (`search-chip-more` — hosts low-traffic fields: feed, embedding model, merge-duplicate-KG-surfaces). Each chip's label switches from `Label ▾` (default) to `Label: detail ▾` (active); **More** shows `More: N` reflecting non-default field count.

The **Enriched answers** toggle previously in the Advanced dialog (UXS-005) is retired here — the `Enriched` filter chip replaces it.

---

## Enriched Answer hero

`EnrichedAnswerHero.vue` sits between the header and the operator bar. Visible when `enriched_search_available: true` from `/api/health` AND the request was made with `enrich_results=true` (shipped in RFC-088 chunk 5).

Contract inherits [UXS-008](UXS-008-enriched-search.md) in full: `gi` domain token border/tint, "AI-generated / grounded" badge, synthesized answer with clickable speaker names (opening Person Landing) + topic tags (opening Topic Entity View), Sources section (collapsible, default 3, "Show all N" toggle), grounded source count indicator ("Based on N grounded insights"), provider attribution ("Synthesised by … "), source-to-result linking (`Used in answer` chip on the matching hit card).

Degradation follows UXS-008: hidden when no grounded insights lifted; muted error state on provider failure; skeleton on latency > 5 s; hidden entirely when enrichment not configured. The hero is **not** rendered by the Cmd-K palette or the LeftPanel launcher (bounded cost).

---

## Result-set operator bar

`ResultSetOperatorBar.vue` sits between the hero and the results. One button per operator; the active operator has `aria-pressed`.

| Operator | `data-testid` | Effect |
| --- | --- | --- |
| Cluster | `search-op-cluster` | Server groups hits via `insight_clusters` / `theme_clusters` (RFC-107 §6); UI renders `ClusterGroupCard`. Fetch `top_k * 3` when active. |
| On graph | `search-op-graph` | Union bbox of derived node ids → graph camera set-focus (`graphNavigation.focusSet`). Camera-fit invariant preserved. |
| Timeline | `search-op-timeline` | Client bucket by `publish_date` month → `SubjectTimelineChart`; bucket-click filters `WorkspaceResults` client-side. |
| Compare | `search-op-compare` | Enabled when ≥ 2 subject types are present in the hit set. Opens a 2-column view sourced from `build_briefing_pack(query, query_type, results, canonical_entity, max_tokens)` per side (RFC-093 shipped API). Judge summary muted below when available. |
| Consensus | `search-op-consensus` | Enabled when at least one Topic is present. Surfaces `topic_consensus` enricher output (ADR-108; shipped 0.91 precision on prod-v2). Cross-speaker corroboration pairs (**not contradictions** — CONTRADICTS edges are v3+, out of scope). |

Operator bar `aria-role="toolbar"`; buttons `aria-role="button"` with visible focus ring. Toolbar disappears when there are 0 hits.

---

## Results

`WorkspaceResults.vue` renders the muted "N results" / "1 result" line (unchanged from UXS-005), the optional `Lift: applied / transcript rows` line (from RFC-072 KL1 lift stats), and the hit list.

Each `ResultCard`:

- **Tier badge** (`search-result-tier`, from UXS-005 PRD-033 FR1.1): **Insight** (`primary`) / **Transcript** (`success`) / **Reference** (`muted`), from `source_tier`.
- **Compound `+ insight` badge** (`search-result-compound`) when the hit is a transcript-lifted-to-insight compound (`lifted` block, RFC-072 KL1).
- **Actions**: `G` (graph focus), `L` (Library episode) — same rules as UXS-005 (L requires `source_metadata_relative_path` + healthy API); `E` (episode id chip, informational). When KG-surface merged, only `G` shows.
- **Lifted GI insight** region (when `lifted` present, UXS-005-inherited): linked insight id/text, speaker/topic labels, quote time range. Includes the `No speaker detected` muted line (`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`, #541) when `lifted.quote` has timestamps but no speaker label.
- **Supporting quotes** (collapsible, UXS-005-inherited): same `No speaker detected` treatment when speaker missing.

`ClusterGroupCard` (when operator === 'cluster'):

- Header: cluster shared entity + member count.
- Expandable body: `ResultCard`s for each member, indented.
- `data-testid="search-cluster-group"` on the header.

---

## Sidebar — Saved + Recent (USERPREFS-1)

**Revised in §S4-shell (2026-07-20):** LeftPanel hosts two collapsible sections:

- **Saved queries** — reads `search.savedQueries` from USERPREFS-1. Each row: `name`, muted preview of `q`. Click = switch to Search tab + re-run with `filters` + `operator` re-applied. Trash icon on hover deletes. Header testid: `left-panel-saved-queries`.
- **Recent queries** — reads `search.recentQueries` (last 20, USERPREFS-1). Each row: `q` truncated. Click = switch to Search tab + re-run. Header testid: `left-panel-recent-queries`.

Empty states: `left-panel-saved-empty` (Saved section, no items), `left-panel-recent-empty` (Recent section, no items). List containers: `left-panel-saved-list`, `left-panel-recent-list`.

Both sections **hide entirely** (no error banner) when `useUserPreferencesStore.available === false` (offline / unauth) — matches USERPREFS-1's silent-degrade rule.

Save affordance: after a successful search in the Workspace, a `Save this query` button in the header opens a mini-dialog for `name`. Persists via `useSavedQueriesStore.save(name, request)` → USERPREFS-1 PATCH.

**Retired testids** (was `workspace-sidebar-*`): `workspace-sidebar-saved`, `workspace-sidebar-recent`.

---

## Active-search-context integration

Every successful Workspace search **publishes** to `useActiveSearchContextStore` (RFC-094 OQ-2, shipped) with `{ q, filters, scope }`. LibraryView and GraphCanvas consumers read this today and weight rows / nodes accordingly. Rail launchers (see below) also publish to the same store on activation. No new store; no changes to consumers.

---

## Rail search launchers

Each subject rail carries a "Search inside this X" affordance in its header (or context menu for Graph nodes). Activating the launcher:

1. Publishes the scope to `useActiveSearchContextStore`.
2. Switches the main tab to Search.
3. Focuses `QueryField` with the scope filter chip visible + the current subject's canonical name pre-filled in a scope hint (not the query text — the user still types their question).

| Rail | Testid | Scope pre-fill |
| --- | --- | --- |
| `EpisodeDetailPanel` | `rail-search-in-episode` | `{episode: metadata_relative_path}` |
| Node view (Topic) / `TopicEntityView` | `rail-search-in-topic` | `{topic: id}` |
| Node view (Person) / `PersonLandingView` | `rail-search-in-person` | `{person: id}` |
| `ShowRailPanel` | `rail-search-in-show` | `{feed: feed_id}` |
| `GraphNodeRailPanel` context menu | `rail-search-in-selection` | `{selection: {node_ids: [selected ids]}}` — server OR-merges each node's natural scope; **no `kg_proximity`** (RFC-091 rejected). |

The rail's existing content is not replaced — this is search-in-addition (e.g. `positions_of(person)` in PersonLanding stays).

---

## In-episode search

`EpisodeDetailPanel` grows a compact `QueryField` below its header, above the sections. Queries the episode-scoped `app_episodes` search endpoint. Hits render inline (no Workspace navigation), each with a jump-to-moment. `data-testid="episode-inline-search-field"` / `episode-inline-search-results`. An "Open in Workspace" chip escape-hatches to the full Workspace with the episode scope pre-filled.

---

## Command palette (shell-wide)

Not in the Workspace itself — see [UXS-005](UXS-005-semantic-search.md) for the compact-launcher role and RFC-107 §4 for the shell-wide palette. The palette's `Open in Workspace` action routes here.

---

## Empty states

- **No corpus configured**: full-width muted card — "Set a corpus path in the status bar to enable search." (Same copy family as UXS-005.) `data-testid="workspace-no-corpus"`.
- **API down**: same as UXS-005 (`Search is unavailable right now`).
- **Query field empty on tab arrival**: sidebar renders (Saved + Recent when available); results area shows a muted centred hint "Type a question and press Enter." No skeleton, no operator bar.
- **Zero results**: muted "No results for `<q>`" line, retained filter chip bar so the user can widen; operator bar hides.

---

## Accessibility

- Workspace `role="region"` `aria-label="Search workspace"`.
- Header `role="search"` (the query form is the primary form).
- Operator bar `role="toolbar"`; each button `aria-pressed` reflects active operator.
- Enriched Answer hero inherits UXS-008 accessibility.
- Sidebar sections use `aria-label="Saved queries"` / `"Recent queries"`; each item is a `role="button"` with visible focus ring and Enter/Space activation.
- Keyboard: `/` and `Cmd-K` open the shell-wide palette (RFC-107 §4); inside the Workspace `Tab` order = query field → filter chips → operator bar → first result → sidebar.
- All new controls carry `data-testid` for the E2E surface map; `aria-*` for screen readers.

---

## Tokens

Inherits UXS-001 tokens. New surface-level rules:

- Workspace background: `canvas`.
- Workspace card padding: standard (matches UXS-001 card padding).
- Hero panel: `surface` background + `gi` left border (4 px solid, inherited from UXS-008).
- Sidebar background: `surface`; section headers `lp-section` (small-caps).
- Operator bar: `surface` background, `border` divider.

---

## E2E contract

New visible labels and selectors require updates to [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) before or with implementation. Key surfaces:

- `SearchTab.vue` (`workspace-root`)
- Header (`workspace-header`, `search-query-type`, `search-filter-bar`, `search-chip-*`)
- Enriched Answer hero (inherits UXS-008 selectors)
- Operator bar (`search-op-cluster`, `search-op-graph`, `search-op-timeline`, `search-op-compare`, `search-op-consensus`)
- Results (`search-results`, `search-result-tier`, `search-result-compound`, `search-cluster-group`)
- Sidebar (`workspace-sidebar-saved`, `workspace-sidebar-recent`, `workspace-save-button`, `workspace-save-dialog`)
- Rail launchers (`rail-search-in-episode`, `-topic`, `-person`, `-show`, `-selection`)
- In-episode search (`episode-inline-search-field`, `episode-inline-search-results`)

Retired (was UXS-005 / VIEWER_IA §Left panel): `left-panel-enter-explore`, `left-panel-back-search`.

Playwright coverage: per-surface Tier-1 mocked specs + Tier-2 production-shaped specs (RFC-107 §T1); Tier-3 real-corpus spec `web/gi-kg-viewer/e2e/validation/search-real-corpus.spec.ts`.

---

## Revision history

| Date       | Change                                                                                                                                          |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-20 | §S4-shell revision — LeftPanel visible on all tabs; Saved + Recent hosted in LeftPanel, not SearchTab; compact launcher retired. See 4d13dce9.  |
| 2026-07-20 | Initial draft — Query Workspace + operator bar + sidebar + hero placement (PRD-045 / RFC-107)                                                   |

---

## §S4-shell revision (2026-07-20)

**Compact SearchPanel launcher retired.** The shell pivot shipped in `4d13dce9` changed the Workspace shape:

- LeftPanel is **now visible on ALL main tabs** (Digest / Library / Search / Graph / Dashboard), not hidden on Search.
- Saved + Recent query sections now live in LeftPanel (via collapsible pattern), not as children of SearchTab via `WorkspaceSidebar.vue`.
- The compact query launcher that previously occupied LeftPanel on non-Search tabs has been retired.
- All Saved + Recent data still flows from USERPREFS-1 keys `search.savedQueries` and `search.recentQueries` (unchanged).
- Rows emit `apply-query` → App switches to Search main tab + runs. This preserves the UX intent (recent/saved lead to Workspace navigation) while co-locating Saved + Recent in one place.
- Keyboard: `focusSearch` in `useViewerKeyboard` made optional; `/` and `Cmd-K` both open the palette.
- Vertical rail "Search / Explore" button relabeled "Saved queries".
- New testids reflect LeftPanel location: `left-panel-saved-queries`, `left-panel-saved-list`, `left-panel-saved-empty`, `left-panel-recent-list`, `left-panel-recent-empty`.
