# RFC-107: Search v3 — Query Workspace (technical design)

- **Status**: Draft
- **PRD**: [PRD-045](../prd/PRD-045-search-v3-query-workspace.md) · **UXS**: introduces [UXS-016](../uxs/UXS-016-query-workspace.md) (Query Workspace); extends [UXS-005](../uxs/UXS-005-semantic-search.md) (compact launcher role); heroifies [UXS-008](../uxs/UXS-008-enriched-search.md); shell IA in [VIEWER_IA.md](../uxs/VIEWER_IA.md)
- **Surface**: `web/gi-kg-viewer` (viewer) · `src/podcast_scraper/server/routes/search.py` (additive query params only)
- **Authors**: Marko
- **Related RFCs**:
  - [RFC-090](RFC-090-hybrid-retrieval.md) — hybrid retrieval (**do not re-enable native combine**; #1205)
  - [RFC-094](RFC-094-search-powered-surfaces-query-layer.md) — relational query layer + `activeSearchContext` + `PanelRetrievalStore` (**shipped**; Search v3 composes with, does not replace)
  - [RFC-088](RFC-088-enrichment-layer-architecture.md) — enrichment layer + `QueryEnricher` (**shipped**)
  - [RFC-093](RFC-093-litm-context-packs.md) — `build_briefing_pack` (**shipped**; Compare operator source)
  - [RFC-092](RFC-092-ml-query-router.md) — query router (5-class intent taxonomy)
  - [RFC-091](RFC-091-kg-proximity-signal.md) — KG proximity as retrieval signal (**REJECTED** 2026-06-03; do NOT use)
  - [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) — canonical identity + compound-lift contract (KL1 shipped; KL5 CONTRADICTS is v3+)
  - [RFC-097](RFC-097-unified-kg-gi-ontology-v2.md) — v2 materialized descriptive edges (typed CONTRADICTS remains v3+)
  - [RFC-062](RFC-062-gi-kg-viewer-v2.md) — viewer v2
- **Related ADRs**:
  - [ADR-095](../adr/ADR-095-viewer-test-pyramid.md) — three-tier pyramid (adopted)
  - [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md) — LanceDB-first (SIGSEGV context)
  - [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) — `topic_consensus` activated; `stance_*` retired
  - [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) — no per-corpus UI state (this RFC's Section 8 conforms)
- **Related incident**: #1205 — LanceDB native hybrid combine SIGSEGV (fix `0fe0854b`)

---

## Abstract

Promote search in the operator viewer from a 288 px sidebar into a **first-class main tab (Query Workspace, UXS-016)** and a **shell-wide command palette (Cmd-K)**, and merge the current Search + Explore mode-switch into one query surface. Add **result-set operators** (cluster / on-graph / timeline / compare / consensus) that unlock existing shipped backend capability (`insight_clusters`, `theme_clusters`, `context_pack.build_briefing_pack`, `topic_consensus`). Add **search-launch handoffs** from every subject rail (publishing to the shipped `activeSearchContext` store from RFC-094), and **in-episode search** on `EpisodeDetailPanel`. Add **saved queries + query history** as a user-scoped surface — **via USERPREFS-1** (`/api/app/preferences`), NOT per-corpus (ADR-119). Adopt the three-tier test pyramid (ADR-095) for search coverage; add a **quality-eval harness** with a labelled query set and a **CDP-based perf-capture harness** mirroring `capture-graph-lcp.{sh,mjs}`. Enforce the #1205 SIGSEGV guardrails (Python-side `search_bm25 + search_vector + rrf_fuse` fan-out; no `_combine_hybrid_results`) as first-class review gates.

## Problem

See PRD-045 §Background. Concretely:

1. **IA**: search sits in a 288 px left column with a slide mode-switch (`shell.leftPanelSurface`) that hides Explore behind Search or vice-versa. It is not addressable as a top-level tab; from Digest/Library/Graph/Dashboard, the operator must scroll to a sidebar to query.
2. **UX**: result cards have no set-level operators; the enriched-answer panel is spec'd but gated behind an Advanced dialog; there is no shell-wide "quick query"; rails have no search launcher.
3. **Substrate under-used**: RFC-094 (relational + `activeSearchContext` + `PanelRetrievalStore`), RFC-088 (`QueryEnricher` + `enrich_results` wiring on `/api/search`), RFC-093 (`build_briefing_pack`), ADR-108 (`topic_consensus`), the shipped intent router (RFC-092) — all shipped, all thinly surfaced.
4. **Test/eval/perf**: no synthetic-corpus search tier-3, no labelled query set, no CDP-based perf baseline.
5. **Reliability**: #1205 (SIGSEGV) proved that the native LanceDB hybrid combine path is not safe under our workload. The v3 changes MUST NOT reintroduce that call chain.
6. **Persistence**: an initial draft of §8 proposed per-corpus JSON for saved queries; ADR-119 explicitly rejects that (corpus is transient). Saved queries must be per-user via USERPREFS-1.

## Design

Ten sub-designs; each mapped to a slice in [`docs/wip/SEARCH-V3-IMPLEMENTATION-PLAN.md`](../wip/SEARCH-V3-IMPLEMENTATION-PLAN.md).

### §1 Shell IA changes

**Tab order:** Digest · Library · **Search** · Graph · Dashboard (default first-load unchanged: Digest).

**LeftPanel role change:**

- On tabs ≠ Search: LeftPanel becomes a **compact query launcher** — the query field + last-N recent queries (from USERPREFS-1 `search.recentQueries`) + "Open in Workspace" chip. No results render here. Enter → switches main tab to Search and runs.
- On the Search tab: LeftPanel is **hidden** (Workspace owns full width).
- The `shell.leftPanelSurface = 'explore'` mode is retired.

**Store additions:** `shellStore.mainTab` gains `'search'`. `useSearchStore` grows a `workspace` sub-state (see §2). `shellStore.leftPanelSurface` collapses to a single `'launcher'` mode outside of Search. `useExploreStore` is folded into `useSearchStore` and deleted.

**Composed shipped stores (RFC-094):** the Workspace and every rail launcher **publish** to `useActiveSearchContextStore` (already at `web/gi-kg-viewer/src/stores/activeSearchContext.ts`) so LibraryView and GraphCanvas consumers react as they do today. `PanelRetrievalStore` (RFC-094 OQ-1) is unchanged — Detail panels keep their per-entity cache.

**Subject persistence rule** (VIEWER_IA.md) holds: switching to Search does not clear the current subject.

### §2 Store shape

Extend the existing `web/gi-kg-viewer/src/stores/search.ts` with a **request/response record** shape:

```ts
interface SearchRequest {
  q: string
  filters: SearchFilters          // topics, speakers, doc_types, since, top_k, grounded_only, feed, min_confidence
  scope?: SearchScope             // 'corpus' | {episode: id} | {topic: id} | {person: id} | {feed: id} | {selection: {node_ids: string[]}}
  operator?: 'none' | 'cluster' | 'timeline' | 'graph' | 'consensus' | 'compare'
  compareSubjects?: [SubjectRef, SubjectRef]   // required when operator === 'compare'
}
interface SearchResponse {
  results: SearchHit[]
  query_type: QueryIntent          // from RFC-092 router; unchanged shape
  source_tiers: Record<Tier, number>
  lift_stats?: LiftStats           // RFC-072 KL1
  enriched?: EnrichedAnswer        // populated when /api/search?enrich_results=true and provider healthy
  operator_result?: OperatorResult // populated per operator (see §6)
}
```

Result-set operators call **the same** query endpoint (see §9) with `operator` set; the server does the aggregation server-side when it needs the full result set (Cluster over a wider hit pool than what the UI paginates). Client-side operators (Timeline over the displayed page) are computed in the store. The `scope: {selection: {node_ids}}` value is a UI-side selection projection — the server resolves each node id to its natural scope (episode / topic / person / feed) and OR-merges them. **Not `kg_proximity`** (RFC-091 rejected).

### §3 Query Workspace (main tab)

Component tree:

```text
SearchTab.vue                          [new — mounts on shellStore.mainTab === 'search']
├─ WorkspaceHeader.vue                 [new]
│    ├─ QueryField.vue                 [existing SearchPanel form, promoted; Preset dropdown added]
│    ├─ IntentChip.vue                 [existing search-query-type, moved to header]
│    └─ SearchFilterBar.vue            [existing, extended with Explore filters — §5]
├─ EnrichedAnswerHero.vue              [new — hoisted from UXS-008 dialog-gated to hero]
├─ ResultSetOperatorBar.vue            [new — Cluster / OnGraph / Timeline / Compare / Consensus]
├─ WorkspaceResults.vue                [new host]
│    ├─ ResultCard.vue                 [existing]
│    ├─ CompoundCard.vue               [existing lifted-region — promoted to card]
│    └─ ClusterGroupCard.vue           [new — when operator === 'cluster']
└─ WorkspaceSidebar.vue                [new — Saved + Recent, both from USERPREFS-1]
```

Uses `w-full` main area (respects VIEWER_IA §Viewport widths); no `w-72` constraint.

### §4 Command palette (Cmd-K / `/`)

New `CommandPalette.vue`, mounted at shell root, driven by `useCommandPalette()` composable. Open on `Cmd-K` / `Ctrl-K` / `/` (existing `useViewerKeyboard` "not-in-editable-control" rule preserved). Debounced live query (default 200 ms) against `/api/search?q=…&top_k=8&palette=1`. Rows carry 3 actions:

- **Open in Workspace** → `router` / tab switch to Search, runs the same request; publishes to `activeSearchContext`.
- **Pin to rail** → `subjectStore.focusHit(hit)` (new dispatcher — routes by hit doc_type to episode/topic/person/graph-node).
- **Show on graph** → existing `G` handoff.

When query is empty, show two groups: **Recent** (from USERPREFS-1 `search.recentQueries`) + **Saved** (from USERPREFS-1 `search.savedQueries`). Both silent-degrade when USERPREFS `available === false` (offline / unauth). Escape / outside-click / action-select closes without disturbing tab or subject.

### §5 Merge Search + Explore

Explore's filters (`topic contains`, `speaker contains`, `limit`, `min confidence`, `grounded only`) become **chips** on `SearchFilterBar.vue` — extend, don't rebuild. Explore presets ("Common questions", "Cross-episode") become a `Preset` dropdown on `QueryField.vue`. Explore's "quick question" natural-language field is the same query field. `ExplorePanel.vue` and `ExploreFilterBar.vue` are retired (files deleted; store `useExploreStore` folded into `useSearchStore`). Test IDs `left-panel-enter-explore` / `left-panel-back-search` are removed; release note in the E2E surface map.

### §6 Result-set operators

Operator bar (`ResultSetOperatorBar.vue`) sits above the result list. Each operator:

| Operator | Backend module (shipped) | Client action |
| --- | --- | --- |
| **Cluster** | `search/insight_clusters.py` + `search/theme_clusters.py` | Server groups hits by cluster; UI renders `ClusterGroupCard`. Fetch `top_k * 3` when Cluster is active (default; config `search.cluster.overfetch_factor`). |
| **Show on graph** | client — reuse `subjectStore.focusHit` per hit; union bbox in `App.vue` graph camera | Compute the union of derived node ids; call graph camera with a set-focus request (new `graphNavigation.focusSet(ids)`). |
| **Timeline** | client — bucket hits by `publish_date` month | Renders `SubjectTimelineChart` with hit counts; bucket-click filters `WorkspaceResults` client-side. |
| **Compare (2 subjects)** | `search/context_pack.py::build_briefing_pack(query, query_type, results, canonical_entity, max_tokens)` (RFC-093 **shipped API**) | Client picks 2 subjects; server runs hybrid scoped to each, calls `build_briefing_pack` twice, returns two `CorpusBriefingPack`s. 2-column view. Judge summary from `search/judged_eval.py` muted below when available. |
| **Consensus** | `topic_consensus` enricher (ADR-108, precision 0.91) — reads `enrichments/topic_consensus.json` (per-corpus enricher output already produced) | Given hits about the same Topic, show cross-speaker corroboration pairs. **NOT contradictions** — that requires typed CONTRADICTS edges (RFC-072 KL5, v3+, out of scope). |

None of these introduce a new native pyarrow.compute call site — every server-side step runs in Python after `rrf_fuse` returns. `make lint-search-v3` (see §S) guards against regressions.

### §7 Enriched-answer hero

Hoist `EnrichedAnswerPanel` (currently gated behind UXS-005 Advanced dialog) into `EnrichedAnswerHero.vue`, positioned above the result list on the Search tab **only**. Contract inherits UXS-008 (badge, sources, provider attribution, degraded states). Consumes the shipped `QueryEnricher` (RFC-088 chunk 5) via `/api/search?enrich_results=true`. Source-to-hit tie-in ("Used in answer" chip) uses hit `doc_id` matching. Palette does not render the hero (bounded cost); LeftPanel launcher does not render it either. The `EnrichmentConfigEditor.vue` (already shipped) remains the config surface — Search v3 does NOT reinvent it.

### §8 Saved queries + query history — USERPREFS-1 backed

**Storage decision (retracts an earlier per-corpus draft):** per-user via USERPREFS-1 (`/api/app/preferences`). Rationale + full ruleset in [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md).

**No new server endpoints.** All round-trips go through the shipped `GET/PUT/PATCH /api/app/preferences`.

**USERPREFS namespaces added:**

- `search.savedQueries: Record<id, SavedQuery>` — `SavedQuery = { id, name, q, filters, operator?, created_at, updated_at, corpusHint? }`.
- `search.recentQueries: RecentQuery[]` — ring buffer, last 20; `RecentQuery = { q, filters, ts, corpusHint? }`. Written on every successful search from the Workspace / palette / launcher. `corpusHint` is a display-only string (feed count + top-feed titles, or `null` when unset), never a filesystem path.

**Client store:** `useSavedQueriesStore` (thin Pinia around `useUserPreferencesStore` — reads/writes the two namespaces above, no independent hydrate). Exposes `save(name, request)`, `remove(id)`, `list()`, `pushRecent(request)`, `listRecent(limit)`, `resetAll()`.

**UI**:
- `WorkspaceSidebar.vue` renders Saved + Recent.
- Cmd-K palette empty-state renders the same two sections.
- Both silent-degrade when USERPREFS `available === false` — the sections hide, no error banner.

**Corpus telemetry (unchanged):** `search/query_log.append_query_event` continues to write per-corpus for Dashboard's QueryActivityChart. That is corpus analytics, not user state (ADR-119).

### §9 API additions

Additive only. No changes to `/api/app/search` (consumer surface — out of scope). No new saved-queries endpoints (§8 uses USERPREFS-1's shipped surface).

- Extend `POST /api/search` (viewer) with optional `operator: 'cluster' | 'consensus' | 'compare'` param. When set, the response carries `operator_result` (typed per operator). No change to the existing hit shape.
- Extend `GET /api/search` with an optional `palette=1` param — for the Cmd-K live-query path (server-side hint to bound response cost — the server may cap top_k regardless).
- `GET /api/search/similar?doc_id=…&top_k=…` — "more like this hit" using `corpus_similar.py` (shipped). Additive.
- For **Compare**: `POST /api/search/compare` — body: `{ subject_a: SubjectRef, subject_b: SubjectRef, q?, max_tokens? }`; response: `{ pack_a: CorpusBriefingPack, pack_b: CorpusBriefingPack }`. Wraps `build_briefing_pack` twice.

**SIGSEGV impact (each new endpoint):** all four call sites route through the Python-side `retrieval.py` fan-out. None touch `_combine_hybrid_results` / `_normalize_scores`. Enforced by `make lint-search-v3`.

### §10 Search-launch handoffs from rails + in-episode search

Additive prop on each rail component — an emit that opens the Workspace with pre-applied scope AND publishes the scope to `useActiveSearchContextStore` (RFC-094 OQ-2, shipped) so Library + Graph consumers react:

| Rail | Emit | Workspace scope pre-fill |
| --- | --- | --- |
| `EpisodeDetailPanel` | `open-search-in-episode` | `scope={episode: metadata_relative_path}` |
| `TopicEntityView` / node view (Topic) | `open-search-in-topic` | `scope={topic: id}` |
| `PersonLandingView` / node view (Person) | `open-search-in-person` | `scope={person: id}` (speaker filter) |
| `ShowRailPanel` | `open-search-in-show` | `scope={feed: feed_id}` |
| `GraphNodeRailPanel` context menu | `open-search-in-selection` | `scope={selection: {node_ids: [selected node ids]}}` — server OR-merges each node's natural scope (episode / topic / person / feed). **Does NOT use `kg_proximity`** (RFC-091 rejected 2026-06-03). |

The rail's existing relational content (e.g. `positions_of(person_id)` for PersonLanding via RFC-094) is unchanged — this is search-in-addition, not search-replaces.

In-episode search (`EpisodeDetailPanel` inline field): renders `WorkspaceResults` inline (no Workspace navigation); scope frozen to that episode via the existing `app_episodes` search endpoint.

## Testing

### §T1 Three-tier pyramid (ADR-095 adopted, no new invention)

- **Tier 1 (fast, mocked, ~30 s)** — per-surface contract specs under `web/gi-kg-viewer/e2e/`:
  - `search-workspace.spec.ts` (tab renders; header + operator bar + results; header intent chip; filter chip bar merges Explore filters)
  - `command-palette.spec.ts` (Cmd-K open/close; live query; three actions per row; recent/saved sections from USERPREFS)
  - `result-set-operators.spec.ts` (cluster/timeline/graph/compare/consensus — mocked responses)
  - `rail-search-launch.spec.ts` (5 rail launchers; each also asserts `activeSearchContext` was published)
  - `saved-queries.spec.ts` (round-trip via mocked USERPREFS-1 endpoints; silent-degrade when unavailable)
- **Tier 2 (production-shaped, ~3–5 min)** — extend `web/gi-kg-viewer/e2e/fixtures/production-shaped/` with search-oriented content:
  - Add ≥ 1 compound-lift pair (transcript + linked insight).
  - Add ≥ 1 `topic_consensus` pair (cross-speaker corroboration).
  - Add ≥ 1 theme cluster with ≥ 5 members.
  - Add labelled query set (see §T2) as a sibling `queries.json`.
  - Extend `scripts/build_production_shaped_fixture.py` with a `--search-slice` flag that appends these; deterministic + idempotent.
  - New spec dir `web/gi-kg-viewer/e2e/search-production/` mirrors Tier 1 row-for-row.
- **Tier 3 (real-corpus, ~5–10 min)** — new `web/gi-kg-viewer/e2e/validation/search-real-corpus.spec.ts`:
  - Runs against `make serve` + operator-supplied corpus.
  - Same 6-point user-visible contract as graph handoff Tier 3 (selection + camera + subject + no console errors), extended for search: (a) enriched answer paints, (b) intent chip matches expected label on labelled subset, (c) cluster operator returns ≥ 1 cluster on a corpus known to have themes, (d) rail launchers open Workspace with correct scope AND publish to `activeSearchContext`.
  - New Make target: `make ci-ui-validation-search CORPUS=…`.
- **Institutional rule (ADR-095) applied**: every Tier-3 bug lands a Tier-2 row before the fix PR merges.

### §T2 Quality eval harness

**Labelled query set** at `tests/fixtures/viewer-validation-corpus/v3/search-queries.json`:

```jsonc
[
  {
    "id": "q001",
    "q": "what did Sam Altman say about compute governance",
    "intent_expected": "entity_lookup",
    "expected_top_k_doc_ids": ["…"],       // hand-labelled, min 3
    "min_ndcg_at_10": 0.6                   // per-query floor
  }
  // ≥ 25 queries covering all 5 intents from RFC-092
]
```

**CLI**: `python scripts/eval/search_quality.py --corpus <path> --queries <path> --top-k 10 --out docs/wip/search-v3/eval/`.

**Metrics** produced per query + aggregated:
- nDCG@10, MRR@10 (over labelled expected set).
- Intent-router accuracy (predicted vs `intent_expected`).
- Tier coverage rate (fraction of queries that return ≥ 1 Insight AND ≥ 1 Transcript in top-10).
- Compound-lift rate (fraction of transcript hits that carry `lifted`).
- Enriched-answer groundedness rate (fraction of enriched-answer sources with `grounded: true`).
- `topic_consensus` precision on the labelled subset (shipped baseline: 0.91 on prod-v2 per ADR-108).

**CI target**: `make eval-search` (offline, uses the checked-in synthetic corpus). Baseline captured on `search-v3` HEAD **before** any UI-facing change lands; per-slice deltas required in PR body.

### §T3 Unit / integration (pyramid base)

Standard vitest + pytest at:
- `web/gi-kg-viewer/src/stores/search.test.ts` (extend for operator + scope shape)
- `web/gi-kg-viewer/src/stores/savedQueries.test.ts` (new — USERPREFS-1 integration + silent-degrade)
- `web/gi-kg-viewer/src/components/search/**/*.test.ts` (per-component)
- `tests/integration/server/test_search_operator_endpoints.py` (new — operator params + compare endpoint contract)
- `tests/unit/search/test_operator_cluster.py` / `test_operator_consensus.py` / `test_search_compare_endpoint.py`
- **SIGSEGV regression** (see §S): `tests/integration/search/test_lancedb_concurrent_no_native_combine.py` — extends the #1205 repro harness.

## Perf capture harness

Mirrors `scripts/dev/capture-graph-lcp.{sh,mjs}` — same shape, same output contract, different targets.

### §P1 Scripts

- `scripts/dev/capture-search-perf.sh` — orchestrator (isolated api on `:8601`, viewer on `:5601`, no port collision with graph).
- `scripts/dev/capture-search-perf.mjs` — Playwright + CDP capturer.
- `scripts/dev/capture-search-api.sh` — pure-API HTTP capturer (no browser), for the query set from §T2, over the intent classes + top_k grid.

### §P2 Scenarios captured

**API (mandatory baseline before slice 1):**

| Scenario | Query source | Metric |
| --- | --- | --- |
| `api-intent-entity_lookup` | 5 queries from §T2 | p50/p95/p99 |
| `api-intent-raw_evidence` | 5 queries | p50/p95/p99 |
| `api-intent-temporal_tracking` | 5 queries | p50/p95/p99 |
| `api-intent-cross_show_synthesis` | 5 queries | p50/p95/p99 |
| `api-intent-semantic` | 5 queries | p50/p95/p99 |
| `api-top_k-{10,25,50,100}` | 3 queries × 4 top_k | p50/p95/p99 |
| `api-concurrent-4` | 4 parallel workers × 5 queries | p50/p95/p99 + SIGSEGV-free assertion |

**UI (mandatory baseline before slice 1):**

| Scenario | Metric |
| --- | --- |
| `ui-workspace-open` | TTI |
| `ui-cmdk-open` | keypress → palette-visible ms |
| `ui-filter-apply` | chip-toggle → results-repaint ms |
| `ui-operator-cluster` | click → cluster render ms |
| `ui-operator-graph` | click → graph camera settle ms |
| `ui-enriched-answer-paint` | request-start → answer-first-byte + full-render ms |

Median-of-3, `.metrics.json` per label, gzipped trace per label, committed under `docs/wip/search-v3/traces/`.

### §P3 Deep-review pass after baseline

Once baseline is committed, open a **deep-review issue** per surface (API + UI) — list top-N latency contributors, action items for slices. Do NOT silently apply optimizations discovered in the review; each opens a follow-up ticket.

## SIGSEGV guardrails (§S)

The #1205 fix (`0fe0854b`) is a live invariant. This RFC encodes it as review contract:

1. **No new call site into `_combine_hybrid_results` or `_normalize_scores`.** Retrieval routes through `search_bm25 + search_vector + rrf_fuse` (`retrieval.py`) exclusively. Any new backend endpoint that touches search uses the fan-out path.
2. **No process-wide query lock.** Concurrency is bounded per-request via the fan-out's own executor. Any new server code that adds a lock across search requests requires an ADR update.
3. **Digest warm-up preserved.** The `corpus_digest` warm-up (one serial band query before parallel fan-out) is NOT removed by any slice.
4. **Regression test.** `tests/integration/search/test_lancedb_concurrent_no_native_combine.py` — extends the #1205 repro harness — asserts under 4-way concurrent search over a real LanceDB corpus: (a) no faulthandler signal, (b) no SIGSEGV, (c) all requests return 200. This test runs on every PR that touches `search/backends/`, `search/retrieval.py`, `search/hybrid_search.py`, or `server/routes/corpus_digest.py` (path-conditional required check).
5. **SIGSEGV impact line in PR body.** Every slice PR answers: *does this slice add a call site into the native combine, or into a new native pyarrow.compute path?* Default = NO; any YES requires an ADR update AND a repro-test row.
6. **Grep guard.** `make lint-search-v3` (new; single-purpose lint) fails when `_combine_hybrid_results` or `_normalize_scores` is imported outside a whitelist (currently: nowhere). Recorded in `.github/lint/search-v3-forbidden-imports.txt`.

## Migration / rollout

- **Slice order** — see [`docs/wip/SEARCH-V3-IMPLEMENTATION-PLAN.md`](../wip/SEARCH-V3-IMPLEMENTATION-PLAN.md).
- **Feature flag** — none. Search v3 is an operator-viewer refactor; no toggle behind users. LeftPanel Search remains functional until slice 1 lands the tab; the mode-switch is retired only when the tab is in.
- **E2E surface map** — updated in each slice; the old Explore test IDs get a "retired in RFC-107 slice 2" note.
- **Docs** — VIEWER_IA.md gains a Search-tab paragraph; UXS-005 grows a "compact launcher" section (retiring its main-column role); UXS-008 grows a "hero placement" section; UXS-016 is the primary UX doc for the Workspace.

## Alternatives considered

1. **Global command bar only, no Search tab.** Cheaper. Rejected — result-set operators + enriched-answer hero need main-area real estate; cmd-K alone can't host them.
2. **Search as a right-rail mode.** Rejected — collides with subject rail (VIEWER_IA §Persistence).
3. **Keep Explore separate, merge later.** Rejected — Explore drift is what caused the "two UIs, one column" problem; deferring re-encodes the debt.
4. **Re-enable native LanceDB hybrid combine.** Rejected on `T7` / #1205 grounds — that path is the SIGSEGV; a future upstream fix + independent verification could revisit, out of scope here.
5. **Per-corpus JSON at `<corpus_root>/.viewer/saved_queries.json`.** Rejected — the corpus is transient (moving to a DB). ADR-119 formalizes the rule; §8 uses USERPREFS-1 instead.
6. **New database table for saved queries.** Rejected — USERPREFS-1 already exists, is battle-tested for graph-lenses / theme / lp.interests / lp.audioSyncOffsets / corpusLensPreset, and adding a feature to it is a client-side namespace addition with no server work.
7. **Use `kg_proximity` scope pre-fill for graph-selection search launcher.** Rejected — RFC-091 formally rejected KG proximity as a retrieval signal (measured negative effect on nDCG and cross-show diversity, 2026-06-03). §10 uses selection → union-of-natural-scopes instead.
8. **Full "Contradictions" operator (contradicts-as-assertion).** Rejected in this epic — requires typed CONTRADICTS edges (RFC-072 KL5) which are v3+; the shipped `nli_contradiction` was disabled at 0 % precision and reimagined as `topic_consensus` (ADR-108). §6 ships Consensus, not Contradictions.
9. **Streaming enriched answer in v1.** Rejected — blocking-then-render is enough for baseline; streaming as follow-up (OQ3).

## Open questions

- **OQ1** Same as PRD OQ1: default first-load tab. Behind a per-user USERPREFS-1 pref (`shell.defaultTab`).
- **OQ2 — RESOLVED** Saved queries live in USERPREFS-1 (ADR-119).
- **OQ3** Enriched-answer streaming — follow-up.
- **OQ4** Compare-2 UI shape — RFC-107 amendment when slice S8 designs land. Default: side-by-side.
- **OQ5** Palette shortcut — this RFC picks `Cmd-K` / `Ctrl-K` / `/` (repointed) unless operator objects.
- **OQ6** Cluster server-side depth — how far to over-fetch when operator === 'cluster' (config; default `top_k * 3`).
- **OQ7** Should `search.recentQueries` be Workspace-per-tab (fresh per open) OR user-cross-device (roams)? RFC-107 default: user-cross-device (USERPREFS-1). Palette empty-state can filter to "just this session" if operator asks.

## References

- [PRD-045](../prd/PRD-045-search-v3-query-workspace.md) — parent
- [RFC-090](RFC-090-hybrid-retrieval.md) — hybrid retrieval (native combine forbidden — #1205)
- [RFC-094](RFC-094-search-powered-surfaces-query-layer.md) — relational layer + `activeSearchContext` + `PanelRetrievalStore` (shipped substrate)
- [RFC-088](RFC-088-enrichment-layer-architecture.md) — enrichment layer + QueryEnricher (shipped)
- [RFC-093](RFC-093-litm-context-packs.md) — `build_briefing_pack` (shipped; Compare source)
- [RFC-092](RFC-092-ml-query-router.md) — intent taxonomy
- [RFC-091](RFC-091-kg-proximity-signal.md) — **REJECTED**; do NOT use for retrieval or scope pre-fill
- [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) — compound-lift contract (KL1)
- [RFC-097](RFC-097-unified-kg-gi-ontology-v2.md) — v2 edges; CONTRADICTS is v3+
- [ADR-095](../adr/ADR-095-viewer-test-pyramid.md) — three-tier pyramid (adopted)
- [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md) — LanceDB-first (SIGSEGV context)
- [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) — `topic_consensus` activated; `stance_*` retired
- [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) — no per-corpus UI state (this RFC's §8 conforms)
- [VIEWER_IA.md](../uxs/VIEWER_IA.md) — shell IA (tab + LeftPanel role change)
- [UXS-005](../uxs/UXS-005-semantic-search.md) — semantic search (compact-launcher role)
- [UXS-008](../uxs/UXS-008-enriched-search.md) — enriched search (heroified)
- [UXS-016](../uxs/UXS-016-query-workspace.md) — Query Workspace (primary UX)
- [GRAPH_PERF_TRACE_RUNBOOK.md](../guides/GRAPH_PERF_TRACE_RUNBOOK.md) — perf-capture template
- USERPREFS-1: `docs/wip/USERPREFS-1.md`
- #1205 — LanceDB SIGSEGV incident + fix `0fe0854b`
