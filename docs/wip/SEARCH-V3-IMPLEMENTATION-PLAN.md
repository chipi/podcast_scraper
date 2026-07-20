# Search v3 — implementation plan (slices)

**Parent PRD**: [PRD-045](../prd/PRD-045-search-v3-query-workspace.md)
**Parent RFC**: [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md)
**Primary UX**: [UXS-016](../uxs/UXS-016-query-workspace.md)
**Persistence rule**: [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) — no per-corpus UI state
**Consensus signal**: [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) — `topic_consensus` (0.91 precision on prod-v2)
**Branch**: `search-v3` (already created off `origin/main` @ `59eecd90`)
**Scope**: operator viewer only. Consumer surface out of scope.

---

## Slice map (S0 → S8)

Total nine slices; S0 is the baseline capture (mandatory before any code lands); S1–S8 are the ship path. Each slice is a self-contained PR onto `search-v3`.

| Slice | Title | Ships | Depends on |
| --- | --- | --- | --- |
| **S0** | Baseline: perf + eval + Tier-2 fixture extension | metrics.json for API + UI; `make eval-search` baseline; extended `production-shaped` fixture + labelled query set | — |
| **S1** | Merge Search + Explore into one query surface | Explore filters as chips; Explore store folded; mode-switch retired | S0 |
| **S2** | Search as 5th main tab (Query Workspace shell) | `SearchTab.vue`, `WorkspaceHeader/Results/Sidebar`, LeftPanel becomes compact launcher; publishes to `activeSearchContext` (RFC-094) | S1 |
| **S3** | Cmd-K / `/` command palette | `CommandPalette.vue`, three-action rows, recent + saved sections (both from USERPREFS-1) | S2 |
| **S4** | Result-set operators (Cluster / Timeline / OnGraph / Consensus) | operator bar, server-side aggregation for cluster; `topic_consensus` surface | S2 |
| **S5** | Enriched-answer hero | `EnrichedAnswerHero.vue` on Workspace, source-to-hit tie-in; consumes RFC-088 QueryEnricher | S2 |
| **S6** | Search-launch from rails + in-episode search | 5 rail launchers publishing to `activeSearchContext`; `EpisodeDetailPanel` inline search | S2 |
| **S7** | Saved queries + Recent (USERPREFS-1) | `useSavedQueriesStore` thin wrapper over `useUserPreferencesStore`; `search.savedQueries` + `search.recentQueries` namespaces; **no new server endpoints** | S3 |
| **S8** | Compare (2 subjects) operator | Compare view sourced from `POST /api/search/compare` wrapping `build_briefing_pack` (RFC-093) twice | S4 |

**Slice count target (PRD ask 3–7):** the plan runs 8 shipping slices + 1 baseline slice. If we want to compress, S3 can absorb S7 and S8 can defer to a follow-up RFC.

---

## Slice detail

### S0 — Baseline (mandatory)

**Scope:**
- Extend `scripts/build_production_shaped_fixture.py` with `--search-slice` (adds compound-lift pair, `topic_consensus` pair, theme cluster ≥ 5). Regenerate the fixture; commit deltas.
- Land `tests/fixtures/viewer-validation-corpus/v3/search-queries.json` — ≥ 25 labelled queries across 5 intents.
- Land `scripts/eval/search_quality.py` + `make eval-search` (metrics per RFC-107 §T2).
- Land `scripts/dev/capture-search-perf.{sh,mjs}` + `scripts/dev/capture-search-api.sh`.
- Capture median-of-3 baseline for the 6 UI + 7 API scenarios listed in RFC-107 §P2; commit `.metrics.json` under `docs/wip/search-v3/traces/S0-baseline-*/`.
- Extend `tests/integration/search/` with `test_lancedb_concurrent_no_native_combine.py` — reuse the #1205 repro harness.
- Land `.github/lint/search-v3-forbidden-imports.txt` + `make lint-search-v3`.

**Acceptance:**
- `make eval-search` produces a report; nDCG@10 recorded per query; `topic_consensus` precision recorded.
- 13 metrics.json committed (7 API + 6 UI).
- SIGSEGV regression test passes on 4-way concurrent load.
- `make lint-search-v3` passes on `main` and on branch.

**Gates:** T1 for fixture extension only. No T2/T3 needed (no rendered surface changes). Perf baseline **IS** the deliverable. SIGSEGV impact line: **NO** (test-only + tooling).

**Risk:** medium — the labelled query set needs care to be honest, not rigged.

---

### S1 — Merge Search + Explore

**Scope:**
- Extend `SearchFilterBar.vue` with `topic-contains`, `speaker-contains`, `limit`, `min-confidence`, `grounded-only` chips.
- Add `Preset` dropdown to `QueryField.vue` (Explore presets migrated).
- Fold `useExploreStore` into `useSearchStore`.
- Delete `ExplorePanel.vue`, `ExploreFilterBar.vue`, `components/explore/**`.
- Remove `shell.leftPanelSurface = 'explore'` mode; simplify `LeftPanel.vue` to always render Search.
- Update E2E surface map — retire `left-panel-enter-explore` / `left-panel-back-search` test IDs.

**Acceptance:**
- Tier 1 mocked contract updated; Tier 2 spec updated (Explore surface tests migrated to Search filter-bar tests).
- No behavior regression — every Explore query producible via new chips + preset.
- `make eval-search` delta ≥ 0 (no quality regression).
- Perf delta on `ui-filter-apply` reported.

**Gates:** T1 required, T2 required, T3 not applicable, `make eval-search` diff required, `capture-search-perf` UI diff required. **SIGSEGV impact: NO.**

**Risk:** medium — deletes 2 top-level components; keyboard tests must migrate.

---

### S2 — Search as 5th main tab

**Scope:**
- Add `'search'` to `shellStore.mainTab`; App tabs render new `SearchTab.vue`.
- Land `WorkspaceHeader.vue` + `WorkspaceResults.vue` + `WorkspaceSidebar.vue` (per UXS-016 component tree).
- LeftPanel becomes compact launcher on non-Search tabs; hidden on Search tab.
- Move intent chip + evidence toggle from `SearchPanel.vue` header to `WorkspaceHeader.vue`; retire `SearchPanel.vue` (or reduce to compact launcher shim).
- **Every Workspace query publishes to `useActiveSearchContextStore`** (RFC-094 OQ-2, shipped) — Library + Graph react as they do today.
- Update VIEWER_IA.md (already done in this branch) — confirm the diff still aligns.

**Acceptance:**
- Tier 1: new `search-workspace.spec.ts`.
- Tier 2: `search-production/workspace.spec.ts` (from extended fixture).
- Tier 3: recipe row in `handoff-production/`.
- Perf: `ui-workspace-open` TTI reported; regression ≤ 0 %.
- Subject rail persistence rule preserved (Tier 1 assertion).
- `activeSearchContext` publish asserted in a Tier 1 spec.

**Gates:** T1 + T2 + T3 all required, perf UI diff required. **SIGSEGV impact: NO** (viewer-only).

**Risk:** high — largest UI slice; touches shell, IA, keyboard, and every open E2E baseline.

---

### S3 — Cmd-K / `/` command palette

**Scope:**
- New `CommandPalette.vue` + `useCommandPalette` composable; mount at App root.
- Repoint `/` shortcut through `useViewerKeyboard`; add `Cmd-K` / `Ctrl-K`.
- Live query (200 ms debounce) via `/api/search?palette=1&top_k=8` — backend accepts additive param (no shape change).
- Three-action rows (Open in Workspace / Pin to rail / Show on graph).
- Empty-state Recent + Saved sections — both read from USERPREFS-1 (Recent = `search.recentQueries`; Saved wire lands in S7, until then Saved section is empty).

**Acceptance:**
- Tier 1: `command-palette.spec.ts`.
- Tier 2: `search-production/palette.spec.ts` (open Cmd-K, run 3 queries against the fixture).
- Perf: `ui-cmdk-open` ≤ 100 ms (baseline captured in S0).
- Accessibility: focus trap; Escape closes; return to prior focus.

**Gates:** T1 + T2 required, T3 optional (recipe row), perf UI diff required. **SIGSEGV impact: NO.**

**Risk:** low — additive; palette lives above the shell.

---

### S4 — Result-set operators

**Scope:**
- `ResultSetOperatorBar.vue` above `WorkspaceResults`.
- **Cluster**: extend `POST /api/search` with `operator=cluster`; server groups via `insight_clusters.py` / `theme_clusters.py`; client renders `ClusterGroupCard`. Over-fetch when cluster active (default `top_k * 3`, config).
- **Timeline**: client-only; new `SubjectTimelineChart` usage on the hit page.
- **On graph (set)**: `graphNavigation.focusSet(ids)` — union bbox in graph camera; camera-fit invariant preserved.
- **Consensus** (ADR-108): extend `POST /api/search` with `operator=consensus`; server reads `enrichments/topic_consensus.json` (already produced by the shipped enricher, precision 0.91 on prod-v2) and returns cross-speaker corroboration pairs for hits about the same Topic; client renders paired evidence. **NOT `contradictions`** — CONTRADICTS edges are v3+ (out of scope).
- Compare (2 subjects) deferred to S8.

**Acceptance:**
- Tier 1: `result-set-operators.spec.ts` (cluster / timeline / graph / consensus, mocked).
- Tier 2: `search-production/operators.spec.ts` — fixture has the theme cluster + `topic_consensus` pair.
- Perf: `ui-operator-cluster`, `ui-operator-graph` reported.
- SIGSEGV: cluster + consensus run in Python after `rrf_fuse` — **verify no new native compute site** (line in PR body + `make lint-search-v3` still green).

**Gates:** T1 + T2 required, T3 recipe row (cluster on real corpus), perf UI diff required, `make eval-search` re-run (cluster/consensus-adjacent metric surface changes). **SIGSEGV impact: NO** — assert in PR body.

**Risk:** high — touches server aggregation + graph camera; SIGSEGV risk if a lazy hand adds a native combine.

---

### S5 — Enriched-answer hero

**Scope:**
- Hoist `EnrichedAnswerPanel` → `EnrichedAnswerHero.vue`, positioned above results on Search tab only.
- Consumes shipped `QueryEnricher` (RFC-088 chunk 5) via `/api/search?enrich_results=true` — no new endpoint.
- Source-to-hit tie-in ("Used in answer" chip) via `doc_id` matching.
- Keep UXS-008 degraded states (skeleton, hidden, error).
- Retire the UXS-005 "Enriched answers" toggle in Advanced dialog; the header `Enriched` chip on the Workspace filter bar owns visibility going forward. `EnrichmentConfigEditor.vue` (already shipped, RFC-088 chunk 6) is unchanged.

**Acceptance:**
- Tier 1: enriched-answer visibility contract (mocked provider).
- Tier 2: `search-production/enriched-answer.spec.ts` — fixture stubs a provider that returns a grounded answer.
- Perf: `ui-enriched-answer-paint` reported; baseline ≤ 800 ms.
- Groundedness rate = 100 % on labelled queries (via `make eval-search`).

**Gates:** T1 + T2 required, T3 recipe row (real provider), `make eval-search` diff required, perf UI diff required. **SIGSEGV impact: NO** (LLM path independent of LanceDB).

**Risk:** medium — depends on enricher configuration.

---

### S6 — Search-launch from rails + in-episode search

**Scope:**
- Add `open-search-in-*` emits on 5 rails (Episode / Topic / Person / Show / Graph-node).
- App-level handler switches to Search tab with pre-applied scope AND publishes the scope to `useActiveSearchContextStore` (RFC-094 OQ-2, shipped).
- Graph-node launcher: pre-fill `scope={selection: {node_ids: [ids]}}` — server OR-merges each node's natural scope. **Does NOT use `kg_proximity`** (RFC-091 rejected 2026-06-03).
- In-episode: `EpisodeDetailPanel` grows a mini `QueryField` + inline `WorkspaceResults` (episode-scoped via `app_episodes` search endpoint).

**Acceptance:**
- Tier 1: `rail-search-launch.spec.ts` (5 rails × pre-scoped query correctness + `activeSearchContext` publish assertion).
- Tier 2: `search-production/rail-launch.spec.ts` — each rail launch produces correctly-scoped results on the fixture.
- Tier 3: recipe row per rail.
- Perf: none required (additive UI).

**Gates:** T1 + T2 + T3 required. **SIGSEGV impact: NO** (no retrieval changes).

**Risk:** low — additive.

---

### S7 — Saved queries + Recent (USERPREFS-1)

**Scope (ADR-119 conforms):**
- **No new server endpoints.** All persistence goes through the shipped `GET/PUT/PATCH /api/app/preferences`.
- USERPREFS-1 namespaces added (client-side conventions, no server schema change):
  - `search.savedQueries: Record<id, SavedQuery>` — `SavedQuery = { id, name, q, filters, operator?, created_at, updated_at, corpusHint? }`.
  - `search.recentQueries: RecentQuery[]` — ring buffer, last 20, `{ q, filters, ts, corpusHint? }`.
- New `useSavedQueriesStore` (thin Pinia around `useUserPreferencesStore`) — no independent hydrate, no localStorage mirror (USERPREFS-1 owns hydration). Exposes `save(name, request)`, `remove(id)`, `list()`, `pushRecent(request)`, `listRecent(limit)`, `resetAll()`.
- `WorkspaceSidebar.vue` renders Saved + Recent (populated in S3 as an empty-state; here they get data).
- Palette empty-state (S3) sections auto-populate.
- Missing filter values downgrade to defaults with a visible notice.

**Acceptance:**
- Tier 1: `saved-queries.spec.ts` (CRUD via mocked USERPREFS-1 endpoints; silent-degrade when `available === false`).
- Tier 2: `search-production/saved-queries.spec.ts` (fixture-scoped store hydrated from mocked USERPREFS payload).
- Unit: `web/gi-kg-viewer/src/stores/savedQueries.test.ts` (round-trip + silent-degrade + ring-buffer bounds + PATCH namespace correctness).
- No new server tests — the underlying `/api/app/preferences` routes are already tested (USERPREFS-1 sec).
- Perf: none required.

**Gates:** T1 + T2 required, unit required. **SIGSEGV impact: NO** — no retrieval changes.

**Risk:** low — additive; ops surface small; consumes shipped USERPREFS-1.

---

### S8 — Compare (2 subjects) operator

**Scope:**
- Compare button on `ResultSetOperatorBar` enabled when ≥ 2 subject types found in the hit set.
- Picker → 2-column view.
- Backend: new `POST /api/search/compare` endpoint. Body: `{ subject_a: SubjectRef, subject_b: SubjectRef, q?, max_tokens? }`; response: `{ pack_a: CorpusBriefingPack, pack_b: CorpusBriefingPack }`. Wraps `search/context_pack.py::build_briefing_pack(query, query_type, results, canonical_entity, max_tokens)` twice — the SHIPPED RFC-093 API. Judge summary from `search/judged_eval.py` muted below when available.
- Rail integration: pinning a subject prefills the Compare picker.

**Acceptance:**
- Tier 1: `operator-compare.spec.ts` (mocked context pack + judge).
- Tier 2: fixture provides 2 comparable speakers; renders side-by-side.
- Unit: `tests/unit/search/test_search_compare_endpoint.py`.
- Perf: `ui-operator-compare` captured (not gated in v1).
- Groundedness: judge summary hidden when `grounded: false`.

**Gates:** T1 + T2 required, unit required. **SIGSEGV impact: NO** — `build_briefing_pack` runs in Python after `rrf_fuse` returns.

**Risk:** medium — UI shape has real open questions (OQ4 in RFC).

---

## Cross-slice discipline

- **Every slice PR**: Tier-1 update + SIGSEGV impact line + `make lint-search-v3` green.
- **Rendered-surface slices** (S1/S2/S3/S4/S5/S6/S7/S8): + Tier-2 update + perf UI diff in PR body.
- **Retrieval-adjacent slices** (S1/S4/S5): + `make eval-search` diff in PR body.
- **Handoff slices** (S6): + Tier-3 recipe row.
- **New API slices** (S4/S8): + pytest unit + integration.
- **All slices**: no branch push without operator "push" / "ship it" (per `~/.claude/CLAUDE.md` rule 1); rebase onto `main` before push (rule 2).
- **Never re-enable** `_combine_hybrid_results` / `_normalize_scores` in any slice. `make lint-search-v3` enforces.
- **Never use `kg_proximity`** for retrieval or scope pre-fill (RFC-091 rejected).
- **Never persist UI state per-corpus** (ADR-119) — user state → USERPREFS-1; corpus telemetry stays in `search/query_log`.

## Not-covered / open

- **Streaming enriched answers** — OQ3 in RFC-107 (follow-up).
- **Compare UI shape** — OQ4 in RFC-107; S8 designs it.
- **Search-v3 for consumer app** — separate PRD later.
- **Default first-load tab** — OQ1 (stays Digest; per-user override via USERPREFS-1 later).
- **True typed-CONTRADICTS operator** — needs RFC-072 KL5 (v3+), out of scope.
- **New retrieval signals** — explicitly out of scope (PRD-045 Non-Goals).

## References

- [PRD-045](../prd/PRD-045-search-v3-query-workspace.md)
- [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md)
- [UXS-016](../uxs/UXS-016-query-workspace.md)
- [ADR-095](../adr/ADR-095-viewer-test-pyramid.md)
- [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md)
- [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md)
- [GRAPH_PERF_TRACE_RUNBOOK.md](../guides/GRAPH_PERF_TRACE_RUNBOOK.md)
- USERPREFS-1 (shipped): `docs/wip/USERPREFS-1.md`
- #1205 — LanceDB SIGSEGV incident + fix `0fe0854b`
