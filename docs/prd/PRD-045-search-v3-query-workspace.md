# PRD-045: Search v3 — Query Workspace (operator viewer)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.8
- **Parent PRD**: [PRD-031](PRD-031-search.md) (Search product surface), [PRD-033](PRD-033-search-powered-surfaces.md) (Search-powered surfaces)
- **Depends on**: [PRD-032](PRD-032-hybrid-corpus-search.md) / [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) (hybrid retrieval), [PRD-027](PRD-027-enriched-search.md) (enriched search), [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) (relational query layer + `activeSearchContext` + `PanelRetrievalStore` — **shipped**), [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) (enrichment layer / `QueryEnricher` — **shipped**), [RFC-093](../rfc/RFC-093-litm-context-packs.md) (`context_pack.build_briefing_pack` — **shipped**), USERPREFS-1 (`docs/wip/USERPREFS-1.md`, shipped)
- **Related UX spec**: extends [UXS-005](../uxs/UXS-005-semantic-search.md); introduces [UXS-016](../uxs/UXS-016-query-workspace.md); heroifies [UXS-008](../uxs/UXS-008-enriched-search.md); shell IA changes in [VIEWER_IA.md](../uxs/VIEWER_IA.md)
- **Related RFC**: [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md) (technical design)
- **Related ADRs**: [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) (`topic_consensus` activated, `stance_*` retired), [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) (no per-corpus UI state)
- **Related epic**: [#1229](https://github.com/chipi/podcast_scraper/issues/1229)
- **Scope**: operator viewer only (`web/gi-kg-viewer`). Consumer learning-player search is **out of scope** for this PRD.

---

## Summary

Promote **Search** in the operator viewer from a 288 px left-column sidebar into a **first-class main tab and shell-wide query surface** — a "Query Workspace" — and merge the current Search + Explore mode-switch into a single query surface. **Unlock backend capabilities the current UI does not expose but that have already shipped** (`insight_clusters`, `theme_clusters`, `context_pack.build_briefing_pack`, intent-adaptive result shape via the shipped query router, the shipped `topic_consensus` signal, `activeSearchContext` + `PanelRetrievalStore`, `relational_queries`, `QueryEnricher`); add operator-user features that require new work (Cmd-K palette, saved queries via USERPREFS-1, search-launch from every rail, in-episode search, enriched-answer hero placement, result-set operator bar). Ship with a Tier-1/Tier-2/Tier-3 test matrix, a quality-eval harness for well-defined text/corpus outcomes, and a Chrome-DevTools/CDP perf baseline mirroring the graph runbook.

**Hard constraint (from #1205):** the LanceDB native hybrid combine (`_combine_hybrid_results` / `_normalize_scores` via `pyarrow.compute`) segfaults the api under stack-test. Search v3 must NOT re-enable the in-engine hybrid path; retrieval stays on the Python-side `search_bm25 + search_vector + rrf_fuse` fan-out (`retrieval.py`).

**Hard constraint (from ADR-119):** no UI feature persists state scoped to the corpus. Corpus is transient (moving to a DB). User-scoped features live in USERPREFS-1 (`/api/app/preferences` + `useUserPreferencesStore`). Corpus telemetry (e.g. `search/query_log` for Dashboard analytics) is the only per-corpus persistence and is unchanged by this PRD.

## Background & Context

- The operator viewer has 30+ backend search modules (`src/podcast_scraper/search/`) plus a shipped relational + panel-cache layer (RFC-094 shipped v2.6.0–v2.7) plus a shipped enrichment layer (RFC-088 shipped v2.6/v2.7 with ongoing accuracy-gated iteration — ADR-108) plus a shipped LITM briefing-pack builder (RFC-093 shipped v2.7). The viewer surfaces ~5 of these end-user-visibly. The gap is UX, not backend.
- The left panel is a fixed `w-72` (288 px) column that shares its real estate with **Explore** via a slide mode-switch (`shell.leftPanelSurface`). Two overlapping query UIs, one column, one visible at a time — cognitive tax with no ergonomic upside.
- Main tabs answer *browse* questions (Digest / Library / Graph / Dashboard). "What am I trying to find out?" is not a first-class question in the shell.
- Result cards have handoffs OUT (`G` graph focus, `L` library episode, entity → rail) but almost no handoffs IN — Digest, Library, Graph, Dashboard, and the subject rails don't launch scoped searches. Search is a place, not a verb.
- The graph program (v3, #1207) landed a shell redesign, a perf runbook (`GRAPH_PERF_TRACE_RUNBOOK.md`), and a three-tier test pyramid (ADR-095). This PRD adopts the same test-pyramid + perf-capture pattern for search.
- #1205 (SIGSEGV in LanceDB native hybrid combine) is a live hazard: the fix bypassed the native combine and removed the process-wide query lock. Search v3 must inherit those guardrails and not silently regress them.

## Goals

1. **Search becomes a first-class citizen in the operator shell** — a main tab and a shell-wide command surface, not a sidebar.
2. **Merge Search + Explore into one query surface.** Kill the mode-switch. One filter chip bar, one result surface.
3. **Cmd-K / `/` command palette** — summonable from any tab; results have "open in Workspace", "pin to rail", "show on graph".
4. **Result-set operators** — cluster / on-graph / timeline / compare / consensus — unlock backend capability with visible verbs (all built on shipped modules).
5. **Enriched-answer hero** — when enrichment is configured, the answer is the hero of the Workspace, not a card at the bottom of a 288 px scroll. Consumes the shipped `QueryEnricher` + `/api/search enrich_results` wiring (RFC-088 chunk 5).
6. **Saved queries + query history — per-user via USERPREFS-1.** Cross-device, silent-degrade offline, no per-corpus persistence (ADR-119).
7. **Search-launch from every rail** — Episode / Topic / Person / Show / Graph-node rails each carry a "Search inside this" affordance that pre-fills scope and publishes to the shipped `activeSearchContext` store (RFC-094 OQ-2).
8. **In-episode search** on `EpisodeDetailPanel` — episode-scoped search inline in the rail.
9. **Three-tier test coverage** (Tier 1 mocked / Tier 2 production-shaped fixture / Tier 3 real-corpus Playwright), mirroring ADR-095.
10. **Quality eval harness** for search outcomes on a well-defined text corpus (nDCG, MRR, tier coverage, intent-router accuracy, compound-lift rate, enriched-answer groundedness, `topic_consensus` precision on the labelled set).
11. **Perf baseline + regeneration harness**, mirroring `capture-graph-lcp.{sh,mjs}` — first for the API (structured search endpoint per intent), then for the UI (Workspace TTI, filter-apply, cmd-K open, result-set operator latency).
12. **No SIGSEGV regression** — inherit and enforce the #1205 guardrails (Python-side fan-out; no native `_combine_hybrid_results`).

## Non-Goals

- **Consumer learning-player search** — separate PRD later; this is operator-only.
- **New retrieval algorithms** — Search v3 uses the shipped hybrid pipeline (RFC-090) as-is. Any new signal is out of scope.
- **KG-proximity retrieval signal** — **REJECTED** by RFC-091 Decision Record (2026-06-03; measured negative effect on nDCG and cross-show diversity). Search v3 does NOT use `kg_proximity` for retrieval, ranking, or "search around a graph selection" scope pre-fill.
- **New enrichment providers or ML tiers** — surfaces existing enricher + `topic_consensus` shipped signal; does not add providers.
- **Typed CONTRADICTS edges (KL5)** — v3+ per RFC-072 / RFC-097 roadmap; a full "Contradictions" operator remains blocked on those. The Search v3 "Consensus" operator surfaces the shipped `topic_consensus` signal (positive: cross-speaker corroboration) and NOT contradictions-as-assertion (which needs typed CONTRADICTS edges).
- **Re-enabling LanceDB native hybrid combine** — explicitly forbidden until #1205's root cause is fixed upstream (LanceDB / pyarrow) and independently verified.
- **New indexing surfaces** (no new `doc_type`s) — this PRD consumes today's index, does not add rows to it.
- **Cross-corpus search** — single-corpus, same scope as today.
- **Graph rewrites** — search-v3 must not touch `mergeGiKg` id shapes (#1219 owns that arc).
- **Per-corpus UI persistence of any kind** — see ADR-119. Corpus telemetry (`search/query_log` writes for Dashboard analytics) is not UI state and remains unchanged.
- **Working around inherited RFC-072 open items (KL2 Person alias registry / KL3 Topic dedup / KL4 episode duration).** Rail launchers, Compare, and Consensus display honest muted "may miss aliased variants" hints where relevant; the fix belongs upstream (RFC-072), not in Search v3.

## Personas

**Operator-analyst** (primary): jumps between corpora during a work session, needs to find evidence fast, compare speakers/episodes, cluster results, jump to graph or player. Signed in; saved queries roam with them.
**Operator-QA**: uses search to verify a specific pipeline change landed correctly across a corpus; needs saved queries and re-run.
**Operator-explorer**: doesn't know the query yet; wants Explore-style filters (topic contains / speaker contains / limit / min confidence) and presets in the same surface.

## User Stories

- *As an operator, I press Cmd-K from any tab and can query the whole corpus without losing my current view.*
- *As an operator, I open the Search tab and get a full-width Workspace with the enriched answer at the top and hits below.*
- *As an operator, I click "Cluster" on a 40-hit result set and see themes collapse.*
- *As an operator, I open a Topic in the right rail and click "Search inside this topic" — the Workspace opens pre-scoped and publishes to `activeSearchContext` so Library rows and Graph nodes weight accordingly.*
- *As an operator, I save "adverse events on GLP-1s" as a named query; on my second device, signed in as the same user, that query is there.*
- *As an operator, on an Episode rail I type in a mini search field and jump to a moment without leaving the rail.*
- *As an operator, I project a search result set onto the graph and see the subgraph highlighted.*
- *As an operator, I never see the api SIGSEGV under any concurrent search load in stack-test.*

## Functional Requirements

### FR1 — Search as 5th main tab

- **FR1.1** A new **Search** tab in the shell tab order: Digest · Library · **Search** · Graph · Dashboard. Default first-load unchanged (Digest).
- **FR1.2** The Search tab hosts a full-width **Query Workspace** (UXS-016): header (query field + intent chip + filter chip bar), enriched-answer hero (when available), result list with tier badges + compound cards, result-set operator bar, sidebar with saved + recent queries.
- **FR1.3** The LeftPanel Search column is retained as a **compact launcher** on non-Search tabs (query field + last-N recent queries); Enter opens the Workspace on the Search tab with that query already run.
- **FR1.4** Subject rail persistence rule (VIEWER_IA.md) holds: opening the Search tab does not clear the current subject.
- **FR1.5** The Workspace publishes to `activeSearchContext` (RFC-094 OQ-2, shipped) on every query — Library rows and Graph consume it as they do today.

### FR2 — Merge Search + Explore into one query surface

- **FR2.1** Explore's filters (topic contains / speaker contains / limit / min confidence / grounded only) fold into the Workspace filter chip bar as first-class chips; the slide mode-switch is retired.
- **FR2.2** Explore's presets ("Common questions", "Cross-episode") land as a **Preset** dropdown attached to the query field.
- **FR2.3** Explore's "quick question" natural-language field is the same query field — behavior converges.
- **FR2.4** The old `shell.leftPanelSurface = 'explore'` mode is removed; test IDs `left-panel-enter-explore` / `left-panel-back-search` are retired (release-note callout for the E2E surface map).

### FR3 — Command palette (Cmd-K / `/`)

- **FR3.1** `Cmd-K` (macOS) / `Ctrl-K` (other) opens a shell-wide overlay from **any** tab; the existing `/` shortcut is repointed to the palette (still respects "not-in-editable-control" rule from `useViewerKeyboard`).
- **FR3.2** Palette runs a live query (debounced) against the same backend as the Workspace; result rows carry three actions per hit: **Open in Workspace**, **Pin to rail**, **Show on graph**.
- **FR3.3** Palette closes on Escape / outside-click / action-select without disturbing the active tab or subject.
- **FR3.4** Palette exposes recent queries + saved queries (FR6) at the top when the query field is empty — both read from USERPREFS-1.

### FR4 — Result-set operators

Bar above the result list with operators that act on the **returned result set** (client-side over the current page; deeper operators call server-side aggregation over an over-fetched pool):

- **FR4.1 Cluster** — collapse hits into themes via the shipped `insight_clusters` / `theme_clusters` modules. Clusters are expandable; each cluster header carries a mini-count and the shared entity.
- **FR4.2 Show on graph** — extend today's per-hit `G` to a **set** projection: focus the graph camera on the union bbox of all hit-derived nodes, highlight the subgraph.
- **FR4.3 Timeline** — plot the hit set on the shared date lens (`corpusLens`), one bar per bucket; hit count on hover; click a bucket to filter the list.
- **FR4.4 Compare (2 subjects)** — pick two subjects from the hit set → open a compare view sourced from the shipped `search/context_pack.py`. Contract: for each side, run hybrid scoped to that subject, then `build_briefing_pack(query, query_type, results, canonical_entity, max_tokens)`; render the two `CorpusBriefingPack`s side-by-side. Judge summary (from `search/judged_eval.py`) muted below when available.
- **FR4.5 Consensus** — surface the shipped `topic_consensus` enricher (ADR-108, precision 0.91 on prod-v2): given a set of hits about the same Topic, show cross-speaker corroboration pairs. **Not "Contradictions"** — that requires typed CONTRADICTS edges (KL5, v3+, out of scope; see Non-Goals).

### FR5 — Enriched-answer hero

- **FR5.1** When `enriched_search_available: true` from `/api/health` and `/api/search` is called with `enrich_results=true` (shipped by RFC-088 chunk 5), the Workspace shows an **Enriched Answer** panel at the top of the results area. Contract inherits UXS-008.
- **FR5.2** Sources in the answer link to the corresponding hit cards below (visible tie-in — chip on the hit card).
- **FR5.3** Enrichment latency degradation follows UXS-008: skeleton while pending; hidden on hard-fail; raw hits always ship first.
- **FR5.4** Enrichment is **never** invoked from the LeftPanel compact launcher or from the Cmd-K palette — Workspace only (bounds cost and latency).
- **FR5.5** The `EnrichmentConfigEditor.vue` (already shipped, RFC-088 config surface) remains the operator-facing config UI; Search v3 does NOT reinvent it.

### FR6 — Saved queries + query history (USERPREFS-1)

- **FR6.1** Saved queries live under the USERPREFS-1 key `search.savedQueries` as a map `Record<id, SavedQuery>`. `SavedQuery` shape: `{ id, name, q, filters (JSON), operator?, created_at, updated_at, corpusHint? }`. **No new server endpoints** — all round-trips go through `PATCH /api/app/preferences` (shipped).
- **FR6.2** Recent queries live under the USERPREFS-1 key `search.recentQueries` as a ring buffer of the last 20 entries, `{ q, filters, ts, corpusHint? }`. Per-user, cross-device, silent-degrade offline (same USERPREFS-1 semantics).
- **FR6.3** The Dashboard's QueryActivityChart continues to read `search/query_log` (corpus-side telemetry) unchanged — that is corpus analytics, not user state (ADR-119).
- **FR6.4** UI: `WorkspaceSidebar.vue` renders **Saved** + **Recent** sections; Cmd-K palette empty-state shows the same two sections.
- **FR6.5** Re-run applies the saved filters + operator verbatim; missing filter values (e.g. a `since` on a corpus that no longer has that date range) downgrade to defaults with a visible notice.
- **FR6.6** When signed out / offline, saved-queries UI is hidden (same silent-degrade rule as the graph-lenses store); recent-queries falls back to session-local memory only.

### FR7 — Search-launch handoffs from every rail

Every rail launcher publishes the pre-filled scope to `activeSearchContext` (RFC-094 OQ-2, shipped) so downstream consumers (Library rows, Graph nodes) react as they do today; the Search tab activates with the query field pre-filled.

- **FR7.1** `EpisodeDetailPanel` header carries **"Search inside this episode"** → publishes `{episode: metadata_relative_path}` scope.
- **FR7.2** `TopicEntityView` / polymorphic node view header carries **"Search this topic"** → publishes `{topic: id}` scope; may also use the shipped `who_said(topic_id)` from RFC-094 to seed the initial view.
- **FR7.3** `PersonLandingView` header carries **"Search this person"** → publishes `{person: id}` scope (speaker filter); the existing `positions_of(person_id)` from RFC-094 remains the person's rail content — this is search-in-addition, not search-replaces.
- **FR7.4** `ShowRailPanel` header carries **"Search inside this show"** → publishes `{feed: feed_id}` scope.
- **FR7.5** `GraphNodeRailPanel` context menu carries **"Search in scope of this selection"** — pre-fills scope as the **union of the selected nodes' natural filters** (episode filter for episode nodes, topic filter for topic nodes, speaker filter for person nodes). **Does NOT use `kg_proximity`** (RFC-091 rejected 2026-06-03).

### FR8 — In-episode search on EpisodeDetailPanel

- **FR8.1** Compact search field on `EpisodeDetailPanel` (below the header, above the sections) queries `episode`-scoped search via the existing `app_episodes` search endpoint.
- **FR8.2** Hits are inline (no Workspace navigation), each with a jump-to-moment; open-in-Workspace escape hatch preserved.

### FR9 — Test pyramid (mirrors ADR-095)

- **FR9.1** **Tier 1 — Fast matrix (mocked)**: contract tests for every new surface (Workspace, palette, operator bar, rail launchers, saved queries). Runs under `make ci-ui-fast`.
- **FR9.2** **Tier 2 — Production-shaped fixture**: extend `web/gi-kg-viewer/e2e/fixtures/production-shaped/` with a search-oriented slice (25 episodes, mixed doc types, at least one compound-lift pair, at least one `topic_consensus` pair, at least one theme cluster of ≥ 5 members). Regenerable via `scripts/build_production_shaped_fixture.py` (extend, don't fork). Runs under `make ci-ui-full`.
- **FR9.3** **Tier 3 — Real-backend Playwright** over the search UX ("search-tier3"): mirrors the graph tier-3 pattern (`playwright.validation.config.ts`). Runs under `make ci-ui-validation CORPUS=…` and on scheduled cron. **Institutional rule (ADR-095):** every Tier-3 bug lands a Tier-2 matrix row before its fix PR merges.

### FR10 — Quality eval harness (search)

- **FR10.1** A checked-in synthetic corpus (or extended `tests/fixtures/viewer-validation-corpus/v3`) with a labelled **query set**: at least 25 queries covering the 5 intent classes (`entity_lookup / raw_evidence / temporal_tracking / cross_show_synthesis / semantic` — matches the shipped router taxonomy per RFC-092), each with a rubric-labeled top-K expected set and per-query nDCG floor.
- **FR10.2** Metrics: nDCG@10, MRR@10, tier coverage (Insight / Transcript / Reference presence), intent-router accuracy, compound-lift rate, enriched-answer **groundedness rate** (fraction of enriched-answer sources that are marked `grounded: true`), `topic_consensus` precision on the labelled subset.
- **FR10.3** Reproducible CLI: `python scripts/eval/search_quality.py --corpus … --query-set … --out …` → JSON report + optional summary.md. CI target: `make eval-search`.
- **FR10.4** Baseline captured on `search-v3` HEAD **before** any UI-facing change lands; re-captured per slice; regressions block the slice.

### FR11 — Perf baseline + regeneration harness

- **FR11.1** New scripts `scripts/dev/capture-search-perf.{sh,mjs}` mirroring `capture-graph-lcp.{sh,mjs}`:
  - **API surface** — an HTTP capturer that records p50/p95/p99 for `/api/search`, `/api/app/search`, `/api/corpus/search` per intent class + per top_k, over the query set from FR10.
  - **UI surface** — Chrome DevTools/CDP trace of: Workspace-open (TTI), cmd-K-open latency, filter-apply, result-set operator ("Cluster", "Show on graph"), enriched-answer paint.
- **FR11.2** Median-of-3 (same fair-comparison rule as GRAPH_PERF_TRACE_RUNBOOK); outputs `.metrics.json` + gzipped trace per label; committed under `docs/wip/search-v3/traces/`.
- **FR11.3** Baseline against `main` captured on branch tip **before** slice 1; re-captured per slice; deltas reported in each PR body.
- **FR11.4** After baseline, a **deep-review pass** (backend + frontend) surfaces optimization candidates — issues opened, not silently applied.

### FR12 — SIGSEGV guardrails (from #1205)

- **FR12.1** No code path re-enables `_combine_hybrid_results` / `_normalize_scores` in `lancedb_backend`. Retrieval routes through the Python-side `search_bm25 + search_vector + rrf_fuse` fan-out (`retrieval.py`).
- **FR12.2** The Digest warm-up path (one serial band query before parallel fan-out; commit `0fe0854b`) is preserved.
- **FR12.3** No process-wide query lock is re-introduced; concurrency is bounded per-request via the fan-out's own executor.
- **FR12.4** A regression test asserts that under concurrent search load (N ≥ 4 parallel requests over a real LanceDB corpus) the api returns 200 / no faulthandler signal / no SIGSEGV — mirrors the #1205 repro harness at `tests/…/test_lancedb_concurrent_repro` (extend, don't fork).
- **FR12.5** Every RFC-107 §API addition and every RFC-107 §Store/Executor change carries a "SIGSEGV impact" line answering: *does this add a call site into the native combine, or into a new native pyarrow.compute path?* Default = NO; any YES requires an ADR update.

## Success metrics

- **UX**: median session includes ≥ 1 Cmd-K use; ≥ 30 % of search sessions apply a result-set operator; enriched-answer clicks-through-to-source ≥ 50 %; saved-query re-run rate ≥ 20 % of returning-signed-in sessions.
- **Quality**: nDCG@10 ≥ baseline; intent-router accuracy ≥ 80 % on the labelled query set; enriched-answer groundedness rate = 100 % (any lower = bug); `topic_consensus` precision ≥ 0.85 on the labelled subset (shipped baseline is 0.91).
- **Perf**: Workspace TTI ≤ 700 ms on the production-shaped fixture; cmd-K open ≤ 100 ms; enriched-answer first-paint ≤ 800 ms; API p95 per intent ≤ current baseline + 0 %.
- **Reliability**: zero SIGSEGV incidents under stack-test / concurrent search over 30 days after ship.
- **Test-pyramid rule**: every Tier-3 bug across the search UX gets a Tier-2 row before merge (audited retro).

## Test / eval / perf / SIGSEGV gates per slice

Each slice PR MUST include:

1. Tier-1 mocked spec update (contract) — always.
2. Tier-2 production-shaped spec update — when the slice touches a rendered surface.
3. Tier-3 recipe row in `web/gi-kg-viewer/e2e/handoff*/…` — when the slice adds a new rail launcher or shell-wide handoff.
4. `make eval-search` diff (baseline vs slice) in the PR body — when the slice changes retrieval, filters, or enriched-answer.
5. `capture-search-perf` metrics.json diff — when the slice touches API or a rendered surface.
6. **SIGSEGV impact line** — always (yes/no + justification).

Slices missing any of these get rejected at review, not deferred.

## Open questions

- **OQ1** Should the Search tab be default first-load for operator-analysts (behind a per-user USERPREFS-1 pref)? Default remains Digest for now.
- **OQ2 — RESOLVED** Saved queries live in USERPREFS-1 (ADR-119). Per-corpus JSON retracted.
- **OQ3** Enriched-answer streaming (server-side chunked response) vs blocking-then-render — v1 blocking, streaming as follow-up.
- **OQ4** Compare-2-subjects UI: side-by-side vs diff view; punted to RFC-107 §FR4.4.
- **OQ5** Should the palette be `Cmd-K` (universal) or `Cmd-P` (VS Code-like) — bike-shed; RFC-107 §Keybindings picks `Cmd-K` and locks unless operator objects.
- **OQ6** Cluster server-side depth — how far to over-fetch when operator === 'cluster' (config; default `top_k * 3` in RFC-107).

## Related

- [PRD-031](PRD-031-search.md) — Search product surface (parent)
- [PRD-033](PRD-033-search-powered-surfaces.md) — Search-powered surfaces (predecessor)
- [PRD-027](PRD-027-enriched-search.md) — Enriched search (heroified here)
- [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) — hybrid retrieval (do not touch native combine — #1205)
- [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) — relational query layer + `activeSearchContext` + `PanelRetrievalStore` (shipped substrate)
- [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — enrichment layer + QueryEnricher (shipped)
- [RFC-093](../rfc/RFC-093-litm-context-packs.md) — `build_briefing_pack` (shipped; Compare operator source)
- [RFC-092](../rfc/RFC-092-ml-query-router.md) — query router intent taxonomy
- [RFC-091](../rfc/RFC-091-kg-proximity-signal.md) — **REJECTED** (KG proximity as retrieval signal)
- [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) — canonical identity / chunk-to-Insight lift (compound-card contract)
- [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) — v2 materialized descriptive edges (typed CONTRADICTS remains v3+)
- [ADR-095](../adr/ADR-095-viewer-test-pyramid.md) — three-tier test pyramid (adopted)
- [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md) — LanceDB-first (SIGSEGV context)
- [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) — `topic_consensus` activated; `stance_*` retired
- [ADR-119](../adr/ADR-119-no-per-corpus-ui-state.md) — no per-corpus UI state
- [VIEWER_IA.md](../uxs/VIEWER_IA.md) — shell IA (tab order + LeftPanel role change)
- [UXS-005](../uxs/UXS-005-semantic-search.md) — extended (compact launcher role)
- [UXS-008](../uxs/UXS-008-enriched-search.md) — heroified
- [UXS-016](../uxs/UXS-016-query-workspace.md) — Query Workspace (this PRD's primary UX)
- [GRAPH_PERF_TRACE_RUNBOOK.md](../guides/GRAPH_PERF_TRACE_RUNBOOK.md) — perf-capture template (mirrored)
- [ENRICHMENT_LAYER_GUIDE.md](../guides/ENRICHMENT_LAYER_GUIDE.md) — current normative operator-facing enrichment config surface (RFC-088 chunk 6)
- [ENRICHMENT_LAYER_API.md](../api/ENRICHMENT_LAYER_API.md) — `/api/enrichment/config*` routes (independent of Search v3's `/api/search?enrich_results=`)
- USERPREFS-1: `docs/wip/USERPREFS-1.md`
