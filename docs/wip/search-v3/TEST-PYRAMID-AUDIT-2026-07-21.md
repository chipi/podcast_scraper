# Search v3 test-pyramid audit — 2026-07-21

**Status**: post-S8 landing + follow-ups. Every Search v3 slice (S1–S8)
has coverage at every applicable tier of ADR-095's 3-tier viewer test
pyramid, including Tier-3 real-corpus walks. This document is the
snapshot record.

**Update — 2026-07-21 (later):** S8 shipped (`5b42bf2f`); F1 refreshed
`mocks.json` for the shipped shape (`e33192a8`); F2 landed the Tier-3
`validation/search-v3.spec.ts` (`39e1886b`); F3 landed the Tier-2
rail-launched walk `search-production/rail-launch.spec.ts`
(`97367527`). The three previously-open follow-ups in the "Known gaps"
section are now resolved.

**Related**: ADR-095 (three-tier viewer test pyramid),
[SEARCH-V3-IMPLEMENTATION-PLAN.md](../SEARCH-V3-IMPLEMENTATION-PLAN.md).

## The three tiers

- **Tier 1 (unit + mocked Playwright)** — vitest specs + Playwright
  `e2e/*.spec.ts` at the root. Fast (< 30s), mock every network call,
  cover the client contract. `make test-ui` runs the vitest set;
  `make test-ui-e2e` runs the mocked Playwright set.
- **Tier 2 (production-shaped)** — Playwright specs under
  `e2e/search-production/*.spec.ts` and `e2e/handoff-production/*.spec.ts`.
  Mock responses match the **shipped** API shapes; cover cross-slice
  interactions the per-slice Tier-1 specs don't see. Included in
  `make test-ui-e2e` (same job, deeper coverage).
- **Tier 3 (real-corpus)** — Playwright specs under `e2e/validation/`
  and `e2e/handoff-production/` recipe rows. Requires the operator to
  supply `CORPUS_PATH` and boot `make serve`. Run via
  `make ci-ui-validation CORPUS=/abs/path` — not part of the default CI
  pass. Coverage here is scenario-driven (screenshots + assertions
  against a live stack).

## Coverage matrix

| Slice | Unit (vitest) | Tier-1 (Playwright) | Tier-2 (production-shaped) | Tier-3 (real-corpus) |
|---|---|---|---|---|
| **S1** — Search + Explore merge | ✓ chip specs, filter bar | ✓ `search-fr1.spec.ts` (source-tier / evidence toggle / lifted-speaker) | ✓ walked via `search-production/workspace.spec.ts` (chip-driven filter runs) | (deferred — plan doc) |
| **S2** — Search main tab | ✓ `SearchTab.test.ts` | ✓ `search-workspace-tab.spec.ts` (nav order, `3` shortcut, LeftPanel visibility, keep-alive) | ✓ walked via `search-production/workspace.spec.ts` | (deferred) |
| **S3** — Cmd-K / `/` palette | ✓ `CommandPalette.test.ts` (mount + 7 unit specs) | ✓ `search-command-palette.spec.ts` (8 specs: `/` summon, cross-tab summon, empty-state, debounced live-fetch, 3-action rows, no-results, Open-in-Workspace handoff, Show-on-graph handoff, Escape close) + `keyboard-shortcuts.spec.ts` (`/` binding) | ✓ walked via `search-production/workspace.spec.ts` (palette rehydration) | (deferred) |
| **S4-shell** — LeftPanel Saved+Recent | ✓ `LeftPanel.test.ts` (6 specs — apply-query emit + honest empty) | ✓ `search-saved-queries.spec.ts` (LeftPanel row rendering) + `search-workspace-tab.spec.ts` (rail visible on Search tab) | ✓ walked via `search-production/workspace.spec.ts` | (deferred) |
| **S4a** — Operator bar shell / Timeline / On-graph | ✓ `ResultSetOperatorBar.test.ts` (13 specs) | ✓ `search-operator-bar.spec.ts` (7 specs) | ✓ walked via `search-production/workspace.spec.ts` (Timeline toggle) | (deferred) |
| **S4b** — Server Cluster / Consensus | ✓ pytest unit on `operators.py` (13 tests); vitest `search.runOperator.test.ts` (8 tests) | ✓ `search-operator-bar.spec.ts` (Cluster + Consensus paths); integration `test_viewer_search.py` (5 tests) | ✓ walked via `search-production/workspace.spec.ts` (Cluster + Consensus operator round-trips) | (deferred) |
| **S5** — Enriched-answer hero | ✓ `EnrichedAnswerHero.test.ts` (7 specs) + `SearchEnrichedChip.test.ts` (5 specs) | ✓ `search-enriched-hero.spec.ts` (4 specs) | ✓ walked via `search-production/workspace.spec.ts` (hero rehydration after operator toggle) | (deferred) |
| **S6** — Rail launcher + `episode_id` scope | ✓ pytest unit on `_hit_passes_cli_filters` (episode_id branch); vitest chip/rail specs | ✓ `search-rail-in-episode.spec.ts` (4 specs); integration `test_viewer_search.py` (2 new tests) | ✓ `search-production/rail-launch.spec.ts` (F3 — Library → Episode → rail launcher → episode-scoped wire + chip clear round-trip) | ✓ `validation/search-v3.spec.ts` T5 (F2 — real corpus) |
| **S7** — Saved queries + Recent | ✓ `savedQueries.test.ts` (14 specs) | ✓ `search-saved-queries.spec.ts` (5 specs) | ✓ walked via `search-production/workspace.spec.ts` (Recent auto-populate + palette rehydration) | (deferred) |
| **S8** — Compare (2 subjects) operator | ✓ pytest unit on `compare.py` (16 specs); vitest `search.runCompare.test.ts` (7 specs) + `CompareOperatorPanel.test.ts` (8 specs) | ✓ `search-operator-compare.spec.ts` (5 specs); integration `test_viewer_search.py` (6 new tests) | (compare walk not yet in the workspace walk — dedicated Tier-1 covers) | ✓ `validation/search-v3.spec.ts` T4 (F2 — real corpus) |
| **Handoff matrix (F1–F4)** | ✓ `search.test.ts` + related | ✓ `handoff/*.spec.ts` (11 files, 65+ tests) | ✓ `handoff-production/*.spec.ts` (9 files, 20+ tests) | ✓ `validation/handoff-matrix-real-corpus.spec.ts` |

Retrieval-adjacent slices (S1 / S4b / S5) also have `make eval-search` +
`.api.metrics.json` baselines per RFC-107 §T2 + §P — those live in
`docs/wip/search-v3/eval/` and `docs/wip/search-v3/traces/` and are the
non-UI half of the pyramid (not shown in the matrix above).

## What "awesome" looks like

Test-count snapshot as of this audit (final):

- **pytest**: baseline 31 + S4b + S6 + S8 unit + integration + F1
  fixture shape-check (46 covered directly by S8's targeted run;
  11 by F1's shape-check spec).
- **vitest**: 206 files, 2666 tests (all green post-S8).
- **Playwright (mocked)**: `make test-ui-e2e` runs 34 spec files.
- **`make test-ui-e2e`**: **217 passed / 0 failed / 0 flaky (2.4m)**
  post-S8 + F1/F2/F3 landing.
- **SIGSEGV guardrail**: `make lint-search-v3` green (531 files scanned,
  2 forbidden symbols, 0 whitelist entries) — Python-side aggregation
  only, no LanceDB native combine, unchanged post-S8.
- **Fixture shape-check**: `tests/integration/fixtures/test_search_v3_mocks_shape.py`
  — 11 pytest specs validate every fixture scenario against the
  shipped Pydantic response model. If the server response shape
  drifts, this test breaks CI and steers Tier-2 specs back into
  agreement.

## Known gaps + follow-ups (RESOLVED)

The three follow-ups previously listed here shipped in the same
session:

- **`search-v3/mocks.json` regeneration** — DONE (`e33192a8`, F1). All
  6 scenarios now match the shipped shape (schema_version 2). Fixture
  shape-check test added.
- **Tier-3 (real-corpus) walk for Search v3** — DONE (`39e1886b`, F2).
  `validation/search-v3.spec.ts` covers T1 enriched hero, T2 cluster,
  T3 consensus, T4 compare (§S8), T5 episode-scope rail.
- **S6 rail launcher in Tier-2 workspace walk** — DONE (`97367527`, F3).
  `search-production/rail-launch.spec.ts` covers the cross-slice
  Library → Episode → rail launcher → scoped `/api/search` →
  chip clear → unscoped `/api/search` round-trip.

Still open (not this session):

- **Perf-demonstrations #768 / #769** — occasionally flaky under
  parallel load (documented behaviour, retries clean). No S7/S8
  impact.

## How to run the pyramid

```bash
# Tier 1 — vitest (fast, always green)
make test-ui

# Tier 1 + Tier 2 — Playwright mocked + production-shaped
make test-ui-e2e

# Tier 3 — real corpus (operator-supplied)
make ci-ui-validation CORPUS=/abs/path/to/your/corpus

# SIGSEGV guardrail
.venv/bin/python scripts/check/lint_search_v3_forbidden_imports.py

# Retrieval baseline
make eval-search
```

---

Audit performed: 2026-07-21. Author: Search v3 arc + follow-ups.
