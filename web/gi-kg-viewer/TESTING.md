# Viewer testing — what exists & how to run it

A map of the viewer's test tiers (graph included), the commands, and the corpora.
Canonical rationale lives in **ADR-095** (viewer test pyramid), **RFC-086** (test
pyramid + production-shaped fixtures), and the strategy's
[Browser UI E2E](../../docs/architecture/TESTING_STRATEGY.md#browser-ui-e2e-playwright)
section; this file is the operator quick-ref. For **which surfaces/selectors** a
browser spec should hit (not how to run), see
[`e2e/E2E_SURFACE_MAP.md`](./e2e/E2E_SURFACE_MAP.md).

All commands run from `web/gi-kg-viewer/` unless noted. Use `npm run …` /
`node_modules/.bin/…` — never `npx playwright` (it refetches a fresh copy and
gets SIGKILLed).

## The tiers

| Tier | What | Runner | Command |
| --- | --- | --- | --- |
| **1 — Unit** | logic: stores / utils / services (`src/**/*.test.ts`, no DOM) | Vitest | `npm run test:unit` |
| **2 — Component** | `.vue` mount tests (`@vue/test-utils` + happy-dom) | Vitest | `npm run test:unit` (same run) |
| **3 — E2E (mocked)** | real browser, **mocked API** (`page.route`) + offline GI/KG fixtures; own Vite dev server on **5174** | Playwright (`playwright.config.ts`) | `npm run test:e2e` |
| **3/4 — Real-corpus walk** | real browser against a **live `make serve` stack** (UI 5173 + API 8000) + a real on-disk corpus; one scenario per matrix surface + screenshots + console-error assertions | Playwright (`playwright.validation.config.ts`) | `make ci-ui-validation CORPUS=<path>` (from repo root) |

Tiers 1 + 2 share one Vitest run (the file's `// @vitest-environment` decides
DOM vs not). `playwright.config.ts` **ignores** `e2e/validation/**` (those need a
live backend), so the mocked tier never touches a real corpus.

### Run just the graph tests

```bash
# Tier 1+2 — every graph store/util/service/component test (path-substring filter)
npx vitest run graph cy        # ~36 files / ~670 tests

# Tier 3 (mocked) — the graph browser specs
node_modules/.bin/playwright test \
  e2e/graph-*.spec.ts e2e/offline-graph.spec.ts e2e/keyboard-shortcuts.spec.ts \
  e2e/export-png.spec.ts e2e/search-to-graph-mocks.spec.ts e2e/topic-entity-view.spec.ts
```

## Coverage gate (#914)

`npm run test:coverage` runs Vitest with v8 coverage and the **thresholds in
`vite.config.ts`** (currently `statements 78 / branches 66 / functions 77 /
lines 80`). CI's `viewer-unit` job runs this, so a UI-coverage regression fails
the job — parallel to the Python coverage gate, kept on a separate track.

Note the moving denominator: v8 scopes coverage to files *imported by a test*
(no `all`). Mounting container components (shell/panels) pulls their whole
transitive `.vue` import tree into the denominator, so adding component tests
*broadens* scope and can lower the headline % even as absolute coverage grows.
Ratchet the thresholds a few points below the new baseline when coverage climbs.

## Corpora

| Corpus | What | Where |
| --- | --- | --- |
| **Synthetic validation corpus** | in-repo, deterministic, **pre-built API-response JSONs** (`digest.json`, `episodes.json`, `artifacts.json`, `topic-clusters.json`, …). 9 podcasts / 23 episodes / GI+KG nodes / 5 cross-cutting umbrella topics / recent publish dates so the default 7-day lens catches everything. ~300 KB. | `tests/fixtures/viewer-validation-corpus/v{1,2}/corpus` (`FIXTURES_VERSION` selects the version; see `tests/fixtures/VIEWER_VALIDATION_CORPUS.md`) |
| **BYOC (real / prod)** | a raw pipeline-output corpus (`metadata/*.gi.json` + `*.kg.json`). Never named in committed code (copyright/privacy) — always supplied via `CORPUS=`. | e.g. a prod backup under `.test_outputs/manual/…/corpus` |

Regenerate the synthetic corpus (deterministic, idempotent):

```bash
python scripts/build_synthetic_validation_corpus.py
```

## Running the real-corpus validation walk (tier 3/4)

The walk has **no built-in web server** — it drives an already-running stack.

**Against the synthetic corpus** (API root must be the repo root so the fixture is
reachable):

```bash
# terminal 1 — serve with the repo root as the API corpus root
make serve-for-validation

# terminal 2 — drive the validation walk
make ci-ui-validation CORPUS="$(pwd)/tests/fixtures/viewer-validation-corpus"
```

**Against a real / prod corpus** (BYOC):

```bash
# terminal 1
make serve SERVE_OUTPUT_DIR=/abs/path/to/corpus
# terminal 2
make ci-ui-validation CORPUS=/abs/path/to/corpus
```

Specs live in `e2e/validation/` (`real-corpus.spec.ts`,
`handoff-matrix-real-corpus.spec.ts`); screenshots land in
`test-results/` / the Playwright report.

### RFC-086 institutional rule

A bug surfaced by the Tier-3 real-corpus walk must land a **Tier-2 matrix row**
under `e2e/handoff-production/` that reproduces it **before** the fix PR merges.
That keeps the cheap mocked tier as the regression guard for anything the
expensive real walk finds.

## Where things live

```text
src/**/*.test.ts                     # tier 1 (logic) + tier 2 (component mount)
e2e/*.spec.ts                        # tier 3 mocked (page.route + offline fixtures)
e2e/validation/*.spec.ts             # tier 3/4 real-corpus walk
e2e/handoff-production/*.spec.ts     # Tier-2 matrix rows (RFC-086 regression guard)
vite.config.ts  → test.coverage      # #914 coverage gate thresholds
playwright.config.ts                 # mocked e2e (own dev server, 5174)
playwright.validation.config.ts      # real-corpus walk (expects make serve, 5173)
tests/fixtures/viewer-validation-corpus/   # synthetic corpus (in repo root)
scripts/build_synthetic_validation_corpus.py
```
