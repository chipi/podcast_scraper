# Learning player — browser E2E

Playwright specs for the consumer app, run against a **real backend over the committed corpus**.
This file orients you; the contract lives in **[`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md)** — read
that before touching a selector.

## Run

```bash
cd web/learning-player
npx playwright install chromium     # once — this app is Chromium; the viewer is Firefox
npm run test:e2e                    # both projects
npm run test:e2e -- --project=mobile-chrome e2e/follow-show.spec.ts
npm run test:e2e:ui                 # interactive
```

`make test-app-e2e` from the repo root does the same with install steps.

**On a machine where the repo venv cannot run the API** (no ML extras — e.g. Intel macOS), use
[`run-local-stack.sh`](run-local-stack.sh), which starts the API in Docker against the fixture
corpus and then runs Playwright. It **recreates the app-state volume every run**: the suite is not
hermetic across runs, and a surviving volume makes `follow-show.spec` fail on a show a previous run
already followed — which looks exactly like a regression. Do not optimise that away.

## Two projects, and why both matter

`mobile-chrome` (Pixel 7) is listed **first** deliberately — mobile is the primary platform. A spec
that only passes on desktop is half-written.

The two viewports have different navigation: the bottom tab bar is `sm:hidden`, the header icon nav
is desktop-only. Use **`navTo(page, 'search')`** from [`helpers.ts`](helpers.ts) rather than clicking
either directly, and never `page.goto()` for in-app navigation — that is a full page load, which
tears down the SPA and stops audio, so it cannot test client-side routing at all.

## Helpers

| Helper | Use it for |
| --- | --- |
| `signInIsolated(page, who, testInfo)` | A stable per-(test, project) user id, so parallel specs never share state |
| `navTo(page, dest)` | Viewport-agnostic in-app navigation |
| `openTranscript(page)` | The transcript toggle differs by viewport |

## Mocks — the exceptions, and why they exist

The suite's contract is **no mocks**: the Playwright `webServer` boots a real API over
`tests/fixtures/app-validation-corpus/v3` plus the mock podcast host serving the real fixture audio,
so specs exercise the actual server. The audio stub is gone (#1618). Remaining exceptions:

- **5 data-shape stubs** in `perspectives`, `entity-signals` and `search-listener-features`, for
  states the corpus cannot produce (no speaker has >2 insights on one topic, etc). Each is a fixture
  gap, recorded in [`CORPUS-V4-FIXTURE-LADDER.md`](../../../docs/wip/CORPUS-V4-FIXTURE-LADDER.md)
  (#1189) so corpus v4 can retire them.

Do not add a sixth. If you need a state the corpus lacks, extend the corpus — a per-spec stub is
invisible to every other spec and hides the gap.

## `globalSetup.ts` does two things

1. **Wipes per-user state** (`e2e/.app-state`) so local runs match a clean CI checkout. That dir is
   gitignored but persists locally, and a prior run's writes leak into fresh-user assertions.
2. **Builds the two-tier search index if absent.** The corpus is committed but its LanceDB index is
   not (binary, format-coupled). Several routes branch on `has_index`, so a missing index silently
   changes which grounded claim a perspective surfaces. Needs the `[search]` extras and the cached
   MiniLM model, offline.

## The surface map is enforced

`src/__checks__/surface-map.test.ts` fails the **unit** suite when routes or `data-testid`s drift
from [`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md) — in both directions: selectors in code but not in
the map, and map entries that no longer exist. Update the map in the same commit as the UI change.

## Where the data comes from

The corpus, audio, RSS and transcripts all live in the Python half of the repo and are **versioned**
— see the table at the top of [`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md),
[`../../../tests/README.md`](../../../tests/README.md), and
[`../../../tests/fixtures/README.md`](../../../tests/fixtures/README.md). If an asset looks missing
from in here, search from the repo root before concluding it does not exist.

## Related

- [`../../README.md`](../../README.md) — the two web apps and how they differ
- [`../../../docs/guides/E2E_TESTING_GUIDE.md`](../../../docs/guides/E2E_TESTING_GUIDE.md)
- [`live/`](live) — specs that run against a deployed environment rather than fixtures
