# GI/KG viewer — browser E2E

Playwright specs for the operator viewer. This file orients you; the contract lives in
**[`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md)** — read that before touching a selector.

## Run

```bash
cd web/gi-kg-viewer
npx playwright install firefox      # once — this app is FIREFOX; the player is Chromium
npm run test:e2e
npm run test:e2e -- e2e/library.spec.ts
npm run test:e2e:ui                 # interactive
```

`make test-ui-e2e` from the repo root does the same with install steps. The Playwright `webServer`
runs Vite on `127.0.0.1:5174` with `reuseExistingServer: true`, so an already-running dev server is
reused.

**Running on Chromium by mistake is the most common first failure here** — a missing-executable
error that reads like a broken install. It is the other app's browser.

## Layout

| Path | What |
| --- | --- |
| `*.spec.ts` (top level) | Feature specs — digest, library, graph, search, dashboard, topic/entity |
| [`handoff/`](handoff), [`handoff-production/`](handoff-production) | The interaction handoff matrix (FSM behaviours) — see [`HANDOFF_MATRIX.md`](HANDOFF_MATRIX.md) |
| [`search-production/`](search-production) | Search against production-shaped fixtures |
| [`validation/`](validation) | Tier-3 walk against a **live corpus** (`make ci-ui-validation`) — see [`validation/README.md`](validation/README.md) |
| [`perf/`](perf), [`live/`](live) | Performance demonstrations; specs against a deployed environment |
| `helpers.ts`, `fixtures.ts`, `dashboardApiMocks.ts` | Shared setup and the route-mock payloads |

The loose `*_ANALYSIS.md` / `*_RESULTS.md` / `TEST_EXECUTION_REPORT_*.md` files are **historical
investigation notes**, not contracts. `HANDOFF_MATRIX.md` and `E2E_SURFACE_MAP.md` are current.

## Mocks — the architecture, and where it is going

Most specs here **route-mock the API** (`page.route('**/api/…')`) — **37 of 68 spec files, 243
interceptions**, plus four shared helpers that mock on their behalf (`helpers.ts`,
`dashboardApiMocks.ts`, `handoff/_handoff-helpers.ts`, `handoff-production/_helpers.ts`). That was
not a shortcut: this suite's `webServer` has only ever run **Vite alone**, with no backend behind
it, so fulfilling payloads per spec was the only option available.

It is no longer necessary. The **same fixture-bootstrapped API the consumer suite uses** serves
every endpoint this app calls — `/api/corpus/*`, `/api/search`, `/api/index/stats`, `/api/artifacts`.
[`run-local-stack.sh`](run-local-stack.sh) starts it in Docker on `:8012` and points Vite's `/api`
proxy at it:

```bash
./e2e/run-local-stack.sh                          # whole suite against the real backend
./e2e/run-local-stack.sh e2e/library.spec.ts      # one spec
```

Migration is [#1619](https://github.com/chipi/podcast_scraper/issues/1619), spec by spec. It is
**not mechanical**: each mock encodes an assumed backend state, so moving one means either the
fixture produces that state or the assertion changes. A bulk find-and-replace would leave 243 green
tests asserting nothing.

(If you find a smaller number quoted elsewhere — "31 files / 202" appears in the #1619 commit —
that came from a regex that only matched string-literal first arguments and missed calls taking a
variable or a `RegExp`. The counts above are `\b(page|context)\.route\(` over `e2e/**/*.spec.ts`.)

**Do not add new route mocks.** Point the spec at the real backend, and extend the corpus if it
cannot express what you need.

## The surface map is enforced

`src/__checks__/surface-map.test.ts` fails the **unit** suite when routes or `data-testid`s drift
from [`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md), in both directions. Update the map in the same
commit as the UI change.

## Where the data comes from

The corpus and fixture trees live in the Python half of the repo and are **versioned** — see the
table at the top of [`E2E_SURFACE_MAP.md`](E2E_SURFACE_MAP.md),
[`../../../tests/README.md`](../../../tests/README.md), and
[`../../../tests/fixtures/README.md`](../../../tests/fixtures/README.md). If an asset looks missing
from in here, search from the repo root first.

## Related

- [`../../README.md`](../../README.md) — the two web apps and how they differ
- [`../TESTING.md`](../TESTING.md) — the viewer's full test tier map (Vitest → component → e2e → Tier-3)
- [`../../../docs/guides/E2E_TESTING_GUIDE.md`](../../../docs/guides/E2E_TESTING_GUIDE.md)
