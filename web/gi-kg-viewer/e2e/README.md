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

This suite's `webServer` has only ever run **Vite alone**, with no backend behind it, so fulfilling
payloads per spec was the only option available — at its peak, **37 of 68 spec files and 243
interceptions**, plus four shared helpers that mock on their behalf (`helpers.ts`,
`dashboardApiMocks.ts`, `handoff/_handoff-helpers.ts`, `handoff-production/_helpers.ts`).

**Current state (#1619, 2026-08-15): 25 files / 169 interceptions, and 23 spec files run against a
live backend.** Every one of the 36 files that ever mocked now carries a written reason at its top
saying either that it migrated, or precisely what blocks it — measured, not assumed. The remaining
mocks fall into four kinds:

| kind | meaning | examples |
| --- | --- | --- |
| **B — corpus cannot produce it** | **Permanent.** A v4 corpus is NOT planned (decided 2026-08-15), so these are mocked for good — the ladder keeps the requirements only as a record | topic bands, lifted/compound results, cross-show topics, `runs: []` |
| **C — permanently mocked** | a healthy backend cannot produce it on demand | 404s, `no_index`, an index rebuild, an empty search result set |
| **state matrix** | one control seen in several states at once | three-state run counters, >15 feeds, an episode published today |
| **by design** | a constructed fixture is the better test | graph topologies, the handoff matrix's fixed graph |

`docs/wip/CORPUS-V4-FIXTURE-LADDER.md` §B is the authority on what v4 must add.

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

That is not hypothetical — migrating found real defects the mocks were hiding. `helpers.ts`'s
`resetUserPreferences` had **never** worked against a real server (it PUT `{}` where the endpoint
requires `{ preferences: {} }`, → 422); the corpus-path hint turned out to render in two dialogs,
making the old locator a latent strict-mode violation; and several specs asserted properties that
were true only because a stub made them so — `search-rail-in-episode` "proved" the server scopes
retrieval using a mock that refused to return anything unscoped, and `auth-roles` tested the role
gate against a role the test itself wrote.

(If you find a smaller number quoted elsewhere — "31 files / 202" appears in the #1619 commit —
that came from a regex that only matched string-literal first arguments and missed calls taking a
variable or a `RegExp`. The counts above are `\b(page|context)\.route\(` over `e2e/**/*.spec.ts`.)

**Do not add new route mocks.** Point the spec at the real backend, and extend the corpus if it
cannot express what you need.

### Migrating a spec: the two sign-in modes

`mockSignIn(page, role, { liveApi: true })` skips the catch-all so requests reach the backend. It
is enough for every **corpus read** (`/api/corpus/*`, `/api/artifacts`, `/api/relational/*`).

It is **not** enough for the operator plane. `/api/feeds`, `/api/operator-config`, `/api/jobs`,
`/api/scheduled-jobs`, `/api/index/rebuild` and `/api/enrichment/*` need two more things:

1. The server must be started with `PODCAST_SERVE_ENABLE_FEEDS_API=1`,
   `PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API=1`, `PODCAST_SERVE_ENABLE_JOBS_API=1`, or the routes
   are **not mounted at all** — `run-local-stack.sh` does not set them today.
2. `OperatorWriteGuard` requires a **real admin session cookie**. `mockSignIn` only stubs
   `/api/app/auth/status` inside the browser, so it does not satisfy the guard and every operator
   request 403s. Use `signInAsAdmin(page)`, which drives the real mock-OAuth round trip.

Read helpers exist so migrated specs assert against what the server actually holds rather than a
hand-copied payload: `liveCorpusRoot`, `liveFeeds`, `liveFirstEpisode`, `liveFeedMetadataDir`.
**Never hardcode a corpus path** — it is the repo-relative fixture when the API is served natively
and `/corpus` under `run-local-stack.sh`.

### Specs that mutate shared server state must be serial

Four specs now write through the operator API — `status-bar-feeds-operator-mocks`,
`feed-overrides-mocks` (both rewrite `feeds.spec.yaml`), `scheduled-jobs-mocks` and
`cron-preview-mocks` (both rewrite `viewer_operator.yaml`). They seed their own starting state and
read the result back, which is what makes them real persistence tests — but it means they share one
mutable corpus.

Each carries `test.describe.configure({ mode: 'serial' })`, because `playwright.config.ts` sets
`fullyParallel: true` and tests inside one file would otherwise seed over each other.

**Known residual risk:** that is per-file. Playwright has no cross-file mutex, so two of these
files can still run concurrently in different workers and clash on the same YAML. It has not been
observed in a full run, and the window is a few seconds per test — but it is real. If it ever
flakes, the fix is a dedicated Playwright project pinned to `workers: 1` for these four, not a
retry.

### The operator plane writes into the corpus — mind the tracked fixture

`GET /api/operator-config` **creates** `viewer_operator.yaml` in the corpus directory when it is
missing, and enabling the jobs API creates `.viewer/jobs.jsonl.lock`. `.gitignore:82` deliberately
force-includes `tests/fixtures/app-validation-corpus/**`, so a live operator spec run leaves the
tracked fixture dirty. Serve from a **copy** of the corpus when running these locally.

That, not assertion effort, is why the remaining operator specs are still mocked: each either
mutates state (`PUT /api/feeds`, `PUT /api/operator-config`), needs the corpus pre-seeded with a
particular `viewer_operator.yaml`, needs per-test server env (`PODCAST_AVAILABLE_PROFILES`), or —
for the two rebuild cards — would trigger a **real** index / topic-cluster rebuild, which an e2e
test must not do. Each file records which of those applies at its top.

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
