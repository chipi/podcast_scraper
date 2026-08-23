# `web/` — the two front-ends

Two separate Vue 3 + TypeScript + Vite apps against the **same FastAPI backend** and the **same
corpus**. They look similar and behave differently; most confusion in this directory comes from
applying one app's assumptions to the other.

| | [`learning-player/`](learning-player) | [`gi-kg-viewer/`](gi-kg-viewer) |
| --- | --- | --- |
| Audience | **Consumers** — listen, search, capture, learn | **Operators** — inspect the corpus, graph, pipeline |
| Nickname | the app / the player | the viewer |
| API surface | `/api/app/*` | `/api/corpus/*`, `/api/search`, `/api/artifacts`, `/api/index/*` |
| Auth | Sessions, OAuth, per-user state (queue, follows, highlights) | None — operator-local |
| Unit tests | Vitest, ~69 files | Vitest, ~220 files |
| Browser tests | Playwright, **Chromium** — two projects: `mobile-chrome` (Pixel 7) + `desktop-chrome`, ~30 specs | Playwright, **Firefox** — one project, ~68 specs |
| Browser install | `npx playwright install chromium` | `npx playwright install firefox` |
| Backend during e2e | A **real API** over the committed corpus | Historically **route-mocked**; migrating ([#1619](https://github.com/chipi/podcast_scraper/issues/1619)) |
| Run units | `make test-app` | `make test-ui` |
| Run browser | `make test-app-e2e` | `make test-ui-e2e` |
| Design docs | PRD-038/041/042/043, UXS-011…014 | VIEWER_IA, UXS-001…006, RFC-062 |

**The browser difference is not cosmetic.** Each app's Playwright config installs and runs only its
own browser, so `npx playwright test` in the wrong directory fails with a missing-executable error
that reads like a broken install. It is not — it is the other app's browser.

**Mobile is primary for the player, not an afterthought.** Its first Playwright project is a Pixel 7;
a spec that only works at desktop widths is a half-finished spec. The bottom tab bar is mobile-only
(`sm:hidden`) and the header icon nav is desktop-only, so navigation helpers must pick whichever is
on screen — see `navTo()` in the player's `e2e/helpers.ts`.

## The contract docs — read before changing selectors

Each app has an **E2E surface map**: the automation contract listing surfaces, routes, owning specs
and the `data-testid`s / roles / labels tests depend on. They are also the reference for driving the
apps through accessibility-tree tools (Playwright MCP, Chrome DevTools MCP).

- [`learning-player/e2e/E2E_SURFACE_MAP.md`](learning-player/e2e/E2E_SURFACE_MAP.md)
- [`gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`](gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)

These are **enforced**: `src/__checks__/surface-map.test.ts` in each app fails the unit suite when a
route or testid drifts from the map, in both directions — undocumented selectors and stale entries
alike. Update the map in the same commit as the UI change; the guard will not let you forget.

## Where the data comes from

Neither app ships fixtures of its own. Both read the committed corpus and fixture trees in the
Python half of the repo — [`../tests/README.md`](../tests/README.md) is the map, and
[`../tests/fixtures/README.md`](../tests/fixtures/README.md) the detail. Notably: **audio and
transcripts are versioned** (`tests/fixtures/{audio,transcripts}/<FIXTURES_VERSION>/`), and two mock
hosts serve them as a real podcast host would. If a fixture looks missing from in here, search from
the repo root before concluding anything — the trees are one level up and easy to miss.

## Related

- [`../docs/guides/E2E_TESTING_GUIDE.md`](../docs/guides/E2E_TESTING_GUIDE.md) — how every test tier fits together
- [`../docs/guides/CONSUMER_LEARNING_PLAYER_GUIDE.md`](../docs/guides/CONSUMER_LEARNING_PLAYER_GUIDE.md) — the player's architecture
- [`gi-kg-viewer/TESTING.md`](gi-kg-viewer/TESTING.md) — the viewer's full test tier map
- [`../docs/guides/POLYGLOT_REPO_GUIDE.md`](../docs/guides/POLYGLOT_REPO_GUIDE.md) — Python + TypeScript conventions
