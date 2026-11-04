# `web/` — the operator viewer

[`gi-kg-viewer/`](gi-kg-viewer) is a Vue 3 + TypeScript + Vite app over the FastAPI backend and the
corpus: it is how an operator inspects the corpus, the graph and the pipeline. The consumer player
that used to sit beside it lives in its own repository now (ADR-170).

| | [`gi-kg-viewer/`](gi-kg-viewer) |
| --- | --- |
| Audience | **Operators** — inspect the corpus, graph, pipeline |
| API surface | `/api/corpus/*`, `/api/search`, `/api/artifacts`, `/api/index/*` |
| Unit tests | Vitest, ~220 files |
| Browser tests | Playwright, **Firefox** — one project, ~68 specs |
| Browser install | `npx playwright install firefox` |
| Backend during e2e | A **real API** over the committed corpus ([#1619](https://github.com/chipi/podcast_scraper/issues/1619)) |
| Run units | `make test-ui` |
| Run browser | `make test-ui-e2e` |
| Design docs | VIEWER_IA, UXS-001…006, RFC-062 |

## The contract doc — read before changing selectors

[`gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`](gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) is the automation
contract: surfaces, routes, owning specs and the `data-testid`s / roles / labels tests depend on. It
is also the reference for driving the viewer through accessibility-tree tools (Playwright MCP,
Chrome DevTools MCP).

It is **enforced**: `src/__checks__/surface-map.test.ts` fails the unit suite when a route or testid
drifts from the map, in both directions — undocumented selectors and stale entries alike. Update the
map in the same commit as the UI change; the guard will not let you forget.

## Where the data comes from

The viewer ships no fixtures of its own. It reads the committed corpus and fixture trees in the
Python half of the repo — [`../tests/README.md`](../tests/README.md) is the map, and
[`../tests/fixtures/README.md`](../tests/fixtures/README.md) the detail. Notably: **audio and
transcripts are versioned** (`tests/fixtures/{audio,transcripts}/<FIXTURES_VERSION>/`), and two mock
hosts serve them as a real podcast host would. If a fixture looks missing from in here, search from
the repo root before concluding anything — the trees are one level up and easy to miss.

## Related

- [`../docs/guides/E2E_TESTING_GUIDE.md`](../docs/guides/E2E_TESTING_GUIDE.md) — how every test tier fits together
- [`gi-kg-viewer/TESTING.md`](gi-kg-viewer/TESTING.md) — the viewer's full test tier map
- [`../docs/guides/POLYGLOT_REPO_GUIDE.md`](../docs/guides/POLYGLOT_REPO_GUIDE.md) — Python + TypeScript conventions
