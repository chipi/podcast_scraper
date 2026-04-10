# GI / KG Viewer (v2)

Vue 3 + Vite + TypeScript + Tailwind + Pinia SPA for [RFC-062](https://github.com/chipi/podcast_scraper/blob/main/docs/rfc/RFC-062-gi-kg-viewer-v2.md). Replaced the legacy `web/gi-kg-viz/` prototype (now removed).

**Repo context:** This folder is the **Node** side of a **Python + web** monorepo. For how that
fits with the root `Makefile`, `.env` files, and CI, see
[Polyglot repository guide](../../docs/guides/POLYGLOT_REPO_GUIDE.md).

## End users: run API + UI together

Use this when you want **list files**, **semantic search**, **explore**, and **dashboard** (everything that talks to `/api/*`).

1. **Python:** `pip install -e ".[server]"` from the repo root (add `[ml]` too if you need FAISS / embeddings on the server, same as `podcast search`).
2. **UI build (once per clone / after UI changes):**  
   `cd web/gi-kg-viewer && npm install && npm run build`  
   This creates `dist/`. When `dist/` exists, `serve` mounts it at `/` so you only run one backend process.
3. **Start the server** (repo root, venv on):  
   `python -m podcast_scraper.cli serve --output-dir /path/to/your/run`  
   Default URL: **<http://127.0.0.1:8000>**
4. In the app, set **Corpus root folder** to that **same** `--output-dir`, then **List files** → select artifacts → **Load selected into graph**.

**Graph-only, no API:** If you skip the server (or health check fails), use **Choose .gi.json / .kg.json files** under the API panel. You still get the graph (and can use the Dashboard for the loaded graph), but not server-backed list/search/index stats.

**Full-stack dev (hot reload):** `make serve SERVE_OUTPUT_DIR=/path/to/output` — API on port **8000**, Vite on **5173** with `/api` proxied. Open **5173** while developing the UI.

---

## Prerequisites (contributors)

- Node 20+ and npm
- Python env with **`[server]`**: `pip install -e '.[server]'` (use `.[dev,ml,llm,server]` if you match `make init` + viewer APIs)

## Development (API + Vite separately)

1. Start the API (from repo root):

   ```bash
   make serve-api SERVE_OUTPUT_DIR=/path/to/pipeline/output
   ```

   Or: `python -m podcast_scraper.cli serve --output-dir /path/to/output` with `PYTHONPATH=src` if needed.

2. Install UI deps once and run Vite:

   ```bash
   cd web/gi-kg-viewer
   npm install
   npm run dev
   ```

3. Open `http://127.0.0.1:5173`. Requests to `/api/*` are proxied to `http://127.0.0.1:8000`.

### One command (parallel)

From repo root (after `npm install` in this directory):

```bash
make serve SERVE_OUTPUT_DIR=/path/to/output
```

## Which path is “corpus root”?

Use the **pipeline output directory** — the folder that contains `metadata/*.gi.json` and `metadata/*.kg.json` (same idea as `podcast … --output-dir ./my-run`). Paste that **folder** into the UI, click **List files**, then tick the artifacts you want and **Load selected into graph**. You do not paste paths to individual JSON files into the corpus field.

### Multi-feed corpora (unified `search/`)

When you scraped **several feeds into one corpus** (CLI multiple `--rss` / `feeds:` in YAML), the corpus
root is the **parent directory that contains `feeds/`** — not `feeds/<stable_id>/`. Per-feed transcripts
and metadata live under `feeds/<stable_id>/…`; **semantic search** and **Dashboard → vector index** use the
single unified index at **`<corpus_parent>/search/`** (GitHub #505). The viewer’s **Corpus root** field
should still point at that parent so list/search/explore see the whole corpus.

## Optional env

Copy `.env.example` to `.env` and set `VITE_DEFAULT_CORPUS_PATH` to pre-fill that folder in the UI.

## Production build

```bash
npm run build
```

Static files go to `dist/`. When `dist/` exists, `podcast serve` mounts it at `/` so a single process can serve API + UI.

## Graph (M2)

After loading one or more `.gi.json` / `.kg.json` files, the **Cytoscape** graph supports type filters, hide-ungrounded-insights (GI), legend click-to-solo, **Shift+double-click** 1-hop neighborhood focus, **double-click** for the node detail panel (single-click selects), and a node detail panel. Styling matches v1 hex palette for now; UXS-001 token parity is tracked on GitHub #489.

## Dashboard (M3)

Use the **Dashboard** tab for artifact metrics (v1-style key/value rows), a **Chart.js** bar chart of node counts by visual type, and **vector index** stats from `GET /api/index/stats` (FAISS under `<corpus>/search/`). If `/api/health` fails, use **Choose .gi.json / .kg.json files** under API to load graphs offline (index stats stay disabled without the server).

## Corpus Library (RFC-067)

The **Library** tab lists **feeds** and **episodes** from on-disk `*.metadata.json` / YAML under your corpus root (same discovery as the pipeline). Select an episode to see **summary bullets** and **Open in graph** (loads sibling `.gi.json` / `.kg.json` via `GET /api/artifacts/...`) or **Prefill semantic search** (opens Search with **Feed id (substring)** and query from **summary text** / title / bullets / episode title — then run **Search** against the vector index). Feeds that appear in the vector index show an **Indexed** chip: `GET /api/index/stats` returns **deduplicated, sorted, trimmed** `feeds_indexed` values aligned with catalog ids. **Find similar episodes** calls `GET /api/corpus/episodes/similar` (FAISS + deduped peers). Endpoints: `GET /api/corpus/feeds`, `GET /api/corpus/episodes`, `GET /api/corpus/episodes/detail`, `GET /api/corpus/episodes/similar`. Requires a healthy API and corpus path set in the left panel. `GET /api/health` includes `corpus_library_api: true` on current servers; a 404 on `/api/corpus/*` usually means an older API process is still running—restart `make serve-api` / `podcast serve` from this repo after `pip install -e ".[server]"`.

## Corpus Digest (RFC-068)

The **Digest** tab calls **`GET /api/corpus/digest`** (rolling **24h** / **7d** / **since** windows, UTC) for a **feed-diverse** list of recent episodes plus optional **topic** bands when a vector index exists. **Open in graph** / **Prefill semantic search** for an episode are used from the **Library** episode panel after opening a row from Digest. The **Library** tab does **not** embed a second digest strip — **Digest** is the discovery surface. **`corpus_digest_api`** on **`GET /api/health`** gates the **Digest** tab; older APIs show an upgrade notice there.

## Semantic search (M4)

**List artifacts** (`GET /api/artifacts`) may return **`hints`** when the path you entered is under `feeds/` but the unified FAISS index lives at the corpus parent; the Corpus panel shows those hints above the file list.

The sidebar **Semantic search** panel calls `GET /api/search` (same pipeline as `podcast search`): natural-language query, optional doc-type / feed / date / speaker / grounded filters, and `top_k`. Results include **Show on graph** when the hit maps to a graph node (`source_id` for insights, quotes, KG topics/entities). That switches to the **Graph** tab, selects the node, opens the detail panel, and centers the view. Requires a built vector index under `<corpus>/search/` and the embedding model available to the server process.

## Explore & query (M5)

**Explore & query** calls `GET /api/explore`:

- **Topic / speaker explore** — same behavior as `gi explore` (filters, sort, limit, grounded-only, optional strict validation). Leave topic and speaker empty to scan all insights up to the limit.
- **Natural language** — UC4 pattern questions (`gi query`), e.g. “What insights about …?”, “What did … say?”, “Which topics have the most insights?”. Unmatched questions return `no_pattern_match` with hints.

Insight rows include **Show on graph** (uses `insight_id` as the node id). Semantic topic narrowing when an index exists is handled inside the server explore pipeline (not duplicated in the UI).

## Theming

Semantic tokens live in `src/theme/tokens.css` (UXS-001). Tailwind colors in `tailwind.config.js` map to `--ps-*` variables.

## Unit tests (Vitest)

Pure TypeScript utility logic (parsing, merge, metrics, formatting, colors, visual groups,
search-focus mapping) is covered by **Vitest** unit tests co-located with source:

```bash
npm run test:unit           # single run
npm run test:unit:watch     # watch mode
```

From repo root: **`make test-ui`**. Tests: `src/utils/*.test.ts`, `src/stores/*.test.ts`,
`src/api/*.test.ts` (e.g. `corpusLibraryApi.test.ts`). Config:
`vite.config.ts` `test` block. CI job: **`viewer-unit`**.

## Browser E2E (M7)

- **Runner:** Playwright, **browser:** Firefox (see `playwright.config.ts`). Install once: `cd web/gi-kg-viewer && npx playwright install firefox` (CI uses `playwright install --with-deps firefox`).
- **Commands:** `npm run test:e2e` (starts Vite on **port 5174** so it does not clash with `npm run dev` on 5173). From repo root: `make test-ui-e2e` (runs `npm install`, browser install, then tests).
- **Fixtures:** `e2e/fixtures/ci_sample.gi.json` mirrors the pytest GIL CI sample; offline tests abort `/api/health` so the file-picker path is deterministic even if something listens on `:8000`.
- **Scenarios:** offline graph + toolbar, Dashboard tab, theme tokens (dark/light), `/` shortcut with mocked health, `Esc` on graph, PNG export download, mocked API list/load/search → **Show on graph**, **Corpus path hint** when `GET /api/artifacts` returns `hints` (`e2e/corpus-hints.spec.ts`), **Library** tab with corpus mocks + index/similar (`e2e/library.spec.ts`).
- **Surface map:** [e2e/E2E_SURFACE_MAP.md](e2e/E2E_SURFACE_MAP.md) — surfaces, fixtures, and stable Playwright selectors. **UX change order:** map → Playwright specs/helpers → [UXS-001](../../docs/uxs/UXS-001-gi-kg-viewer.md) if the visual contract changes. Checklist: [E2E Testing Guide](../../docs/guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow) ([GitHub #509](https://github.com/chipi/podcast_scraper/issues/509)).

## Polish (M6)

- **Typography:** Inter (UI) and JetBrains Mono (code) load from Google Fonts in `index.html`; stacks match UXS-001.
- **Tokens:** `overlay` / `disabled` (and hover `bg-overlay`) align with UXS-001 hover and disabled guidance.
- **Keyboard:** **`/`** focuses the semantic search query (ignored when focus is in an input/textarea/select). **`Esc`** on the **Graph** tab clears graph selection, closes the node detail panel, exits 1-hop neighborhood view, and clears pending “show on graph” focus — skipped while focus is in an editable field so typing is unaffected.
- **Graph export:** **Export PNG** in the graph toolbar saves the **full** graph at 2× scale with the current theme’s canvas color as background. (Vector SVG export is not in core Cytoscape 3.x; PNG is the supported path.)
- **Layout:** Sidebar stacks on small viewports; graph min-heights taper down below the `sm` breakpoint for shorter screens.
