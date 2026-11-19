# Workbench

Vue 3 + Vite + TypeScript + Tailwind + Pinia SPA ([RFC-062](https://github.com/chipi/podcast_scraper/blob/main/docs/rfc/RFC-062-gi-kg-viewer-v2.md)): the operator's browser UI for a pipeline output folder. Tabs: **Digest**, **Library**, **Search**, **Graph**, **Dashboard**, **Ops**, **Admin**. Desktop only (minimum width 1024 px, [UXS-001](../../docs/uxs/UXS-001-workbench.md)).

**Repo context:** This folder is the **Node** side of a **Python + web** monorepo. For how that
fits with the root `Makefile`, `.env` files, and CI, see
[Polyglot repository guide](../../docs/guides/POLYGLOT_REPO_GUIDE.md).

## End users: run API + UI together

Use this when you want **list**, **search**, **dashboard** and the rest of what talks to `/api/*`.

1. **Python:** `pip install -e ".[dev]"` from the repo root (add `[search]` too if you need LanceDB / embeddings on the server, same as `podcast search`).
2. **UI build (once per clone / after UI changes):**  
   `cd web/workbench && npm install && npm run build`  
   This creates `dist/`. When `dist/` exists, `serve` mounts it at `/` so you only run one backend process.
3. **Start the server** (repo root, venv on):  
   `python -m podcast_scraper.cli serve --output-dir /path/to/your/run`  
   Default URL: **<http://127.0.0.1:8000>**
4. In the status bar, enter the **same** `--output-dir` as the corpus path, then click **List**. The **Corpus artifacts** dialog lists the graph artifacts the server found: select them (**All** / **None**) and click **Load into graph**.

**Graph-only, no API:** If you skip the server (or the health check fails), click **Files** in the status bar to choose graph JSON files from disk. You still get the graph (and can use the Dashboard for the loaded graph), but not server-backed list, search or index stats.

**Offline vs API corpus path:** File-picker loads do **not** fetch `search/topic_clusters.json` or run the **topic-cluster sibling auto-load** (catalog resolve), and they need no server. Anything that reads the corpus (list, Library, Digest, Search, index stats) requires a healthy API and a corpus path pointing at your pipeline output.

**Full-stack dev (hot reload):** `make serve SERVE_OUTPUT_DIR=/path/to/output` — API on port **8000**, Vite on **5173** with `/api` proxied. Open **5173** while developing the UI. The `serve` target also starts a local dev database (`scripts/tools/dev_db.py`) and enables the feeds, operator-config and jobs APIs.

---

## Prerequisites (contributors)

- Node 20+ and npm
- Python env with **`[dev]`** (includes FastAPI): `pip install -e '.[dev]'` (use `.[dev,ml,llm]` to match `make init` for full viewer APIs)

## Development (API + Vite separately)

1. Start the API (from repo root):

   ```bash
   make serve-api SERVE_OUTPUT_DIR=/path/to/pipeline/output
   ```

   Or: `python -m podcast_scraper.cli serve --output-dir /path/to/output` with `PYTHONPATH=src` if needed.

2. Install UI deps once and run Vite:

   ```bash
   cd web/workbench
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

Use the **pipeline output directory** — the folder that contains `metadata/` (the same idea as `podcast … --output-dir ./my-run`). Enter that **folder** in the status bar's corpus path field, then **List** the artifacts and load the ones you want. You do not enter paths to individual JSON files there; **Files** is the picker for those.

### Multi-feed corpora (unified `search/`)

When you scraped **several feeds into one corpus** (CLI multiple `--rss` / `feeds:` in YAML), the corpus
root is the **parent directory that contains `feeds/`** — not `feeds/<stable_id>/`. Per-feed transcripts
and metadata live under `feeds/<stable_id>/…`; **search** and **Dashboard → vector index** use the
single unified index at **`<corpus_parent>/search/`** (GitHub #505). The **corpus path** field
should still point at that parent so list and search see the whole corpus.

## Optional env

Copy `.env.example` to `.env` and set `VITE_DEFAULT_CORPUS_PATH` to pre-fill the corpus path in the UI. `VITE_DEFAULT_GRAPH_LENS_DAYS` sets the Graph time-lens lower bound on first auto-load (a positive integer is "last N days"; `0` or empty is "all time"; see the [Polyglot guide](../../docs/guides/POLYGLOT_REPO_GUIDE.md)).

## Testing

See **[TESTING.md](./TESTING.md)** for the full test tier map — Vitest unit +
`@vue/test-utils` component tests, the `#914` coverage gate, the Playwright e2e
(`npm run test:e2e`), and the Tier-3 real-corpus validation walk
(`make ci-ui-validation`) — plus the corpora and how to run each.

## Production build

```bash
npm run build
```

Static files go to `dist/`. When `dist/` exists, `podcast serve` mounts it at `/` so a single process can serve API + UI.

## Graph

After loading one or more graph artifacts, the **Cytoscape** graph supports type, edge, degree and feed filters, **Shift+double-click** 1-hop neighborhood focus, **double-click** for the node detail panel (single-click selects), and a node detail panel. Styling follows the UXS-001 tokens.

## Dashboard

Use the **Dashboard** tab for corpus **Coverage**, **Intelligence** and **Pipeline** views: artifact metrics, **Chart.js** charts, **vector index** stats from `GET /api/index/stats` (LanceDB under `<corpus>/search/lance_index/`), and, under **Pipeline**, the job history and the **Run** button for pipeline jobs (needs the jobs API, which `make serve` enables). If `/api/health` fails, click **Files** to load graphs offline (index stats stay disabled without the server).

## Library (RFC-067)

The **Library** tab lists **shows** and **episodes** from on-disk `*.metadata.json` / YAML under your corpus root (same discovery as the pipeline). Select an episode to see **summary bullets** and **Open in graph** (loads the sibling graph artifacts via `GET /api/artifacts/...`) or **Prefill semantic search** (opens Search with the **Feed** filter and a query built from the summary **title** + **bullets**, else the **episode title**, length-capped). After episode detail loads, **Similar episodes** calls `GET /api/corpus/episodes/similar` (LanceDB + deduped peers). Endpoints: `GET /api/corpus/feeds`, `GET /api/corpus/episodes`, `GET /api/corpus/episodes/detail`, `GET /api/corpus/episodes/similar`. `GET /api/health` includes `corpus_library_api: true` on current servers; a 404 on `/api/corpus/*` usually means an older API process is still running — restart `make serve-api` / `podcast serve` from this repo after `pip install -e ".[dev]"`.

## Digest (RFC-068)

The **Digest** tab calls **`GET /api/corpus/digest`** (rolling **24h** / **7d** / **since** windows, UTC) for a **feed-diverse** list of recent episodes plus optional **topic** bands when a vector index exists. **Open in graph** / **Prefill semantic search** for an episode are used from the **Library** episode panel after opening a row from Digest. **`corpus_digest_api`** on **`GET /api/health`** gates the **Digest** tab; older APIs show an upgrade notice there.

## Search

The **Search** tab calls `GET /api/search` (same pipeline as `podcast search`): a natural-language query with filters (document type, feed, date, speaker) and `top_k`. A public install indexes four document types: transcript chunks, summary bullets, KG entities and KG topics. Results include **Show on graph** when the hit maps to a graph node (`source_id` for KG topics and entities). That switches to the **Graph** tab, selects the node, opens the detail panel, and centers the view. Requires a built vector index under `<corpus>/search/` and the embedding model available to the server process. **List artifacts** (`GET /api/artifacts`) may return **`hints`** when the path you entered is under `feeds/` but the unified LanceDB index lives at the corpus parent; the **Corpus artifacts** dialog shows those hints above the file list.

## Theming

Semantic tokens live in `src/theme/tokens.css` (UXS-001). Tailwind colors in `tailwind.config.js` map to `--ps-*` variables.

## Unit tests (Vitest)

Pure TypeScript utility logic (parsing, merge, metrics, formatting, colors, visual groups,
search-focus mapping) is covered by **Vitest** unit tests co-located with source:

```bash
# Always cd into this directory first — see "Always run from this directory" below.
cd web/workbench
npm run test:unit           # single run
npm run test:unit:watch     # watch mode
```

From repo root: **`make test-ui`** (the target handles cwd itself). Tests:
`src/utils/*.test.ts`, `src/stores/*.test.ts`, `src/api/*.test.ts` (e.g.
`corpusLibraryApi.test.ts`). Config: `vite.config.ts` `test` block. CI job:
**`viewer-unit`**.

### Always run from this directory

When invoking Vitest or Playwright **outside `make`**, always set CWD to
`web/workbench/` so caches and reports land in the viewer's own
`node_modules/.vite/` and `playwright-report/`. Examples:

```bash
# ✅ Correct — viewer is CWD; everything stays scoped to this subtree
cd web/workbench
npm run test:unit
node_modules/.bin/vitest run src/services/graphHandoffFsm.test.ts

# ❌ Avoid — CWD stays at repo root; Vitest creates a stray
#    `<repo-root>/node_modules/.vite/` cache that does not belong here.
npm exec --prefix web/workbench -- vitest run
```

The `--prefix` flag only resolves the binary; it does **not** change CWD.
A root-level `node_modules/` is always a transient artifact (gitignored
since [#770]) and should be deleted on sight.

## Browser E2E

- **Runner:** Playwright, **browser:** Firefox (see `playwright.config.ts`). Install once: `cd web/workbench && npm install && npx playwright install firefox`.
- **Commands:** `npm run test:e2e` (starts Vite on **port 5174** so it does not clash with `npm run dev` on 5173, plus an API on a fixture corpus). From repo root: `make test-ui-e2e` (runs `npm install`, browser install, then tests).
- **Fixtures:** `e2e/fixtures/ci_sample.kg.json` is a small KG graph (one episode, topic and person) for the offline file-picker path; `tests/fixtures/app-validation-corpus/v3` is the corpus the API boots against.
- **Surface map:** [e2e/E2E_SURFACE_MAP.md](e2e/E2E_SURFACE_MAP.md) — surfaces, fixtures, and stable Playwright selectors. **UX change order:** map → Playwright specs/helpers → [UXS-001](../../docs/uxs/UXS-001-workbench.md) if the visual contract changes. Checklist: [E2E Testing Guide](../../docs/guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow) ([GitHub #509](https://github.com/chipi/podcast_scraper/issues/509)).

## Polish

- **Typography:** Inter (UI) and JetBrains Mono (code) are bundled through `@fontsource` packages; stacks match UXS-001.
- **Tokens:** `overlay` / `disabled` (and hover `bg-overlay`) align with UXS-001 hover and disabled guidance.
- **Keyboard:** **`Cmd/Ctrl+K`** or **`/`** (outside an editable field) opens the command palette; **`1`**–**`5`** switch between Digest, Library, Search, Graph and Dashboard (outside an editable field). **`Esc`** on the **Graph** tab clears the graph focus and interaction state — skipped while focus is in an editable field so typing is unaffected.
- **Graph export:** **Export PNG** (label **PNG** in the graph bottom bar; `aria-label` **Export PNG**) saves the **full** graph at 2× scale with the current theme’s canvas color as background. (Vector SVG export is not in core Cytoscape 3.x; PNG is the supported path.)
