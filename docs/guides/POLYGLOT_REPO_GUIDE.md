# Polyglot repository layout (Python + web viewer)

This repository is **Python-first**: the package, CLI, tests, and `Makefile` live at the **repo
root**. The **GI/KG viewer** ([RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)) is a **separate**
**Node** project under [`web/gi-kg-viewer/`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/).

Treat them as two toolchains that share one git tree, not one unified `npm` workspace (by design).

---

## Quick comparison

| Area | Location | Install / run |
| ---- | -------- | ------------- |
| **Python app** (CLI, pipeline, FastAPI server) | Repo root (`pyproject.toml`, `src/`) | `python3 -m venv .venv`, `source .venv/bin/activate`, `make init` or `pip install -e ".[…]"` |
| **Viewer UI** (Vue 3 + Vite + TypeScript) | `web/gi-kg-viewer/` (`package.json`) | `cd web/gi-kg-viewer && npm install`; `npm run dev` / `npm run build` / `npm run test:*` |

**End-to-end viewer (API + built SPA in one process):** install `[server]`, build `dist/` under
`web/gi-kg-viewer`, then `python -m podcast_scraper.cli serve --output-dir …`. See
[Server Guide](SERVER_GUIDE.md) and [web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md).

---

## Environment files (two purposes)

| File | Purpose |
| ---- | ------- |
| **`config/examples/.env.example`** | Copy to **repo root** `.env` for **Python**: API keys, `CACHE_DIR`, logging, optional `PODCAST_SCRAPER_*`, etc. ([CONFIGURATION.md](../api/CONFIGURATION.md); [twelve-factor config](../api/CONFIGURATION.md#twelve-factor-app-alignment-config)). |
| **`web/gi-kg-viewer/.env.example`** | Copy to **`web/gi-kg-viewer/.env`** for **Vite** only (e.g. `VITE_DEFAULT_CORPUS_PATH`). Vite loads `.env*` next to the app by default. |

Root `.gitignore` already ignores `.env`, `.env.local`, and similar patterns so secrets are not
committed from either location.

---

## Git ignores (viewer)

**Node / Vite / Playwright** artifacts for the viewer (`node_modules/`, `dist/`, Playwright
reports, log files, etc.) are listed under **`web/gi-kg-viewer/`** in the **repository root**
[`.gitignore`](https://github.com/chipi/podcast_scraper/blob/main/.gitignore). There is no
separate `web/gi-kg-viewer/.gitignore`.

---

## Makefile targets (run from repo root)

All of these assume an **activated** Python venv when Python is involved.

The viewer directory is **`WEB_VIEWER_DIR`** in the root [`Makefile`](https://github.com/chipi/podcast_scraper/blob/main/Makefile)
(default **`web/gi-kg-viewer`**). You can override it for a one-off command, e.g.
`make serve-ui WEB_VIEWER_DIR=path/to/viewer`.

| Target | What it does |
| ------ | ------------- |
| `make serve SERVE_OUTPUT_DIR=…` | **Parallel:** FastAPI (`serve-api`) + Vite dev (`serve-ui`); UI usually on **5173**, API on **8000**. |
| `make serve-api SERVE_OUTPUT_DIR=…` | FastAPI only. |
| `make serve-ui` | Vite only in `web/gi-kg-viewer` (proxies `/api` → **8000**). |
| `make test-ui` | Vitest unit tests for TS utils under `web/gi-kg-viewer` (no browser). |
| `make test-ui-e2e` | Playwright E2E (Firefox); installs npm deps and browsers as needed. |

**CI** runs the same `web/gi-kg-viewer` paths; see `.github/workflows/python-app.yml`.

---

## Where to read next

- **Contributor setup:** [CONTRIBUTING.md](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)
- **Daily dev patterns:** [Development Guide](DEVELOPMENT_GUIDE.md) — [GI / KG browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype)
- **FastAPI routes and `serve`:** [Server Guide](SERVER_GUIDE.md) — `/api/*` overview; OpenAPI UI at **`/docs`** when the API is up
- **Vitest + Playwright:** [Testing Guide](TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2)
- **One-page commands:** [Quick Reference](QUICK_REFERENCE.md)
