# GI / KG artifact viewer (local)

Interactive prototype for **Grounded Insight Layer** (`*.gi.json`) and **Knowledge Graph** (`*.kg.json`) artifacts ([GitHub #445](https://github.com/chipi/podcast_scraper/issues/445)).

## Documentation (project guides)

Canonical usage write-up and cross-links live in the main docs (this file is the **implementation** reference):

| Doc | Content |
| --- | --- |
| [Development Guide — GI / KG browser viewer](https://github.com/chipi/podcast_scraper/blob/main/docs/guides/DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) | **Start here:** `make serve-gi-kg-viz`, workflow, pointers to GIL/KG/CLI |
| [Grounded Insights Guide](https://github.com/chipi/podcast_scraper/blob/main/docs/guides/GROUNDED_INSIGHTS_GUIDE.md) | `gi.json`, `gi` CLI; § *Browser visualization (prototype)* |
| [Knowledge Graph Guide](https://github.com/chipi/podcast_scraper/blob/main/docs/guides/KNOWLEDGE_GRAPH_GUIDE.md) | `kg.json`, `kg` CLI; *Consumption and integration* |
| [CLI API](https://github.com/chipi/podcast_scraper/blob/main/docs/api/CLI.md) | § *GI / KG artifacts — browser viewer (prototype)* after `gi` / `kg` examples |

Published site (when deployed): [Guides index](https://chipi.github.io/podcast_scraper/guides/) → Development Guide.

## Run

From the **repository root**:

```bash
make serve-gi-kg-viz
```

Open <http://127.0.0.1:8765/> and pick a viewer:

| Page | Purpose |
| --- | --- |
| `index.html` | Hub |
| `graph-vis.html` | **vis-network** — physics, filters, color legend, double-click neighborhood focus, Chart.js |
| `graph-cyto.html` | **Cytoscape.js** — cose layout, same filters, legend, double-click focus, Chart.js |
| `json-only.html` | Metrics + Chart.js + raw JSON (no graph libs) |

Using `file://` often blocks or restricts **CDN** scripts; prefer the Makefile server.

## Features

- **File picker** — multi-select JSON artifacts (no backend). One file drives the graph at a
  time; with multiple files loaded, use **Merge** (see below) to combine episodes (IDs are
  distinct per episode in the pipeline).
- **Folder load** — **Choose folder…** walks the directory (recursive). **GI / KG** (Both · GI only ·
  KG only, default **Both**) controls which `*.gi.json` / `*.kg.json` files are loaded, which rows
  appear in the list, and how merge works. Uses **File System Access API** when the browser
  supports it; otherwise the **webkitdirectory** fallback. Changing mode or the auto-merge
  checkbox updates the URL (see below).
- **Name filter + merge** — **Name filter** narrows the list by substring (visible files only
  count toward merge). One **Merge** button picks the action from context: in **Both** mode with
  ≥1 GI and ≥1 KG it offers **Merge GI + KG**; otherwise **Merge all GI** (≥2 visible GI) or
  **Merge all KG** (≥2 visible KG). Combined graph uses `g:` / `k:` id prefixes and **one
  Episode node per shared UUID** when the same episode appears in both layers (including
  after multi-file GI + multi-file KG merge).
- **Overview** — episode, counts, GIL grounding stats / KG extraction metadata, edges-by-type.
- **Graph pages** (`graph-vis.html`, `graph-cyto.html`) — **node color legend**; filter by **node type**, **hide ungrounded insights** (GIL); **double-click** a node to show only its 1-hop neighborhood (double-click again or empty canvas to restore); **Fit** and **re-layout**; pan/zoom (library default). Shared: `shared.js`, `graph-legend.js`.
- **Chart.js** — horizontal bar of node counts for the **current (filtered) view**.
- **CLI** — collapsible examples; full reference: [CLI.md](https://github.com/chipi/podcast_scraper/blob/main/docs/api/CLI.md).

## URL query parameters

With **`make serve-gi-kg-viz`** (uses `scripts/gi_kg_viz_server.py`), the viewer can load
artifacts from a **repo-relative directory** without using the folder picker:

| Param | Values | Meaning |
| --- | --- | --- |
| `data` | path under repo root | Recursively load all `*.gi.json` / `*.kg.json` under that directory (POSIX `/`, URL-encoded as needed). |
| `layer` | `gi`, `kg`, `both` | Which extensions to include (default `both`). |
| `merged` | `1`, `true`, `yes` | After load: if `layer` is `both` and ≥1 GI and ≥1 KG, merge GI+KG; else merge GI if ≥2 (when allowed), else KG if ≥2. |

`python -m http.server` alone has no `/_api/` — `?data=` is ignored there (no error).

**Note:** `?data=` only tells the dev server to **fetch** JSON from a repo path over HTTP. It does **not** open Finder or the OS folder picker; use **Choose folder…** in the viewer for that.

Examples:

- `graph-vis.html?data=.test_outputs/manual/foo/run_xyz/metadata&layer=both&merged=1`
- `graph-vis.html?layer=kg&merged=1` (picker defaults only)

Cross-page links with `data-viz-nav` preserve `data`, `layer`, and `merged`.

## Regenerate data (CLI)

See the in-page **CLI** section or [CLI.md](https://github.com/chipi/podcast_scraper/blob/main/docs/api/CLI.md) § Grounded insights / Knowledge Graph (`gi validate`, `gi export`, `kg validate`, `kg export`, `gi explore`, etc.).

## Layout

- `shared.js` — parse, metrics, filter logic, vis/cyto adapters, merge helper.
- `viz-query.js` — parse `data` / `layer` / `merged`, rewrite `data-viz-nav` links, `replaceState`.
- `../scripts/gi_kg_viz_server.py` — static viz + `/_api/gi-kg-list` + `/_repo/…` for `?data=`.
- `viewer-shell.js` — file list, file picker, folder picker, merge buttons.
- `graph-drawers.js` — collapsible left/right panels on graph pages.
- `filters-panel.js`, `metrics-charts.js`, `cli-hints.js`, `graph-legend.js` — UI helpers.
- `styles.css` — shared styling.

## Offline / vendoring

To work without CDN, download the same UMD builds into e.g. `vendor/` and point `<script src="...">` in the HTML files to local paths.
