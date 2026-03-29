# GI / KG artifact viewer (local)

Interactive prototype for **Grounded Insight Layer** (`*.gi.json`) and **Knowledge Graph** (`*.kg.json`) artifacts ([GitHub #445](https://github.com/chipi/podcast_scraper/issues/445)).

## Run

From the **repository root**:

```bash
make serve-gi-kg-viz
```

Open <http://127.0.0.1:8765/> and pick a viewer:

| Page | Purpose |
| --- | --- |
| `index.html` | Hub |
| `graph-vis.html` | **vis-network** — physics, filters, Chart.js |
| `graph-cyto.html` | **Cytoscape.js** — cose layout, same filters + chart |
| `json-only.html` | Metrics + Chart.js + raw JSON (no graph libs) |

Using `file://` often blocks or restricts **CDN** scripts; prefer the Makefile server.

## Features

- **File picker** — multi-select JSON artifacts (no backend).
- **Overview** — episode, counts, GIL grounding stats / KG extraction metadata, edges-by-type.
- **Graph pages** — filter by **node type**, **hide ungrounded insights** (GIL); **Fit** and **re-layout** controls; pan/zoom (library default).
- **Chart.js** — horizontal bar of node counts for the **current (filtered) view**.
- **CLI** — collapsible examples; full reference: `docs/api/CLI.md`.

## Regenerate data (CLI)

See the in-page **CLI** section or `docs/api/CLI.md` § Grounded insights / Knowledge Graph (`gi validate`, `gi export`, `kg validate`, `kg export`, `gi explore`, etc.).

## Layout

- `shared.js` — parse, metrics, filter logic, vis/cyto adapters.
- `viewer-shell.js` — file list + picker.
- `filters-panel.js`, `metrics-charts.js`, `cli-hints.js` — UI helpers.
- `styles.css` — shared styling.

## Offline / vendoring

To work without CDN, download the same UMD builds into e.g. `vendor/` and point `<script src="...">` in the HTML files to local paths.
