# Run comparison tool (RFC-047)

Lightweight Streamlit UI for comparing ML evaluation runs from `data/eval/` artifacts (`metrics.json`, `predictions.jsonl`, optional `diagnostics.jsonl`).

The pip extra is **`[compare]`**. Older instructions may reference `run_compare`; that extra name is retired.

## Setup

Install optional dependencies (from repo root):

```bash
pip install -e '.[compare]'
```

## Usage

From the repository root:

```bash
make run-compare
```

Optional environment variable:

- `BASELINE` — run id or `data/eval`-relative path; if that run is in the current selection, the **Baseline (for deltas)** dropdown starts on it.

On load, **all** runs matching the category filter are selected; use **Select all** / **Deselect all** in the sidebar to change that quickly.

Or run Streamlit directly:

```bash
BASELINE=... python -m streamlit run tools/run_compare/app.py --server.port=8501
```

## Layout

- **Left sidebar (collapsible):** categories, **Runs to compare** with **Select all** / **Deselect all**, baseline for deltas.
- **Top navigation (above the title):** text links **Home · KPIs · Delta · Episodes** using `?page=home|kpis|delta|episodes` (current page shown in bold); main area uses tighter top padding so the nav sits near the top.
- **Home:** token/latency charts use **ROUGE-comparable** episodes only; then **ROUGE (aggregate)** bar chart from `metrics.json` `vs_reference`, and **ROUGE (per episode)** box plots vs `data/eval/references/<ref_id>/predictions.jsonl` (recomputed with `rouge-score`, same setup as scoring). Charts follow `docs/guides/TUFTE_CHART_CRITIQUE.md` (data-ink, small multiples instead of legends where possible, bar baseline 0, no bubble area lie). Optional map/reduce diagnostics unchanged.
- **KPIs:** single wide table including **ROUGE-L F1** when `vs_reference` is present in `metrics.json`.
- **Delta:** baseline vs candidates; ROUGE-L deltas appear when both runs report `rougeL_f1`.
- **Episodes:** only the **intersection** of ROUGE-comparable episodes across all selected runs (apples-to-apples); filter, pagination, side-by-side text, unified diffs.
- **Top right:** **Artifact availability** popover (click to open — same “pull out” idea as the collapsible filter sidebar on the left; Streamlit has no native right sidebar).

See `docs/rfc/RFC-047-run-comparison-visual-tool.md` for the full design.
