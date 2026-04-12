# Run comparison tool (RFC-047)

Lightweight Streamlit UI for comparing ML evaluation runs from `data/eval/` artifacts
(`metrics.json`, `predictions.jsonl`, optional `diagnostics.jsonl`) and, on the **Performance**
page, frozen profiles from `data/profiles/*.yaml` (RFC-064) joined by release key (RFC-066).

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

- `BASELINE` — identifies a run by **`data/eval`-relative path** (`rel_label`, same string the app uses as the internal run key), by **run id** (directory basename), or by the legacy **`[category] rel_label`** form. If that run is in the current selection, the **Baseline (for deltas & episode diffs)** control on **Delta** / **Episodes** starts on it.

**Run labels in the UI:** the category filter still scopes which runs load, but dropdowns and charts do **not** repeat `[category]` on every line. Chart and table labels are **shortened** by stripping a shared prefix across the current selection and showing an ellipsis plus the **differing tail** (with numeric suffixes if two runs would still collide). **Baseline** selects (**Delta** / **Episodes** in the main column, and **Performance** in the sidebar) use **tail-first** path strings (last segments first, no leading `…`) so clipping still surfaces the end of the path; hover open-list options for the full value. The **Artifact availability** popover shows the short title with the full `rel_label` under it.

On load, **all** runs matching the category filter are selected; use the **checkbox table** (full `data/eval` paths) plus **Select all** / **Deselect all** in the sidebar to change that quickly.

Or run Streamlit directly:

```bash
BASELINE=... python -m streamlit run tools/run_compare/app.py --server.port=8501
```

## Layout

- **Left sidebar (collapsible):** categories, **Type** (paragraph vs bullets vs other — inferred from path names like `*_paragraph_*` / `*_bullets_*`), **Runs to compare** as a **checkbox table** (full paths, not tag multiselect) with **Select all** / **Deselect all**. A warning appears if you select runs that mix types. **Baseline** for quality deltas is **not** in the sidebar — it lives on **Delta** and **Episodes** (main column).
- **Top navigation:** page links **Home · KPIs · Delta · Episodes · Performance** (`?page=…`) sit in the **main** column (first row, with extra top padding so they clear Streamlit’s header). **Artifact availability** is a Streamlit popover on the same row (right column). Nav is **not** `position: fixed` so it does not block the sidebar toggle or header controls. **Eval run comparison** + intro live at the **top of the left sidebar**.
- **Home:** a **Jump** line links to in-page anchors for each chart block. Token/latency charts use **ROUGE-comparable** episodes only. **Latency vs output length** uses **tokens on the x-axis** and **latency on the y-axis**, shows how many episodes have row-level timing vs not, and one **scatter** (color = run); per-episode timing comes only from `metadata.processing_time_seconds` in `predictions.jsonl` (not from aggregate `metrics.json`). **ROUGE (aggregate)** shows how many selected runs have **`vs_reference`** mean F1 in **`metrics.json`**, warns when those runs use **different reference ids**, and one **grouped bar** chart (color = run). **ROUGE (per episode)** is box plots vs `data/eval/references/<ref_id>/predictions.jsonl` (recomputed with `rouge-score`, same setup as scoring). Charts follow `docs/guides/TUFTE_CHART_CRITIQUE.md` (data-ink, bar baseline 0, no bubble area lie). Optional map/reduce diagnostics unchanged.
- **KPIs:** single wide table (same `st.dataframe` behavior as **Delta**) including **ROUGE-L F1** when `vs_reference` is present in `metrics.json`.
- **Delta:** choose **baseline** at the top, then baseline vs candidates; ROUGE-L deltas appear when both runs report `rougeL_f1`.
- **Episodes:** same **baseline** control; **Jump** links for browse vs summaries vs diffs; only the **intersection** of ROUGE-comparable episodes across all selected runs (apples-to-apples). After you pick an episode: **summaries** for that `episode_id` in up to **three columns** (or **tabs** when more runs), then **unified diffs** vs baseline in expanders.
- **Top right:** **Artifact availability** popover (click to open — same “pull out” idea as the collapsible filter sidebar on the left; Streamlit has no native right sidebar).
- **Performance** (`?page=performance`): does **not** require two eval runs; sidebar starts with the same **Eval run comparison** title + Performance intro, then filters (hostname, dataset, releases, baseline for resource deltas). Main column includes a **Jump** row for KPIs, optional **Monitor traces** (when present), delta, trends, scatter, and coverage. Joins eval runs to profiles when `fingerprint.json` has `run_context.release` (or top-level `release`) matching the profile’s `release` field; otherwise the eval **run id** (directory name) is the join key. Opens with **one stacked figure** (three small-multiple bar panels: peak RSS, total wall time, episode count — shared x = release; **green → red** encodes best→worst **within each panel**: lower RSS and lower wall time are greener; on the episodes row, color follows **seconds per episode** while bar height stays episode count, `docs/guides/TUFTE_CHART_CRITIQUE.md`), then **RFC-065 monitor traces** (table + download expander) for any selected profile that has a sibling **`<stem>.monitor.log`** or **`rfc065_monitor`** in **`<stem>.stage_truth.json`**, then resource delta table, per-stage trend lines, quality-vs-cost scatter (needs both eval + profile), and a coverage table.

See `docs/rfc/RFC-047-run-comparison-visual-tool.md` for the quality UI design and `docs/rfc/RFC-066-run-compare-performance-tab.md` for the performance tab.
