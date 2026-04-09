"""Streamlit UI for RFC-047 run comparison (Issue #373)."""

from __future__ import annotations

import os
import sys
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except ImportError as _import_err:
    raise SystemExit(
        "Missing dependencies for compare UI. Install with: pip install -e '.[compare]'"
    ) from _import_err

from tools.run_compare.data import (
    artifact_status,
    compute_per_episode_rouge_rows,
    delta_direction_good,
    discover_runs,
    extract_aggregate_rouge,
    extract_kpis,
    get_summary_text,
    index_predictions,
    load_diagnostics_jsonl,
    load_metrics,
    load_predictions_jsonl,
    merge_run_summary,
    pick_shared_reference_id,
    predictions_to_chart_rows,
    reference_predictions_path,
    repo_root_from_here,
    rouge_comparable_episode_ids,
    RunEntry,
)

PAGE_SIZE = 20

_METRIC_KEYS = [
    ("success_rate", "Success rate"),
    ("failed_count", "Failed episodes"),
    ("rougeL_f1", "ROUGE-L F1 (vs ref)"),
    ("avg_output_tokens", "Avg output tokens"),
    ("avg_latency_s", "Avg latency (s)"),
    ("truncation_rate", "Truncation rate"),
    ("speaker_label_leak_rate", "Speaker leak rate"),
    ("boilerplate_leak_rate", "Boilerplate leak rate"),
]

_NAV_PAGES = ("Home", "KPIs", "Delta", "Episodes")

# URL ?page=… slugs ↔ nav labels (real links, not radio bullets)
_SLUG_TO_PAGE = {
    "home": "Home",
    "kpis": "KPIs",
    "delta": "Delta",
    "episodes": "Episodes",
}
_PAGE_TO_SLUG = {label: slug for slug, label in _SLUG_TO_PAGE.items()}

_ROUGE_METRIC_ORDER = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

_NAV_CSS = """
<style>
.runcompare-nav a {
  color: inherit;
  text-decoration: none;
  border-bottom: 1px solid transparent;
}
.runcompare-nav a:hover {
  border-bottom: 1px solid currentColor;
}
/* Less top padding so main nav sits near the top of the page */
section.main div.block-container {
  padding-top: 0.5rem !important;
  padding-bottom: 1rem !important;
}
section.main div.block-container > div:first-child {
  margin-top: 0 !important;
}
</style>
"""


def _page_from_query() -> str:
    """Active page from ``?page=home|kpis|delta|episodes`` (default: Home)."""
    qp = st.query_params
    raw = qp.get("page", "home")
    if isinstance(raw, list):
        raw = raw[0] if raw else "home"
    slug = str(raw).lower().strip()
    return _SLUG_TO_PAGE.get(slug, "Home")


def _render_nav_row(
    current_page: str,
    selected: Optional[Sequence[RunEntry]],
) -> None:
    """Top row: text links (left) + artifact popover when runs are loaded (right)."""
    st.markdown(_NAV_CSS, unsafe_allow_html=True)
    sep = ' <span style="opacity:0.45;">·</span> '
    parts: List[str] = []
    for label in _NAV_PAGES:
        if label == current_page:
            parts.append(f'<span style="font-weight:600;">{label}</span>')
        else:
            slug = _PAGE_TO_SLUG[label]
            parts.append(f'<a href="?page={slug}" target="_self">{label}</a>')
    nav_html = sep.join(parts)
    left, right = st.columns([5, 1])
    with left:
        st.markdown(
            f'<div class="runcompare-nav" style="font-size:1.05rem;">{nav_html}</div>',
            unsafe_allow_html=True,
        )
    with right:
        if selected is not None and len(selected) >= 2:
            with st.popover(
                "Artifact availability",
                use_container_width=True,
                help="Per-run files present on disk (same idea as the collapsible filter sidebar).",
            ):
                st.caption("Files present under each run directory")
                for e in selected:
                    st.markdown(f"**{_format_entry_label(e)}**")
                    st.json(artifact_status(e.path))
        else:
            st.caption("")


def _format_entry_label(e: RunEntry) -> str:
    return f"[{e.category}] {e.rel_label}"


def _sync_run_multiselect_state(flabels: List[str]) -> None:
    """Keep multiselect session state in sync with ``flabels``; default = all visible runs."""
    ms_key = "run_compare_multiselect"
    sig_key = "run_compare_flabels_sig"
    sig = tuple(flabels)
    if st.session_state.get(sig_key) == sig:
        return
    st.session_state[sig_key] = sig
    prev = st.session_state.get(ms_key)
    if isinstance(prev, list):
        kept = [x for x in prev if x in flabels]
        st.session_state[ms_key] = kept if kept else list(flabels)
    else:
        st.session_state[ms_key] = list(flabels)


def _parse_env_baseline(entries: Sequence[RunEntry]) -> Optional[RunEntry]:
    """Resolve ``BASELINE`` env to a run entry if it matches a discovered run."""
    baseline_env = (os.environ.get("BASELINE") or "").strip()
    if not baseline_env:
        return None
    by_id = {e.run_id: e for e in entries}
    by_rel = {e.rel_label: e for e in entries}
    return by_id.get(baseline_env) or by_rel.get(baseline_env)


def _load_runs_data(
    selected: Sequence[RunEntry],
) -> Optional[Dict[str, Dict[str, Any]]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for e in selected:
        try:
            m = merge_run_summary(load_metrics(e.path), e.path)
            preds = load_predictions_jsonl(e.path / "predictions.jsonl")
            loaded[_format_entry_label(e)] = {
                "entry": e,
                "metrics": m,
                "kpis": extract_kpis(m),
                "preds": preds,
            }
        except (OSError, ValueError, KeyError) as ex:
            st.error(f"Failed to load {_format_entry_label(e)}: {ex}")
            return None
    return loaded


def _cache_key(selected_labels: Sequence[str], baseline_label: str) -> str:
    return "|".join(sorted(selected_labels)) + "||" + baseline_label


def _legend_below_chart(fig: go.Figure, *, bottom_margin: int = 120) -> None:
    """Put legend under the plot area so the plot uses full width (not squeezed by right legend)."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=bottom_margin),
    )


def _apply_tufte_base(fig: go.Figure) -> None:
    """High data-ink: white ground, minimal chrome (see docs/guides/TUFTE_CHART_CRITIQUE.md)."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222222", size=12),
    )


def _tufte_xy_faint_grid_x(fig: go.Figure) -> None:
    """Hairline vertical grid on x (read values); no y grid."""
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=0.5,
        zeroline=True,
        zerolinewidth=0.5,
        zerolinecolor="#dddddd",
    )
    fig.update_yaxes(showgrid=False, zeroline=False)


def _kpi_table(selected: Sequence[RunEntry], loaded: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Wide table: one row per run (avoids cramped metric tiles)."""
    rows: List[Dict[str, Any]] = []
    for e in selected:
        k = loaded[_format_entry_label(e)]["kpis"]
        rows.append(
            {
                "Run": _format_entry_label(e),
                "Success %": round(k["success_rate"] * 100, 2),
                "Failed #": int(k["failed_count"]),
                "Avg tokens": (
                    None
                    if k.get("avg_output_tokens") is None
                    else round(float(k["avg_output_tokens"]), 2)
                ),
                "Latency s": (
                    None if k.get("avg_latency_s") is None else round(float(k["avg_latency_s"]), 2)
                ),
                "Trunc rate": k.get("truncation_rate"),
                "Spk leak": k.get("speaker_label_leak_rate"),
                "Boiler rate": k.get("boilerplate_leak_rate"),
                "ROUGE-L": (
                    None if k.get("rougeL_f1") is None else round(float(k["rougeL_f1"]), 4)
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_delta_table(
    base_kpis: Dict[str, Any],
    candidates: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
) -> None:
    delta_rows: List[Dict[str, Any]] = []
    for key, label in _METRIC_KEYS:
        bv = base_kpis.get(key)
        if bv is None:
            continue
        for e in candidates:
            ck = loaded[_format_entry_label(e)]["kpis"].get(key)
            if ck is None:
                continue
            delta = float(ck) - float(bv)
            good = delta_direction_good(key, delta)
            delta_rows.append(
                {
                    "metric": label,
                    "baseline": bv,
                    "candidate": _format_entry_label(e),
                    "value": ck,
                    "delta": delta,
                    "good": good,
                }
            )
    if not delta_rows:
        st.info("No comparable metrics for delta (check baseline and candidates).")
        return
    st.dataframe(
        pd.DataFrame(delta_rows),
        width="stretch",
        hide_index=True,
    )


_PLOTLY_CHART_LEFT_CSS = """
<style>
div[data-testid="stPlotlyChart"] {
  text-align: left !important;
}
div[data-testid="stPlotlyChart"] > div {
  width: 100% !important;
  align-items: flex-start !important;
}
div[data-testid="stPlotlyChart"] iframe {
  margin-left: 0 !important;
  margin-right: auto !important;
  display: block;
}
</style>
"""


def _render_rouge_charts(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
) -> None:
    """Aggregate ROUGE from metrics.json; per-episode ROUGE vs reference predictions."""
    eval_root = repo_root_from_here() / "data" / "eval"

    agg_rows: List[Dict[str, Any]] = []
    for e in selected:
        label = _format_entry_label(e)
        hit = extract_aggregate_rouge(loaded[label]["metrics"])
        if not hit:
            continue
        _ref_id, rouges = hit
        label_map = {
            "rouge1_f1": "ROUGE-1",
            "rouge2_f1": "ROUGE-2",
            "rougeL_f1": "ROUGE-L",
        }
        for k, mname in label_map.items():
            v = rouges.get(k)
            if v is not None:
                agg_rows.append({"run": label, "metric": mname, "value": float(v)})

    if agg_rows:
        st.subheader("ROUGE (aggregate from metrics)")
        st.caption(
            "From `metrics.json` vs_reference — same means as scoring. "
            "Bars are baseline 0; values labeled on the bars (no color legend)."
        )
        dfa = pd.DataFrame(agg_rows)
        vmax = float(dfa["value"].max())
        n_runs_agg = dfa["run"].nunique()
        wrap_agg = min(4, max(2, n_runs_agg))
        fig_agg = px.bar(
            dfa,
            x="metric",
            y="value",
            facet_col="run",
            facet_col_wrap=wrap_agg,
            category_orders={"metric": _ROUGE_METRIC_ORDER},
            text="value",
        )
        fig_agg.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            cliponaxis=False,
            marker=dict(line=dict(width=0.5, color="#333333")),
        )
        _apply_tufte_base(fig_agg)
        fig_agg.update_layout(
            showlegend=False,
            height=max(300, 240 + 60 * ((n_runs_agg + wrap_agg - 1) // wrap_agg)),
            margin=dict(t=40, b=48, l=48, r=24),
        )
        fig_agg.update_yaxes(
            range=[0, max(0.05, 1.0, vmax * 1.12)],
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=0.5,
            zeroline=True,
            zerolinewidth=0.5,
            zerolinecolor="#cccccc",
        )
        fig_agg.update_xaxes(showgrid=False, title="")
        st.plotly_chart(fig_agg, width="stretch")
    else:
        st.caption("No aggregate ROUGE in metrics (add `vs_reference` ROUGE scores to runs).")

    ref_id, consistent = pick_shared_reference_id(
        loaded,
        ordered_labels=[_format_entry_label(e) for e in selected],
    )
    if not ref_id:
        return
    if not consistent:
        st.warning(
            "Selected runs list different `vs_reference` ids; per-episode ROUGE uses the "
            "first run's reference."
        )
    ref_path = reference_predictions_path(eval_root, ref_id)
    if not ref_path.is_file():
        st.caption(f"Per-episode ROUGE skipped: no file `references/{ref_id}/predictions.jsonl`.")
        return

    ref_preds = load_predictions_jsonl(ref_path)
    ref_by_id = index_predictions(ref_preds)
    peri_rows: List[Dict[str, Any]] = []
    for e in selected:
        label = _format_entry_label(e)
        k = loaded[label]["kpis"]
        preds = loaded[label]["preds"]
        failed_ids = k.get("failed_episodes") or []
        peri_rows.extend(compute_per_episode_rouge_rows(label, preds, failed_ids, ref_by_id))

    if not peri_rows:
        st.caption(
            "Per-episode ROUGE: no overlapping episodes with reference text, or "
            "`rouge-score` not installed."
        )
        return

    dfp = pd.DataFrame(peri_rows)
    long = dfp.melt(
        id_vars=["run", "episode_id"],
        value_vars=["rouge1_f1", "rouge2_f1", "rougeL_f1"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(
        {
            "rouge1_f1": "ROUGE-1",
            "rouge2_f1": "ROUGE-2",
            "rougeL_f1": "ROUGE-L",
        }
    )
    st.subheader("ROUGE (per episode vs reference)")
    st.caption(
        f"Reference **`{ref_id}`** — recomputed with `rouge-score` for comparable episodes. "
        "One column per metric (small multiples); boxes + all points — no legend."
    )
    fig_per = px.box(
        long,
        x="value",
        y="run",
        facet_col="metric",
        facet_col_spacing=0.05,
        orientation="h",
        points="all",
        category_orders={"metric": _ROUGE_METRIC_ORDER},
    )
    _apply_tufte_base(fig_per)
    fig_per.update_layout(
        showlegend=False,
        height=max(280, 80 + 28 * long["run"].nunique()),
        margin=dict(t=36, b=48, l=8, r=16),
    )
    fig_per.update_xaxes(
        title="F1",
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=0.5,
        zeroline=True,
        zerolinewidth=0.5,
        zerolinecolor="#dddddd",
    )
    fig_per.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig_per, width="stretch")


def _render_charts_a_b(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
) -> None:
    st.markdown(_PLOTLY_CHART_LEFT_CSS, unsafe_allow_html=True)
    chart_rows: List[Dict[str, Any]] = []
    for e in selected:
        label = _format_entry_label(e)
        k = loaded[label]["kpis"]
        preds = loaded[label]["preds"]
        failed_ids = k.get("failed_episodes") or []
        chart_rows.extend(predictions_to_chart_rows(label, preds, failed_ids))
    df_chart = pd.DataFrame(chart_rows)
    if df_chart.empty:
        st.warning(
            "No ROUGE-comparable episodes for token/latency charts (need non-empty summaries, "
            "no inference error, not in gate failures)."
        )
    else:
        st.subheader("Output tokens (per episode, estimated)")
        st.caption(
            "Comparable episodes only. **Insight:** compare medians (bar in box) and spread; "
            "jittered points are every episode — variation is the story, not a single mean."
        )
        fig_a = px.box(
            df_chart,
            x="output_tokens_est",
            y="run",
            points="all",
            orientation="h",
        )
        _apply_tufte_base(fig_a)
        fig_a.update_layout(
            showlegend=False,
            autosize=True,
            height=max(280, 80 + 28 * df_chart["run"].nunique()),
            margin=dict(l=8, r=16, t=8, b=52),
            xaxis=dict(
                title="Estimated output tokens",
                automargin=True,
                side="bottom",
                anchor="y",
            ),
            yaxis=dict(
                title="",
                automargin=True,
                side="left",
                ticklabelposition="outside",
            ),
        )
        _tufte_xy_faint_grid_x(fig_a)
        fig_a.update_xaxes(automargin=True)
        fig_a.update_yaxes(automargin=True)
        st.plotly_chart(fig_a, width="stretch")

        st.subheader("Latency vs output length")
        sub = df_chart.dropna(subset=["latency_s"])
        if sub.empty:
            st.caption("No latency data in predictions metadata.")
        else:
            st.caption(
                "**Insight:** one panel per run (small multiples) — same scales, no color legend. "
                "Fixed marker size avoids bubble area distortion (Tufte lie factor)."
            )
            n_runs_b = int(sub["run"].nunique())
            wrap_b = min(4, max(2, n_runs_b))
            fig_b = px.scatter(
                sub,
                x="latency_s",
                y="output_tokens_est",
                facet_col="run",
                facet_col_wrap=wrap_b,
                hover_data=["episode_id"],
            )
            fig_b.update_traces(
                marker=dict(size=7, opacity=0.62, line=dict(width=0.45, color="rgba(0,0,0,0.2)")),
                hovertemplate=(
                    "Latency: %{x:.2f} s<br>"
                    "Tokens: %{y:.0f}<br>"
                    "Episode: %{customdata[0]}<extra></extra>"
                ),
            )
            _apply_tufte_base(fig_b)
            fig_b.update_layout(
                showlegend=False,
                height=max(320, 260 + 220 * ((n_runs_b + wrap_b - 1) // wrap_b)),
                margin=dict(t=36, b=48, l=52, r=16),
                xaxis_title="Latency (s)",
                yaxis_title="Estimated output tokens",
            )
            fig_b.update_xaxes(
                showgrid=True,
                gridcolor="#e8e8e8",
                gridwidth=0.5,
                zeroline=True,
                zerolinewidth=0.5,
                zerolinecolor="#dddddd",
            )
            fig_b.update_yaxes(
                showgrid=True,
                gridcolor="#e8e8e8",
                gridwidth=0.5,
                zeroline=True,
                zerolinewidth=0.5,
                zerolinecolor="#dddddd",
            )
            st.plotly_chart(fig_b, width="stretch")

    _render_rouge_charts(selected, loaded)


def _render_chart_c(selected: Sequence[RunEntry]) -> None:
    any_diag = any((e.path / "diagnostics.jsonl").is_file() for e in selected)
    if not any_diag:
        st.caption("Map/reduce diagnostics: no `diagnostics.jsonl` in selected runs.")
        return
    st.subheader("Map/reduce diagnostics (per episode)")
    for e in selected:
        p = e.path / "diagnostics.jsonl"
        if not p.is_file():
            continue
        rows = load_diagnostics_jsonl(p)
        if not rows:
            continue
        dfp = pd.DataFrame(rows)
        if "episode_id" not in dfp.columns:
            continue
        fig_c = go.Figure()
        for col in ("avg_map_tokens", "reduce_input_tokens", "final_tokens"):
            if col in dfp.columns:
                fig_c.add_trace(go.Bar(name=col, x=dfp["episode_id"], y=dfp[col].astype(float)))
        fig_c.update_layout(barmode="group", title=_format_entry_label(e))
        _legend_below_chart(fig_c, bottom_margin=100)
        st.plotly_chart(fig_c, width="stretch")


def _render_episode_drilldown(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
    bl_key: str,
) -> None:
    comp_sets: List[set] = []
    for e in selected:
        lbl = _format_entry_label(e)
        preds = loaded[lbl]["preds"]
        failed = loaded[lbl]["kpis"].get("failed_episodes") or []
        comp_sets.append(rouge_comparable_episode_ids(preds, failed))
    episode_ids = sorted(set.intersection(*comp_sets)) if comp_sets else []
    if not episode_ids:
        st.warning(
            "No episode is ROUGE-comparable across all selected runs "
            "(non-empty summary, no error, not in gate failures). "
            "Expand selection or fix failed runs."
        )
        return
    st.caption(
        "Episodes listed are the intersection of ROUGE-comparable episodes per run "
        "(apples-to-apples with aggregate ROUGE)."
    )
    q = st.text_input("Filter episode id", "")
    if q:
        episode_ids = [eid for eid in episode_ids if q in eid]
    total = len(episode_ids)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start = (page - 1) * PAGE_SIZE
    slice_ids = episode_ids[start : start + PAGE_SIZE]
    st.caption(f"Showing {len(slice_ids)} of {total} episodes (page {page}/{n_pages})")
    if not slice_ids:
        st.warning("No episodes match the filter.")
        return
    ep = st.selectbox("Episode", options=slice_ids, index=0)

    side = {
        _format_entry_label(e): index_predictions(loaded[_format_entry_label(e)]["preds"])
        for e in selected
    }
    texts: List[Tuple[str, str, Optional[str]]] = []
    for e in selected:
        label = _format_entry_label(e)
        pred = side[label].get(ep)
        if not pred:
            texts.append((label, "", f"Missing episode {ep}"))
            continue
        txt, err = get_summary_text(pred)
        texts.append((label, txt, err))

    c_ep = st.columns(len(selected))
    for i, (col, (label, txt, err)) in enumerate(zip(c_ep, texts)):
        with col:
            st.markdown(f"**{label}**")
            if err:
                st.error(err)
            else:
                st.text_area(
                    "Summary",
                    value=txt,
                    height=320,
                    key=f"sum_ep_{i}_{ep}",
                    label_visibility="collapsed",
                )

    base_txt = next((t for lab, t, er in texts if lab == bl_key and not er), "")
    if not base_txt:
        return
    for lab, cand_txt, err in texts:
        if lab == bl_key or err or not cand_txt:
            continue
        diff_lines = list(
            unified_diff(
                base_txt.splitlines(),
                cand_txt.splitlines(),
                fromfile="baseline",
                tofile=lab,
                lineterm="",
            )
        )
        if diff_lines:
            with st.expander(f"Unified diff vs {lab}"):
                st.code("\n".join(diff_lines), language="diff")


def main() -> None:
    st.set_page_config(page_title="Run compare", layout="wide", initial_sidebar_state="expanded")

    entries = discover_runs()
    page = _page_from_query()

    if not entries:
        st.markdown(_NAV_CSS, unsafe_allow_html=True)
        _render_nav_row(page, None)
        st.title("Eval run comparison")
        st.caption("RFC-047 — baseline vs candidate(s) from data/eval artifacts")
        st.warning("No runs found under data/eval (runs/, baselines/, references/).")
        return

    labels = [_format_entry_label(e) for e in entries]
    label_to_entry = dict(zip(labels, entries))
    env_base = _parse_env_baseline(entries)

    # One injection before sidebar so main column top padding applies before content.
    st.markdown(_NAV_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Filters")
        cat_filter = st.multiselect(
            "Categories",
            options=["run", "baseline", "reference"],
            default=["run", "baseline", "reference"],
        )
        filtered = [e for e in entries if e.category in cat_filter]
        flabels = [_format_entry_label(e) for e in filtered]
        if not flabels:
            st.error("No runs match the category filter.")
            selected_labels = []
            baseline_pick = ""
        else:
            _sync_run_multiselect_state(flabels)
            ms_key = "run_compare_multiselect"
            hdr = st.columns([1.4, 2.4])
            with hdr[0]:
                st.markdown("**Runs to compare** (2+)")
            with hdr[1]:
                bulk = st.radio(
                    "Apply to list",
                    options=["—", "Select all", "Deselect all"],
                    horizontal=True,
                    key="run_compare_bulk",
                    help="Choose an action once; selection resets to — after applying.",
                )
            if bulk == "Select all":
                st.session_state[ms_key] = list(flabels)
                st.session_state["run_compare_bulk"] = "—"
                st.rerun()
            if bulk == "Deselect all":
                st.session_state[ms_key] = []
                st.session_state["run_compare_bulk"] = "—"
                st.rerun()
            selected_labels = st.multiselect(
                "Runs to compare",
                options=flabels,
                key=ms_key,
                label_visibility="collapsed",
                placeholder="Choose runs...",
            )
            baseline_pick = ""
            if selected_labels:
                baseline_idx = 0
                if env_base:
                    want = _format_entry_label(env_base)
                    if want in selected_labels:
                        baseline_idx = selected_labels.index(want)
                baseline_pick = st.selectbox(
                    "Baseline (for deltas)",
                    options=selected_labels,
                    index=min(baseline_idx, max(0, len(selected_labels) - 1)),
                )

    if not flabels:
        _render_nav_row(page, None)
        st.title("Eval run comparison")
        st.caption("RFC-047 — baseline vs candidate(s) from data/eval artifacts")
        st.error("No runs match the category filter (see sidebar).")
        return

    if len(selected_labels) < 2:
        _render_nav_row(page, None)
        st.title("Eval run comparison")
        st.caption("RFC-047 — baseline vs candidate(s) from data/eval artifacts")
        st.info("Select at least two runs in the sidebar (left).")
        return

    selected = [label_to_entry[x] for x in selected_labels if x in label_to_entry]
    baseline = label_to_entry[baseline_pick]
    candidates = [e for e in selected if e.path != baseline.path]
    bl_key = _format_entry_label(baseline)

    ck = _cache_key(selected_labels, baseline_pick)
    if st.session_state.get("run_compare_cache_key") != ck or "loaded" not in st.session_state:
        loaded = _load_runs_data(selected)
        if loaded is None:
            _render_nav_row(page, selected)
            st.title("Eval run comparison")
            st.caption("RFC-047 — baseline vs candidate(s) from data/eval artifacts")
            return
        st.session_state["loaded"] = loaded
        st.session_state["run_compare_cache_key"] = ck
    else:
        loaded = st.session_state["loaded"]

    base_kpis = loaded[bl_key]["kpis"]

    _render_nav_row(page, selected)
    st.title("Eval run comparison")
    st.caption("RFC-047 — baseline vs candidate(s) from data/eval artifacts")
    st.markdown("---")

    if page == "Home":
        _render_charts_a_b(selected, loaded)  # tokens/latency then ROUGE charts
        _render_chart_c(selected)
    elif page == "KPIs":
        st.subheader("KPI summary")
        st.caption("One row per run — scroll horizontally if needed.")
        df_kpi = _kpi_table(selected, loaded)
        st.dataframe(df_kpi, width="stretch", hide_index=True)
    elif page == "Delta":
        st.subheader("Delta vs baseline")
        st.caption(f"Baseline: **{bl_key}**")
        _render_delta_table(base_kpis, candidates, loaded)
    else:
        st.subheader("Episode drill-down")
        st.caption(f"Baseline for diffs: **{bl_key}**")
        _render_episode_drilldown(selected, loaded, bl_key)


if __name__ == "__main__":
    main()
