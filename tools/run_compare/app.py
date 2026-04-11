"""Streamlit UI for RFC-047 run comparison (Issue #373)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import uuid
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    import streamlit.components.v1 as st_components
    from plotly.subplots import make_subplots
except ImportError as _import_err:
    raise SystemExit(
        "Missing dependencies for compare UI. Install with: pip install -e '.[compare]'"
    ) from _import_err

from tools.run_compare.data import (
    artifact_status,
    compact_baseline_select_labels,
    compact_run_display_names,
    compute_per_episode_rouge_rows,
    delta_direction_good,
    discover_profiles,
    discover_runs,
    extract_aggregate_rouge,
    extract_kpis,
    filter_joined_releases,
    get_summary_text,
    index_predictions,
    infer_run_type_bucket,
    invert_compact_display_map,
    join_releases,
    JoinedRelease,
    load_diagnostics_jsonl,
    load_metrics,
    load_predictions_jsonl,
    merge_run_summary,
    pick_shared_reference_id,
    predictions_to_chart_rows,
    profile_stage_delta_rows,
    profile_trend_long_rows,
    ProfileEntry,
    reference_predictions_path,
    repo_root_from_here,
    rouge_comparable_episode_ids,
    RUN_TYPE_LABELS,
    RUN_TYPE_ORDER,
    RunEntry,
)

PAGE_SIZE = 20


def _episode_text_area_key(run_index: int, episode_id: str) -> str:
    """Stable Streamlit key: episode ids may contain ``/``, spaces, etc."""
    h = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:16]
    return f"rc_ep_sum_{run_index}_{h}"


# Checkbox table for "Runs to compare" (full paths; not tag/pill multiselect).
_RUN_COMPARE_RUNS_EDITOR_KEY = "run_compare_runs_editor"

# Sidebar run labels: wider than charts so paths stay readable; truncation keeps the tail only.
_SIDEBAR_RUN_LABEL_MAX_CHARS = 96
# Baseline select (main column on Delta / Episodes): wider than old sidebar widget.
_MAIN_BASELINE_SELECT_MAX_CHARS = 56

_CATEGORY_LABELS = {
    "run": "Run",
    "baseline": "Baseline",
    "reference": "Reference",
}
_CATEGORY_OPTION_TOOLTIPS = {
    "run": "Eval runs under data/eval/runs/",
    "baseline": "Baselines under data/eval/baselines/",
    "reference": "Reference outputs under data/eval/references/",
}

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

_NAV_PAGES = ("Home", "KPIs", "Delta", "Episodes", "Performance")

# URL ?page=… slugs ↔ nav labels (real links, not radio bullets)
_SLUG_TO_PAGE = {
    "home": "Home",
    "kpis": "KPIs",
    "delta": "Delta",
    "episodes": "Episodes",
    "performance": "Performance",
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
.runcompare-nav {
  margin-top: 0 !important;
}
/*
 * Clear Streamlit’s fixed top chrome (header + toolbar ≈ 3–3.5rem). Values below
 * ~2.5rem often leave the first markdown (nav) visually clipped until you scroll.
 * Do not use position:fixed + high z-index for nav — it can block the sidebar toggle
 * and header controls (Deploy / hamburger) after collapsing the sidebar.
 */
.stAppViewContainer .stMain .block-container,
section.main div.block-container {
  padding-top: 3.5rem !important;
  padding-bottom: 1rem !important;
}
section.main div.block-container > div:first-child {
  margin-top: 0 !important;
}

/*
 * Sidebar width when expanded. When collapsed, avoid overriding width/max-width —
 * Streamlit’s own rules control the reopen control; aggressive overrides broke toggling.
 */
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: min(18rem, 96vw) !important;
  width: clamp(18rem, 25.5vw, 24rem) !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
  min-width: 0 !important;
}
section[data-testid="stSidebar"] > div {
  width: 100% !important;
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 0.75rem !important;
  padding-left: 0.65rem !important;
  padding-right: 0.65rem !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.4rem !important;
}
/* Multiselect / select dropdown panel: use more viewport height */
section[data-testid="stSidebar"] [data-baseweb="menu"] {
  max-height: min(28rem, 70vh) !important;
}
/* Single-value sidebar selects: let Streamlit clip; labels use tail-first compact strings. */
section[data-testid="stSidebar"] [data-baseweb="select"]:not(:has([data-baseweb="tag"])) {
  max-width: 100% !important;
}
</style>
"""

# Doc links (GitHub main); matches references in docs/guides/EXPERIMENT_GUIDE.md.
_DOCS_BASE = "https://github.com/chipi/podcast_scraper/blob/main"


def _quality_intro_markdown() -> str:
    return (
        "This tool compares **eval runs** side by side. Each run is a directory under "
        "`data/eval/` (`runs/`, `baselines/`, `references/`) with `metrics.json`, "
        "`predictions.jsonl`, and optional files such as `diagnostics.jsonl`. "
        "Pick categories below and select **two or more runs**. "
        "On **Delta** and **Episodes**, pick a **baseline** run for comparisons (Home and KPIs "
        "use every selected run).\n\n"
        "Use **Home** for charts (tokens, latency, ROUGE), **KPIs** for a summary table, "
        "**Delta** for changes vs baseline, and **Episodes** for per-episode text "
        "comparison. Open **Performance** (`?page=performance`) to relate runs to frozen "
        "profiles under `data/profiles/*.yaml`.\n\n"
        "Guides: "
        f"[Experiment guide — visual run comparison]({_DOCS_BASE}/docs/guides/"
        "EXPERIMENT_GUIDE.md#visual-run-comparison-rfc-047) · "
        f"[RFC-047 design]({_DOCS_BASE}/docs/rfc/"
        "RFC-047-run-comparison-visual-tool.md) · "
        f"[Tool README]({_DOCS_BASE}/tools/run_compare/README.md)."
    )


def _performance_intro_markdown() -> str:
    return (
        "**Performance** joins **frozen profiles** (`data/profiles/*.yaml`, RFC-064) with "
        "eval runs when the **release** join key matches (see README). Use the filters below "
        "to narrow hosts and datasets, then compare KPIs, deltas, and trends.\n\n"
        f"[RFC-066]({_DOCS_BASE}/docs/rfc/RFC-066-run-compare-performance-tab.md) · "
        f"[Performance profile guide]({_DOCS_BASE}/docs/guides/PERFORMANCE_PROFILE_GUIDE.md) "
        f"· [Experiment guide]({_DOCS_BASE}/docs/guides/EXPERIMENT_GUIDE.md)."
    )


def _render_quality_sidebar_title_and_intro() -> None:
    st.markdown("### Eval run comparison")
    st.markdown(_quality_intro_markdown())
    st.divider()


def _render_performance_sidebar_title_and_intro() -> None:
    st.markdown("### Eval run comparison")
    st.markdown(_performance_intro_markdown())
    st.divider()


def _section_anchor(html_id: str) -> None:
    st.markdown(
        f'<div id="{html_id}" style="scroll-margin-top:1rem;"></div>', unsafe_allow_html=True
    )


def _page_jump_nav(links: Sequence[Tuple[str, str]]) -> None:
    """In-page links (HTML anchors). Shown only when there are 2+ sections."""
    if len(links) < 2:
        return
    parts = [
        f'<a href="#{html_id}" style="text-decoration:none;">{label}</a>'
        for html_id, label in links
    ]
    st.markdown(
        '<p style="font-size:0.92rem;margin:0 0 0.75rem 0;line-height:1.45;">'
        "<strong>Jump</strong>: " + " · ".join(parts) + "</p>",
        unsafe_allow_html=True,
    )


def _home_jump_nav_items(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
    disp_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    rows: List[Dict[str, Any]] = []
    for e in selected:
        rk = _run_key(e)
        dname = disp_map.get(rk, rk)
        k = loaded[rk]["kpis"]
        preds = loaded[rk]["preds"]
        failed_ids = k.get("failed_episodes") or []
        rows.extend(predictions_to_chart_rows(dname, preds, failed_ids))
    df_chart = pd.DataFrame(rows)
    if not df_chart.empty:
        items.append(("rc-h-tokens", "Output tokens"))
        items.append(("rc-h-latency", "Latency vs length"))
    items.append(("rc-h-rouge-agg", "ROUGE aggregate"))
    eval_root = repo_root_from_here() / "data" / "eval"
    ref_hit, _ = pick_shared_reference_id(loaded, ordered_labels=[_run_key(e) for e in selected])
    if ref_hit and reference_predictions_path(eval_root, ref_hit).is_file():
        items.append(("rc-h-rouge-per", "ROUGE per episode"))
    if any((e.path / "diagnostics.jsonl").is_file() for e in selected):
        items.append(("rc-h-diag", "Map/reduce"))
    return items


def _inject_sidebar_option_tooltips(display_to_full: Dict[str, str]) -> None:
    """Attach ``title`` to sidebar multiselect options/tags (Streamlit has no per-option help)."""
    if not display_to_full:
        return
    b64 = base64.b64encode(json.dumps(display_to_full, ensure_ascii=False).encode("utf-8")).decode(
        "ascii"
    )
    tip_id = f"runcompare-tip-{uuid.uuid4().hex[:10]}"
    html = f"""<div id="{tip_id}" data-b64="{b64}"></div>
<script>
(function () {{
  const anchor = document.getElementById("{tip_id}");
  if (!anchor) return;
  const raw = anchor.getAttribute("data-b64");
  if (!raw) return;
  let map = {{}};
  try {{ map = JSON.parse(atob(raw)); }} catch (e) {{ return; }}
  const doc = window.parent.document;
  function cleanTagText(t) {{
    return (t || "")
      .replace(/[\u00d7×✕]/g, "")
      .replace(/\\s+/g, " ")
      .trim();
  }}
  function apply() {{
    const sb = doc.querySelector('section[data-testid="stSidebar"]');
    if (!sb) return;
    sb.querySelectorAll('[role="option"]').forEach(function (el) {{
      const k = (el.textContent || "").trim();
      const tip = map[k];
      if (tip) el.setAttribute("title", tip);
    }});
    sb.querySelectorAll('[data-baseweb="tag"]').forEach(function (el) {{
      const k = cleanTagText(el.textContent);
      const tip = map[k];
      if (tip) el.setAttribute("title", tip);
    }});
  }}
  apply();
  const sb = doc.querySelector('section[data-testid="stSidebar"]');
  if (sb) {{
    const obs = new MutationObserver(function () {{ apply(); }});
    obs.observe(sb, {{ subtree: true, childList: true, characterData: true }});
    setTimeout(function () {{ obs.disconnect(); }}, 120000);
  }}
}})();
</script>"""
    st_components.html(html, height=0)


def _joined_release_tooltip_row(j: JoinedRelease) -> str:
    bits: List[str] = [f"release: {j.release}"]
    ev = j.eval_entry
    if ev is not None:
        bits.append(f"eval: {ev.rel_label}")
    pe = j.profile_entry
    if pe is not None:
        bits.append(f"profile: {pe.path.name} · {pe.hostname} · {pe.dataset_id}")
    return " · ".join(bits)


def _render_quality_intro() -> None:
    """Main-column intro when the quality sidebar is not shown (error states only)."""
    st.markdown(_quality_intro_markdown())


def _render_performance_intro() -> None:
    """Main-column intro when the Performance sidebar is not shown (error states only)."""
    st.markdown(_performance_intro_markdown())


def _page_from_query() -> str:
    """Active page from ``?page=home|…|performance`` (default: Home)."""
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
                pop_disp = compact_run_display_names([x.rel_label for x in selected])
                for e in selected:
                    st.markdown(f"**{pop_disp.get(e.rel_label, e.rel_label)}**")
                    st.caption(e.rel_label)
                    st.json(artifact_status(e.path))
        else:
            st.caption("")


def _run_key(e: RunEntry) -> str:
    """Stable dict/session key: ``data/eval``-relative path (includes runs/ vs baselines/)."""
    return e.rel_label


def _sync_type_filter_state(type_options: List[str]) -> None:
    """Keep run-type multiselect aligned with category-filtered runs."""
    sig_key = "run_compare_type_sig"
    sf_key = "run_compare_type_filter"
    sig = tuple(type_options)
    if st.session_state.get(sig_key) == sig:
        return
    st.session_state[sig_key] = sig
    prev = st.session_state.get(sf_key)
    if isinstance(prev, list):
        kept = [x for x in prev if x in type_options]
        st.session_state[sf_key] = kept if kept else list(type_options)
    else:
        st.session_state[sf_key] = list(type_options)


def _sync_baseline_pick(
    selected_labels: Sequence[str],
    env_base: Optional[RunEntry],
) -> None:
    """Keep ``run_compare_quality_baseline`` inside the current run selection."""
    key = "run_compare_quality_baseline"
    valid = set(selected_labels)
    cur = st.session_state.get(key)
    if cur in valid:
        return
    if env_base:
        want = _run_key(env_base)
        if want in valid:
            st.session_state[key] = want
            return
    st.session_state[key] = selected_labels[0] if selected_labels else ""


def _render_delta_episodes_baseline_select(selected_labels: Sequence[str]) -> None:
    """Baseline run for Delta table and Episodes diffs only (not Home / KPIs charts)."""
    disp = compact_baseline_select_labels(
        list(selected_labels),
        max_chars=_MAIN_BASELINE_SELECT_MAX_CHARS,
    )
    st.selectbox(
        "Baseline (for deltas & episode diffs)",
        options=list(selected_labels),
        format_func=lambda r: disp.get(r, r),
        key="run_compare_quality_baseline",
        help=(
            "Reference run for the Delta table and unified diffs on Episodes. "
            "Home and KPIs charts always include every selected run."
        ),
    )


def _sync_run_multiselect_state(flabels: List[str]) -> None:
    """Keep run selection in sync with ``flabels``; default = all visible runs."""
    ms_key = "run_compare_multiselect"
    sig_key = "run_compare_flabels_sig"
    sig = tuple(flabels)
    if st.session_state.get(sig_key) == sig:
        return
    st.session_state[sig_key] = sig
    st.session_state.pop(_RUN_COMPARE_RUNS_EDITOR_KEY, None)
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
    legacy = {f"[{e.category}] {e.rel_label}": e for e in entries}
    return by_id.get(baseline_env) or by_rel.get(baseline_env) or legacy.get(baseline_env)


def _load_runs_data(
    selected: Sequence[RunEntry],
) -> Optional[Dict[str, Dict[str, Any]]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for e in selected:
        try:
            m = merge_run_summary(load_metrics(e.path), e.path)
            preds = load_predictions_jsonl(e.path / "predictions.jsonl")
            loaded[_run_key(e)] = {
                "entry": e,
                "metrics": m,
                "kpis": extract_kpis(m),
                "preds": preds,
            }
        except (OSError, ValueError, KeyError) as ex:
            st.error(f"Failed to load {_run_key(e)}: {ex}")
            return None
    return loaded


def _cache_key(selected_labels: Sequence[str]) -> str:
    """Cache loaded run payloads; baseline does not change artifact contents."""
    return "|".join(sorted(selected_labels))


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


def _kpi_table(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
    disp_map: Dict[str, str],
) -> pd.DataFrame:
    """Wide table: one row per run (avoids cramped metric tiles)."""
    rows: List[Dict[str, Any]] = []
    for e in selected:
        rk = _run_key(e)
        k = loaded[rk]["kpis"]
        rows.append(
            {
                "Run": disp_map.get(rk, rk),
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
    disp_map: Dict[str, str],
) -> None:
    delta_rows: List[Dict[str, Any]] = []
    for key, label in _METRIC_KEYS:
        bv = base_kpis.get(key)
        if bv is None:
            continue
        for e in candidates:
            rk = _run_key(e)
            ck = loaded[rk]["kpis"].get(key)
            if ck is None:
                continue
            delta = float(ck) - float(bv)
            good = delta_direction_good(key, delta)
            delta_rows.append(
                {
                    "metric": label,
                    "baseline": bv,
                    "candidate": disp_map.get(rk, rk),
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
    disp_map: Dict[str, str],
) -> None:
    """Aggregate ROUGE from metrics.json; per-episode ROUGE vs reference predictions."""
    eval_root = repo_root_from_here() / "data" / "eval"

    agg_rows: List[Dict[str, Any]] = []
    agg_ref_by_run: List[Tuple[str, str]] = []
    label_map = {
        "rouge1_f1": "ROUGE-1",
        "rouge2_f1": "ROUGE-2",
        "rougeL_f1": "ROUGE-L",
    }
    for e in selected:
        rk = _run_key(e)
        dname = disp_map.get(rk, rk)
        hit = extract_aggregate_rouge(loaded[rk]["metrics"])
        if not hit:
            continue
        ref_id, rouges = hit
        agg_ref_by_run.append((dname, ref_id))
        for k, mname in label_map.items():
            v = rouges.get(k)
            if v is not None:
                agg_rows.append({"run": dname, "metric": mname, "value": float(v)})

    n_sel = len(selected)
    runs_with_agg = {r["run"] for r in agg_rows}
    n_with_agg = len(runs_with_agg)
    _section_anchor("rc-h-rouge-agg")
    st.subheader("ROUGE (aggregate from metrics)")
    st.caption(
        f"Selected runs: **{n_sel}**. "
        f"With stored **vs_reference** mean F1 in **`metrics.json`**: **{n_with_agg}**"
        + (
            f"; **{n_sel - n_with_agg}** missing that block (see KPIs **ROUGE-L** or note below)."
            if n_sel > n_with_agg
            else "."
        )
    )
    if agg_ref_by_run and len({r[1] for r in agg_ref_by_run}) > 1:
        st.warning(
            "These runs use **different `vs_reference` ids** in `metrics.json`. "
            "Aggregate bars compare **different reference scorings** — treat as indicative only, "
            "or narrow the selection to one reference."
        )
    if agg_rows:
        dfa = pd.DataFrame(agg_rows)
        vmax = float(dfa["value"].max())
        y_top = max(0.05, 1.0, vmax * 1.12)
        present_agg = set(dfa["run"].unique())
        run_order_agg: List[str] = []
        seen_agg: Set[str] = set()
        for e in selected:
            dname = disp_map.get(_run_key(e), _run_key(e))
            if dname in present_agg and dname not in seen_agg:
                run_order_agg.append(dname)
                seen_agg.add(dname)
        for r in sorted(present_agg):
            if r not in seen_agg:
                run_order_agg.append(r)
                seen_agg.add(r)

        st.caption(
            "Run-level **mean** F1 from `metrics.json` → `vs_reference` (what scoring wrote). "
            "Grouped bars on one **shared y-scale**; values on bars; legend = run. "
            "The **baseline** control on **Delta** / **Episodes** does **not** change these bars — "
            "each was scored against that run’s stored reference at eval time."
        )
        bar_line = dict(width=0.5, color="#333333")
        fig_agg = px.bar(
            dfa,
            x="metric",
            y="value",
            color="run",
            barmode="group",
            category_orders={"metric": _ROUGE_METRIC_ORDER, "run": run_order_agg},
            text="value",
        )
        fig_agg.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            cliponaxis=False,
            marker=dict(line=bar_line),
        )
        _apply_tufte_base(fig_agg)
        fig_agg.update_layout(
            showlegend=True,
            barmode="group",
            height=420,
            margin=dict(t=20, b=100, l=48, r=20),
            xaxis_title="",
            yaxis_title="Mean F1",
            legend=dict(
                title="",
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
            ),
        )
        fig_agg.update_yaxes(
            range=[0, y_top],
            showgrid=True,
            gridcolor="#e8e8e8",
            gridwidth=0.5,
            zeroline=True,
            zerolinewidth=0.5,
            zerolinecolor="#cccccc",
        )
        fig_agg.update_xaxes(showgrid=False)
        st.plotly_chart(fig_agg, width="stretch")
    else:
        st.info(
            "No selected run has aggregate ROUGE in **`metrics.json`**. "
            "You need **`vs_reference`**: a dict keyed by **reference id**, with at least one of "
            "**`rouge1_f1`**, **`rouge2_f1`**, **`rougeL_f1`** set (and no `error` on that block). "
            "That appears when the eval is scored **against a reference**; runs without "
            "`--reference` / `REFERENCE_IDS` often have `vs_reference: null`.\n\n"
            "**Per-episode ROUGE** below can still work from `predictions.jsonl` + "
            "`references/<id>/predictions.jsonl` when this block is missing.\n\n"
            f"Details: [Experiment guide — vs_reference]({_DOCS_BASE}/docs/guides/"
            "EXPERIMENT_GUIDE.md#vs_reference-metrics)."
        )

    ref_id, consistent = pick_shared_reference_id(
        loaded,
        ordered_labels=[_run_key(e) for e in selected],
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
    _section_anchor("rc-h-rouge-per")
    peri_rows: List[Dict[str, Any]] = []
    for e in selected:
        rk = _run_key(e)
        dname = disp_map.get(rk, rk)
        k = loaded[rk]["kpis"]
        preds = loaded[rk]["preds"]
        failed_ids = k.get("failed_episodes") or []
        peri_rows.extend(compute_per_episode_rouge_rows(dname, preds, failed_ids, ref_by_id))

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
    disp_map: Dict[str, str],
) -> None:
    st.markdown(_PLOTLY_CHART_LEFT_CSS, unsafe_allow_html=True)
    st.caption(
        "These charts use **every run** in your sidebar selection. "
        "**Baseline** is chosen on **Delta** and **Episodes** only — it does not change this page."
    )
    chart_rows: List[Dict[str, Any]] = []
    for e in selected:
        rk = _run_key(e)
        dname = disp_map.get(rk, rk)
        k = loaded[rk]["kpis"]
        preds = loaded[rk]["preds"]
        failed_ids = k.get("failed_episodes") or []
        chart_rows.extend(predictions_to_chart_rows(dname, preds, failed_ids))
    df_chart = pd.DataFrame(chart_rows)
    if df_chart.empty:
        st.warning(
            "No ROUGE-comparable episodes for token/latency charts (need non-empty summaries, "
            "no inference error, not in gate failures)."
        )
    else:
        _section_anchor("rc-h-tokens")
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

        _section_anchor("rc-h-latency")
        st.subheader("Latency vs output length")
        n_comp = int(len(df_chart))
        sub = df_chart.dropna(subset=["latency_s"])
        n_lat = int(len(sub))
        st.caption(
            f"ROUGE-comparable episodes here: **{n_comp}**. "
            f"With per-episode latency in **`predictions.jsonl`** "
            f"(`metadata.processing_time_seconds`): **{n_lat}**."
        )
        if sub.empty:
            st.info(
                "Each point would be one **episode**: **estimated output tokens** (x) vs "
                "**latency (s)** (y). "
                "There is no row-level timing in your artifacts yet, so this plot is empty "
                "while the token box plot above can still show data. "
                "Aggregate latency in **`metrics.json`** does not populate this chart.\n\n"
                f"See [Experiment guide — predictions]({_DOCS_BASE}/docs/guides/"
                "EXPERIMENT_GUIDE.md#understanding-predictions) and emit "
                "`metadata.processing_time_seconds` from your eval pipeline "
                "(e.g. `scripts/eval/run_experiment.py`)."
            )
        else:
            present = set(sub["run"].unique())
            run_order: List[str] = []
            seen_order: Set[str] = set()
            for e in selected:
                dname = disp_map.get(_run_key(e), _run_key(e))
                if dname in present and dname not in seen_order:
                    run_order.append(dname)
                    seen_order.add(dname)
            for r in sorted(present):
                if r not in seen_order:
                    run_order.append(r)
                    seen_order.add(r)

            st.caption(
                "**Read:** longer outputs (right) vs time (up). One chart, **same scales**, "
                "color = run; fixed marker size (no bubble area distortion)."
            )
            grid_kw = dict(
                showgrid=True,
                gridcolor="#e8e8e8",
                gridwidth=0.5,
                zeroline=True,
                zerolinewidth=0.5,
                zerolinecolor="#dddddd",
            )
            fig_o = px.scatter(
                sub,
                x="output_tokens_est",
                y="latency_s",
                color="run",
                category_orders={"run": run_order},
                hover_data=["episode_id"],
            )
            fig_o.update_traces(
                marker=dict(
                    size=8,
                    opacity=0.58,
                    line=dict(width=0.45, color="rgba(0,0,0,0.18)"),
                ),
                hovertemplate=(
                    "Tokens: %{x:.0f}<br>"
                    "Latency: %{y:.2f} s<br>"
                    "Episode: %{customdata[0]}<extra></extra>"
                ),
            )
            _apply_tufte_base(fig_o)
            fig_o.update_layout(
                showlegend=True,
                height=440,
                margin=dict(t=16, b=96, l=56, r=20),
                xaxis_title="Estimated output tokens",
                yaxis_title="Latency (s)",
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig_o.update_xaxes(**grid_kw)
            fig_o.update_yaxes(**grid_kw)
            st.plotly_chart(fig_o, width="stretch")

    _render_rouge_charts(selected, loaded, disp_map)


def _render_chart_c(selected: Sequence[RunEntry], disp_map: Dict[str, str]) -> None:
    any_diag = any((e.path / "diagnostics.jsonl").is_file() for e in selected)
    if not any_diag:
        st.caption("Map/reduce diagnostics: no `diagnostics.jsonl` in selected runs.")
        return
    _section_anchor("rc-h-diag")
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
        rk = _run_key(e)
        fig_c.update_layout(barmode="group", title=disp_map.get(rk, rk))
        _legend_below_chart(fig_c, bottom_margin=100)
        st.plotly_chart(fig_c, width="stretch")


def _render_episode_drilldown(
    selected: Sequence[RunEntry],
    loaded: Dict[str, Dict[str, Any]],
    bl_key: str,
    disp_map: Dict[str, str],
) -> None:
    comp_sets: List[set] = []
    for e in selected:
        rk = _run_key(e)
        preds = loaded[rk]["preds"]
        failed = loaded[rk]["kpis"].get("failed_episodes") or []
        comp_sets.append(rouge_comparable_episode_ids(preds, failed))
    episode_ids = sorted(set.intersection(*comp_sets)) if comp_sets else []
    if not episode_ids:
        st.warning(
            "No episode is ROUGE-comparable across all selected runs "
            "(non-empty summary, no error, not in gate failures). "
            "Expand selection or fix failed runs."
        )
        return
    _page_jump_nav(
        [
            ("rc-e-browse", "Browse"),
            ("rc-e-summaries", "Summaries"),
            ("rc-e-diffs", "Diffs"),
        ]
    )
    _section_anchor("rc-e-browse")
    st.caption(
        "Intersection of ROUGE-comparable episodes across runs (same idea as aggregate ROUGE)."
    )
    q_col, page_col, ep_col = st.columns([5, 2, 5], gap="small")
    with q_col:
        q = st.text_input(
            "Filter",
            "",
            key="rc_ep_id_filter",
            help="Keep episodes whose id contains this substring (empty = all).",
            placeholder="episode id substring",
        )
    if q:
        episode_ids = [eid for eid in episode_ids if q in eid]
    total = len(episode_ids)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    with page_col:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            value=1,
            step=1,
            key="rc_ep_page",
        )
    start = (page - 1) * PAGE_SIZE
    slice_ids = episode_ids[start : start + PAGE_SIZE]
    if not slice_ids:
        st.warning("No episodes match the filter.")
        return
    with ep_col:
        ep = st.selectbox(
            "Episode",
            options=slice_ids,
            index=0,
            key="rc_ep_pick",
        )
    st.caption(f"**{len(slice_ids)}** on this page of **{total}** · page **{page}/{n_pages}**")

    _section_anchor("rc-e-summaries")
    st.markdown(
        "**Summaries** — same episode in each column/tab; **Unified diffs** (below) vs baseline."
    )
    side = {_run_key(e): index_predictions(loaded[_run_key(e)]["preds"]) for e in selected}
    texts: List[Tuple[str, str, Optional[str]]] = []
    for e in selected:
        rk = _run_key(e)
        pred = side[rk].get(ep)
        if not pred:
            texts.append((rk, "", f"Missing episode {ep}"))
            continue
        txt, err = get_summary_text(pred)
        texts.append((rk, txt, err))

    def _one_summary_block(i: int, rk: str, txt: str, err: Optional[str]) -> None:
        st.markdown(f"**{disp_map.get(rk, rk)}**")
        if err:
            st.error(err)
            return
        st.text_area(
            "Generated summary (read-only)",
            value=txt,
            height=min(300, max(120, 32 + 12 * (1 + txt.count("\n")))),
            key=_episode_text_area_key(i, ep),
            label_visibility="collapsed",
            disabled=True,
        )

    n_runs = len(selected)
    if n_runs <= 3:
        c_ep = st.columns(n_runs)
        for i, (col, (rk, txt, err)) in enumerate(zip(c_ep, texts)):
            with col:
                _one_summary_block(i, rk, txt, err)
    else:
        tabs = st.tabs([f"{i + 1}. {disp_map.get(rk, rk)}" for i, (rk, _, _) in enumerate(texts)])
        for i, (tab, (rk, txt, err)) in enumerate(zip(tabs, texts)):
            with tab:
                _one_summary_block(i, rk, txt, err)

    base_txt = next((t for rk, t, er in texts if rk == bl_key and not er), "")
    if not base_txt:
        return
    _section_anchor("rc-e-diffs")
    st.markdown(
        f"**Unified diffs vs baseline** (`{disp_map.get(bl_key, bl_key)}`) — one expander per "
        "other run."
    )
    for rk, cand_txt, err in texts:
        if rk == bl_key or err or not cand_txt:
            continue
        cand_disp = disp_map.get(rk, rk)
        diff_lines = list(
            unified_diff(
                base_txt.splitlines(),
                cand_txt.splitlines(),
                fromfile="baseline",
                tofile=cand_disp,
                lineterm="",
            )
        )
        if diff_lines:
            with st.expander(f"Unified diff vs {cand_disp}"):
                st.code("\n".join(diff_lines), language="diff")


def _load_kpis_optional(entry: RunEntry) -> Optional[Dict[str, Any]]:
    """Load KPI dict for an eval run, or None if artifacts are missing or invalid."""
    try:
        m = merge_run_summary(load_metrics(entry.path), entry.path)
        return extract_kpis(m)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
        return None


def _sync_perf_release_state(options: List[str]) -> None:
    """Keep performance release multiselect aligned with filter options."""
    sig_key = "perf_release_options_sig"
    ms_key = "perf_release_multiselect"
    sig = tuple(options)
    if st.session_state.get(sig_key) == sig:
        return
    st.session_state[sig_key] = sig
    prev = st.session_state.get(ms_key)
    if isinstance(prev, list):
        kept = [x for x in prev if x in options]
        st.session_state[ms_key] = kept if kept else list(options)
    else:
        st.session_state[ms_key] = list(options)


def _performance_sidebar(
    joined: List[JoinedRelease], profiles_all: List[ProfileEntry]
) -> Tuple[List[str], str, List[JoinedRelease]]:
    """Sidebar widgets; returns selected release keys, baseline label, filtered join rows."""
    selected_releases: List[str] = []
    baseline_release = ""
    filtered: List[JoinedRelease] = []

    with st.sidebar:
        _render_performance_sidebar_title_and_intro()
        st.header("Performance filters")
        host_opts = sorted({p.hostname for p in profiles_all if p.hostname})
        ds_opts = sorted({p.dataset_id for p in profiles_all if p.dataset_id})
        tip_m: Dict[str, str] = {}
        if host_opts:
            for h in host_opts:
                tip_m[h] = f"Profiles with environment.hostname = {h!r}."
            hf = st.multiselect(
                "Hostname",
                options=host_opts,
                default=host_opts,
                key="perf_hostname_ms",
                help="Restrict to profiles captured on these hosts. Hover an option for details.",
            )
            host_filter = list(hf) if set(hf) != set(host_opts) else None
        else:
            host_filter = None
        if ds_opts:
            for d in ds_opts:
                tip_m[d] = f"Profiles with dataset_id = {d!r}."
            dsf = st.multiselect(
                "Dataset",
                options=ds_opts,
                default=ds_opts,
                key="perf_dataset_ms",
                help="Restrict to these profile dataset_id values. Hover an option for details.",
            )
            ds_filter = list(dsf) if set(dsf) != set(ds_opts) else None
        else:
            ds_filter = None

        filtered = filter_joined_releases(joined, hostnames=host_filter, datasets=ds_filter)
        rel_options = [j.release for j in filtered]
        if not rel_options:
            st.error("No releases left after hostname/dataset filters.")
        else:
            for j in filtered:
                tip_m[j.release] = _joined_release_tooltip_row(j)
            _sync_perf_release_state(rel_options)
            selected_releases = st.multiselect(
                "Releases",
                options=rel_options,
                key="perf_release_multiselect",
                help="Join keys (release). Hover an option for eval path and profile file.",
            )
            prof_for_baseline = [
                j
                for j in filtered
                if j.release in selected_releases and j.profile_entry is not None
            ]
            if prof_for_baseline:
                opts = [j.release for j in prof_for_baseline]
                perf_baseline_disp = compact_baseline_select_labels(
                    opts, max_chars=_MAIN_BASELINE_SELECT_MAX_CHARS
                )
                baseline_release = st.selectbox(
                    "Baseline (resource deltas)",
                    options=opts,
                    format_func=lambda r: perf_baseline_disp.get(r, r),
                    index=0,
                    key="perf_baseline_release",
                    help="Hover an open-list option for full release / eval / profile details.",
                )
                for full_key, short in perf_baseline_disp.items():
                    if full_key in tip_m:
                        tip_m[short] = tip_m[full_key]
        _inject_sidebar_option_tooltips(tip_m)

    return selected_releases, baseline_release, filtered


def _lerp_rgb_green_amber_red(t: float) -> Tuple[int, int, int]:
    """Map ``t`` in ``[0, 1]`` to RGB: 0 = green (good), 1 = red (worse), amber mid."""
    t = max(0.0, min(1.0, t))
    green = (27, 118, 45)
    amber = (230, 180, 0)
    red = (183, 28, 28)
    if t < 0.5:
        u = t * 2.0
        return (
            int(green[0] * (1.0 - u) + amber[0] * u),
            int(green[1] * (1.0 - u) + amber[1] * u),
            int(green[2] * (1.0 - u) + amber[2] * u),
        )
    u = (t - 0.5) * 2.0
    return (
        int(amber[0] * (1.0 - u) + red[0] * u),
        int(amber[1] * (1.0 - u) + red[1] * u),
        int(amber[2] * (1.0 - u) + red[2] * u),
    )


def _bar_colors_lower_is_better(values: Sequence[float]) -> List[str]:
    """Per-bar hex: min value = green, max = red; flat series uses neutral gray."""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        return ["#6b6b6b"] * len(values)
    out: List[str] = []
    for v in values:
        t = (float(v) - vmin) / (vmax - vmin)
        r, g, b = _lerp_rgb_green_amber_red(t)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def _render_perf_env_warnings(prof_selected: List[JoinedRelease]) -> None:
    hosts = {j.profile_entry.hostname for j in prof_selected if j.profile_entry}
    if len(hosts) > 1:
        st.warning(
            "Selected profiles come from **different machines** — treat absolute RSS/CPU "
            "comparisons as indicative only. Prefer filtering to one hostname in the sidebar."
        )
    dsets = {j.profile_entry.dataset_id for j in prof_selected if j.profile_entry}
    if len(dsets) > 1:
        st.warning(
            "Selected profiles use **different dataset_id** values — wall times are not "
            "directly comparable across datasets."
        )


def _render_perf_kpis(prof_selected: List[JoinedRelease]) -> None:
    """Tufte-style opening view: one figure, small multiples, no metric tiles (see TUFTE guide)."""
    _section_anchor("rc-p-kpis")
    st.subheader("Resource KPIs (frozen profiles)")
    st.caption(
        "Same releases in all three panels (ordered by profile time). Bars start at zero. "
        "**Color** (per panel): greener = lower peak RSS; greener = less total wall time; "
        "on **Episodes**, greener = lower wall time **per episode** (throughput), bar **height** "
        "is still episode count. Hover for host and dataset_id."
    )
    ordered = sorted(
        prof_selected,
        key=lambda j: (j.profile_entry.sort_ts if j.profile_entry else 0.0, j.release),
    )
    releases = [j.release for j in ordered]
    rss: List[float] = []
    wall: List[float] = []
    eps: List[int] = []
    hosts: List[str] = []
    dsets: List[str] = []
    for j in ordered:
        p = j.profile_entry
        assert p is not None
        t = p.totals
        rss.append(float(t.get("peak_rss_mb") or 0))
        wall.append(float(t.get("wall_time_s") or 0))
        eps.append(int(p.episodes_processed))
        hosts.append(p.hostname or "—")
        dsets.append(p.dataset_id or "—")
    customdata = list(zip(hosts, dsets))
    colors_rss = _bar_colors_lower_is_better(rss)
    colors_wall = _bar_colors_lower_is_better(wall)
    per_ep_wall = [wall[i] / float(eps[i]) if eps[i] else float(wall[i]) for i in range(len(eps))]
    colors_eps = _bar_colors_lower_is_better(per_ep_wall)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_titles=("Peak RSS (MB)", "Total wall (s)", "Episodes processed"),
    )
    fig.add_trace(
        go.Bar(
            x=releases,
            y=rss,
            marker_color=colors_rss,
            showlegend=False,
            hovertemplate="%{x}<br>%{y:.0f} MB<br>host: %{customdata[0]}<br>"
            "dataset: %{customdata[1]}<extra></extra>",
            customdata=customdata,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=releases,
            y=wall,
            marker_color=colors_wall,
            showlegend=False,
            hovertemplate="%{x}<br>%{y:.1f} s<br>host: %{customdata[0]}<br>"
            "dataset: %{customdata[1]}<extra></extra>",
            customdata=customdata,
        ),
        row=2,
        col=1,
    )
    eps_hover_cd = list(zip(hosts, dsets, per_ep_wall))
    fig.add_trace(
        go.Bar(
            x=releases,
            y=eps,
            marker_color=colors_eps,
            showlegend=False,
            hovertemplate="%{x}<br>%{y} episodes<br>%{customdata[2]:.2f} s/episode<br>"
            "host: %{customdata[0]}<br>dataset: %{customdata[1]}<extra></extra>",
            customdata=eps_hover_cd,
        ),
        row=3,
        col=1,
    )
    _apply_tufte_base(fig)
    fig.update_layout(
        height=min(520, 160 + 55 * len(releases)),
        bargap=0.28,
        showlegend=False,
        margin=dict(t=36, b=48),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=0.5,
        zeroline=True,
        zerolinewidth=0.5,
        zerolinecolor="#dddddd",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickangle=-32)
    st.plotly_chart(fig, width="stretch")


def _render_perf_delta(prof_selected: List[JoinedRelease], baseline_release: str) -> None:
    _section_anchor("rc-p-delta")
    st.subheader("Resource delta vs baseline")
    base_prof = next(
        (j.profile_entry for j in prof_selected if j.release == baseline_release),
        None,
    )
    cand_profs = [
        j.profile_entry
        for j in prof_selected
        if j.profile_entry is not None and j.release != baseline_release
    ]
    if base_prof is None:
        st.caption("Pick a baseline release that has a profile.")
        return
    if not cand_profs:
        st.caption("Select at least two releases with profiles to see deltas.")
        return
    st.caption(f"Baseline profile: **{baseline_release}**")
    drows = profile_stage_delta_rows(base_prof, cand_profs)
    if not drows:
        st.caption("No overlapping stage metrics for delta.")
        return
    pdf = pd.DataFrame(drows)
    display_cols = [
        "stage",
        "metric",
        "baseline",
        "candidate_release",
        "value",
        "delta",
        "delta_pct",
        "good",
    ]
    pdf = pdf[[c for c in display_cols if c in pdf.columns]]
    st.dataframe(pdf, width="stretch", hide_index=True)


def _render_perf_trends(prof_selected: List[JoinedRelease]) -> None:
    _section_anchor("rc-p-trends")
    st.subheader("Trends (wall time & peak RSS per stage)")
    st.caption(
        "Releases ordered by profile date. Gaps in lines mean a stage was absent in that "
        "profile."
    )
    profiles_for_trend = [j.profile_entry for j in prof_selected if j.profile_entry is not None]
    trend = profile_trend_long_rows(profiles_for_trend)
    if not trend:
        st.caption("No per-stage rows to chart.")
        return
    dft = pd.DataFrame(trend)
    rel_order = [
        p.release for p in sorted(profiles_for_trend, key=lambda x: (x.sort_ts, x.release))
    ]
    tw = dft.dropna(subset=["wall_time_s"])
    if not tw.empty:
        fig_w = px.line(
            tw,
            x="release",
            y="wall_time_s",
            color="stage",
            markers=True,
            category_orders={"release": rel_order},
        )
        _apply_tufte_base(fig_w)
        fig_w.update_layout(xaxis_title="Release", yaxis_title="Wall time (s)")
        st.plotly_chart(fig_w, width="stretch")
    tr = dft.dropna(subset=["peak_rss_mb"])
    if not tr.empty:
        fig_r = px.line(
            tr,
            x="release",
            y="peak_rss_mb",
            color="stage",
            markers=True,
            category_orders={"release": rel_order},
        )
        _apply_tufte_base(fig_r)
        fig_r.update_layout(xaxis_title="Release", yaxis_title="Peak RSS (MB)")
        st.plotly_chart(fig_r, width="stretch")


def _scatter_resource_value(totals: Dict[str, Any], cy: str) -> float:
    if cy == "total_wall_time_s":
        return float(totals.get("wall_time_s") or 0)
    if cy == "total_peak_rss_mb":
        return float(totals.get("peak_rss_mb") or 0)
    return float(totals.get("avg_wall_time_per_episode_s") or 0)


def _render_perf_scatter(prof_selected: List[JoinedRelease]) -> None:
    _section_anchor("rc-p-scatter")
    st.subheader("Quality vs resource cost")
    st.caption("Dots need **both** an eval run and a frozen profile for the same release key.")
    qx = st.selectbox(
        "Quality (x)",
        options=["rougeL_f1", "success_rate", "avg_latency_s"],
        format_func=lambda k: {
            "rougeL_f1": "ROUGE-L F1",
            "success_rate": "Success rate",
            "avg_latency_s": "Avg latency (s)",
        }[k],
        key="perf_scatter_qx",
    )
    cy = st.selectbox(
        "Resource (y)",
        options=["total_wall_time_s", "total_peak_rss_mb", "avg_wall_per_episode_s"],
        format_func=lambda k: {
            "total_wall_time_s": "Total wall time (s)",
            "total_peak_rss_mb": "Total peak RSS (MB)",
            "avg_wall_per_episode_s": "Avg wall time / episode (s)",
        }[k],
        key="perf_scatter_cy",
    )
    scatter_rows: List[Dict[str, Any]] = []
    for j in prof_selected:
        if j.eval_entry is None or j.profile_entry is None:
            continue
        kpis = _load_kpis_optional(j.eval_entry)
        if kpis is None:
            continue
        qv = kpis.get(qx)
        if qv is None:
            continue
        prof = j.profile_entry
        scatter_rows.append(
            {
                "release": j.release,
                "quality": float(qv),
                "resource": _scatter_resource_value(prof.totals, cy),
            }
        )
    if not scatter_rows:
        st.caption(
            "No releases with both eval metrics and a profile (or missing selected quality field)."
        )
        return
    dfs = pd.DataFrame(scatter_rows)
    fig_s = px.scatter(
        dfs,
        x="quality",
        y="resource",
        text="release",
        hover_data=["release"],
    )
    _apply_tufte_base(fig_s)
    fig_s.update_traces(textposition="top center")
    fig_s.update_layout(
        xaxis_title=qx.replace("_", " "),
        yaxis_title=cy.replace("_", " "),
        showlegend=False,
    )
    st.plotly_chart(fig_s, width="stretch")


def _render_perf_coverage(filtered: List[JoinedRelease], selected_releases: List[str]) -> None:
    _section_anchor("rc-p-coverage")
    st.subheader("Coverage")
    cov_rows: List[Dict[str, Any]] = []
    sel = set(selected_releases)
    for j in filtered:
        if j.release not in sel:
            continue
        cov_rows.append(
            {
                "release": j.release,
                "eval": "yes" if j.eval_entry else "no",
                "profile": "yes" if j.profile_entry else "no",
            }
        )
    if cov_rows:
        st.dataframe(pd.DataFrame(cov_rows), width="stretch", hide_index=True)


def _render_performance_main(
    entries: List[RunEntry],
    profiles_all: List[ProfileEntry],
) -> None:
    """RFC-066: frozen profiles + optional join to eval quality metrics."""
    st.markdown(_NAV_CSS, unsafe_allow_html=True)
    joined, join_warns = join_releases(entries, profiles_all)
    selected_releases, baseline_release, filtered = _performance_sidebar(joined, profiles_all)

    _render_nav_row("Performance", None)

    if join_warns:
        with st.expander("Join warnings (duplicate keys)", expanded=False):
            for w in join_warns:
                st.caption(w)

    if not profiles_all:
        st.warning(
            "No frozen profiles found under **data/profiles/*.yaml**. "
            "See RFC-064 and `make profile-freeze`."
        )
        return

    if not selected_releases:
        st.info("Select at least one release in the sidebar (Performance filters).")
        return

    prof_selected = [
        j for j in filtered if j.release in selected_releases and j.profile_entry is not None
    ]
    if not prof_selected:
        st.info("None of the selected releases have a frozen profile YAML.")
        return

    _page_jump_nav(
        [
            ("rc-p-kpis", "Resource KPIs"),
            ("rc-p-delta", "Resource delta"),
            ("rc-p-trends", "Trends"),
            ("rc-p-scatter", "Quality vs cost"),
            ("rc-p-coverage", "Coverage"),
        ]
    )
    _render_perf_env_warnings(prof_selected)
    _render_perf_kpis(prof_selected)
    _render_perf_delta(prof_selected, baseline_release)
    _render_perf_trends(prof_selected)
    _render_perf_scatter(prof_selected)
    _render_perf_coverage(filtered, selected_releases)


def main() -> None:
    st.set_page_config(page_title="Run compare", layout="wide", initial_sidebar_state="expanded")

    entries = discover_runs()
    profiles_all = discover_profiles()
    page = _page_from_query()

    if not entries and not profiles_all:
        st.markdown(_NAV_CSS, unsafe_allow_html=True)
        _render_nav_row(page, None)
        st.title("Eval run comparison")
        _render_quality_intro()
        st.warning(
            "No eval runs under data/eval and no frozen profiles under data/profiles/*.yaml."
        )
        return

    if page == "Performance":
        _render_performance_main(entries, profiles_all)
        return

    if not entries:
        st.markdown(_NAV_CSS, unsafe_allow_html=True)
        _render_nav_row(page, None)
        st.title("Eval run comparison")
        _render_quality_intro()
        st.warning(
            "No runs found under data/eval (runs/, baselines/, references/). "
            "Open **?page=performance** if you only have frozen profiles."
        )
        return

    rel_to_entry = {e.rel_label: e for e in entries}
    env_base = _parse_env_baseline(entries)

    # One injection before sidebar so main column top padding applies before content.
    st.markdown(_NAV_CSS, unsafe_allow_html=True)

    with st.sidebar:
        _render_quality_sidebar_title_and_intro()
        st.header("Filters")
        cat_filter = st.multiselect(
            "Categories",
            options=["run", "baseline", "reference"],
            default=["run", "baseline", "reference"],
            format_func=lambda c: _CATEGORY_LABELS.get(c, c),
            help="Hover an option for which subtree under data/eval it includes.",
        )
        filtered = [e for e in entries if e.category in cat_filter]
        present_types = {infer_run_type_bucket(e.rel_label) for e in filtered}
        type_options = [b for b in RUN_TYPE_ORDER if b in present_types]
        if type_options:
            _sync_type_filter_state(type_options)
            type_pick = st.multiselect(
                "Type",
                options=type_options,
                format_func=lambda b: RUN_TYPE_LABELS.get(b, b),
                key="run_compare_type_filter",
                help=(
                    "Limit the run list using keywords in the path (e.g. *_paragraph_* vs "
                    "*_bullets_*). Compare within one type for meaningful ROUGE/length deltas."
                ),
            )
            active_types = type_pick if type_pick else list(type_options)
        else:
            active_types = []
        filtered = [
            e
            for e in filtered
            if not type_options or infer_run_type_bucket(e.rel_label) in active_types
        ]
        flabels = [e.rel_label for e in filtered]
        sidebar_disp = compact_run_display_names(flabels, max_chars=_SIDEBAR_RUN_LABEL_MAX_CHARS)
        if not flabels:
            st.error("No runs match the current filters (categories and type).")
            selected_labels = []
        else:
            _sync_run_multiselect_state(flabels)
            ms_key = "run_compare_multiselect"
            st.markdown("**Runs to compare** (pick at least **2**)")
            st.caption(
                "Use the checkboxes — **Run** shows the full **data/eval** path (not the short "
                "chart labels)."
            )
            b1, b2 = st.columns(2, gap="small")
            with b1:
                if st.button(
                    "Select all",
                    key="run_compare_select_all",
                    use_container_width=True,
                    help="Select every run in the list.",
                ):
                    st.session_state[ms_key] = list(flabels)
                    st.session_state.pop(_RUN_COMPARE_RUNS_EDITOR_KEY, None)
                    st.rerun()
            with b2:
                if st.button(
                    "Deselect all",
                    key="run_compare_deselect_all",
                    use_container_width=True,
                    help="Clear the run selection.",
                ):
                    st.session_state[ms_key] = []
                    st.session_state.pop(_RUN_COMPARE_RUNS_EDITOR_KEY, None)
                    st.rerun()
            cur_sel = st.session_state.get(ms_key)
            if not isinstance(cur_sel, list):
                cur_sel = list(flabels)
            sel_set = {x for x in cur_sel if x in flabels}
            df_runs = pd.DataFrame(
                {
                    "Compare": [lab in sel_set for lab in flabels],
                    "Run": flabels,
                }
            )
            edited = st.data_editor(
                df_runs,
                column_config={
                    "Compare": st.column_config.CheckboxColumn(
                        "",
                        width=44,
                        help="Include this run in charts and tables.",
                    ),
                    "Run": st.column_config.TextColumn(
                        "Run (data/eval path)",
                        help="Full path under data/eval.",
                    ),
                },
                disabled=["Run"],
                hide_index=True,
                key=_RUN_COMPARE_RUNS_EDITOR_KEY,
                num_rows="fixed",
                height=min(420, 72 + 36 * len(flabels)),
            )
            if edited is not None and len(edited.index) == len(flabels):
                flags = edited["Compare"].fillna(False)
                selected_labels = [flabels[i] for i in range(len(flabels)) if bool(flags.iloc[i])]
                st.session_state[ms_key] = selected_labels
            else:
                selected_labels = [x for x in cur_sel if x in flabels]
                st.session_state[ms_key] = selected_labels
        tip_map: Dict[str, str] = {
            _CATEGORY_LABELS[k]: v for k, v in _CATEGORY_OPTION_TOOLTIPS.items()
        }
        tip_map.update(invert_compact_display_map(sidebar_disp))
        _inject_sidebar_option_tooltips(tip_map)

    if not flabels:
        _render_nav_row(page, None)
        st.error("No runs match the current filters (categories and type). See sidebar.")
        return

    if len(selected_labels) < 2:
        _render_nav_row(page, None)
        st.info("Select at least two runs in the sidebar (left).")
        return

    selected = [rel_to_entry[x] for x in selected_labels if x in rel_to_entry]
    _sync_baseline_pick(selected_labels, env_base)
    chart_disp = compact_run_display_names([e.rel_label for e in selected])

    ck = _cache_key(selected_labels)
    if st.session_state.get("run_compare_cache_key") != ck or "loaded" not in st.session_state:
        loaded = _load_runs_data(selected)
        if loaded is None:
            _render_nav_row(page, selected)
            return
        st.session_state["loaded"] = loaded
        st.session_state["run_compare_cache_key"] = ck
    else:
        loaded = st.session_state["loaded"]

    _render_nav_row(page, selected)
    type_buckets = {infer_run_type_bucket(x) for x in selected_labels}
    if len(type_buckets) > 1:
        st.warning(
            "Selected runs use **mixed types** (paragraph vs bullets vs other). "
            "ROUGE and length metrics are **not directly comparable** across types — pick "
            "one **Type** in the sidebar, or trim the run list."
        )

    if page == "Home":
        _page_jump_nav(_home_jump_nav_items(selected, loaded, chart_disp))
        _render_charts_a_b(selected, loaded, chart_disp)
        _render_chart_c(selected, chart_disp)
    elif page == "KPIs":
        st.subheader("KPI summary")
        st.caption("One row per run — scroll horizontally if needed.")
        st.dataframe(
            _kpi_table(selected, loaded, chart_disp),
            width="stretch",
            hide_index=True,
        )
    elif page == "Delta":
        _render_delta_episodes_baseline_select(selected_labels)
        bl_key = str(st.session_state["run_compare_quality_baseline"])
        base_kpis = loaded[bl_key]["kpis"]
        candidates = [e for e in selected if e.rel_label != bl_key]
        st.subheader("Delta vs baseline")
        st.caption(
            f"Showing deltas vs **{chart_disp.get(bl_key, bl_key)}** — other selected runs are "
            "candidates."
        )
        _render_delta_table(base_kpis, candidates, loaded, chart_disp)
    else:
        _render_delta_episodes_baseline_select(selected_labels)
        bl_key = str(st.session_state["run_compare_quality_baseline"])
        st.subheader("Episode drill-down")
        st.caption(
            f"Unified diffs use **{chart_disp.get(bl_key, bl_key)}** as the left-hand baseline."
        )
        _render_episode_drilldown(selected, loaded, bl_key, chart_disp)


if __name__ == "__main__":
    main()
