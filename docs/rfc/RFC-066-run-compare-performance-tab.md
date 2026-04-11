# RFC-066: Run Comparison Tool — Performance Tab

## Status

**Draft** (split from RFC-064)

## RFC Number

066

## Authors

Podcast Scraper Team

## Date

2026-04-09 (stub) · 2026-04-10 (expanded)

## Related RFCs

- `docs/rfc/RFC-064-performance-profiling-release-freeze.md` — Frozen profiles
  and release freeze framework (provides the data)
- `docs/rfc/RFC-065-live-pipeline-monitor.md` — Sibling RFC; live monitoring
  dashboard (also split from RFC-064)
- `docs/rfc/RFC-047-run-comparison-visual-tool.md` — Existing Streamlit
  comparison tool (extended by this RFC)
- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` — Quality
  benchmarking framework (quality side of the comparison)

## Related Issues

- [#511](https://github.com/chipi/podcast_scraper/issues/511) — Tracking issue
- [#510](https://github.com/chipi/podcast_scraper/issues/510) — RFC-064 epic
  (frozen profiles prerequisite)

---

## Abstract

Extend the existing RFC-047 Streamlit run comparison tool
(`tools/run_compare/app.py`) with a **Performance** navigation page that
displays resource cost metrics from frozen profiles (RFC-064) alongside the
existing quality comparison views.

Where RFC-047 answers *"did quality change between runs?"*, this extension adds
*"did resource cost change?"* — joining quality evaluation data (`data/eval/`)
and performance profiles (`data/profiles/`) by release tag into a single visual
tool.

Split from RFC-064 to keep the release freeze framework focused on its core
deliverable (frozen profiles + diff tool) and to avoid mixing Streamlit UI
design with profiling infrastructure.

---

## Motivation

After RFC-064 shipped the `freeze_profile.py` script and `diff_profiles.py`
terminal diff, we have frozen YAML profiles accumulating in `data/profiles/`.
The terminal diff is useful for quick two-profile comparisons, but it cannot:

- Show trends across many releases at a glance.
- Overlay quality metrics (ROUGE, success rate) on the same view as resource
  cost, so the developer can see whether a quality improvement came at a
  resource cost or vice versa.
- Let the developer interactively filter by hostname, dataset, or stage.

The run comparison tool already provides interactive Streamlit UI, Plotly
charts, and a nav structure. Adding a Performance page is the natural extension.

---

## Design

### Join Key

Quality runs in `data/eval/runs/` and performance profiles in
`data/profiles/*.yaml` are joined by **release tag**. Both artifact types carry
the release tag in their metadata:

- Frozen profiles: top-level `release` field (e.g. `v2.6.0`).
- Eval runs: the run directory name or `fingerprint.json` `release` field.

The tool matches them automatically when both exist for the same tag. When only
one side exists, the tool shows what it has and marks the other as "missing".

### Profile Discovery

A new function `discover_profiles()` in `tools/run_compare/data.py` scans
`data/profiles/*.yaml`, parses each YAML, and returns a list of
`ProfileEntry` dataclass objects:

```python
@dataclass(frozen=True)
class ProfileEntry:
    release: str
    date: str
    dataset_id: str
    hostname: str
    path: Path
    stages: Dict[str, Dict[str, float]]
    totals: Dict[str, float]
    environment: Dict[str, Any]
```

Profiles are sorted by date (oldest first) for trend charts.

### Joined Data Model

A `JoinedRelease` dataclass pairs an optional `RunEntry` (quality) with an
optional `ProfileEntry` (performance) by release tag:

```python
@dataclass(frozen=True)
class JoinedRelease:
    release: str
    eval_entry: Optional[RunEntry]
    profile_entry: Optional[ProfileEntry]
```

The join is a full outer join: releases with only quality data appear (profile
column blank), and releases with only a profile appear (quality column blank).

### Navigation

Add **Performance** to the existing nav bar:

```text
Home · KPIs · Delta · Episodes · Performance
```

The `?page=performance` slug routes to the new page. Existing pages are
unchanged.

### Performance Page Layout

The page has four sections, rendered top to bottom:

#### 1. KPI Tiles (Top Row)

For each selected release that has a profile, display a row of metric tiles:

| Tile | Source |
| ---- | ------ |
| Peak RSS (MB) | `totals.peak_rss_mb` |
| Total wall time (s) | `totals.wall_time_s` |
| Avg wall time / episode (s) | `totals.avg_wall_time_per_episode_s` |
| Episodes processed | `episodes_processed` |
| Hostname | `environment.hostname` |

When two or more releases are selected, show delta arrows (green/red) between
baseline and candidates, reusing the same `delta_direction_good` pattern from
the quality delta table (lower is better for RSS and wall time).

#### 2. Stage Delta Table

Same colored-delta pattern as RFC-047's quality delta table, applied to
per-stage resource metrics:

| Column | Description |
| ------ | ----------- |
| Stage | Pipeline stage name (e.g. `summarization`) |
| Metric | `wall_time_s`, `peak_rss_mb`, or `avg_cpu_pct` |
| Baseline | Value from the baseline release |
| Candidate | Value from the candidate release |
| Delta | Absolute difference |
| Delta % | Percentage change |
| Good? | Green check / red X (lower is better for all three) |

The baseline is selected from the sidebar dropdown (same pattern as quality
baseline). When more than two releases are selected, each non-baseline release
gets its own candidate column.

#### 3. Trend Chart (Multi-Release Line Chart)

Line chart with releases on the x-axis (ordered by date) and one line per
stage. Two sub-charts stacked vertically:

- **Wall time per stage** — y-axis in seconds.
- **Peak RSS per stage** — y-axis in MB.

Stages that are absent in some releases show gaps (no interpolation). A
hostname filter in the sidebar restricts the chart to profiles from the same
machine (with a warning banner when mixed-machine profiles are displayed
together).

Chart style follows `docs/guides/TUFTE_CHART_CRITIQUE.md`: white background,
minimal chrome, small multiples where appropriate, no bubble-area distortion.

#### 4. Quality vs Cost Scatter

A two-axis scatter plot where each dot is one release:

- **X-axis**: user-selectable quality metric from a dropdown:
  - ROUGE-L F1 (default, from `metrics.json` `vs_reference`)
  - Success rate
  - Avg latency (s)
- **Y-axis**: user-selectable resource metric from a dropdown:
  - Total wall time (s) (default)
  - Peak RSS (MB)
  - Avg wall time per episode (s)

Only releases that have **both** a quality eval run and a frozen profile are
plotted. Dots are labeled with the release tag. Hover shows full details.

This chart answers the key question: *"Did the quality improvement come at a
resource cost?"*

### Cross-Machine Handling

Profiles carry `environment.hostname`. When the selected set includes profiles
from more than one hostname:

1. A **warning banner** appears at the top of the Performance page:
   *"Selected profiles come from different machines — absolute RSS/CPU values
   are not directly comparable. Filter by hostname in the sidebar for
   meaningful comparisons."*
2. The sidebar gains a **Hostname** filter (multi-select, default: all).
3. Trend charts and delta tables still render but with the warning visible.

### Missing Stages in Trend Charts

Some pipeline stages may be disabled in certain releases (e.g. `gi_generation`
skipped when GI is off). The trend chart handles this by:

- Showing a gap in the line for that stage at that release.
- Not interpolating across gaps.
- A footnote: *"Gaps indicate stages not present in that release's profile."*

### Minimum Profile Count

- **1 profile**: KPI tiles and stage breakdown render (no trend, no delta).
- **2 profiles**: Delta table and quality-vs-cost scatter become available.
- **3+ profiles**: Trend chart becomes meaningful (line chart with 2 points is
  technically renderable but not very useful; still shown).

### Cross-Dataset Profiles

Profiles carry `dataset_id`. The sidebar gains a **Dataset** filter
(multi-select, default: all). When comparing profiles from different datasets,
a warning appears: *"Profiles use different datasets — wall times are not
directly comparable across different episode counts/sizes."*

---

## Data Layer Changes (`tools/run_compare/data.py`)

New functions:

| Function | Purpose |
| -------- | ------- |
| `discover_profiles(profiles_root)` | Scan `data/profiles/*.yaml`, return `List[ProfileEntry]` |
| `load_profile(path)` | Parse one YAML into `ProfileEntry` |
| `join_releases(runs, profiles)` | Full outer join by release tag → `List[JoinedRelease]` |
| `profile_delta_rows(base, candidates)` | Stage-level delta rows for the delta table |
| `profile_trend_rows(profiles)` | Long-form rows for Plotly trend chart |

No changes to existing quality data loading functions.

---

## UI Layer Changes (`tools/run_compare/app.py`)

- Import new data functions and `ProfileEntry` / `JoinedRelease`.
- Add `"Performance"` to `_NAV_PAGES` and slug mappings.
- New `_render_performance_page(...)` function containing the four sections.
- Sidebar additions: hostname filter, dataset filter (only shown when
  Performance page is active).
- Baseline dropdown extended to work with `JoinedRelease` objects.

---

## Dependencies

No new dependencies. Everything needed is already in `[compare]`:

- `streamlit` — UI framework
- `plotly` — charts
- `pandas` — data manipulation
- `pyyaml` — YAML parsing (already a core dependency)

Frozen profile YAML files from RFC-064 (`data/profiles/*.yaml`) are the data
source.

---

## Implementation Plan

### Phase 1: Data Layer

1. Add `ProfileEntry` dataclass and `discover_profiles()` to `data.py`.
2. Add `load_profile()` with YAML parsing and validation.
3. Add `join_releases()` full outer join.
4. Add `profile_delta_rows()` and `profile_trend_rows()` helpers.
5. Unit tests for all new data functions.

### Phase 2: Performance Page

1. Add Performance to nav bar and slug routing.
2. Implement KPI tiles section.
3. Implement stage delta table.
4. Implement trend chart (wall time + RSS).
5. Implement quality-vs-cost scatter.

### Phase 3: Filters and Warnings

1. Hostname filter in sidebar.
2. Dataset filter in sidebar.
3. Cross-machine warning banner.
4. Cross-dataset warning banner.
5. Missing-stage gap handling in trend chart.

---

## Testing Strategy

- **Unit tests** (`tests/unit/tools/test_run_compare_data.py`): test
  `discover_profiles`, `load_profile`, `join_releases`, delta/trend row
  helpers with fixture YAML files.
- **Manual QA**: run `make run-compare` with 2+ frozen profiles in
  `data/profiles/` and verify all four sections render correctly.
- **No E2E Playwright tests** — the run comparison tool is a developer-facing
  Streamlit app, not the GI/KG viewer.

---

## Open Questions (Resolved)

| # | Question | Resolution |
| - | -------- | ---------- |
| 1 | New tab or separate app? | New nav page in existing app — consistent UX, shared sidebar filters, single `make run-compare` entry point. |
| 2 | Quality metric on scatter x-axis? | User-selectable dropdown (ROUGE-L F1 default, plus success rate and avg latency). |
| 3 | Cross-machine profiles? | Hostname filter in sidebar + warning banner when mixed. |
| 4 | Per-episode or stage aggregates? | Stage aggregates only — frozen profiles are per-stage; per-episode breakdown belongs in the quality pages. |
| 5 | Missing stages in trend charts? | Gaps (no interpolation) with footnote. |
| 6 | Minimum profiles for trend chart? | 1 profile = tiles only; 2 = delta + scatter; 3+ = trend lines. |
| 7 | Cross-dataset comparison? | Dataset filter in sidebar + warning when mixed. |

---

## References

- **Parent RFC**: `docs/rfc/RFC-064-performance-profiling-release-freeze.md`
- **Extended Tool**: `tools/run_compare/app.py` (RFC-047)
- **Quality Data**: `data/eval/runs/`, `data/eval/baselines/`
- **Performance Data**: `data/profiles/*.yaml`
- **Chart Style**: `docs/guides/TUFTE_CHART_CRITIQUE.md`
- **Tracking Issue**: [#511](https://github.com/chipi/podcast_scraper/issues/511)
