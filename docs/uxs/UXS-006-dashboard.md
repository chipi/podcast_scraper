# UXS-006: Dashboard

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states

- **Normative layout & charts**: see **## Dashboard implementation specification** below (full product spec merged into this UXS).
- **Related PRDs**:
  - [PRD-025: Corpus Intelligence Dashboard](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
- **Related RFCs**:
  - [RFC-071: Corpus Intelligence Dashboard](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/dashboard/DashboardView.vue`
  - `web/gi-kg-viewer/src/components/dashboard/BriefingCard.vue`
  - `web/gi-kg-viewer/src/components/dashboard/IndexStatusCard.vue`
  - `web/gi-kg-viewer/src/components/dashboard/TopicClustersStatusBlock.vue`
  - `web/gi-kg-viewer/src/utils/chartRegister.ts`
  - `web/gi-kg-viewer/src/stores/indexStats.ts`, `web/gi-kg-viewer/src/stores/dashboardNav.ts`
- **Shell IA:** [VIEWER_IA.md](VIEWER_IA.md) — canonical shell layout, navigation axes, subject rail, status bar, first-run behavior

---

## Summary

For shell layout, the three navigation axes, subject rail persistence and clearing, status bar, and first-run empty corpus behavior, see **[VIEWER_IA.md](VIEWER_IA.md)**. This document specifies the **Dashboard** main tab only (briefing card, Coverage / Intelligence / Pipeline sub-tabs, charts, and layout below).

The **Dashboard** main tab is **briefing + three tabs only**: **Coverage** (default),
**Intelligence**, **Pipeline**. Corpus artifact picking (**List**, **All** / **None**,
**Load into graph**) lives on the **status bar** (**List** opens
`data-testid="artifact-list-dialog"`). Deep links to **Library**, **Digest**, and
**Graph** use `dashboardNav` handoffs consumed when those tabs activate. Chart.js uses
shared Tufte-style defaults from `chartRegister.ts`.

---

## Layout

1. **Briefing card** (`data-testid="briefing-card"`) — last run / health / short actions; always above tabs.
2. **Tablist** `aria-label="Dashboard tabs"` — **Coverage** | **Intelligence** | **Pipeline**.
3. **Coverage** — coverage by month, feed coverage table, artifact activity (from listed artifacts), **Index status** (`data-testid="index-status-card"`) with **Update index** and **Full rebuild** (`index-status-update`, `index-status-full-rebuild`).
4. **Intelligence** — digest snapshot, **Topic clusters** status (`topic-clusters-status-block`), topic landscape, top voices (when API available). Topic momentum / emerging connections **omitted** until RFC-073 data ships (no placeholder UI).
5. **Pipeline** — run history strip, duration trend, stage timings, numeric outcomes, episodes per run; optional per-feed run heatmap only when server exposes stable per-feed fields.

The legacy **CorpusDataWorkspace** / **Pipeline | Content intelligence** split on Dashboard is removed; those components are not part of this surface.

---

## Corpus artifacts (status bar)

**List** (when health is ok and a corpus path is set) fetches `GET /api/artifacts` and opens the **Corpus artifacts** dialog. **Load into graph** switches the main tab to **Graph** when load succeeds (same behavior as the former workspace).

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — **Dashboard** tab,
`briefing-card`, **Dashboard tabs** tablist, **Index status** card, **`artifact-list-dialog`**;
`openCorpusDataWorkspace` only switches to **Dashboard** and waits for the briefing card.

---

## Revision history

| Date       | Change                                                                           |
| ---------- | -------------------------------------------------------------------------------- |
| 2026-04-06 | Initial content (in UXS-001)                                                     |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-006                                   |
| 2026-04-19 | Corpus workspace on Dashboard; left rail query-only IA                           |
| 2026-04-19 | Dashboard: briefing + tabs; artifacts via status bar dialog                      |
| 2026-04-20 | §6.0 jobs: operator `--config` + optional `profile:` merge (#593)                |
| 2026-04-20 | §6.0 jobs (cont.): feeds via `--feeds-spec` when `feeds.spec.yaml` exists        |
| 2026-04-21 | §6.0: Operator YAML = **Config**; shallow PUT validation (feed + secret keys)    |

---

## Dashboard implementation specification

**Status:** Ready for implementation
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/components/dashboard/`
**Related docs:** UXS-006, RFC-071, TUFTE_CHART_CRITIQUE.md, UXS-001
**Replaces:** Current UXS-006 content (rewrite, not amendment)

---

### 1. Overview

The Dashboard is restructured around two principles:

**5-second answers.** The primary question — "is everything OK, and what
should I do?" — is answered by a permanent briefing card above the tabs,
before any chart is read.

**Actionability.** Every data point either has an attached action or is
explicitly decorative context. Observations without a "so what?" belong
in depth charts, not on the primary surface.

**Structure:**

```text
├── BRIEFING CARD  (permanent, always visible, above tabs)
│   Last run · Corpus health · Action items
│
└── TABS
    ├── Coverage      — what fraction of my corpus has intelligence coverage?
    ├── Intelligence  — what is my corpus telling me?
    └── Pipeline      — how is my pipeline performing?
```yaml

Three tabs instead of the current two. The old "Status" concept becomes
the briefing card. The old "Content Intelligence" becomes Coverage +
Intelligence split by purpose.

---

### 2. Global Chart Rules (Tufte)

These apply to every chart in the dashboard without exception. They
are not per-chart suggestions — they are design system constraints.

#### 2.1 Non-negotiable rules

**No legends.** Every series is labelled directly — end-of-line for
line charts, end-of-bar for bar charts, positioned annotation for
everything else. No legend box anywhere in the dashboard.

**Title = insight.** Every chart title states the finding, not the
variable name. "Coverage by month" is a label. "N months below average
GI coverage" is a title. Mandatory, not optional.

**No gridlines.** Remove all gridlines from all charts. Reference
lines (average, target) are a single `border`-token hairline annotated
directly — not a grid.

**No dual y-axes.** If two series have different scales, they get
separate charts. No exceptions.

**Remove top and right spines.** Only left and bottom axis lines on
all Chart.js charts.

**Y-axis starts at 0 on all bar charts.** No truncated axes.

**Mandatory insight line.** Remove the word "optional" from UXS-001
and UXS-006. Every chart has an insight line below it. If the data
does not support a clear takeaway, the chart does not belong on the
dashboard.

**Charts earn their place.** Before adding a chart, ask: does this
need visual encoding, or is a number or table sufficient? Three numbers
are not a chart. If the answer is "a table," build a table.

**No chartjunk.** No 3D effects, no gradient fills, no drop shadows
on chart elements, no decorative borders around chart areas.

#### 2.2 Chart.js global config

Apply these defaults in `chartRegister.ts` so every chart component
inherits them without per-component overrides:

```typescript
Chart.defaults.plugins.legend.display = false
Chart.defaults.plugins.tooltip.enabled = true

Chart.defaults.scales.linear = {
  ...Chart.defaults.scales.linear,
  grid: { display: false },
  border: { display: false },
  ticks: { maxTicksLimit: 5, color: 'var(--ps-muted)' }
}

Chart.defaults.scales.category = {
  ...Chart.defaults.scales.category,
  grid: { display: false },
  border: { display: false },
  ticks: { color: 'var(--ps-muted)' }
}
```

#### 2.3 Direct end-label plugin

A shared Chart.js `afterDraw` plugin for direct series labelling.
Register once in `chartRegister.ts`. Used by all multi-series line
charts:

```typescript
const endLabelPlugin = {
  id: 'endLabel',
  afterDraw(chart: Chart) {
    const ctx = chart.ctx
    chart.data.datasets.forEach((dataset, i) => {
      const meta = chart.getDatasetMeta(i)
      if (!meta.hidden && meta.data.length > 0) {
        const lastPoint = meta.data[meta.data.length - 1]
        ctx.fillStyle = dataset.borderColor as string
        ctx.font = '10px Inter, system-ui'
        ctx.textAlign = 'left'
        ctx.fillText(
          dataset.label ?? '',
          lastPoint.x + 6,
          lastPoint.y + 3
        )
      }
    })
  }
}
Chart.register(endLabelPlugin)
```yaml

---

### 3. Briefing Card

#### 3.1 Purpose

Permanent section at the top of the Dashboard page, above the tab bar.
Always visible regardless of which tab is active. Answers the three
primary questions in under 5 seconds:

- "Did the last run succeed?" (Last run section)
- "Is my corpus healthy?" (Corpus health section)
- "What should I do?" (Action items section)

#### 3.2 Layout

```text
│  CORPUS      213 episodes · 67% GI · 82% indexed · 3 feeds      │
│  ──────────────────────────────────────────────────────────────  │
│  → 71 episodes have no GI artifacts          [View in Library]   │
│  → Index last rebuilt 14 days ago            [Rebuild now]       │
│  → 2 episodes failed in last run             [View failures]     │
└──────────────────────────────────────────────────────────────────┘
```

- Background: `elevated` token (lifts the card above the page surface)
- Border: `border` token, `rounded-sm`
- Padding: `p-4`
- Divider between health and action items: `border-t border-border mt-3 pt-3`
- Section labels ("LAST RUN", "CORPUS"): `text-[10px] font-semibold
  tracking-wider muted` small caps

- Data values: `text-sm surface-foreground`

#### 3.3 Last run section

Sources: `GET /api/corpus/runs/summary` most recent item.

**Format:** `● [status] · [N] episodes · [duration] · [age]  [Details →]`

Status badge:

- `● Success` — `success` token dot
- `● Partial` — `warning` token dot (some feeds failed in multi-feed run)
- `● Failed` — `danger` token dot

Age: relative time — "2 hours ago", "yesterday", "3 days ago". Not a
raw timestamp. Computed client-side from run timestamp.

Duration: human-readable — "3m 24s", not "204s".

Multi-feed: "3 of 4 feeds succeeded" replaces episode count when
`multi_feed_summary` is present and any feed failed.

"Details →" link: navigates to Pipeline tab.

**Empty state** (no runs found): `muted` text — "No pipeline runs
found. Run `podcast scrape` to begin."

`data-testid="briefing-last-run"`

#### 3.4 Corpus health section

Sources: `GET /api/corpus/coverage` (NEW endpoint — see Section 8) +
`GET /api/index/stats`.

**Format:** `[total] episodes · [gi]% GI · [indexed]% indexed · [feeds] feeds`

Each metric is a link:

- Episode count → Library tab
- GI% → Coverage tab (scrolls to coverage chart)
- Indexed% → Coverage tab (scrolls to index section)
- Feed count → Library tab (feed filter)

Warning threshold: if GI% < 50% or indexed% < 60%, that metric
renders in `warning` token colour. Immediate visual signal without
reading action items.

Thresholds are tunable parameters (UXS-001 table):

| Parameter | Default | Status |
| --- | --- | --- |
| GI coverage warning threshold | 50% | Open |
| Index coverage warning threshold | 60% | Open |

`data-testid="briefing-corpus-health"`

#### 3.5 Action items section

Max 3 items. Triage order: pipeline failures first (blocking),
coverage gaps second, operational staleness third.

Sources: assembled client-side from coverage data + index stats +
run summary.

**Item format:** `→ [plain-language finding]  [primary action link]  [secondary action link]`

Action item rules:

- Plain language, present tense: "71 episodes have no GI artifacts"
  not "GI coverage: 67%"

- Primary action link: the most direct next step
- Max one secondary action link
- Never says "run enrich" or references RFC-073 enrichers — out of scope

**Possible items (ordered by priority):**

| Condition | Text | Actions |
| --- | --- | --- |
| Last run has failures | "N episodes failed in last run" | [View failures] → Library |
| Last run failed entirely | "Last run failed — no episodes processed" | [View in Pipeline] |
| GI coverage < 50% | "N episodes have no GI artifacts" | [View in Library] |
| Index not built | "Vector index has not been built" | [Build index] |
| Index stale > 7 days | "Index last rebuilt N days ago" | [Rebuild now] |
| Feed not indexed | "Feed X is not in the vector index" | [View coverage] |
| No runs in > 7 days | "No pipeline runs in N days" | [View in Pipeline] |
| Topic clusters missing | "Topic clusters not built" | [View in Coverage] |

"Rebuild now" and "Build index" trigger `POST /api/index/rebuild`
directly (existing endpoint). All others are navigation links.

**All-clear state:** When no items qualify: a single line —
`● Everything looks good` — using `success` token dot. This positive
confirmation is as important as finding items — it tells you to stop
looking.

`data-testid="briefing-action-items"`
`data-testid="briefing-action-item"` (per item)
`data-testid="briefing-all-clear"` (empty state)

---

### 4. Coverage Tab

Answers: "What fraction of my corpus has intelligence coverage, and
where are the gaps?"

Four sections, top to bottom.

#### 4.1 GI coverage by month

**Chart type:** Single-series bar chart.

**What it shows:** GI coverage percentage per publish month (0–100%).
One bar per month. Bars are the primary story — what fraction of
episodes published that month have GI artifacts.

**Not** a stacked chart. Not episode counts. Coverage rate is the
signal, not volume.

**Tufte compliance:**

- Single series, no legend needed
- Y-axis: 0–100%, labelled "Coverage %", max 5 tick labels
- X-axis: month labels, `text-[10px]`
- Reference line: horizontal hairline at corpus average coverage %,
  annotated directly — "avg 67%" placed at the right end of the line

- Bars below average: `warning` token fill
- Bars at or above average: `gi` token fill
- No gridlines, no top/right spines
- Direct annotation on the lowest bar: "Lowest: [month] — N%"
- Insight line below chart (mandatory): computed from data, e.g.
  "3 months below average — [month], [month], [month] need attention"

  or "Coverage improving: up [X]pp since [earliest month]"

**Interaction:** clicking any bar navigates to Library filtered to
episodes from that month without GI artifacts (`since` + end of
month + `topic_cluster_only=false` + filter for no-GI).

**Source:** `GET /api/corpus/coverage` → `by_month` array (NEW).

**Component:** `CoverageByMonthChart.vue` (new, replaces
`VerticalBarChart.vue` for this use case)

**Empty state:** "No episode metadata found. Run the pipeline to
generate corpus data."

`data-testid="coverage-by-month-chart"`

#### 4.2 Feed breakdown

**Not a chart — a table with inline data bars.**

**What it shows:** Per-feed coverage summary. Each row = one feed.
Columns: Feed name | Episodes | GI coverage | KG coverage | Indexed.

GI coverage and KG coverage columns show a 40px inline progress bar
(CSS, not Chart.js) plus the percentage number. This gives visual
comparison without a separate chart.

**Tufte note:** 40px inline bars are proportional (0–100% always),
direct labelled (number next to bar), and scannable. No legend, no
axis, no gridlines.

**Sort order:** ascending by GI coverage — worst-covered feeds at
top, since those are the action targets.

**Row interaction:** clicking a row navigates to Library filtered to
that feed.

**Insight line:** "Feed [name] has lowest GI coverage at [N]% — [M]
episodes without GI artifacts."

**Source:** `GET /api/corpus/coverage` → `by_feed` array (NEW).

**Component:** `FeedCoverageTable.vue` (new)

`data-testid="feed-coverage-table"`
`data-testid="feed-coverage-row"` (per row)

#### 4.3 Artifact activity

**Chart type:** Grouped bar chart — new artifacts per day, last 30
days.

**Replaces:** The current cumulative GI+KG line chart.

**Why:** Cumulative lines always trend upward and look healthy even
when work has stopped. A recency chart makes silence visually obvious.

**What it shows:** For each day in the last 30 days: count of new
GI artifacts + count of new KG artifacts written that day.

**Tufte compliance:**

- Two series: `gi` token bars and `kg` token bars, grouped per day
- Series labelled directly: "GI" and "KG" as `text-[10px]` text
  positioned above the first bar of each series (not a legend)

- X-axis: day labels, only show every 7th label to avoid clutter
- Y-axis: starts at 0, labelled "New artifacts"
- No gridlines, no spines except left + bottom
- Silence is visible: days with no bars are visually obvious gaps
- If last 14 consecutive days have no bars: mandatory insight reads
  "No new artifacts in 14 days — pipeline may not be running"

- Otherwise insight: "Last GI: [date] · Last KG: [date]"
- Annotate the most recent active day: small dot + date label

**Source:** `GET /api/artifacts` → `mtime_utc` values per artifact,
bucketed by day client-side. Existing `artifactMtimeBuckets.ts` —
extend to produce daily counts rather than cumulative.

**Component:** `ArtifactActivityChart.vue` (new — replaces
`CategoryLineChart.vue` for this purpose)

`data-testid="artifact-activity-chart"`

#### 4.4 Index status

**Not a chart — a card.**

The question "is my index current?" doesn't need a chart. It needs
three facts:

```text
Feeds in index: 3 of 3
```python

- "⚠ Rebuild recommended" appears when index staleness heuristic
  from `index/stats` is true. Uses `warning` token.

- "Rebuild now" button when stale (triggers existing endpoint).
- "Last rebuild error: [message]" in `danger` token when
  `rebuild_last_error` is present.

- Disabled while `rebuild_in_progress` is true (shows "Rebuilding…"
  spinner inline).

No Chart.js involved. Pure card with MetricsPanel pattern.

**Source:** `GET /api/index/stats` (existing).

`data-testid="index-status-card"`

---

### 5. Intelligence Tab

Answers: "What is my corpus telling me?"

Five sections. Some require enricher data and degrade gracefully.
Enricher triggers ("run enrich") are out of scope — degraded states
show what's missing, not how to generate it.

#### 5.1 Corpus snapshot

**Always available.**

An expanded version of the digest compact view. Shows the rolling
window summary (default 7d) with a little more breathing room than
the Digest tab's compact layout.

Content:

- Window line: "Last 7 days — 12 new episodes across 3 feeds"
- Top 3 topic bands from the digest, one line each: topic label +
  episode count ("AI policy — 4 episodes")

- Each topic band line clickable → Digest tab (which loads that topic)
- "Open Digest →" link at bottom

Not a chart. A structured text summary using the digest API data.

**Source:** `GET /api/corpus/digest` with `window=7d`, `compact=false`,
`max_rows=3`.

`data-testid="intelligence-snapshot"`

#### 5.2 Topic landscape

**Available when topic clusters have been built.**

A compact grid of topic clusters. Each cluster card:

- Cluster canonical label (`text-sm font-semibold`)
- Member topic count badge ("4 topics")
- Episode count (summed from `members[].episode_ids`, deduplicated)
- Clicking → Graph tab, focused on that TopicCluster compound node

Grid: `sm:2` / `xl:3` columns. Same grid pattern as Digest topic bands
but cards are smaller and denser — no hit rows, just cluster identity.

**Insight line:** "N topic clusters covering M distinct topics."

**Degraded state** (404 from topic-clusters endpoint): `muted` text —
"Topic clusters not yet built for this corpus." No action prompt.

**Source:** `GET /api/corpus/topic-clusters` (existing).

`data-testid="intelligence-topic-landscape"`

#### 5.3 Top voices

**Available when `GET /api/corpus/persons/top` exists (NEW endpoint —
see Section 8).**

Top 5 persons by insight count. Each person card:

- Display name (`text-sm font-semibold`)
- Episode count badge: "23 episodes"
- Insight count badge: "67 insights" (`gi` token)
- Top 3 topic chips (`kg` token border) — topics from the endpoint
- "View profile →" link → Person Landing (UXS-010) in subject rail

Cards in a horizontal scrollable row on narrow viewports, 2-column
grid on wide viewports.

**Insight line:** "N speakers with grounded insights — [top person]
leads with N insights across M episodes."

**Degraded state** (endpoint not available or 0 persons): `muted` text —
"No speaker intelligence found for this corpus."

**Source:** `GET /api/corpus/persons/top?limit=5` (NEW).

`data-testid="intelligence-top-voices"`

#### 5.4 Topic momentum

**Enricher-gated — degrades gracefully.**

Available when `temporal_velocity` enricher data is present (RFC-073).

Top 5 topics by recent activity. Each row:

- Topic name (`text-sm`)
- Sparkline: 8-point mini line chart, last 8 weeks of mention
  frequency. `series-1` token stroke, no fill. No axes, no labels —

  the shape is the signal.

- Trend badge: accelerating (`success`) / stable (`muted`) /
  declining (`warning`) — same as UXS-007

- Episode count: "N episodes" in `muted`
- Clicking → Topic Entity View (UXS-007) in subject rail

**Tufte note on sparklines:** y-scale normalises to each topic's own
range (0 to its max). This shows *relative* trend shape, not absolute
volume. Volume is shown as the episode count number. This is acceptable
for a "relative momentum" display — the spec must note this so
implementers don't try to share y-scales (which would make
low-volume topics invisible).

**Insight line:** "N topics accelerating in the last 30 days."

**Degraded state:** `muted` text — "Topic momentum data not available
for this corpus." No enricher prompt.

**Source:** RFC-073 enricher output (not yet available).

`data-testid="intelligence-topic-momentum"`

#### 5.5 Emerging connections

**Enricher-gated — degrades gracefully.**

Available when `topic_cooccurrence` enricher data is present (RFC-073).

Top 4 topic pairs by unexpected co-occurrence strength. Each row:

- "Topic A ↔ Topic B" (`text-sm`)
- Episode count: "14 episodes" (`muted`)
- Clicking → Graph tab, both topic nodes highlighted + focused,
  subgraph showing their shared connections

**Insight line:** "N topic pairs co-occur more than expected given
their individual frequency."

**Degraded state:** `muted` text — "Topic co-occurrence data not
available for this corpus." No enricher prompt.

**Source:** RFC-073 enricher output (not yet available).

`data-testid="intelligence-emerging-connections"`

---

### 6. Pipeline Tab

Answers: "How is my pipeline performing?"

Now freed from being the default tab (briefing card handles the quick
answer), the Pipeline tab can be dense and detailed.

#### 6.0 HTTP pipeline jobs (RFC-077)

When **`jobs_api`** is true on health, the tab includes **`PipelineJobsCard`**
(`data-testid="pipeline-jobs-card"`): list **HTTP-triggered** pipeline jobs for
the current corpus, **Run** (enqueue), **Reconcile**, **Refresh**, per-row **Cancel**
when status is `queued` or `running`, and inline help when the API is off. This
surface is **in addition to** run-history charts fed from **`GET /api/corpus/runs/summary`**
(see below). Spec: [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md).

**Config a job uses:** the subprocess passes **`--config`** to the server-resolved operator YAML path (same file as status bar **Operator YAML** / **Config**). That file may contain **`profile: <preset>`** — effective pipeline settings are **packaged preset defaults merged with explicit keys** in the file ([GitHub #593](https://github.com/chipi/podcast_scraper/issues/593)). **Feeds** for the job come from corpus **`feeds.spec.yaml`** via **`--feeds-spec`** when that file exists, not from duplicate root keys in operator YAML (those keys are rejected on **`PUT /api/operator-config`** — **top-level** denylist only, same as secrets). Any **Run** / help copy in this tab should say **operator file + optional profile**, not “edit feeds here.”

#### 6.1 Run history strip

**Not a chart — a row of status dots.**

Last 10 runs rendered as coloured dots in a horizontal row. No axes,
no labels — maximum information density.

- `●` success token = run succeeded
- `●` warning token = partial success (some feeds failed)
- `●` danger token = run failed

Dots are ordered left-to-right oldest to newest. Most recent dot is
slightly larger (12px vs 8px) to draw the eye to current state.

Native `title` tooltip per dot: "Run [date] · [N] episodes ·
[duration] · [status]"

Clicking a dot: expands that run's detail inline below the strip,
replacing the current "latest run detail" section with the selected
run. Clicking again collapses. Only one run expanded at a time.

**Insight line:** "Last 10 runs: [N] success, [M] partial, [P] failed"
— or simply "All 10 runs succeeded" in `success` token if clean.

**Source:** `GET /api/corpus/runs/summary` (existing).

`data-testid="pipeline-run-history-strip"`
`data-testid="pipeline-run-dot"` (per dot)

#### 6.2 Duration trend

**Chart type:** Bar chart, last 5 runs' durations.

**What it shows:** Run-to-run duration variability. Is the pipeline
getting slower?

**Tufte compliance:**

- 5 bars — simple, no legend needed
- Y-axis: starts at 0, labelled in minutes, max 4 ticks
- X-axis: run dates, `text-[10px]`
- Reference line: horizontal hairline at 5-run average, directly
  labelled "avg [X]m [Y]s"

- Latest bar: `primary` fill. Others: `surface-foreground` at 60%
  opacity. Draws eye to the current run vs history.

- If latest bar > average + 20%: `warning` fill on latest bar
- No gridlines, no top/right spines
- Direct bar labels: duration value above each bar ("3m 24s") in
  `text-[10px] muted`

- **Title = insight** (mandatory): "Latest run [X]% [faster/slower]
  than 5-run average" — computed. If within 10% either way: "Run

  duration stable — avg [X]m [Y]s"

**Source:** `GET /api/corpus/runs/summary` last 5 items sorted by
timestamp.

**Component:** Reuse/update existing chart component.

`data-testid="pipeline-duration-trend"`

#### 6.3 Latest run breakdown

Two sections side-by-side on wide viewports, stacked on narrow.

**Stage timings (left):**

**Chart type:** Horizontal bar chart, one bar per pipeline stage,
sorted by duration descending.

**What it shows:** Where time was spent in the most recently selected
run (from the history strip — defaults to latest).

**Tufte compliance:**

- Horizontal bars — stage names are text, need space to breathe
- Sorted descending: longest bar at top, shortest at bottom
- Direct value labels: duration ("1m 23s") printed at the right end
  of each bar — **no x-axis needed** once values are labelled

- Bar fill: `primary` token for the longest bar (the bottleneck),
  `surface-foreground` at 60% opacity for all others. The bottleneck

  is visually distinct without any annotation.

- No gridlines at all — bar length communicates value
- No spines — labels make axes redundant
- **Title = insight** (mandatory): "[Stage name]: [X]% of total run
  time" — always names the bottleneck

**Episode outcomes (right):**

**Not a chart.** Three numbers are not a chart.

```text
✗  2  failed  [View failures →]
```

`text-sm` numbers, intent token colours for each count. "View
failures →" link appears only when failures > 0, navigates to
Library filtered for that run's failed episodes.

**Source:** `GET /api/corpus/runs/summary` for the selected run
(or run detail endpoint if stage timings need fuller data — see
open questions in Section 9).

`data-testid="pipeline-stage-timings"`
`data-testid="pipeline-episode-outcomes"`

#### 6.4 Episodes per run

**Chart type:** Bar chart, all runs in summary (up to 150, capped).

**What it shows:** Episode count processed per run over time. Shows
pipeline activity and cadence.

**Tufte compliance:**

- Single series, no legend needed
- Y-axis: starts at 0, "Episodes" label, max 5 ticks
- X-axis: run dates (sparse labelling — every Nth label)
- No gridlines, no top/right spines
- Total corpus as text annotation below chart: "Total: [N] episodes
  across [M] runs" — **not** a cumulative overlay on the same axis

  (that was the dual-axis lie)

- **Title = insight**: "Average [N] episodes per run" or "Processing
  volume declining — last 3 runs below average" if trend is negative

**Source:** `GET /api/corpus/runs/summary` (existing).

**Component:** Reuse `VerticalBarChart.vue` with updated config.

`data-testid="pipeline-episodes-per-run"`

#### 6.5 Feed processing history

**Chart type:** Heatmap grid (small multiples).

Only shown for multi-feed corpora. Hidden for single-feed.

**What it shows:** Per-feed run success/partial/failure over the last
5 runs. Rows = feeds, columns = runs (left-to-right oldest to newest).

**Tufte compliance:**

- Cell colour: `success` fill / `warning` fill / `danger` fill
- Cell glyph: ✓ / ⚠ / ✗ inside each cell — colour alone is not
  sufficient (accessibility + Tufte "don't rely on colour only")

- Cell size: ~28px × 28px. Gap spacing between cells instead of
  borders (whitespace is cleaner than lines)

- Feed names: left-aligned row headers, `text-xs`
- Run dates: top-aligned column headers, `text-[10px]`
- No legend — ✓/⚠/✗ glyphs are self-explanatory
- **Insight line** (mandatory): "Feed [name] failed [N] of last 5
  runs" — triggered when any feed has ≥ 2 failures in the window.

  If all clean: "All feeds succeeded in last 5 runs" (`success`
  token).

**Source:** `GET /api/corpus/runs/summary` per-feed breakdown
(verify field availability in `CorpusRunSummaryItem` — see Section 9).

**Component:** `FeedRunHistoryGrid.vue` (new)

`data-testid="pipeline-feed-history-grid"`

---

### 7. Tab Order and Defaults

Coverage is the default tab (not Pipeline as today) because it answers
the most frequent ongoing question — "how complete is my intelligence
coverage?" — whereas Pipeline is consulted reactively after running.

The briefing card's "Details →" link on the Last run section jumps to
the Pipeline tab.

---

### 8. New API Endpoints Required

#### 8.1 `GET /api/corpus/coverage`

**Priority:** CRITICAL. Unblocks briefing card scorecard + Coverage tab.

Single catalog scan checking GI/KG artifact presence per episode,
grouped by month and feed. Same implementation pattern as
`/api/corpus/stats`.

**Response schema:**

```python

class CoverageByMonthItem(BaseModel):
    month: str           # "YYYY-MM"
    total: int
    with_gi: int
    with_kg: int
    with_both: int

class CoverageFeedItem(BaseModel):
    feed_id: str
    display_title: str
    total: int
    with_gi: int
    with_kg: int

class CorpusCoverageResponse(BaseModel):
    total_episodes: int
    with_gi: int
    with_kg: int
    with_both: int
    with_neither: int
    by_month: List[CoverageByMonthItem]
    by_feed: List[CoverageFeedItem]

```

**Implementation:** One filesystem scan. For each episode metadata
file: check sibling `*.gi.json` and `*.kg.json` existence, group by
`publish_date` month and `feed_id`. No GI/KG file reading — just
existence checks. Fast.

**Test:** `tests/unit/podcast_scraper/server/test_viewer_corpus_coverage.py`

#### 8.2 `GET /api/corpus/persons/top`

**Priority:** MEDIUM. Enables Intelligence tab Top Voices section.
Intelligence tab degrades gracefully without it.

Partial GI artifact scan: reads Person node counts from loaded GI
artifacts, aggregates by person canonical id, returns top N by
insight count.

**Response schema:**

```python

class TopPersonItem(BaseModel):
    person_id: str            # "person:{slug}"
    display_name: str
    episode_count: int
    insight_count: int
    top_topics: List[str]     # canonical topic ids, max 3

class CorpusTopPersonsResponse(BaseModel):
    persons: List[TopPersonItem]
    total_persons: int        # total distinct persons in corpus

```yaml

**Implementation:** Requires reading GI artifacts. Two options:
(a) scan all `*.gi.json` files, extract Person node counts — full scan,
slower; (b) use the loaded graph in memory if available — faster but
state-dependent. Prefer (a) for correctness — this is a background
fetch, latency is acceptable.

**Test:** `tests/unit/podcast_scraper/server/test_viewer_corpus_persons.py`

---

### 9. Open Questions

These are not blocking Phase 1–2 but need resolution before the
implementation is complete.

**Stage timings granularity:** The current `CorpusRunSummaryItem`
schema provides "compact metrics per run." Verify whether it includes
per-stage timing breakdown or only total duration. If stage timings
are only in the full `run.json` file and not in the compact summary,
a `GET /api/corpus/runs/{run_id}` detail endpoint may be needed for
the stage timings chart. Check the existing `corpus_run_summary.json`
schema before adding a new endpoint.

**Feed per-run breakdown:** `FeedRunHistoryGrid` requires per-feed
success/failure per run. Verify whether `CorpusRunSummaryItem` includes
per-feed breakdown. If not, this chart silently hides (it is conditional
on multi-feed corpus anyway) and is tracked as a follow-up.

**`GET /api/corpus/coverage` performance:** If the corpus has thousands
of episodes, the existence-check scan may be slow. Add a
`Cache-Control: max-age=60` header and consider a lightweight cache
in the route handler (invalidated on corpus path change). Profile with
a real large corpus during implementation.

---

### 10. Files to Touch

#### New components

```text
web/gi-kg-viewer/src/components/dashboard/ArtifactActivityChart.vue
web/gi-kg-viewer/src/components/dashboard/FeedRunHistoryGrid.vue
web/gi-kg-viewer/src/components/dashboard/IntelligenceSnapshot.vue
web/gi-kg-viewer/src/components/dashboard/TopicLandscape.vue
web/gi-kg-viewer/src/components/dashboard/TopVoices.vue
web/gi-kg-viewer/src/components/dashboard/TopicMomentum.vue
web/gi-kg-viewer/src/components/dashboard/EmergingConnections.vue
```

#### Modified components

```text
web/gi-kg-viewer/src/components/dashboard/DashboardView.vue
  — briefing card always rendered above tabs

web/gi-kg-viewer/src/utils/chartRegister.ts
  — global Chart.js defaults (no legends, no gridlines)
  — endLabelPlugin registered globally

web/gi-kg-viewer/src/utils/artifactMtimeBuckets.ts
  — extend to produce daily new-artifact counts (not just cumulative)
```

#### New API modules

```text
web/gi-kg-viewer/src/api/corpusPersonsApi.ts
  — fetchTopPersons(limit)
```

#### New server routes

```text
src/podcast_scraper/server/schemas.py
  — CorpusCoverageResponse, CoverageByMonthItem, CoverageFeedItem
  — CorpusTopPersonsResponse, TopPersonItem
```

Add coverage and persons/top routes to `app.py`.

#### New server tests

Add or extend integration tests for the new coverage and persons routes
(mirror patterns used for existing corpus API tests).

#### UXS amendments (after implementation)

```text
docs/uxs/UXS-006-dashboard.md
  — Full rewrite (current file is thin; this spec replaces it)
  — Remove API/Data left panel section (moved to status bar)
  — Add briefing card spec
  — Add three-tab structure
  — Add Tufte compliance rules (cross-reference TUFTE_CHART_CRITIQUE.md)

docs/uxs/UXS-001-gi-kg-viewer.md
  — Remove "optional" from insight line rule — mandatory
  — Add to tunable parameters:
      GI coverage warning threshold    50%    Open
      Index coverage warning threshold 60%    Open
      Action items max                 3      Open
      Top voices limit                 5      Open

docs/guides/SERVER_GUIDE.md
  — Add GET /api/corpus/coverage and GET /api/corpus/persons/top
    to the API reference table
```

#### E2E surface map

```text
  — coverage-by-month-chart, feed-coverage-table, artifact-activity-chart
  — index-status-card
  — intelligence-snapshot, intelligence-topic-landscape
  — intelligence-top-voices, intelligence-topic-momentum
  — intelligence-emerging-connections
  — pipeline-run-history-strip, pipeline-run-dot
  — pipeline-duration-trend, pipeline-stage-timings
  — pipeline-episode-outcomes, pipeline-episodes-per-run
  — pipeline-feed-history-grid
```yaml

---

### 11. Implementation Phases

#### Phase 1 — Briefing card + Coverage tab

**Goal:** The two most impactful surfaces. Requires `GET
/api/corpus/coverage` new endpoint.

Steps:

1. Implement `GET /api/corpus/coverage` server route + tests
2. `BriefingCard.vue` — all three sections wired to coverage + runs/summary
3. Coverage tab: `CoverageByMonthChart.vue`, `FeedCoverageTable.vue`
4. Apply global Chart.js Tufte defaults in `chartRegister.ts`
5. `ArtifactActivityChart.vue` (replaces cumulative line)
6. `IndexStatusCard` section (from existing index/stats data)

**Checkpoint:** Briefing card shows last run + corpus health + action
items. Coverage tab shows month chart, feed table, activity chart,
index card. No regressions in existing Pipeline tab.

#### Phase 2 — Pipeline tab restructure

**Goal:** Improved Pipeline tab with Tufte-compliant charts.

Steps:

1. `RunHistoryStrip.vue` with click-to-expand per run
2. `DurationTrendChart.vue` with insight title + reference line
3. Stage timings chart with direct value labels + bottleneck highlight
4. Episode outcomes → inline numbers in run card (remove chart)
5. `EpisodesPerRunChart.vue` (remove cumulative overlay)
6. `FeedRunHistoryGrid.vue` (multi-feed only)

**Checkpoint:** Pipeline tab fully restructured. All Tufte rules
applied. Episode outcomes is no longer a chart.

#### Phase 3 — Intelligence tab

**Goal:** Intelligence tab with available data + graceful degradation.

Steps:

1. `IntelligenceSnapshot.vue` (digest API, existing data)
2. `TopicLandscape.vue` (topic-clusters API, existing data)
3. Implement `GET /api/corpus/persons/top` + `TopVoices.vue`
4. `TopicMomentum.vue` stub with degraded state (enricher not yet
   available)

5. `EmergingConnections.vue` stub with degraded state

**Checkpoint:** Intelligence tab shows snapshot + topic landscape
immediately. Top voices shows when persons/top endpoint ships. Momentum
and connections show degraded states (no enricher prompts).

#### Phase 4 — UXS + E2E updates

1. Rewrite `UXS-006-dashboard.md` to match implementation
2. Update `UXS-001-gi-kg-viewer.md` tunable params + mandatory insight
   line rule

3. Update `SERVER_GUIDE.md` API table
4. Update `E2E_SURFACE_MAP.md` with all new testids
5. Run `make test-ui-e2e`

---

### 12. What This Does Not Change

- Chart.js library itself — same version, same registration pattern
- `indexStats.ts` store — unchanged
- Digest and Library tabs — unchanged
- Subject rail — unchanged
- Token system — all chart colours use existing `--ps-*` variables
- `DashboardOverviewSection.vue` is absorbed into `BriefingCard.vue`
  and can be removed after migration
