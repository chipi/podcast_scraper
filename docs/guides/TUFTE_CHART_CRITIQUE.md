# Tufte Chart Critique Guide

A sparring partner for data visualization. When asked to critique a chart, score it against these principles and return specific, actionable fixes — preferably as code.

---

## How to Use This File

In Cursor: `@docs/guides/TUFTE_CHART_CRITIQUE.md` or `@TUFTE_CHART_CRITIQUE.md`, then describe or paste your chart code.

Ask things like:

- *"Critique this chart against the Tufte rubric"*
- *"Score this and show me the fixed version"*
- *"Quick Tufte check — what's the worst offender here?"*

---

## The 7 Principles (with scoring)

Each principle scores **Pass / Warn / Fail**. At the end, give an overall verdict and a prioritized fix list.

---

### 1. Data-Ink Ratio

> Every drop of ink should serve the data. Remove anything that doesn't.

**Check for:**

- Grid lines (especially heavy ones) — use faint or remove entirely
- Chart borders / spines (top, right axes rarely needed)
- Tick marks pointing outward (inward or none is cleaner)
- Background fills, panel shading
- Redundant axis labels (if units are in the title, drop them from ticks)

**Pass:** Ink is nearly all data. Minimal or no decoration.
**Warn:** Some decorative elements present but not overwhelming.
**Fail:** Gridlines, borders, background, AND a legend all competing with the data.

---

### 2. Direct Labeling (Kill the Legend)

> Labels belong on the data, not in a separate key the eye must travel to.

**Check for:**

- Legend boxes that require eye travel — replace with end-of-line labels or callouts
- Any chart with ≤5 series has no excuse for a legend
- Color-only differentiation for colorblind users (add shape or position too)

**Pass:** Every series labeled directly at its endpoint or nearest natural position.
**Warn:** Legend present but data is simple enough to label directly.
**Fail:** Legend box with 3+ colors and no direct labels anywhere.

**Fix pattern (matplotlib):**

```python
# Instead of plt.legend()
for line, label in zip(lines, labels):
    ax.annotate(label, xy=(x[-1], line.get_ydata()[-1]),
                xytext=(5, 0), textcoords='offset points', va='center')
```

---

### 3. Show the Data (Don't Summarize Away Variation)

> Aggregates hide stories. Show distributions, not just means.

**Check for:**

- Bar charts showing only means — where are the error bars, distributions, or individual points?
- Pie charts (almost always wrong — use a bar chart or just a table)
- Smoothed lines that erase meaningful spikes
- Truncated Y-axis that exaggerates or minimizes variation

**Pass:** Raw data visible, or aggregation is clearly justified and labeled.
**Warn:** Summarized but context/uncertainty shown.
**Fail:** Means-only bars, no variance, truncated axis with no callout.

---

### 4. Annotate the Insight

> A chart without a stated insight is a puzzle, not a communication.

**Check for:**

- Is the key takeaway surfaced in the title or subtitle? (not "Sales by Quarter" but "Q3 reversed a 3-quarter decline")
- Are notable points annotated? (peaks, inflections, anomalies)
- Is there a reference line for context? (target, average, prior year)

**Pass:** Subtitle states the insight. Key data points annotated.
**Warn:** Title is neutral/descriptive but an annotation saves it.
**Fail:** Title is a variable name. No annotations. Reader must find the story themselves.

**Fix pattern:**

```python
ax.set_title("Field", fontsize=13)
ax.set_subtitle("Key insight stated here in plain language", fontsize=10, color='#555')
ax.annotate('Peak: 37.1% (1983)', xy=(1983, 37.1),
            xytext=(1975, 40), arrowprops=dict(arrowstyle='->', color='#333'))
```

---

### 5. No Chartjunk

> Decoration is noise. Noise obscures signal.

**Immediate fails:**

- 3D effects on any 2D data
- Drop shadows on bars or lines
- Gradient fills on bars
- Decorative icons or clip art
- Exploded pie slices
- More than ~5 colors in a palette

**Pass:** No decoration. Color serves only to distinguish categories.
**Warn:** Minor decoration (slight gradient, mild shadow) but data still readable.
**Fail:** 3D bars, gradient fills, drop shadows, heavy gridlines — all at once.

---

### 6. Lie Factor

> The visual representation of a number should match its actual magnitude.

**Check for:**

- Y-axis not starting at zero for bar charts (misleading area)
- Bubble sizes not scaled to area (scaled to radius = 2× visual lie)
- Dual Y-axes with different scales implying false correlation
- Chopped time axes that hide long-term context

**Pass:** Visual magnitude proportional to data. Truncation clearly labeled if used.
**Warn:** Non-zero baseline on line chart (acceptable) but not called out.
**Fail:** Bar chart from 95–100%, making a 1% difference look like 5×.

**Lie Factor formula:** `(visual size change) / (data value change)` — should be ~1.0

---

### 7. Small Multiples Over Animation / Toggling

> Show comparisons side-by-side, not sequentially.

**Check for:**

- Interactive filters hiding data that could sit alongside
- Animated transitions between states that should be compared
- Subgroup overlaps that would be clearer as separate panels

**Pass:** Comparisons are spatial (side-by-side panels).
**Warn:** Single chart is fine but a small multiple would reveal more.
**Fail:** "Click to switch between X and Y" when X and Y should be shown together.

---

## Critique Output Format

When critiquing a chart, return this structure:

```text
## Tufte Critique

| Principle           | Score | Notes                              |
|---------------------|-------|------------------------------------|
| Data-Ink Ratio      | | Clean, minimal spines              |
| Direct Labeling     | | Legend box with 4 colors           |
| Show the Data       | | Means shown, no variance           |
| Annotate Insight    | | Title is "Revenue". No subtitle.   |
| No Chartjunk        | | No decoration                      |
| Lie Factor          | | Axis starts at 0                   |
| Small Multiples     | | Could benefit from faceting        |

**Verdict:** Structurally sound, but this chart doesn't communicate — it just displays.

## Priority Fixes
1. [CRITICAL] Replace legend with direct end-labels → code below
2. [HIGH] Rewrite title as insight: "Revenue declined 3 quarters before Q3 recovery"
3. [MEDIUM] Annotate the Q3 inflection point with value + date
4. [LOW] Remove top and right spines

## Fixed Code
[paste corrected snippet here]
```

---

## Quick Reference Cheatsheet

| Never | Instead |
| -------- | ---------- |
| Legend box | Direct end-labels |
| 3D bars | Flat bars |
| "Sales by Quarter" title | "Q3 reversed a 3-quarter decline" |
| Gradient fills | Solid colors |
| Dual Y-axis | Normalize or use small multiples |
| Bar chart from 95% | Start at 0 or use dot plot |
| Pie chart | Bar chart or table |
| 8+ colors | ≤5 colors + direct labels |
| Heavy gridlines | No lines or hairline gray |
| Outward tick marks | Inward or none |

---

## Tone for Critiques

Be direct. Tufte himself is blunt. Don't soften "this chart fails" into "this chart could potentially be improved in some areas." Name what's wrong, explain why it harms comprehension, and show the fix.

A chart that correctly plots data but fails to communicate is still a failed chart.
