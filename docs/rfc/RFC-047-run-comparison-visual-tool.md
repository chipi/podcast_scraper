# RFC-047: Lightweight Run Comparison & Diagnostics Tool

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: ML evaluation team, Core developers
- **Related RFCs**:
  - `docs/rfc/RFC-015-ai-experiment-pipeline.md` (AI Experiment Pipeline)
  - `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` (Benchmarking Framework)
  - `docs/rfc/RFC-045-ml-model-optimization-guide.md` (ML Model Optimization)
  - `docs/rfc/RFC-046-materialization-architecture.md` (Materialization Architecture)

## Abstract

This RFC proposes a **lightweight run comparison and diagnostics tool** - a perfect place for a small "run compare" tool that stays tiny while being incredibly useful. The goal isn't a full dashboard; it's a one-page "what changed?" view that answers, in 30 seconds: Is this run better or worse than baseline? Why (length, gates, repetition, latency, map/reduce starvation)? Which episodes regressed?

As the evaluation framework matures with frequent experiments across different models, parameters, and preprocessing contracts, it becomes slow and error-prone to understand whether a new run is actually better than a baseline. This tool provides a focused, one-page visual interface that makes comparisons fast and actionable.

**Architecture Alignment:** This tool operates entirely on existing evaluation artifacts (`metrics.json`, `predictions.jsonl`, `fingerprint.json`, and optionally `run_summary.json`, `diagnostics.jsonl`) and requires no backend services or infrastructure changes. It is explicitly *not* a full dashboard, but a focused diagnostic tool (~150-300 LOC).

## Problem Statement

As the ML evaluation framework matures, we now run frequent experiments across:

- Different models (small vs large, ML vs LLM)
- Different decoding parameters
- Different preprocessing/materialization contracts

While `metrics.json`, `predictions.jsonl`, and logs provide full fidelity, **it is currently slow and error-prone to understand whether a new run is actually better than a baseline**.

**Current Pain Points:**

1. **Slow diagnosis**: Comparing runs requires manually reading multiple JSON files and logs
2. **Error-prone**: Easy to miss regressions or improvements when scanning raw data
3. **No visual context**: Hard to spot patterns (compression issues, truncation, outliers)
4. **Root cause unclear**: Difficult to determine if issues are in map, reduce, or preprocessing

**Impact of Not Solving This:**

- Baseline promotion decisions take too long and are less confident
- Regressions go undetected or are discovered late
- Architectural decisions lack clear visual evidence
- Developer velocity suffers from manual data analysis

**Use Cases:**

1. **Debugging regressions**: A new config produces worse summaries - quickly identify whether the problem is map compression, reduce starvation, truncation, or preprocessing drift
2. **Baseline promotion decisions**: Compare candidate run vs frozen baseline, verify gates are clean, verify output length and coherence improved
3. **Model comparison**: Compare small vs large models, ML vs OpenAI, map/reduce architecture variants

## Goals

1. **Fast diagnosis**: Answer "is this run better or worse?" in 30 seconds
2. **Root cause visibility**: Surface root causes (length, gates, repetition, latency, map/reduce starvation)
3. **Low maintenance**: Keep it tiny (~150-300 LOC) while being incredibly useful
4. **Artifact-based**: Work entirely on existing run artifacts (no new data collection)
5. **No infrastructure**: Require no backend services or persistent storage
6. **One-page focus**: Not a full dashboard - focused "what changed?" view

## Constraints & Assumptions

**Constraints:**

- Must work with existing artifacts in `data/eval/runs/`, `data/eval/baselines/`, `data/eval/references/`
- Must not require database or backend services
- Must be fast to load and render (< 5 seconds for typical dataset)
- Must support comparing 2-N runs simultaneously

**Assumptions:**

- Existing artifacts (`metrics.json`, `predictions.jsonl`, `fingerprint.json`) contain sufficient data
- Optional artifacts (`run_summary.json`, `diagnostics.jsonl`) can be added for performance
- Static HTML or lightweight web framework is sufficient (no need for real-time updates)
- Tool is used by developers, not end users (can require local setup)

## Design & Implementation

### 1. Core UI (One-Page Layout)

The tool provides a focused, single-page "what changed?" view that answers key questions in 30 seconds:

- Is this run better or worse than baseline?
- Why? (length, gates, repetition, latency, map/reduce starvation)
- Which episodes regressed?

#### 1.1 Run Selector

- **Auto-discover on startup**: Tool scans `data/eval/runs/`, `data/eval/baselines/`, `data/eval/references/` and shows all available runs
- **Filter by type**: User can filter to show only runs, only baselines, only references, or all
- **Availability indicators**: Tool shows which artifacts are present for each run (e.g., "✓ has diagnostics.jsonl", "⚠ missing run_summary.json")
- **Select 2–N runs**: User chooses which runs/baselines/references to compare (any combination)
- **User designates baseline**: User explicitly selects which run is the baseline (others are candidates)
- **Flexible comparison**: Compare anything vs anything - runs vs baselines, baselines vs references, etc.
- **Support multiple candidates**: Compare one baseline against multiple candidate runs

#### 1.2 Summary KPI Tiles (Top Row)

For each run, display compact tiles with:

- ✅ **Success rate** (episodes processed / total episodes)
- ⛔ **Failed episodes count**
- 📏 **Avg output tokens**
- ⏱ **Avg latency per episode** (seconds)
- 🧹 **Gate failures** (speaker label, truncation, boilerplate counts)

**Purpose**: Immediate "safe or broken" signal

#### 1.3 Delta Table (Baseline vs Candidate)

A side-by-side comparison table with colored deltas:

| Metric | Baseline | Candidate | Δ |
| --- | --- | --- | --- |
| Avg output tokens | 470 | 190 | −280 (red) |
| Truncation rate | 0.0 | 0.2 | +0.2 (red) |
| Avg latency (s) | 40 | 62 | +22 (yellow) |
| Speaker label leak | 0.0 | 0.0 | 0.0 (green) |

**Purpose**: Fastest way to reason about changes - make tradeoffs explicit

#### 1.4 Three High-Value Charts

**Chart A — Output Tokens Distribution**

- Box plot or histogram per run
- **Purpose**: Instantly shows if "fatter map" worked (final outputs got longer)
- **Reveals**: Compression issues, outliers (e.g., one episode still tiny)

**Chart B — Latency vs Output Length**

- Scatter plot: x=latency (seconds), y=output tokens (one dot per episode)
- **Purpose**: Shows "am I paying 2× time for no gain?"
- **Reveals**: Cost vs quality tradeoff, efficiency issues

**Chart C — Map/Reduce Starvation Diagnostics** (if `diagnostics.jsonl` available)

- Per-episode grouped bars showing:
  - Avg map summary tokens
  - Reduce input tokens/chars
  - Final output tokens
- **Purpose**: Best chart for diagnosing current problems
- **Reveals at a glance**:
  - Map is too compressive (map tokens too low)
  - Reduce input is too small (reduce input too low)
  - Final output is capped (final tokens hit max)

**Note**: This is the most valuable chart for debugging map/reduce architecture issues.

#### 1.5 Episode Drill-Down (Killer Feature)

Click an episode ID to show:

- **Side-by-side comparison**: Baseline summary vs candidate summary with **diff highlighting**
  - Highlight differences between baseline and candidate
  - Show additions (green), deletions (red), or changes (yellow)
- **Token counts**: Input, map output, reduce input, final output
- **Gate flags**: Truncation, speaker label leak, boilerplate leak
- **Failed episodes**: If episode failed completely, show error message in red, highlight in episode list

**Purpose**: Fast qualitative evaluation - see actual output quality differences with visual diff highlighting

**Note**: Failed episodes are highlighted in red in the episode list and show error messages in drill-down view instead of summaries.

### 2. Input Artifacts

The tool operates on existing artifacts in:

```text
data/eval/runs/<run_id>/
data/eval/baselines/<baseline_id>/
data/eval/references/<reference_id>/
```

**Required files:**

- `metrics.json` - Aggregated metrics and gate failures
- `predictions.jsonl` - Per-episode predictions and outputs
- `fingerprint.json` - System fingerprint and configuration

**Optional (enhanced features):**

- `run_summary.json` - Pre-computed aggregates (if missing, tool computes on-the-fly)
- `diagnostics.jsonl` - Map/reduce diagnostic stats per episode (if missing, Chart C is hidden)

**Tool Behavior:**
The tool adapts to what's available:

- If `run_summary.json` exists → use it (fast)
- If `run_summary.json` missing → compute aggregates from `metrics.json` and `predictions.jsonl` (slower but works)
- If `diagnostics.jsonl` exists → show Chart C (Map/Reduce Diagnostics)
- If `diagnostics.jsonl` missing → hide Chart C, show other charts

### 3. Data Model Enhancements (Optional but Recommended)

#### 3.1 `run_summary.json`

A tiny, structured summary file written by the experiment runner to avoid repeatedly recomputing aggregates. This makes the compare tool trivial to implement:

```json
{
  "run_id": "baseline_ml_dev_authority_smoke_v1",
  "dataset_id": "curated_5feeds_smoke_v1",
  "materialization_id": "summarization_canonical_v1",
  "models": {
    "map": "bart-base",
    "reduce": "led-base"
  },
  "avg_output_tokens": 470.4,
  "avg_latency_s": 40.2,
  "gate_failures": {
    "truncation": 0,
    "speaker_label": 0,
    "boilerplate": 0
  }
}
```

**Location**: `data/eval/runs/<run_id>/run_summary.json`

**Generation**: Written automatically by experiment runner after metrics computation.

#### 3.2 `diagnostics.jsonl`

Per-episode diagnostic stats for deep root-cause analysis. Enables Chart C (Map/Reduce Starvation Diagnostics) without parsing logs:

```json
{"episode_id": "p01_e01", "chunks": 6, "avg_map_tokens": 180, "reduce_input_tokens": 980, "reduce_input_chars": 4500, "final_tokens": 520}
```

**Location**: `data/eval/runs/<run_id>/diagnostics.jsonl`

**Generation**: Written by experiment runner during inference (optional but recommended for ML model runs).

**Note**: If diagnostics are missing, Chart C is skipped (tool still works with other charts).

### 4. Implementation Options

#### Option A — Streamlit (Recommended)

**Description**: Single Python app using Streamlit framework (~300-500 LOC)

**Pros:**

- Fastest to build
- Excellent UX for exploration
- Interactive widgets and filtering
- Perfect for developer workflow

**Cons:**

- Adds dependencies (`streamlit`, `plotly`)
- Requires running server locally (`streamlit run app.py`)

**Chart Library**: Plotly (via `st.plotly_chart()`)

- Rich interactivity (zoom, pan, hover tooltips)
- Professional-looking charts
- Better for complex visualizations (scatter plots, grouped bars)

**Implementation:**

```python
# tools/run_compare/app.py
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Load run_summary.json and predictions.jsonl
# Render KPI tiles, delta table, 3 Plotly charts, episode drill-down
```

**Usage:**

**Manual (flexible):**

```bash
cd tools/run_compare
streamlit run app.py
# Opens browser with run selector - user chooses what to compare
```

**Make task (convenience):**

```bash
make run-compare BASELINE=baseline_ml_dev_authority CANDIDATES="run1 run2 run3"
# Launches Streamlit with pre-selected runs (user can still change selection)
```

#### Option B — Static HTML Generator

**Description**: Python script outputs `compare.html` using Plotly offline

**Pros:**

- No server required
- Shareable artifact (can commit to repo)
- No runtime dependencies

**Cons:**

- Less interactive
- Must regenerate on changes

#### Option C — Markdown / CLI

**Description**: Generates `compare.md` with tables and basic stats

**Pros:**

- Minimal dependencies
- Git-friendly
- Easy to review in PRs

**Cons:**

- Limited visualization
- No interactivity

**Recommendation**: Start with Option A (Streamlit) - fastest to build, great UX, perfect for developer workflow. Can evolve to Option B (static HTML) later if shareability becomes important.

### 5. Proposed Location in Repo

```text
tools/run_compare/
  app.py              # Main Streamlit app (~300-500 LOC)
  README.md           # Usage instructions
```

**Minimal structure**: Single file is sufficient for v1 (< 500 LOC). Can split into modules later if needed.

**Large Dataset Handling:**

- For datasets > 20 episodes, episode drill-down uses pagination
- Charts show all episodes but with aggregated tooltips to avoid overcrowding
- Episode selector supports filtering/search for large lists

**Data access**: Tool reads directly from:

- `data/eval/runs/<run_id>/`
- `data/eval/baselines/<baseline_id>/`
- `data/eval/references/<reference_id>/`

**Make task integration**: Add to `Makefile`:

```makefile
run-compare:
 @echo "Launching run comparison tool..."
 @BASELINE=$(BASELINE) CANDIDATES="$(CANDIDATES)" \
  streamlit run tools/run_compare/app.py \
  --server.headless=false \
  --server.port=8501
```

**Python code reads environment variables:**

```python
import os
baseline = os.environ.get("BASELINE")
candidates = os.environ.get("CANDIDATES", "").split() if os.environ.get("CANDIDATES") else []
```

## Key Decisions

1. **One-Page Layout**
   - **Decision**: Focused, single-page interface (not a full dashboard)
   - **Rationale**: Fast to build, easy to maintain, clear purpose

2. **Artifact-Based (No New Data Collection)**
   - **Decision**: Tool operates on existing evaluation artifacts
   - **Rationale**: No infrastructure changes, works with current system

3. **Optional Data Model Enhancements**
   - **Decision**: `run_summary.json` and `diagnostics.jsonl` are optional
   - **Rationale**: Tool works with existing artifacts, enhancements improve performance

4. **Streamlit as Initial Implementation**
   - **Decision**: Use Streamlit for fast iteration
   - **Rationale**: Fast to build, excellent UX, can evolve to static HTML later

## Alternatives Considered

1. **Full Dashboard (Grafana, Superset, etc.)**
   - **Description**: Production-grade dashboard with time-series tracking
   - **Pros**: Powerful, scalable, persistent storage
   - **Cons**: Heavy infrastructure, overkill for current needs
   - **Why Rejected**: Tool should be lightweight and low-maintenance

2. **Jupyter Notebook**
   - **Description**: Interactive notebook for analysis
   - **Pros**: Flexible, familiar to data scientists
   - **Cons**: Not shareable, requires manual execution
   - **Why Rejected**: Need a tool that's always ready, not a notebook

3. **CLI Tool Only**
   - **Description**: Command-line tool that prints tables
   - **Pros**: Minimal, no dependencies
   - **Cons**: Limited visualization, harder to spot patterns
   - **Why Rejected**: Visual comparison is key value proposition

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Data loading and parsing from artifacts
- **Integration tests**: Full tool execution with sample runs
- **Visual regression**: Screenshot tests for charts (if using static HTML)

**Test Organization:**

- Location: `tests/integration/tools/test_run_compare.py`
- Fixtures: Sample run artifacts in `tests/fixtures/eval_runs/`

**Test Execution:**

- Integration tests run in CI-full (requires sample artifacts)
- Unit tests run in CI-fast

## Rollout & Monitoring

**Rollout Plan:**

1. **Phase 1**: Implement Streamlit app with core features (KPI tiles, delta table, basic charts)
2. **Phase 2**: Add episode drill-down and map/reduce diagnostics
3. **Phase 3**: Add optional `run_summary.json` and `diagnostics.jsonl` generation to experiment runner
4. **Phase 4**: Document usage and add to developer workflow

**Monitoring:**

- Track tool usage (if possible via analytics)
- Collect feedback on missing features
- Monitor performance (load time, render time)

**Success Criteria:**

1. ✅ A regression can be diagnosed in < 1 minute
2. ✅ Baseline promotion decisions are faster and more confident
3. ✅ Map vs reduce issues are visually obvious
4. ✅ The tool remains small and maintainable (< 500 LOC)

## Relationship to Other RFCs

This RFC (RFC-047) complements the evaluation infrastructure:

1. **RFC-015: AI Experiment Pipeline** - Defines experiment config structure and artifacts; this RFC visualizes those artifacts
2. **RFC-041: Benchmarking Framework** - Defines metrics and comparison; this RFC makes comparisons visual and fast
3. **RFC-045: ML Model Optimization** - Documents preprocessing impact; this RFC helps diagnose optimization results
4. **RFC-046: Materialization Architecture** - Formalizes preprocessing; this RFC helps compare materialization variants

**Key Distinction:**

- **RFC-041**: Defines *what* metrics to collect and *how* to compare
- **RFC-047**: Provides *visual tool* to *quickly understand* comparisons

Together, these RFCs provide a complete evaluation workflow: run experiments → collect metrics → visualize and diagnose.

## Benefits

1. **Fast diagnosis**: Regressions identified in < 1 minute instead of manual log parsing
2. **Confident decisions**: Baseline promotion decisions are faster and more data-driven
3. **Root cause visibility**: Map vs reduce issues are visually obvious
4. **Developer velocity**: Less time spent on manual data analysis
5. **Low maintenance**: Small, focused tool that's easy to evolve

## Migration Path

1. **Phase 1**: Implement tool with existing artifacts (no changes to experiment runner)
2. **Phase 2**: Add optional `run_summary.json` generation to experiment runner (performance optimization)
3. **Phase 3**: Add optional `diagnostics.jsonl` generation for map/reduce diagnostics
4. **Phase 4**: Document usage in `docs/guides/EXPERIMENT_GUIDE.md`

## Open Questions

1. **Should the tool support time-series tracking across many runs?**
   - Proposed: No, keep it focused on 2-N run comparison (out of scope for v1)

2. **Should the tool generate shareable reports?**
   - Proposed: Yes, export to HTML/PDF for sharing (future enhancement)

3. **Should the tool integrate with CI/CD?**
   - Proposed: No, keep it as a local developer tool (out of scope)

## References

- **Related RFC**: `docs/rfc/RFC-015-ai-experiment-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md`
- **Related RFC**: `docs/rfc/RFC-045-ml-model-optimization-guide.md`
- **Related RFC**: `docs/rfc/RFC-046-materialization-architecture.md`
- **Source Code**: `scripts/eval/run_experiment.py` (generates artifacts)
- **Source Code**: `src/podcast_scraper/evaluation/scorer.py` (generates metrics.json)
