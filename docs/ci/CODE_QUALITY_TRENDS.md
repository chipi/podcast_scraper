# Code Quality Trends

This page describes how the project tracks **code quality metrics over time** (complexity and
maintainability trends across git history). This complements the snapshot metrics from
[radon](https://radon.readthedocs.io/) (see [RFC-031](../rfc/RFC-031-code-complexity-analysis-tooling.md))
by adding historical trend analysis. The project uses **radon 5.1.x** (not 6.x) so that
[wily](https://github.com/tonybaloney/wily) can be installed in the same environment.

## Overview

**What we track:**

- **Cyclomatic complexity** — How complex the code is (branching paths). Lower is better.
- **Maintainability index** — How easy the code is to maintain. Higher is better (0–100 scale).
- **Trends over commits** — Whether metrics are improving or degrading compared to previous commits.

**Tool:** [wily](https://github.com/tonybaloney/wily) builds a baseline from git history and
reports complexity/maintainability over time. Radon provides the current snapshot; wily provides
the trend.

**Note:** Wily tracks **code quality** only. It does not track test execution time, CI duration,
or system performance. For performance tracking, see `make test-track` and
[Resource Usage](RESOURCE_USAGE.md).

## Local Usage

### Prerequisites

- Full git history (no shallow clone). For CI we use `fetch-depth: 0`.
- Dev dependencies installed (includes `wily` and `radon`). For local runs that match CI and
  dashboard generation, install with LLM extras too: `pip install -e ".[dev,llm]"`.

### Build baseline and view trends

```bash
# Build wily baseline (last 50 commits by default)
make complexity-track
```

This runs:

1. `wily build src/podcast_scraper --max-revisions 50` — Builds the baseline in `.wily/`.
2. `wily report src/podcast_scraper/` — Prints overall package trends.

### View trends for a specific file

```bash
# Trends for a single module
wily report src/podcast_scraper/workflow/orchestration.py

# Compare two commits
wily diff HEAD~5 HEAD src/podcast_scraper/workflow/orchestration.py
```

### Example questions you can answer

- "Did my refactor actually reduce complexity?"
- "Which files are getting harder to maintain over time?"
- "When did `orchestration.py` become so complex?"
- "Compare complexity before vs after a major refactoring."

## Interpreting metrics

### Cyclomatic complexity

- **1–5** — Simple, low risk.
- **6–10** — Moderate; consider simplifying.
- **11+** — High; refactoring recommended.

Trend: **↑** (increasing) may indicate growing complexity; **↓** (decreasing) is usually good.

### Maintainability index

- **20–100** — Good (A/B).
- **10–19** — Moderate (C).
- **0–9** — Poor (D/E); hard to maintain.

Trend: **↓** (decreasing) may indicate maintainability degradation; **↑** (increasing) is good.

### When to be concerned

- Complexity trend **+0.5 or more** over a few commits for a critical file.
- Maintainability **dropping by 2+ points** over recent commits.
- Files repeatedly appearing in **files_degrading** in the metrics dashboard.

### Low-MI modules (accepted)

The following modules have **maintainability index (MI) in the C (0–9) or B (10–19) range** and are
documented here rather than refactored, per [Issue #432](https://github.com/chipi/podcast_scraper/issues/432) Phase 4. New low-MI code should be avoided where practical; existing exceptions are reviewed periodically.

| Module | MI (rank) | Reason accepted |
| ------ | -------- | ---------------- |
| `config.py` | 0 (C) | Large Pydantic config with many fields and validators; structure is intentional. |
| `cli.py` | 0 (C) | CLI entrypoint with many subcommands and options; split would fragment UX. |
| `providers/ml/summarizer.py` | 0 (C) | ML summarization pipeline (model load, map/reduce); complexity is domain-inherent. |
| `workflow/metadata_generation.py` | 0 (C) | Metadata rules and branching; candidate for future extraction of smaller helpers. |
| `providers/ml/ml_provider.py` | 8.3 (C) | ML provider orchestration; bridges workflow and summarizer/speaker detection. |
| `workflow/orchestration.py` | 12.5 (B) | Orchestration dispatcher; many branches by design (stage routing, error handling). |

### C901 complexity exceptions (per-file-ignores)

The following modules are listed in `.flake8` under `per-file-ignores` with **C901** (McCabe complexity)
so that flake8 does not fail on high-complexity functions. Per [Issue #432](https://github.com/chipi/podcast_scraper/issues/432) Phase 5, each exception is documented here. Radon grades: F = highest complexity, D/C/B/A = decreasing.

| File / pattern | Worst radon (example) | Reason accepted |
| -------------- | -------------------- | ---------------- |
| `config.py` | Config._validate_cross_field_settings D (24) | Large Pydantic config; cross-field and env validators by design. |
| `episode_processor.py` | download_media_for_transcription E (31) | Episode pipeline: download, transcribe, cache; many branches by design. |
| `workflow.py` | (legacy; see orchestration) | Superseded by `workflow/orchestration.py`; keep ignore for any remaining ref. |
| `speaker_detection.py` | detect_speaker_names D (24) | NER + heuristics; host/guest detection with many rules. |
| `whisper_integration.py` | (see whisper_utils / ml_provider) | Whisper wiring; complexity in providers/ml. |
| `*summarizer.py` | summarize_long_text F (44), create_summarization_provider F (62) | ML pipeline (map/reduce, chunking); factory branches over providers. |
| `*metadata_generation.py` | (many C/D) | Metadata rules and branching; see Low-MI table. |
| `*orchestration.py` | _both_providers_use_mps D (29), run_pipeline entry | Stage routing, device/logging setup; see Low-MI table. |
| `*processing.py` | prepare_episode_download_args E (39) | Episode processing concurrency and host detection. |
| `*run_index.py` | _find_metadata_file C (14) | Run index and status mapping. |
| `cli.py` | _build_config F (87), main F (49) | CLI entrypoint and config building; see Low-MI table. |

New C901 ignores should be avoided; add a row here and a short in-file comment when necessary.

## CI integration

### When wily runs

- **Branch:** Only on `main`, `release/2.4`, and `release/2.5` (not on pull requests).
- **Job:** In the `coverage-unified` job, after code quality metrics (radon) are generated.
- **Full history:** The job checks out with `fetch-depth: 0` so wily can build the baseline.

### What CI produces

- **Wily baseline** — Stored in `.wily/` and uploaded as an artifact (e.g. `wily-reports`).
- **Trend reports** — Text reports under `reports/wily/` (e.g. overall and key-file trends).
- **Trend data for dashboard** — `reports/wily/trends.json` (used by the metrics pipeline to show
  complexity/maintainability trends and files degrading/improving).

### Viewing CI results

- **Metrics dashboard:** [Unified Metrics Dashboard](METRICS.md) shows complexity and maintainability
  with trend indicators (↑/↓) and lists of files degrading/improving when wily data is available.
- **Artifacts:** In the GitHub Actions run, download the `wily-reports` artifact to inspect
  `.wily/` and `reports/wily/*.txt` locally.

Trends appear in the dashboard after at least two CI runs (first run builds the baseline; second
run can compute deltas).

## Troubleshooting

### "No revisions found" or empty report

- Ensure the repo is not a shallow clone: `git fetch --unshallow` (or in CI use `fetch-depth: 0`).
- Run `wily build src/podcast_scraper` first; then `wily report`.

### Baseline corrupted or invalid

Reset and rebuild:

```bash
rm -rf .wily/
wily build src/podcast_scraper --max-revisions 50
```

### Wily fails in CI

Wily steps are **non-blocking** (`continue-on-error: true`). If wily fails, the rest of CI still
runs; the metrics dashboard will show "N/A" for trend fields until wily succeeds in a later run.

### Trends show "N/A" on dashboard

- First run after enabling wily: no previous revision to compare; trends appear from the next run.
- Wily job failed or was skipped: check the workflow run for the `Build wily baseline` step.

## Related documentation

- [RFC-031: Code Complexity Analysis Tooling](../rfc/RFC-031-code-complexity-analysis-tooling.md) —
  Radon integration and snapshot metrics.
- [CI Metrics](METRICS.md) — Dashboard and alert thresholds.
- [RFC-025: Test Metrics and Health Tracking](../rfc/RFC-025-test-metrics-and-health-tracking.md) —
  Metrics pipeline and history.
