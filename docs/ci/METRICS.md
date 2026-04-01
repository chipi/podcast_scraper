# CI & Code Quality Metrics

This project publishes a **small public dashboard** on GitHub Pages plus deeper **local / CI
artifact** workflows for code quality. The dashboard is meant for **at-a-glance health** (latest
run + a thin history), not a full observability product.

**Companion:** [Code quality trends](CODE_QUALITY_TRENDS.md) covers **wily** and **radon** over git
history (different from the dashboard’s per-snapshot radon chart).

## [Unified metrics dashboard](https://chipi.github.io/podcast_scraper/metrics/)

Single HTML page with a **data source** selector:

- **CI metrics** — Last metrics bundle produced when **main** or configured **release** branches
  run the unified coverage / metrics job (`python-app.yml`).
- **Nightly metrics** — Same layout, fed by the scheduled **nightly** workflow (`nightly.yml`).

### What you actually see

| Area | Content |
| ---- | ------- |
| **Header** | Last update time, commit short SHA, branch, link to the **GitHub Actions run** (when `workflow_run` URL is present in JSON). |
| **Alerts** | Rule-based messages from `generate_metrics.py` when enough history exists (runtime, coverage, test count, flaky count, complexity, etc.). |
| **Summary cards** | Single-run snapshot: test counts, pass rate, pytest wall time, combined **line** coverage vs threshold, **radon** package averages (complexity, maintainability), interrogate docstring %, vulture/codespell counts, flaky count, optional **sample pipeline** timings when collection succeeds. |
| **Test run history** (chart) | **Only if `history-*.jsonl` has two or more snapshots:** pytest duration + line coverage % across successive CI/nightly metrics deploys (one appended row per run). See the dashboard subtitle for how `history-*.jsonl` is built. |
| **Code quality history** (chart) | Same <code>history-*.jsonl</code> as test run history; mean cyclomatic complexity + mean maintainability index from radon per snapshot. See the dashboard subtitle (not wily; card-only fields listed there). |
| **Pipeline chart** | Shown only when historical snapshots include pipeline metrics (`collect_pipeline_metrics.py`). |
| **Tables** | Top slowest tests and flaky tests **for the latest run only**. **Slowest:** combined from pytest JSON **and** all `junit*.xml` in `reports/` (dedupe by name, max duration); CI/nightly workflows emit JUnit per job so xdist-sparse JSON does not cap the list. **Flaky:** aggregated across `pytest.json` **and** every `pytest-*.json` shard (same `nodeid` keeps any “passed after rerun” signal). **Flaky** means passed after **pytest-rerunfailures** retry; `pytest-json-report` uses top-level `outcome: rerun` with `call.outcome: passed` (or legacy `rerun: true`). |

### What is *not* on this dashboard

- **No LLM / API usage charts** — Not part of the published JSON or HTML (remove any expectation of
  token or cost graphs here).
- **No flaky-test *trend* line** — Flaky count is on the **cards** and in **alerts**, not plotted
  over time (would need an explicit schema change to add).
- **No wily / per-file git history** — The “code quality” chart is **radon averages for that CI
  snapshot**, not wily’s multi-commit report. For that, use **`make complexity-track`** locally or
  wily CI artifacts; see [Code quality trends](CODE_QUALITY_TRENDS.md).

### Why charts can look “empty”

- **History** is built from `history-ci.jsonl` / `history-nightly.jsonl`. Few pushes ⇒ few points.
- With **fewer than two** snapshots, the UI shows a short explanation instead of drawing lines.
- **Docstrings**, **dead code**, and **spelling** are **card-only** (not duplicated on charts).
- **CI → Code quality history** flat at **0** with **no** maintainability line: the unified metrics job
  must have **radon** (and the capture tools for docstrings/vulture/codespell) on `PATH`. The
  **`coverage-unified`** job installs them explicitly so `generate_metrics.py` can read
  `reports/complexity.json` / `reports/maintainability.json`. **Nightly** already installed those
  tools; snapshots before the fix can still show zeros until new CI runs upload fresh **`metrics`**.

### Data source differences

| | CI | Nightly |
| --- | --- | --- |
| **Trigger** | Push to main / release branches | Schedule (and configured events) |
| **Tests** | Unit + integration + E2E (as configured in workflow) | Broader suite including nightly-only tests |
| **Models** | Smaller / CI-oriented | Can use heavier paths (see workflow docs) |
| **Pipeline sample** | Often 1 episode | Same collector; may differ by job success |

### Metrics collection

| Source | Workflow | Latest file | History file |
| ------ | -------- | ----------- | ------------- |
| CI | `python-app.yml` | `latest-ci.json` | `history-ci.jsonl` |
| Nightly | `nightly.yml` | `latest-nightly.json` | `history-nightly.jsonl` |

Both deploy the same `index.html` on GitHub Pages. The page prefers **`dashboard-data.json`**
(single bundle built by `consolidate_dashboard_data.py` from the four files below) so the browser
does one fetch; `latest-*.json` and `history-*.jsonl` remain for workflows and legacy fallback.

### File layout (`metrics/` on `gh-pages`)

```text
metrics/
├── index.html              # Unified dashboard
├── dashboard-data.json     # CI + nightly latest + history arrays (preferred by the page)
├── latest-ci.json
├── history-ci.jsonl        # One JSON object per line (compact)
├── latest-nightly.json
└── history-nightly.jsonl
```

**JSONL:** Each line must be one JSON object. CI appends via
`scripts/dashboard/append_metrics_history_line.py`. Legacy multi-line appends are normalized with
`repair_metrics_jsonl.py --in-place` in workflows. Local repair:
`python scripts/dashboard/repair_metrics_jsonl.py metrics/history-ci.jsonl --in-place`.

**Why history might not grow on GitHub:** Workflows used to load prior rows with
`git show gh-pages:metrics/history-*.jsonl`. Publishing via **`actions/deploy-pages`** updates the
**live site** but **not** necessarily the **`gh-pages` git branch**, so every run could start from an
empty file and only append one line. **`nightly.yml`** and **`python-app.yml`** now load from the
**published Pages URL** first (`scripts/dashboard/fetch_metrics_file_from_pages.sh`), then fall back
to git. Custom Pages base URL: set **`METRICS_PAGES_BASE`** in the workflow env if needed.

**Strict local check:** `make metrics-preview-check` rebuilds the preview and exits non-zero if a
`history-*.jsonl` looks like pretty-printed JSON instead of JSONL.

**Why local CI chart points may stay low:** `make fetch-ci-metrics` only adds a bundle when that
workflow run uploaded the **`metrics`** artifact. Many successful **`python-app.yml`** runs never
produce it (job skipped, failed before upload, or older runs predate the artifact). Artifacts also
**expire** after **90** days (`python-app.yml`). You cannot download more CI points than GitHub still
stores. Raising **`N`** scans more run IDs but does not create artifacts that were never uploaded or
that expired. After **`history-ci.jsonl`** on Pages grows (workflow loads from live Pages + appends),
you can refresh local **`metrics/history-ci.jsonl`** from the site or rely on new pushes.

### Local nightly history (dashboard chart points)

CI history for preview comes from downloaded **`artifacts/ci-metrics-runs/run-*`** bundles; nightly
history comes only from **`metrics/history-nightly.jsonl`** on disk. To pull the same accumulated
nightly files the workflow publishes (many lines ⇒ many chart points after rebuild):

```bash
make fetch-nightly-metrics
make build-metrics-dashboard-preview
```

For **many nightly chart points** locally (like **`make fetch-ci-metrics N=80`** for more CI bundles), download
several successful nightly artifacts and merge **`latest-nightly.json`** from each into proper
JSONL:

```bash
make fetch-nightly-metrics N=25
make build-metrics-dashboard-preview
```

If **`artifacts/nightly-metrics-runs/run-*`** exists, **`build-metrics-dashboard-preview`** merges
those bundles into the preview even when **`metrics/history-nightly.jsonl`** is short.

This uses the latest successful **`nightly.yml`** **`nightly-metrics`** artifact when **`gh`** can
download it; otherwise it **`curl`**s **`latest-nightly.json`** and **`history-nightly.jsonl`**
from GitHub Pages (URL from **`gh repo view`** or **`git remote`**, or override with
**`GHPAGES_METRICS_BASE`** e.g. `https://chipi.github.io/podcast_scraper/metrics`).

If the **latest artifact** has a **short** `history-nightly.jsonl` but **Pages** is ahead, use Pages
only: **`FETCH_NIGHTLY_PREFER_PAGES=1 make fetch-nightly-metrics`**.

### Slowest / flaky fixes (pipelines)

| Issue | Change |
| ----- | ------ |
| **Slowest list too short or empty on CI** | `coverage-unified` copies `junit*.xml` from coverage artifacts into `reports/` before `generate_metrics.py`. `extract_slowest_tests` **always** merges timed rows from JSON **and** `junit*.xml` (not only when JSON had zero rows). |
| **Nightly missing JUnit for unit/integration/E2E** | `nightly.yml` pytest invocations now pass `--junitxml=reports/junit-*.xml` (aligned with `python-app.yml`). |
| **Flaky always zero** | `extract_test_metrics_from_reports_dir` reads all `pytest-*.json` plus `pytest.json` and merges by `nodeid` so a shard with `outcome: rerun` is not overwritten by a merged “clean pass” row. |
| **How many slow rows** | Default **10** in `latest-*.json` (`--slowest-top-n`); same as the dashboard table (`SLOWEST_TESTS_TABLE_MAX` in `generate_dashboard.py`). |

### Verify metrics fixes locally

You need a `reports/` tree similar to CI: pytest JSON (`pytest.json` and/or `pytest-*.json`) and, for slowest, optional `junit*.xml`. Easiest sources: run a test layer that writes `reports/`, or unzip a **`metrics`** / **`coverage-unified`** / **`nightly-*-reports`** artifact and copy `reports/` here.

```bash
# Generate metrics JSON (adjust paths if you use a temp dir)
python scripts/dashboard/generate_metrics.py \
  --reports-dir reports \
  --output /tmp/latest-metrics-check.json \
  --slowest-top-n 10

# Quick sanity: non-empty slowest when JUnit exists; flaky from shards
python3 -c "import json; d=json.load(open('/tmp/latest-metrics-check.json')); m=d['metrics']; print('slowest count:', len(m.get('slowest_tests') or [])); print('flaky:', m['test_health'].get('flaky')); print('total tests:', m['test_health'].get('total'))"

# Optional: validate a downloaded bundle directory (see script docstring)
python scripts/dashboard/validate_metrics_bundle.py path/to/bundle
```

**Unit tests** (dashboard scripts only):

```bash
python -m pytest tests/unit/scripts/dashboard/test_generate_metrics_slowest.py \
  tests/unit/scripts/dashboard/test_generate_metrics_flaky.py -q --no-cov
```

**Full preview** (charts + tables): after `reports/` or fetched JSONL is in place, run
`make build-metrics-dashboard-preview` (or your usual preview target from [Workflows](WORKFLOWS.md)).

**Important:** The preview only **displays** whatever is in `artifacts/dashboard-preview/` (built from
`artifacts/ci-metrics-runs/run-*` and nightly runs). It does **not** re-execute `generate_metrics.py` on
your laptop. If **CI slowest** stays empty, fetch a **newer** bundle after a green `main` run:
`make fetch-ci-metrics`, then rebuild the preview. The unified dashboard banner explains this when it
detects “many tests but zero slowest rows” in the CI snapshot.

## Alert thresholds

When enough history exists, the generator can flag:

- Runtime up materially vs recent median
- Coverage down vs recent average
- Test count shift
- Flaky tests up
- Complexity / maintainability drift

Details: [RFC-025: Test metrics and health tracking](../rfc/RFC-025-test-metrics-and-health-tracking.md).

## Related docs

- [Code quality trends](CODE_QUALITY_TRENDS.md) — wily, radon, local trends vs dashboard snapshots.
- [Workflows](WORKFLOWS.md) — when jobs run and what they produce.
