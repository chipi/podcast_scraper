# Metrics documentation and dashboard — redesign (WIP)

**Status:** **Partially implemented** (2026-03): `dashboard-data.json`, `consolidate_dashboard_data.py`, unified HTML bundle-first fetch + legacy fallback, workflows + `make metrics-preview-check`, doc/nav updates. **2026-03-31:** slowest + flaky metrics pipeline fixes documented in [ci/METRICS.md](../ci/METRICS.md) (see *Slowest / flaky fixes* and *Verify metrics fixes locally*).

**Goal:** One **robust** story for **test & CI metrics** (dashboard + local preview), **clear pairing** of the two **CI docs** you care about, and room for **additional** pages where needed — plus validation so bad local files fail fast instead of “mystery charts.”

---

## 1. Documentation pairs (corrected)

**Primary pair (this redesign focuses here):**

| Page | Role |
| ---- | ---- |
| [ci/METRICS.md](../ci/METRICS.md) | **Unified dashboard** (GitHub Pages): CI vs nightly snapshots, `latest-*.json`, `history-*.jsonl`, what the charts/cards mean |
| [ci/CODE_QUALITY_TRENDS.md](../ci/CODE_QUALITY_TRENDS.md) | **Wily + radon over git history**: local `make complexity-track`, per-file trends, how this differs from the dashboard’s “code quality history” chart |

They already cross-link; the redesign keeps **two separate pages** (or more if we add a tiny index) — no need to merge them into one long doc.

**Separate product (leave as its own page):**

| Page | Role |
| ---- | ---- |
| [guides/METRICS_GUIDE.md](../guides/METRICS_GUIDE.md) | **Experiment / eval** metrics (`run_experiment`, `metrics.json`, scorer) — not the CI dashboard |

**Problem this WIP still solves:** The **dashboard** depends on **four files** and **strict JSONL**; local copies often break that contract. That is independent of how many MkDocs pages we have.

---

## 2. Data sources we actually have (test / CI dashboard)

| Source | Where it lives | Format | Notes |
| ------ | -------------- | ------ | ----- |
| **GitHub Pages** | `gh-pages` branch `metrics/` | `index.html` + `latest-*.json` + `history-*.jsonl` | Canonical public view; JSONL one compact object per line |
| **CI workflow output** | Artifacts: `metrics` zip, `pytest-*` JSON | Per-job + merged `pytest.json` | Metrics job generates `latest-ci.json`; history merged from prior gh-pages |
| **Local preview** | `artifacts/dashboard-preview/` | Copy of CI bundle + `metrics/` nightly | Built by `build_local_metrics_preview.sh` |
| **Local dev copies** | `metrics/*.json`, `*.jsonl` | Often **wrong** (pretty-printed blob in `.jsonl`) | Root cause of “one dot” / empty charts |

**Nightly vs CI** is a **logical** split (two `latest-*` / `history-*` pairs), not two different schemas — same `generate_metrics.py` shape.

---

## 3. Root causes of pain (so we fix the right layer)

1. **JSONL contract** is strict; humans/tools save **one pretty JSON** into `history-*.jsonl` → parser recovers **one** object → one chart point.
2. **Local preview** mixes **downloaded `run-*`** (CI) with **`metrics/`** (nightly) without a single **validated bundle** step.
3. **Browser** loads **four URLs**; cache + toggle order caused confusion; partially mitigated with `cache: 'no-store'` and load sequencing.
4. **Optional nav clarity:** MkDocs nav could group **CI metrics** (METRICS + CODE_QUALITY_TRENDS) next to each other so the pair is obvious; experiment guide stays under Guides.

---

## 4. Proposed documentation information architecture

**Keep separate pages** (your preference: two or more is fine).

| Change | Detail |
| ------ | ------ |
| **METRICS.md** | Optional subtitle or intro line: “Companion: [Code quality trends](../ci/CODE_QUALITY_TRENDS.md) (wily / git history).” Already points there for wily; can tighten once. |
| **CODE_QUALITY_TRENDS.md** | Already contrasts itself with METRICS.md; keep. |
| **METRICS_GUIDE.md** | One-line pointer at top to [Test dashboard](../ci/METRICS.md) — done. |
| **mkdocs.yml** | Optional: nest under CI or rename nav labels for clarity, e.g. “Test dashboard (GitHub Pages)” and “Code quality trends (wily)”. |

**Optional:** Short `docs/ci/README.md` or index bullet list linking METRICS + CODE_QUALITY_TRENDS only — only if you want a third **navigation** hop without merging content.

---

## 5. Proposed technical architecture (robust preview + parity with CI)

### 5.1 Normalization pipeline (single entry point)

Introduce a **Python step** used by **both** local preview and (optionally) CI before deploy:

**Inputs:**

- CI: newest `artifacts/ci-metrics-runs/run-*/` **or** `metrics/latest-ci.json` + `history-ci.jsonl`
- Nightly: `metrics/latest-nightly.json` + `history-nightly.jsonl`

**Behavior:**

1. Load history with `metrics_jsonl.load_metrics_history` (already tolerates some legacy shapes).
2. If history parses to **≤1** record and file has **multiple physical lines** of `{`-heavy content → emit **warning or error**: “File looks like pretty JSON, not JSONL.”
3. Emit a **single artifact** for the browser, e.g. `dashboard-data.json`:

   ```json
   {
     "generated_at": "...",
     "ci": { "latest": {...}, "history": [ ... ] },
     "nightly": { "latest": {...}, "history": [ ... ] }
   }
   ```

4. **Dashboard JS** performs **one** `fetch('dashboard-data.json')` and switches source in memory — no four-file drift, no JSONL parsing in the browser for history.

**CI / gh-pages:** Same generator runs in workflow; deploy `dashboard-data.json` next to `index.html`. Keep **individual** `latest-*.json` / `history-*.jsonl` for backward compatibility and `append_metrics_history_line.py`, or deprecate after one release (decision below).

### 5.2 Validation target

**`make metrics-preview-check`** (name TBD):

- Runs after normalization (or as part of `build_local_metrics_preview`)
- Exits non-zero if:
  - `history-*.jsonl` fails JSONL repair contract, or
  - parsed history count **≠** expected minimum for “chart smoke” (optional flag)

### 5.3 Slowest tests / pytest shards (**updated 2026-03-31**)

Implemented (not only planned):

- **Shard-first** pytest JSON for slowest when shards exist; **plus** all **`junit*.xml`** under `reports/` are always parsed and **merged** (dedupe by test name, keep max duration). Avoids “5 slow tests only” when JSON has sparse xdist timings but JUnit has full `testcase@time` data.
- **CI (`python-app.yml`):** per-job `--junitxml=reports/junit-….xml`; artifacts include those files; **`coverage-unified`** copies `junit*.xml` from downloaded coverage artifacts into `reports/` before `generate_metrics.py`.
- **Nightly (`nightly.yml`):** same `--junitxml` pattern on unit / integration / E2E so merged nightly `reports/` matches CI behavior.
- **`generate_metrics.py`:** `--slowest-top-n` (default **10**, aligned with the dashboard table); log line prints how many slow rows were written.

**Flaky:** `extract_test_metrics_from_reports_dir` merges all `pytest-*.json` + `pytest.json` by `nodeid` so rerun-pass signals are not lost when merged JSON looks like a clean pass.

**Local verification:** See [ci/METRICS.md § Verify metrics fixes locally](../ci/METRICS.md#verify-metrics-fixes-locally) — `generate_metrics.py` on a real `reports/` tree + the two unit test modules above.

---

## 6. Phased rollout (recommended)

| Phase | Scope | Risk |
| ----- | ----- | ---- |
| **P0** | Small doc tweaks: METRICS ↔ CODE_QUALITY_TRENDS cross-promotion; optional one-line pointer on METRICS_GUIDE; optional mkdocs nav labels | Low |
| **P1** | `consolidate_dashboard_data.py` + extend `build_local_metrics_preview.sh` to write `dashboard-data.json`; HTML uses single fetch (feature-flag or cutover) | Medium |
| **P2** | Wire same generator into `python-app.yml` / gh-pages deploy; optional retention of old files | Medium |
| **P3** | `make metrics-preview-check` in CI optional job or pre-commit | Low |

---

## 7. Open decisions

1. **Single-file cutover:** Drop four separate fetches on GitHub Pages immediately vs keep both during transition.
2. **History source of truth:** Continue appending JSONL on CI vs move to a small SQLite / parquet (probably **not** needed).
3. **Experiment metrics:** Leave in eval pipeline only, or add a **future** read-only panel in the same HTML (out of scope unless product asks).

---

## 8. Success criteria

- Local: `make build-metrics-dashboard-preview` **warns or fails** on bad `history-*.jsonl` shape.
- Local: Nightly chart shows **as many points as** parsed history records (given real gh-pages history).
- Docs: Under **CI/CD**, **Test dashboard (GitHub Pages)** vs **Code quality trends (wily)** read as a deliberate pair; **experiment** metrics stay under Guides (`METRICS_GUIDE.md`).

---

## 9. Checklist: what to run locally after pulling fixes

| Check | Command / action |
| ----- | ---------------- |
| Slowest + flaky extraction | `python scripts/dashboard/generate_metrics.py --reports-dir reports --output /tmp/m.json` with a populated `reports/` (see [METRICS.md](../ci/METRICS.md#verify-metrics-fixes-locally)) |
| Unit tests | `python -m pytest tests/unit/scripts/dashboard/test_generate_metrics_slowest.py tests/unit/scripts/dashboard/test_generate_metrics_flaky.py -q --no-cov` |
| Dashboard HTML | `make build-metrics-dashboard-preview` (with `metrics/*.jsonl` / fetched bundles as you already use) |
| JSONL shape | `make metrics-preview-check` |

---

## Related

- [RFC-025](../rfc/RFC-025-test-metrics-and-health-tracking.md)
- [RFC-026](../rfc/RFC-026-metrics-consumption-and-dashboards.md)
- [METRICS_GAP_ANALYSIS.md](METRICS_GAP_ANALYSIS.md)
