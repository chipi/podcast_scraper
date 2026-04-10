# Performance Profile Guide (RFC-064)

**Status:** Work in progress — evolves with tooling and release practice.

This guide is the **operator manual** for **frozen performance profiles**: how to capture
them, how to interpret them, and how that work relates to **quality evaluation** under
`data/eval/`. For normative design and artifact schema, see
[RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md).

**Related guides**

- **[Experiment Guide](EXPERIMENT_GUIDE.md)** — datasets, baselines, experiments, and
  **output quality** (ROUGE, gates). Profiling is a **parallel** track, not a substitute.
- **[Performance](PERFORMANCE.md)** — runtime tuning (preprocessing cache, transcription,
  etc.), not release YAML profiles.
- **[Performance reports](performance-reports/index.md)** — published profile snapshots
  (tables, caveats), sibling to [Evaluation Reports](eval-reports/index.md).

---

## What this system is for

| Track | Question | Typical artifacts |
| ----- | -------- | ----------------- |
| **Eval / experiments** (`data/eval/`) | Did **quality** change? | Runs, baselines, `fingerprint.json`, reports |
| **Performance profiles** (`data/profiles/`) | Did **resource cost** change? | `vX.Y.Z-<variant>.yaml` (RSS, CPU%, wall time per stage) |

Profiles answer: peak RSS, wall time, CPU% **per pipeline stage** (plus environment
metadata) under a **fixed** pipeline config, so you can diff **release to release** on
the same machine.

---

## Prerequisites

- Repo root, dev environment with **`psutil`** (e.g. `pip install -e '.[dev]'`).
- **`make profile-freeze`** / **`make profile-diff`** (see repository root
  [`Makefile`](https://github.com/chipi/podcast_scraper/blob/main/Makefile)).
- For **E2E mock RSS** presets: no real feed; `freeze_profile.py` starts
  `tests.e2e.fixtures.e2e_http_server.E2EHTTPServer` when the config uses a placeholder
  URL (see below).
- For **OpenAI / Anthropic** presets: real API keys as usual. RSS can still be mock;
  API calls are real unless you configure mock bases yourself.
- For **Ollama** presets: models pulled locally (`ollama pull …`).

---

## Methodology (short)

### Capture conditions

- Prefer **quiet machine**, **AC power** on laptops, minimal heavy background apps.
- **Comparability:** treat profiles on the **same `hostname`** as directly comparable;
  across machines, use trends on that host only (absolute RSS differs by hardware).

### Warm-up

- By default, **`freeze_profile.py`** runs **`run_pipeline`** once with
  **`max_episodes: 1`** to a **temp** output dir, then the **measured** run with your
  real `output_dir` and **psutil** sampling. Goal: reduce cold-start / first-load spikes
  in the timed run.
- Skip with **`--skip-warmup`** / **`SKIP_WARMUP=1`** when debugging.

### RSS / CPU attribution

- **Wall time per stage** comes from pipeline **`metrics.json`** (same source as
  dashboards).
- **Per-stage RSS and CPU%** are derived by **splitting** the measured wall-clock window
  **proportionally** by those stage times. Parallel work can **mis-attribute** samples;
  the profile **`totals.sampling_note`** states this.
- **Short stages and `vector_indexing`:** if a stage’s proportional window contains **no** psutil
  samples (common for a very small tail slice), **`peak_rss_mb`** would read **0** even when wall
  time is non-zero. The freeze tool **reuses** the **global sampled peak** for
  **`vector_indexing`** only in that case so the YAML stays interpretable; it is an upper-bound
  hint, not a stage-isolated measurement.

### Sampling interval

- **`freeze_profile.py`** polls RSS/CPU on a fixed interval (default **0.5 s**; was **1.0 s** in
  early tooling). Use **`--sample-interval`** or **`SAMPLE_INTERVAL=`** on **`make profile-freeze`**
  for finer boundaries (e.g. **`0.25`**) at slightly higher overhead.

### Companion `*.stage_truth.json`

- Each freeze also writes **`data/profiles/<VERSION>.stage_truth.json`** unless disabled with
  **`--no-stage-truth-snapshot`** / **`NO_STAGE_TRUTH=1`**. It holds a **trimmed `metrics.json`**
  excerpt, **`wall_seconds_by_stage`**, psutil-derived **`resource_by_stage_psutil`**, the
  **run wall clock**, **sum of mapped stage walls** (parallelism hint), **`sample_interval_s`**, and
  the path to the **source `metrics.json`** for audits.

### Apple Silicon / MPS

- Process **RSS** can **under-count** GPU memory on **MPS**. The profile may include
  **`rss_measurement_note`** in `environment` when device is `mps`.

### What v1 does **not** do

- **No eval materialized dataset path** in the freeze tool: capture uses **`run_pipeline`**
  from a normal **`Config`** (RSS or E2E fixture URL). Reproducibility is **config +
  fixture + git tag + host**, not transcript-hash materialization like
  [Experiment Guide](EXPERIMENT_GUIDE.md) Step 1a. A future extension could add a
  dataset-driven entry point if needed.

---

## Process: capture a profile

1. Choose a **pipeline YAML** under
   [`config/profiles/`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md).
   Recommended: **`capture_e2e_*.yaml`** (aligned with
   `config/acceptance/sample_acceptance_e2e_fixture_single.yaml`).
2. Choose **`VERSION`** (e.g. `v2.6.0-ml-dev`) — becomes `release` in the YAML and default
   output path `data/profiles/<VERSION>.yaml`.
3. Set **`DATASET_ID`** to something **truthful** for metadata (e.g. `e2e_podcast1_mtb_n2`),
   not `indicator_v1` unless that is what you ran.
4. Run:

   ```bash
   make profile-freeze VERSION=v2.6.0-ml-dev \
     PIPELINE_CONFIG=config/profiles/capture_e2e_ml_dev.yaml \
     DATASET_ID=e2e_podcast1_mtb_n2
   ```

5. **Commit** `data/profiles/<VERSION>.yaml` and **`data/profiles/<VERSION>.stage_truth.json`**
   with the release (or your profiling PR).

**Optional Makefile variables:** `OUTPUT=`, `SKIP_WARMUP=1`, `E2E_FEED=podcast1_multi_episode`,
`SAMPLE_INTERVAL=0.25`, `NO_STAGE_TRUTH=1`
(forces fixture name when you want something other than the default **`podcast1_mtb`**
auto-selected from placeholder RSS).

**Retroactive profiles:** check out an old tag, run the same command with a config valid
for that era, merge the YAML on `main`. See
[`data/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/data/profiles/README.md).

---

## Diff two profiles

```bash
make profile-diff FROM=v2.5.0-ml-dev TO=v2.6.0-ml-dev
```

Paths default to `data/profiles/<FROM>.yaml` and `data/profiles/<TO>.yaml`. Optional
**`data/profiles/regression_rules.yaml`** can annotate large deltas (advisory; RFC Phase 1).

---

## What gets recorded in the profile YAML

| Section | Contents |
| ------- | -------- |
| **Top-level** | `release`, `date`, `dataset_id`, `episodes_processed` |
| **environment** | `generate_provider_fingerprint()` fields (git, package version, libraries, device, …) plus **host**: hostname, CPU model, core counts, RAM, OS, Python; optional MPS RSS note |
| **stages** | Per stage (non-zero wall only): `wall_time_s`, `peak_rss_mb`, `avg_cpu_pct` |
| **totals** | `peak_rss_mb`, `wall_time_s`, `avg_wall_time_per_episode_s`, `sampling_note` |

**Pipeline `metrics.json`** (input to the freeze) also includes **`vector_index_seconds`**
after auto vector indexing when enabled.

---

## Interpreting the profile

### `totals.wall_time_s` vs summing `stages.*.wall_time_s`

- **`totals.wall_time_s`** is the **real wall-clock** duration of the measured
  **`run_pipeline`** call (what `freeze_profile.py` times with `perf_counter`).
- **`stages.<name>.wall_time_s`** comes from **`metrics.json`**: aggregated counters
  (for example episode-level stage times **added**), mapped into the RFC-064 stage
  names.

The pipeline runs **parallel work** (for example multiple processing slots and
summary workers). Stage totals are **not** a serial schedule, so **the sum of
per-stage wall times can be much larger than `totals.wall_time_s`**. That is
expected. For release comparisons, treat **`totals.wall_time_s`** and
**`totals.peak_rss_mb`** as the headline numbers on a given host; use per-stage
deltas as **signals**, not as a strict partition of wall-clock time.

### Missing stages in `stages:`

Only stages with **non-zero** mapped wall time appear. Common reasons:

- **Cached transcripts** — `transcribe_count` / averages can be **zero**, so
  **transcription** (and sometimes **media_download** / **audio_preprocessing**)
  may **not** appear even though RSS and download ran earlier in the run.
- **Very small** scrape/parse/download times — may round to **zero** in the
  metrics fields that feed the profile.

So: **a stage omitted from the YAML does not mean that step was skipped**; it
often means **no measurable wall time** was recorded for that bucket in
`metrics.json` for this capture (cache hits are the usual case for transcription).

### RSS, CPU%, and short stages

- **Proportional RSS/CPU attribution** (see above) uses coarse **~1 s** `psutil`
  samples. A **very short** stage (for example **vector_indexing**) can show
  **`peak_rss_mb: 0`** or **`avg_cpu_pct: 0`** after rounding or because almost
  no sample fell in that window — especially if embedding runs on **MPS** (process
  RSS is already a lower bound for GPU memory).
- **High wall time** with **low `avg_cpu_pct`** is normal when work is **API
  bound**, **I/O bound**, or **GPU bound**; the sampler mostly sees the Python
  process idle between calls.

---

## Suggested variant matrix (release)

See **[`config/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md)** for the
**`capture_e2e_*.yaml`** table (ML dev/prod, OpenAI, Anthropic, Ollama). **Minimal**
subset if time is tight: **ml-dev + ml-prod + one cloud**.

---

## Troubleshooting

| Issue | What to check |
| ----- | ------------- |
| `metrics.json` not found | `metrics_output` must not be disabled; `output_dir` must be writable. |
| E2E import error | Run from **repo root**; dev install must include test layout (`tests.e2e`). |
| Placeholder RSS but no server | URL must contain **`example.invalid`** or **`e2e-placeholder`**, or pass **`E2E_FEED`**. |
| Sum of stage walls >> `totals.wall_time_s` | **Normal** under parallelism; see [Interpreting the profile](#interpreting-the-profile). |
| **Transcription** (or ingest) missing from `stages` | Often **cache hits** or tiny metrics; see [Interpreting the profile](#interpreting-the-profile). |

---

## Reference links

| Doc | Role |
| --- | ---- |
| [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) | Design, stage model, non-goals |
| [`config/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md) | Capture config index and preset matrix |
| [`data/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/data/profiles/README.md) | Artifact directory and interpretation |
| [`scripts/eval/freeze_profile.py`](https://github.com/chipi/podcast_scraper/blob/main/scripts/eval/freeze_profile.py) | CLI (`--e2e-feed`, warm-up, …) |
| [`scripts/eval/diff_profiles.py`](https://github.com/chipi/podcast_scraper/blob/main/scripts/eval/diff_profiles.py) | Terminal diff |
