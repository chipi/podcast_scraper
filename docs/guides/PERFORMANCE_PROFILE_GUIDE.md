# Performance Profile Guide (RFC-064)

**Status:** Work in progress — evolves with tooling and release practice.

This guide is the **operator manual** for **frozen performance profiles**: how to capture
them, how to interpret them, and how that work relates to **quality evaluation** under
`data/eval/`. For normative design and artifact schema, see
[RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md).

**Related guides**

- **[Experiment Guide](EXPERIMENT_GUIDE.md)** — datasets, baselines, experiments, and
  **output quality** (ROUGE, gates). Profiling is a **parallel** track, not a substitute.
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** -- eval run promotion (baselines,
  silver references). Profile promotion follows the same philosophy but
  with lighter artifacts.
- **[Performance](PERFORMANCE.md)** — runtime tuning (preprocessing cache, transcription,
  etc.), not release YAML profiles.
- **[Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md)** — RFC-065: optional **`--monitor`**
  subprocess, **`.pipeline_status.json`**, **`.monitor.log`** when stderr is not a TTY (or when
  **`PODCAST_SCRAPER_MONITOR_FILE_LOG`** is set); optional **`.[monitor]`** for **memray** / **py-spy**.
  The freeze tool can turn the monitor on for the **measured** run only and archive the log; see
  [Live monitor during profile freeze](#live-monitor-during-profile-freeze-rfc-065) below.
- **[CONFIGURATION.md](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512)** — config table
  including **`PODCAST_SCRAPER_MONITOR_FILE_LOG`**.
- **[Performance reports](performance-reports/index.md)** — published profile snapshots
  (tables, caveats), sibling to [Evaluation Reports](eval-reports/index.md).

---

## What this system is for

| Track | Question | Typical artifacts |
| ----- | -------- | ----------------- |
| **Eval / experiments** (`data/eval/`) | Did **quality** change? | Runs, baselines, `fingerprint.json`, reports |
| **Performance profiles** (`data/profiles/`) | Did **resource cost** change? | `vX.Y.Z-<variant>.yaml` (RSS, CPU%, wall time per stage) |
| **Optional:** same profile capture | How did **RSS / CPU / stage** evolve **over time** during the run? | Same YAML freeze, plus optional **`<VERSION>.monitor.log`** (RFC-065 ticks) and **`rfc065_monitor`** in **`stage_truth.json`** |

Profiles answer: peak RSS, wall time, CPU% **per pipeline stage** (plus environment
metadata) under a **fixed** pipeline config, so you can diff **release to release** on
the same machine. The optional monitor adds a **time-resolved** trail (stage changes,
sampled RSS and CPU%) alongside that aggregate snapshot.

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
- The warm-up run forces **`monitor: false`** so the RFC-065 subprocess does not run
  there; optional monitor applies **only** to the measured run.
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

### Live monitor during profile freeze (RFC-065)

Use this when you want a **time-resolved** view of the same run that produces the frozen YAML: which
**pipeline stage** was active, how **RSS** and **CPU%** moved tick-by-tick, and a **per-stage summary**
inside the monitor (see [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md)). It does **not** replace
the profile YAML or **`freeze_profile.py`**’s own in-process **ResourceSampler** (used for
proportional stage attribution); you get **two** observers: the **child** monitor (RFC-065, aligned
with **`.pipeline_status.json`**) and the **freeze script** sampler (RFC-064 attribution).

#### Enabling the monitor

| Mechanism | Effect |
| :--- | :--- |
| **`make profile-freeze … MONITOR=1`** | Passes **`--monitor`** to **`freeze_profile.py`**. |
| **`freeze_profile.py --monitor`** | Turns the monitor on for the **measured** run even if the pipeline YAML has **`monitor: false`**. |
| **`monitor: true`** in the pipeline YAML | Measured run uses the monitor **without** the Makefile flag (**`MONITOR`** is optional if YAML already enables it). |

Effective switch: **`monitor` for measured run** = **`(pipeline YAML monitor) OR (--monitor / MONITOR=1)`**.

Default: **off** — no extra subprocess, no **`.monitor.log`** from the monitor unless you opt in.

#### File logging: `PODCAST_SCRAPER_MONITOR_FILE_LOG`

During a normal CLI run, the monitor writes **`.monitor.log`** only when **stderr** is **not** a TTY;
otherwise it uses **`rich.Live`** on stderr and may **not** create a log file.

For profile capture, **`freeze_profile.py`** sets environment variable
**`PODCAST_SCRAPER_MONITOR_FILE_LOG=1`** for the duration of the **measured** **`run_pipeline`**
call (and restores the previous value afterward). The monitor subprocess then **always** appends
ticks to **`.monitor.log`** under the pipeline’s **effective output directory**, so an archived log
exists even when you run **`make profile-freeze`** from an interactive terminal.

Normative detail: [CONFIGURATION.md — Live pipeline monitor](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512).

#### Artifacts after a freeze with the monitor on

1. **Under `output_dir` (and run-scoped subdirs, if any):** **`.pipeline_status.json`** (final state
   after the run), and the **newest** **`.monitor.log`** found under that tree (the freeze tool picks
   by modification time if several exist).
2. **Next to the profile YAML:** **`data/profiles/<VERSION>.monitor.log`** — copy of that log for
   version control and promotion.

If the monitor was enabled but **no** **`.monitor.log`** is found, **`freeze_profile.py`** logs a
warning and **`rfc065_monitor.archived_log`** in **`stage_truth.json`** is **null** with a short
**`note`** (should be rare when file logging is forced).

#### `stage_truth.json`: `rfc065_monitor`

When the monitor ran for the measured capture, **`stage_truth.json`** includes an **`rfc065_monitor`**
object (omitted entirely when the monitor was off):

| Field | Meaning |
| :--- | :--- |
| **`enabled`** | Always **true** when this object is present. |
| **`forced_file_log_env`** | Name of the env var the freeze tool set (**`PODCAST_SCRAPER_MONITOR_FILE_LOG`**). |
| **`source_log`** | Absolute path to the **`.monitor.log`** that was copied (under **`output_dir`**). |
| **`archived_log`** | Repo-relative path to **`data/profiles/<VERSION>.monitor.log`** when copy succeeded; **null** if missing. |
| **`lines`**, **`bytes`** | Size of the archived file (after copy). |
| **`note`** | Present when **`archived_log`** is **null** — explains missing log. |

If you pass **`NO_STAGE_TRUTH=1`** / **`--no-stage-truth-snapshot`**, the **`stage_truth.json`** file
is not written, but **`<VERSION>.monitor.log`** is still produced and copied when the monitor was on.

#### Promotion to `references/`

**`make profile-promote`** copies optional companions next to the source YAML:

- **`*.stage_truth.json`** (required unless **`NO_STAGE_TRUTH_REQUIRED=1`**).
- **`*.monitor.log`** when present — becomes **`references/<promoted_id>.monitor.log`**.

Commit promoted references with the same discipline as YAML + **`stage_truth`**: treat large logs as
optional (skip monitor on routine captures if repo size matters).

### Companion `*.stage_truth.json`

- Each freeze also writes **`data/profiles/<VERSION>.stage_truth.json`** unless disabled with
  **`--no-stage-truth-snapshot`** / **`NO_STAGE_TRUTH=1`**. It holds a **trimmed `metrics.json`**
  excerpt, **`wall_seconds_by_stage`**, psutil-derived **`resource_by_stage_psutil`**, the
  **run wall clock**, **sum of mapped stage walls** (parallelism hint), **`sample_interval_s`**, the
  path to the **source `metrics.json`** for audits, and when applicable **`rfc065_monitor`** (see
  above).

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

   With **RFC-065 monitor** ticks archived next to the profile (measured run only):

   ```bash
   make profile-freeze VERSION=v2.6.0-ml-dev \
     PIPELINE_CONFIG=config/profiles/capture_e2e_ml_dev.yaml \
     DATASET_ID=e2e_podcast1_mtb_n2 \
     MONITOR=1
   ```

5. **Commit** `data/profiles/<VERSION>.yaml` and **`data/profiles/<VERSION>.stage_truth.json`**
   with the release (or your profiling PR). If you used **`MONITOR=1`** (or **`monitor: true`**
   in the pipeline YAML), also commit **`data/profiles/<VERSION>.monitor.log`** when you want that
   trace in git; omit it for smaller PRs if the YAML alone is enough.

**Optional Makefile variables:** `OUTPUT=`, `SKIP_WARMUP=1`, `E2E_FEED=podcast1_multi_episode`,
`SAMPLE_INTERVAL=0.25`, `NO_STAGE_TRUTH=1`, **`MONITOR=1`**
(forces fixture name when you want something other than the default **`podcast1_mtb`**
auto-selected from placeholder RSS).

**Retroactive profiles:** check out an old tag, run the same command with a config valid
for that era, merge the YAML on `main`. See
[`data/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/data/profiles/README.md).

---

## Profile lifecycle

Profiles follow a **capture -- iterate -- promote -- compare -- archive**
lifecycle, modelled on the eval promotion workflow but with lighter
artifacts (a profile is a single YAML, not a bundle of predictions and
metrics).

```text
capture ──> iterate ──> promote ──> compare
                                       │
                                   archive (when superseded)
```

### Directory layout

```text
data/profiles/
├── README.md                     # index table + lifecycle docs
├── references/                   # promoted immutable profiles
│   ├── v2.6.0-openai.yaml
│   ├── v2.6.0-openai.stage_truth.json
│   └── v2.6.0-openai.monitor.log   # optional; when capture used MONITOR=1 / monitor: true
├── _archive/                     # superseded references (history)
│   ├── README.md                 # archive log table
│   └── v2.5.0-openai.yaml
├── v2.6-wip-openai.yaml          # working profiles (root)
├── v2.6-wip-ml-dev.yaml
└── issue-477/                    # feature-branch working profiles
    ├── README.md
    ├── issue477-staged-gpt4o.yaml
    └── issue477-staged-gpt4o.stage_truth.json
```

- **Working profiles** live at the root or in feature subdirectories.
  They are mutable and disposable.
- **Reference profiles** live in `references/`. They are immutable once
  promoted.
- **Archived references** live in `_archive/`. They are kept for
  historical traceability but must not be used for new comparisons.

### Naming conventions

| Type | Pattern | Example |
| :--- | :--- | :--- |
| Working | `v2.6-wip-<variant>.yaml` | `v2.6-wip-openai.yaml` |
| Feature | `<feature>/<name>.yaml` | `issue-477/issue477-staged-gpt4o.yaml` |
| Reference | `v2.6.0-<variant>.yaml` (semver, no "wip") | `v2.6.0-openai.yaml` |

The **stage truth companion** always shares the stem:
`<same-stem>.stage_truth.json`. An optional **monitor** archive uses the same stem:
`<same-stem>.monitor.log`.

---

## Promote a profile to reference

When a working profile is ready to become a release anchor, promote it
into `data/profiles/references/`.

```bash
make profile-promote \
  SOURCE=data/profiles/v2.6-wip-openai.yaml \
  PROMOTED_ID=v2.6.0-openai \
  REASON="Release v2.6.0 OpenAI reference profile"
```

**What this does:**

1. Validates the source YAML has required fields (`release`, `date`,
   `dataset_id`, `stages`, `totals`).
2. Rejects `promoted_id` values containing "wip" (guards against
   accidental promotions of working names).
3. Requires a companion `stage_truth.json` next to the source (override
   with `NO_STAGE_TRUTH_REQUIRED=1`).
4. Stamps a `promoted` metadata block into the YAML:

   ```yaml
   promoted:
     promoted_id: v2.6.0-openai
     promoted_from: data/profiles/v2.6-wip-openai.yaml
     promoted_at: "2026-04-12T14:30:00Z"
     reason: "Release v2.6.0 OpenAI reference profile"
   ```

5. Copies the YAML and **`stage_truth.json`** into **`references/`**. If
   **`<source-stem>.monitor.log`** exists beside the source YAML, it is copied to
   **`references/<promoted_id>.monitor.log`**.

**Preview first** with `DRY_RUN=1`:

```bash
make profile-promote \
  SOURCE=data/profiles/v2.6-wip-openai.yaml \
  PROMOTED_ID=v2.6.0-openai \
  REASON="Release v2.6.0 OpenAI reference" \
  DRY_RUN=1
```

**Immutability:** once a reference exists in `references/`, the script
refuses to overwrite it. To replace a reference, archive the old one
first (see below).

### Archiving a superseded reference

When a new reference replaces an old one:

1. Move the old YAML (and stage_truth.json) from `references/` to
   `_archive/`.
2. Add a row to the archive log in `_archive/README.md`.
3. Update the active reference table in `data/profiles/README.md`.
4. Promote the new reference with `make profile-promote`.

---

## Diff two profiles

```bash
make profile-diff FROM=v2.5.0-ml-dev TO=v2.6.0-ml-dev
```

Paths default to `data/profiles/<FROM>.yaml` and `data/profiles/<TO>.yaml`.

To compare across directories (e.g. a reference against a feature
profile), use the path relative to `data/profiles/`:

```bash
make profile-diff \
  FROM=references/v2.6.0-openai \
  TO=issue-477/issue477-staged-gpt4o
```

Optional **`data/profiles/regression_rules.yaml`** can annotate large
deltas (advisory; RFC Phase 1).

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

**Companion files** (same **`VERSION`** stem as the YAML, under **`data/profiles/`** unless
**`OUTPUT=`** overrides the YAML path):

| File | When |
| ---- | ---- |
| **`<VERSION>.stage_truth.json`** | Default; omit with **`NO_STAGE_TRUTH=1`**. |
| **`<VERSION>.monitor.log`** | Only when the **measured** run had the RFC-065 monitor on (**`MONITOR=1`**, **`--monitor`**, or **`monitor: true`** in pipeline YAML). |

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
| Monitor on but **`rfc065_monitor.archived_log`** is **null** | Check **`freeze_profile`** stderr for a warning; confirm **`output_dir`** is writable and the measured run completed. With **`NO_STAGE_TRUTH=1`**, the log may still exist as **`<VERSION>.monitor.log`** without JSON metadata. |

---

## Reference links

| Doc | Role |
| --- | ---- |
| [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) | Design, stage model, non-goals |
| [`config/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md) | Capture config index and preset matrix |
| [`data/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/data/profiles/README.md) | Artifact directory, index table, lifecycle |
| [`scripts/eval/freeze_profile.py`](https://github.com/chipi/podcast_scraper/blob/main/scripts/eval/freeze_profile.py) | CLI (`--e2e-feed`, warm-up, `--monitor`, `PODCAST_SCRAPER_MONITOR_FILE_LOG` during measured run, …) |
| [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) | RFC-065 operator guide and artifact paths |
| [`scripts/eval/promote_profile.py`](https://github.com/chipi/podcast_scraper/blob/main/scripts/eval/promote_profile.py) | Promote working profile to reference |
| [`scripts/eval/diff_profiles.py`](https://github.com/chipi/podcast_scraper/blob/main/scripts/eval/diff_profiles.py) | Terminal diff |
| [Experiment Guide](EXPERIMENT_GUIDE.md) | Eval promotion (baselines, silvers) -- same philosophy |
