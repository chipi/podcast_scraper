# RFC-064: Performance Profiling and Release Freeze Framework

## Status

**Draft**

## RFC Number

064

## Authors

Podcast Scraper Team

## Date

2026-04-09

## Related RFCs

- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` — Quality benchmarking framework; this RFC is the resource cost sibling
- `docs/rfc/RFC-047-run-comparison-visual-tool.md` — Run comparison tool; a future RFC will extend it with a performance tab
- `docs/rfc/RFC-056-autoresearch-optimization-loop.md` — Autoresearch loop; benefits from per-stage resource visibility

## Related ADRs

- ADR-014: Codified Comparison Baselines
- ADR-027: Deep Provider Fingerprinting
- ADR-040: Explicit Golden Dataset Versioning

---

## Abstract

This RFC introduces a **performance profiling and release freeze framework** for `podcast_scraper` — the resource cost sibling to the quality benchmarking system established in RFC-041.

Where RFC-041 answers "did output quality change?", this RFC answers "did resource cost change?" — specifically peak memory (RSS), wall time per stage, and CPU utilization. The framework builds on top of the project's existing per-episode timing infrastructure (`workflow.metrics.Metrics`, `EpisodeStageTimings`), the `ProviderFingerprint` environment capture, and the acceptance benchmark reporting (`generate_performance_benchmark.py`). It adds what those systems lack: a **frozen, versioned, cross-release profile artifact** that makes resource cost comparable across releases under identical conditions.

The framework has two components:

1. **`data/profiles/` — frozen release profiles**: YAML snapshots of peak RSS, wall time, and CPU utilization per pipeline stage, committed at release time and versioned by release tag.
2. **`scripts/eval/freeze_profile.py` — profile capture and diff tooling**: A headless script that runs the reference dataset, captures resource metrics via `psutil` alongside the existing `Metrics` collector, and writes the frozen profile. A companion diff script compares any two profiles.

**Out of scope for this RFC** (deferred to future RFCs):

- **Live monitoring dashboard** (`--monitor` CLI flag with tmux/Terminal.app split, interactive flamegraph capture) — macOS-specific developer tooling; separate RFC.
- **RFC-047 performance tab** (Streamlit extension joining quality and resource metrics) — separate RFC to keep scope focused.

---

## Motivation

As `podcast_scraper` evolves — new providers, heavier models, additional pipeline stages (GI, KG, vector search) — resource cost can drift silently. A new release might improve summary quality while doubling peak RAM, or reduce WER while tripling Whisper wall time. Without a performance baseline system, these regressions are invisible until they cause operational problems during overnight autoresearch runs or exceed memory limits on the development machine.

RFC-041 already handles quality regression detection with frozen baselines and regression gates. This RFC applies the same discipline to resource cost.

### What Exists Today

The project already has significant performance instrumentation. This RFC builds on top of it rather than replacing it:

| Existing System | What It Captures | What It Lacks |
| --- | --- | --- |
| `workflow.metrics.Metrics` | Per-stage wall time, per-episode timing, device tracking, LLM token/cost | No memory (RSS), no CPU utilization, no cross-release comparison, no frozen artifact |
| `EpisodeStageTimings` | Per-episode breakdown: download, transcribe, extract names, cleaning, summarize | Timing only, no resource measurement, no freeze |
| `ProviderFingerprint` | Model name/version/hash, device, precision, git commit, library versions | ML reproducibility oriented, not machine-level resource profiling |
| `generate_performance_benchmark.py` | Provider-grouped timing from acceptance runs, baseline comparison | Tied to acceptance sessions, not release versions; no memory/CPU |
| `profile_e2e_test_memory.py` | Per-test-case memory profiling via `psutil` | Test-scoped, not pipeline-scoped; no artifact format |
| `generate_metrics.py` | JUnit/pytest timings, pipeline performance in HTML dashboards | CI/test metrics, not pipeline resource profiling |

**The gap this RFC fills:** None of the above systems produce a frozen, versioned artifact that captures memory + CPU + timing for the full pipeline under controlled conditions, comparable across releases. That is what `data/profiles/` provides.

---

## Goals

- Detect resource regressions (RSS memory, wall time, CPU utilization) across releases under identical conditions
- Build on existing `Metrics` and `EpisodeStageTimings` infrastructure — reuse timing data, add resource dimensions
- Reuse `ProviderFingerprint` for environment capture rather than inventing a new fingerprint format
- Produce a simple, YAML-first, git-committable artifact per release
- Support retroactive freeze of past releases using existing reference datasets
- Provide a terminal diff tool for quick release-over-release comparison

## Non-Goals

- Live monitoring dashboard or interactive profiling (future RFC)
- RFC-047 Streamlit extension with performance tab (future RFC)
- Real-time alerting or dashboards
- Cloud infrastructure or persistent storage
- Profiling of LLM API calls (network latency is not a local resource cost; token/cost tracking already exists in `Metrics`)
- Hard CI gates on resource metrics (data collection first; enforcement is optional and comes later)
- Per-process disk I/O tracking (not available via `psutil` on macOS; see Measurement Methodology)

---

## Design

### 1. Frozen Release Profiles (`data/profiles/`)

#### Concept

A **frozen profile** is a YAML snapshot of resource usage captured by running the fixed reference dataset (`indicator_v1` — same primary dataset used in RFC-041) through the full pipeline at a specific release. It is committed to the repo alongside the release tag and never modified after commit.

The freeze is a single entry point: `make profile-freeze VERSION=v2.6.0`. It runs headless, captures metrics, and writes the artifact.

#### Directory Structure

```text
data/profiles/
  v2.5.0.yaml
  v2.4.0.yaml
  v2.3.0.yaml
  ...
  README.md
```

One file per release, flat structure, easy to scan and diff.

#### Artifact Format

The artifact combines three data sources:

1. **Resource metrics** (new) — peak RSS and CPU utilization, captured via `psutil` polling during the run
2. **Stage timings** (existing) — wall time per stage, sourced from `workflow.metrics.Metrics` and the saved `metrics.json`
3. **Environment fingerprint** (existing) — machine and software details, sourced from `ProviderFingerprint` plus additional machine-level fields

```yaml
# data/profiles/v2.6.0.yaml

release: "v2.6.0"
date: "2026-04-09T14:32:00Z"
dataset_id: "indicator_v1"
episodes_processed: 20

# Environment — extends ProviderFingerprint with machine-level fields
environment:
  hostname: "marko-m4pro"
  cpu: "Apple M4 Pro"
  cpu_cores_physical: 14
  cpu_cores_logical: 14
  ram_total_gb: 48
  os: "macOS 15.3"
  python: "3.12.2"
  package_version: "2.5.0"
  git_sha: "a3f8c21"
  git_dirty: false
  device: "mps"
  precision: "fp32"
  library_versions:
    torch: "2.5.1"
    transformers: "4.46.3"
    whisper: "20231117"
    spacy: "3.8.4"

# Per-stage resource profile
# Stages match the actual pipeline architecture (episode-loop, not batch)
# Timings sourced from metrics.json; RSS/CPU from psutil polling
stages:
  rss_feed_fetch:
    wall_time_s: 2.1
    peak_rss_mb: 280
    avg_cpu_pct: 12.3

  media_download:
    wall_time_s: 18.4
    peak_rss_mb: 310
    avg_cpu_pct: 8.5

  audio_preprocessing:
    wall_time_s: 4.2
    peak_rss_mb: 312
    avg_cpu_pct: 45.2

  transcription:
    wall_time_s: 193.5
    peak_rss_mb: 3100
    avg_cpu_pct: 68.4
    note: "Includes model load + inference across all episodes"

  speaker_detection:
    wall_time_s: 8.3
    peak_rss_mb: 890
    avg_cpu_pct: 55.1

  transcript_cleaning:
    wall_time_s: 12.6
    peak_rss_mb: 680
    avg_cpu_pct: 22.4

  summarization:
    wall_time_s: 94.3
    peak_rss_mb: 2200
    avg_cpu_pct: 42.8

  gi_generation:
    wall_time_s: 45.2
    peak_rss_mb: 1800
    avg_cpu_pct: 38.5

  kg_extraction:
    wall_time_s: 32.1
    peak_rss_mb: 1650
    avg_cpu_pct: 35.2

  vector_indexing:
    wall_time_s: 3.8
    peak_rss_mb: 920
    avg_cpu_pct: 62.1

totals:
  peak_rss_mb: 3100
  wall_time_s: 414.5
  avg_wall_time_per_episode_s: 20.7
```

#### Stage Model — Aligned with the Actual Pipeline

The pipeline processes episodes in a loop (not in batch-per-stage). The stage model in the profile reflects this reality. Each stage aggregates timing across all episodes:

- **`rss_feed_fetch`** — maps to `Metrics.time_scraping` + `time_parsing`
- **`media_download`** — aggregated from `Metrics.download_media_time_by_episode`
- **`audio_preprocessing`** — aggregated from `Metrics.preprocessing_times`
- **`transcription`** — aggregated from `Metrics.transcribe_time_by_episode`; includes model load time (first episode) and inference (all episodes), because the model stays loaded across the episode loop
- **`speaker_detection`** — aggregated from `Metrics.extract_names_time_by_episode`
- **`transcript_cleaning`** — aggregated from `Metrics.cleaning_time_by_episode`
- **`summarization`** — aggregated from `Metrics.summarize_time_by_episode`
- **`gi_generation`** — aggregated from `Metrics.gi_times`
- **`kg_extraction`** — aggregated from `Metrics.kg_times`
- **`vector_indexing`** — wall time of `maybe_index_corpus()` (not currently tracked in `Metrics`; the freeze script measures this directly)

Wall time per stage comes from the saved `metrics.json` file (written by the pipeline to the output directory on every run). The freeze script adds the resource dimensions (RSS, CPU) by running a `psutil` polling thread alongside the pipeline and correlating samples to stages using timing boundaries.

Stages that are disabled in the reference config (e.g., if GI/KG is off) are omitted from the profile. The profile records what actually ran.

**Note on totals:** `totals.wall_time_s` is the end-to-end wall clock time of the pipeline run, not the sum of stage wall times. Stage times can overlap (e.g., concurrent download and transcription threads) or have gaps (setup, cleanup, thread synchronization), so the total may differ from the sum of stages.

#### Environment Fingerprint — Reusing ProviderFingerprint

The `environment` block reuses `ProviderFingerprint` fields (git commit, library versions, device, precision) and adds machine-level fields that `ProviderFingerprint` does not capture (hostname, CPU model, core count, total RAM, OS version). The freeze script calls `generate_provider_fingerprint()` and merges the result with `platform` module data.

All profiles captured on the same machine are directly comparable. Profiles from different machines are valid for trend analysis within that machine's history but should not be cross-compared for absolute values.

#### Retroactive Freeze

Past releases can be frozen retroactively by checking out the release tag and running the freeze script against the reference dataset:

```bash
for tag in v2.3.0 v2.4.0 v2.5.0; do
  git checkout $tag
  make profile-freeze VERSION=$tag
done
git checkout main
```

The artifact is honest about its provenance — the environment fingerprint tells the full story.

---

### 2. Freeze Script and Diff Tool

#### 2a. Freeze Script (`scripts/eval/freeze_profile.py`)

Single entry point for profile capture. Runs headless.

```bash
make profile-freeze VERSION=v2.6.0
```

Equivalent to:

```bash
.venv/bin/python3 scripts/eval/freeze_profile.py \
  --version v2.6.0 \
  --dataset indicator_v1 \
  --output data/profiles/v2.6.0.yaml
```

**What the script does:**

1. Loads the reference dataset config (same as RFC-041 benchmark runs)
2. Calls `generate_provider_fingerprint()` and collects machine-level metadata via `platform` and `psutil`
3. Starts a `psutil` polling thread (1-second interval) that records RSS and CPU% for the process
4. Runs a warm-up pass (see Measurement Methodology)
5. Calls `run_pipeline(cfg)` — the real pipeline, not a mock
6. On pipeline completion, reads the saved `metrics.json` from the output directory for per-stage wall times (the pipeline writes this file automatically via `Metrics.save_to_file()`)
7. Correlates `psutil` samples to stages using the timing boundaries from `metrics.json`
8. Writes the YAML artifact to `data/profiles/<version>.yaml`

**Accessing pipeline metrics:** `run_pipeline()` returns `Tuple[int, str]` (count and summary), not the `Metrics` object directly. The freeze script reads timing data from the `metrics.json` file that the pipeline writes to the output directory on every run. This is the same file used by the dashboard and analysis scripts — no new data path is needed.

The polling thread is lightweight (one `psutil.Process` call per second using `oneshot()` for efficiency). Its overhead on timing is negligible — well under 1% of wall time for any stage longer than a few seconds.

#### 2b. Profile Diff (`scripts/eval/diff_profiles.py`)

```bash
make profile-diff FROM=v2.5.0 TO=v2.6.0
```

Prints a structured terminal diff of two YAML profiles:

```text
Profile diff: v2.5.0 → v2.6.0 (indicator_v1, marko-m4pro)

Stage                  Metric              v2.5.0     v2.6.0     Δ
─────────────────────────────────────────────────────────────────────
transcription          peak_rss_mb         2980       3100       +4.0%
transcription          wall_time_s         201.2      193.5      -3.8%
summarization          peak_rss_mb         2200       2200       +0.0%
summarization          wall_time_s         94.3       94.3       +0.0%
totals                 peak_rss_mb         3050       3100       +1.6%
totals                 wall_time_s         398.2      414.5      +4.1%
```

The diff tool is a pure YAML comparison — no dependencies beyond PyYAML and `rich` (for terminal formatting). It reads two profile files, computes deltas, and prints a table.

If `data/profiles/regression_rules.yaml` exists, the diff tool annotates rows that exceed thresholds. If the rules file does not exist, the diff still works — it just shows raw deltas without annotations.

---

### 3. Regression Rules (Optional)

Regression rules are **not shipped in v1**. The first priority is data collection — freeze profiles for several releases and observe natural variance before setting thresholds.

When ready, rules can be added to `data/profiles/regression_rules.yaml`:

```yaml
# data/profiles/regression_rules.yaml
# Optional — diff tool works without this file

thresholds:
  peak_rss_mb:
    max_delta_pct: 20
    severity: warning

  wall_time_s:
    max_delta_pct: 15
    severity: warning

  avg_wall_time_per_episode_s:
    max_delta_pct: 15
    severity: warning

stages:
  transcription:
    peak_rss_mb:
      max_delta_pct: 10
      severity: warning
```

The diff tool reads this file if present and annotates violations. Thresholds are always advisory (warnings) — never hard gates. Enforcement can be added in a future phase once baseline history demonstrates what "normal" variance looks like on the target machine.

**Why deferred:** Without 3-5 release profiles to establish natural variance, any threshold is arbitrary. Wall time on a developer laptop varies 5-15% between runs due to background processes, thermal throttling, and memory pressure. Setting thresholds before understanding this variance risks false positives that erode trust in the system.

---

### 4. Measurement Methodology

Resource profiling on a developer machine is inherently noisy. This section documents the methodology to maximize comparability.

#### Capture Conditions

- **Close resource-heavy applications** before running `make profile-freeze` (browsers, IDEs, Docker containers not needed for the run). The profile artifact does not enforce this — it is a discipline, documented in `data/profiles/README.md`.
- **Run on AC power** (laptops throttle CPU on battery).
- **Single run per freeze** in v1. Averaging multiple runs is a future enhancement. A single run under controlled conditions is sufficient for detecting large regressions (>15-20% delta).

#### Warm-Up

The freeze script performs a **warm-up pass** before the measured run. This ensures model weights are loaded into memory, filesystem caches are warm, and MPS shaders are compiled. The warm-up works as follows:

1. The freeze script runs `run_pipeline(cfg)` once with `max_episodes=1` (overriding the reference config)
2. The warm-up run's output and metrics are written to a temporary directory and discarded
3. The measured run then processes the full reference dataset (20 episodes for `indicator_v1`) with the `psutil` polling thread active

This two-pass approach works because `run_pipeline()` is a standalone function call — the freeze script simply calls it twice with different configs. Model weights loaded during warm-up remain in memory (the freeze script does not unload them between passes).

#### What RSS Measures (and Does Not Measure)

`psutil.Process.memory_info().rss` captures the process's **resident set size** — physical memory pages currently in RAM. This is the correct metric for "how much RAM does this pipeline need?"

**Known limitation — MPS (Metal Performance Shaders):** On Apple Silicon with `device=mps`, GPU memory allocations for model weights and intermediate tensors are **not reflected in RSS**. They appear in the system's "wired" memory but not in the process's RSS. This means:

- Whisper and summarization model weights loaded on MPS are partially invisible to RSS measurement
- The RSS value underestimates true memory pressure on MPS
- For accurate GPU memory tracking on macOS, `sudo powermetrics` or Instruments would be needed — out of scope for v1

The profile artifact documents the device (`environment.device: "mps"`) so consumers know that RSS on MPS is a lower bound. On `device=cpu`, RSS is accurate. On `device=cuda`, `torch.cuda.memory_allocated()` could supplement RSS — a future enhancement.

#### CPU Measurement

`psutil.Process.cpu_percent(interval=None)` sampled at 1-second intervals. Averaged per stage. On Apple Silicon with efficiency/performance cores, this reflects the blend of core types used. It is useful for trend comparison (did CPU utilization increase?) but not for absolute core-pinning analysis.

#### Disk I/O — Not Available on macOS

`psutil.Process.io_counters()` is **not supported on macOS** — the macOS kernel does not expose per-process I/O statistics to userspace. This is a platform limitation, not a psutil bug.

Consequently, per-stage disk I/O fields (`disk_read_mb`, `disk_write_mb`) are **not included** in profiles captured on macOS. On Linux or Windows, where `io_counters()` is available, the freeze script will include these fields automatically.

System-wide disk I/O (`psutil.disk_io_counters()`) is available on macOS but cannot be attributed to the pipeline process specifically, so it is not used.

---

## Relationship to Acceptance Benchmarks

The acceptance test infrastructure (`scripts/acceptance/generate_performance_benchmark.py`) produces performance reports grouped by provider/model configuration. These reports serve a different purpose than frozen profiles:

| Concern | Acceptance Benchmarks | Frozen Profiles (this RFC) |
| --- | --- | --- |
| Purpose | Compare providers/models within a single run | Compare the same config across releases |
| Trigger | After acceptance test sessions | At release time |
| Metrics | Wall time per provider, pass/fail rates | RSS, CPU, wall time per stage |
| Grouping | By provider/model | By pipeline stage |
| Artifact | Session-scoped JSON/markdown report | Release-versioned YAML committed to git |
| Baseline | Optional session-to-session comparison | Release-to-release comparison |

The two systems are complementary. Acceptance benchmarks answer "which provider is faster for this run?" Frozen profiles answer "is this release heavier than the last one?"

A future enhancement could feed acceptance run timing data into the profile format for richer cross-validation, but v1 keeps them independent.

---

## Relationship to RFC-041

| Concern | RFC-041 | RFC-064 |
| --- | --- | --- |
| What it measures | Output quality (WER, ROUGE, gates) | Resource cost (RAM, CPU, wall time) |
| Freeze artifact | `data/eval/baselines/<id>/` | `data/profiles/<version>.yaml` |
| Freeze trigger | Baseline promotion workflow | `make profile-freeze VERSION=x` |
| Reference dataset | `indicator_v1` (primary), `shortwave_v1` (secondary) | `indicator_v1` (primary dataset only for consistency) |
| Fingerprinting | `fingerprint.json` per run | `environment:` block reusing `ProviderFingerprint` |
| Visual comparison | RFC-047 quality tab | Terminal diff tool (`make profile-diff`) |

The two systems are **intentionally separate runs**. Profiling overhead (psutil polling) is negligible, but quality evaluation concerns (scoring, regression gates) are orthogonal to resource measurement. They share the same reference episodes and fingerprinting conventions but execute independently.

**Why only `indicator_v1`:** RFC-041 uses both `indicator_v1` (clean baseline) and `shortwave_v1` (medium noise) for quality evaluation because noise characteristics affect quality metrics differently. For resource profiling, the dataset content matters less — what matters is consistent episode count, audio duration, and pipeline config. Using a single dataset simplifies the profile artifact and avoids doubling the freeze time. If a second dataset proves valuable for resource profiling, it can be added later.

---

## Directory Structure

```text
data/profiles/
  v2.3.0.yaml
  v2.4.0.yaml
  v2.5.0.yaml
  README.md

scripts/eval/
  freeze_profile.py            # Profile capture script
  diff_profiles.py             # Profile diff tool
```

No new `src/` modules are introduced in v1. The freeze script is a standalone script in `scripts/eval/` that imports from `podcast_scraper.workflow` and `podcast_scraper.evaluation.fingerprint`. The `psutil` polling logic lives in the freeze script itself — it does not need to be a reusable library module until the live monitor RFC requires it.

---

## Dependencies

### No New Python Packages

- **`psutil`** — already in `[dev]` dependencies (`psutil>=5.9.0,<8.0.0`)
- **`rich`** — already a core dependency (used for CLI output formatting)
- **`PyYAML`** — already a core dependency

No new optional dependency group is needed for v1.

### No New Infrastructure

All artifacts are local YAML files committed to git. No backend services, no database, no cloud storage.

---

## Implementation Phases

### Phase 0: Frozen profiles + freeze script (this RFC, v1 scope)

- `data/profiles/` directory and `README.md` (capture conditions, methodology)
- `scripts/eval/freeze_profile.py` — headless profile capture with warm-up pass
- `scripts/eval/diff_profiles.py` — terminal diff tool
- `make profile-freeze` and `make profile-diff` Makefile targets
- Add `vector_indexing` timing to `Metrics` (currently not tracked; needed for complete stage coverage)
- Retroactive freeze of recent releases
- Integration with future release guide (see Forward References)

### Phase 1: Regression rules (after baseline history exists)

- `data/profiles/regression_rules.yaml` with thresholds derived from observed variance
- Diff tool annotates threshold violations
- Still advisory — no hard CI gates

### Future RFCs (out of scope)

- **RFC-065: Live Pipeline Monitor** (`docs/rfc/RFC-065-live-pipeline-monitor.md`) — `--monitor` CLI flag with tmux/Terminal.app split, `rich` live dashboard, `py-spy`/`memray` flamegraph capture. macOS-specific developer tooling.
- **RFC-066: Run Comparison Tool — Performance Tab** (`docs/rfc/RFC-066-run-compare-performance-tab.md`) — Streamlit extension joining quality and resource metrics by release tag, with trend charts and quality/cost scatter plots.

---

## Forward References

### Release Guide

This RFC assumes a release guide will be created that documents the full release checklist. The profile freeze step (`make profile-freeze`) is one part of that checklist, alongside:

- Full CI pass (`make ci`)
- Quality evaluation reports in `data/eval/`
- Performance profiles in `data/profiles/`
- Release notes draft (`make release-docs-prep`)

The release guide is a separate deliverable — not part of this RFC. This RFC defines the profiling artifact and tooling; the release guide defines when and how to use them.

---

## Key Decisions

**Build on existing infrastructure, don't replace it.** The freeze script reads timing data from the pipeline's `metrics.json` output and environment data from `ProviderFingerprint`. It adds resource dimensions (RSS, CPU) via `psutil` polling. No existing code is modified — the freeze script is additive.

**Stage model matches the episode-loop architecture.** The pipeline processes episodes sequentially (or in parallel batches), not in batch-per-stage. Stages like "transcription" represent the aggregate of all episode transcriptions, including model load on first use. The profile documents this clearly.

**All pipeline stages are represented.** The stage list includes media download, audio preprocessing, transcription, speaker detection, transcript cleaning, summarization, GI generation, KG extraction, and vector indexing — matching the full set of per-episode operations tracked in `Metrics`. Stages that are disabled in the reference config are omitted from the profile.

**Single entry point, headless.** `make profile-freeze` is the only way to create a profile in v1. No interactive mode, no CLI flags on the main `podcast_scraper` command. This keeps the scope minimal and the artifact format unambiguous.

**Metrics accessed via `metrics.json`, not in-process.** The freeze script reads timing data from the pipeline's saved `metrics.json` file rather than accessing the `Metrics` object in-process. This avoids modifying `run_pipeline()` to return the metrics object and reuses the existing data path used by dashboards and analysis scripts.

**No thresholds in v1.** Data collection first. Thresholds are meaningless without baseline history showing natural variance. The diff tool works without `regression_rules.yaml` — it shows raw deltas. Rules are added when the data justifies them.

**Separate runs for quality and performance.** Mixing quality evaluation and resource profiling in one run would complicate both. They share reference datasets but execute independently.

**RSS underestimates on MPS — documented, not solved.** GPU memory on Apple Silicon is not visible to `psutil`. The profile documents the device so consumers know RSS is a lower bound on MPS. Solving this requires platform-specific tooling (Instruments, `powermetrics`) that belongs in the live monitor RFC, not here.

**No per-process disk I/O on macOS.** `psutil.Process.io_counters()` is not supported on macOS. Disk I/O fields are included only when the platform supports them (Linux, Windows). This is a platform limitation, not a design choice.

---

## Conclusion

RFC-064 adds the missing resource cost dimension to the existing evaluation framework. It builds on the project's existing timing infrastructure (`Metrics`, `EpisodeStageTimings`), environment fingerprinting (`ProviderFingerprint`), and reference datasets (RFC-041) rather than creating parallel systems. The result is a frozen, versioned profile artifact per release that makes resource regressions visible — the first step toward a complete release-time picture of quality and performance.
