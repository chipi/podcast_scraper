# Performance Reports

> **Frozen resource-cost snapshots** (RFC-064): wall time, peak RSS, and per-stage
> attribution from `make profile-freeze`, separate from **quality** evaluation in
> [Evaluation Reports](../eval-reports/index.md).

This index explains how to read performance reports and lists each published snapshot.
For capture workflow, methodology, and interpretation caveats, see the
[Performance Profile Guide](../PERFORMANCE_PROFILE_GUIDE.md) and
[RFC-064](../../rfc/RFC-064-performance-profiling-release-freeze.md).

---

## What we measure (vs evaluation reports)

| Track | Question | Artifacts |
| ----- | -------- | --------- |
| **Evaluation reports** | Did **summary quality** change? | ROUGE, embeddings, runs under `data/eval/` |
| **Performance reports** | Did **resource cost** change? | `data/profiles/<tag>.yaml` (RSS, CPU%, wall per stage) |

Performance profiles do **not** replace eval runs: they answer a different question.

---

## Methodology (short)

- **Capture:** `make profile-freeze VERSION=<tag> PIPELINE_CONFIG=config/profiles/<preset>.yaml`
  with optional `SKIP_WARMUP=1`, `DATASET_ID=…` (see
  [`config/profiles/README.md`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md)).
- **Fixture:** E2E presets use mock RSS (`podcast1_mtb`), **`max_episodes: 2`**, unless a
  report states otherwise.
- **Comparability:** Compare profiles on the **same `hostname`** only; absolute RSS is
  not portable across machines.
- **Interpretation:** `totals.wall_time_s` is end-to-end wall clock; summed stage walls
  can exceed it under **parallelism**. Missing stages often mean **zero** time in
  `metrics.json` (e.g. transcript cache). See
  [Interpreting the profile](../PERFORMANCE_PROFILE_GUIDE.md#interpreting-the-profile).

---

## Report library

Each report is an immutable snapshot: new runs get a **new** file and table row.

| Report | Date | Dataset label | Variants | Notes |
| ------ | ---- | --------------- | -------- | ----- |
| [E2E WIP v1 (2026-04)](PERF_E2E_WIP_V1_2026_04.md) | Apr 2026 | `e2e_podcast1_mtb_n2` | 11 pipeline presets | WIP tags `v2.6-wip-*`; many runs used `SKIP_WARMUP=1` |

---

## How to add a new report

1. Capture profiles (see [Performance Profile Guide](../PERFORMANCE_PROFILE_GUIDE.md)).
2. Commit `data/profiles/<version>.yaml` when ready for the record.
3. Add `docs/guides/performance-reports/PERF_<CAMPAIGN>_V<NN>_<YYYY_MM>.md` with tables,
   hostname, flags (`SKIP_WARMUP`, models), and caveats.
4. Add a row to **Report library** above and an entry under **Performance Reports** in
   `mkdocs.yml`.

---

## Related documentation

- [Performance Profile Guide](../PERFORMANCE_PROFILE_GUIDE.md) — operator manual
- [Performance](../PERFORMANCE.md) — runtime tuning (not release YAML)
- [Evaluation Reports](../eval-reports/index.md) — quality metrics and provider sweeps
- [AI Provider Comparison Guide](../AI_PROVIDER_COMPARISON_GUIDE.md) — provider choice
- [RFC-064](../../rfc/RFC-064-performance-profiling-release-freeze.md) — design
