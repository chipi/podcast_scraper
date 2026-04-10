# Frozen release profiles (RFC-064)

YAML snapshots under this directory capture **resource cost** (peak RSS, CPU%, wall time per
pipeline stage) for a fixed reference run, versioned by release tag (for example `v2.6.0.yaml`).

See [RFC-064](https://github.com/chipi/podcast_scraper/blob/main/docs/rfc/RFC-064-performance-profiling-release-freeze.md) and the **[Performance Profile Guide](../../docs/guides/PERFORMANCE_PROFILE_GUIDE.md)** (methodology, E2E vs real RSS, troubleshooting).

## Capture workflow

1. Use a **pipeline config** YAML under [`config/profiles/`](../config/profiles/).
   Presets **`capture_e2e_*.yaml`** use the same **E2E mock RSS** pattern as
   `sample_acceptance_e2e_fixture_single.yaml` (no real feed; `freeze_profile`
   starts the fixture server automatically).
2. Close heavy background apps, use **AC power** on laptops, and run when the machine is idle.
3. From the repo root:

   ```bash
   make profile-freeze VERSION=v2.6.0-ml-dev \
     PIPELINE_CONFIG=config/profiles/capture_e2e_ml_dev.yaml \
     DATASET_ID=e2e_podcast1_mtb_n2
   ```

4. Commit `data/profiles/<version>.yaml` and the companion
   `data/profiles/<version>.stage_truth.json` (trimmed metrics excerpt, per-stage walls, psutil
   attribution, sampling interval) with the release. Disable the JSON with `NO_STAGE_TRUTH=1` on
   `make profile-freeze` if needed.

`make profile-diff FROM=v2.5.0 TO=v2.6.0` prints a terminal comparison of two profiles.

## Interpretation

- **Same hostname** — profiles are directly comparable for regression spotting.
- **Different machines** — useful for trends on that machine only; do not compare absolute RSS
  across different hardware.
- **`device: mps` (Apple Silicon)** — RSS is a **lower bound**; GPU memory is not included in
  process RSS. See RFC-064 Measurement Methodology.

## Retroactive freeze

Check out a release tag, use a pipeline config appropriate for that era, run `make profile-freeze`,
then merge the YAML on `main`.

## Optional regression rules

If `data/profiles/regression_rules.yaml` exists, `diff_profiles.py` can annotate large deltas
(advisory only; see RFC-064 Phase 1).
