# Frozen release profiles (RFC-064)

YAML snapshots under this directory capture **resource cost** (peak RSS,
CPU%, wall time per pipeline stage) for a fixed reference run, versioned
by release tag (for example `v2.6.0.yaml`).

See [RFC-064](https://github.com/chipi/podcast_scraper/blob/main/docs/rfc/RFC-064-performance-profiling-release-freeze.md)
and the
**[Performance Profile Guide](../../docs/guides/PERFORMANCE_PROFILE_GUIDE.md)**
(methodology, E2E vs real RSS, troubleshooting).

## Active reference profiles

Promoted profiles live in `references/`. They are immutable once
promoted.

| Reference | Model | Dataset | Episodes | Date | Promoted from |
| :--- | :--- | :--- | :---: | :--- | :--- |
| *(none yet -- promote with `make profile-promote`)* | | | | | |

## Lifecycle

```text
capture ──> iterate ──> promote ──> compare
                                       │
                                   archive (when superseded)
```

1. **Capture** -- `make profile-freeze` writes a working profile to
   `data/profiles/<version>.yaml`
2. **Iterate** -- feature branches capture to subdirectories
   (e.g. `issue-477/`)
3. **Promote** -- `make profile-promote` copies a working profile into
   `references/` with a `promoted` metadata block. References are
   immutable.
4. **Compare** -- `make profile-diff FROM=... TO=...` compares any two
   profiles (working or reference)
5. **Archive** -- when a reference is superseded, move it to `_archive/`
   with a note in `_archive/README.md`. Never delete.

**Optional RFC-065 monitor during capture:** `make profile-freeze … MONITOR=1` (or
`freeze_profile.py --monitor`) archives **`<VERSION>.monitor.log`** next to the YAML and records
metadata in **`stage_truth.json`**. See the
[Performance Profile Guide](../../docs/guides/PERFORMANCE_PROFILE_GUIDE.md).

## Naming conventions

- **Working profiles:** `v2.6-wip-<variant>.yaml` or
  `<feature>/<name>.yaml`
- **Reference profiles:** `v2.6.0-<variant>.yaml` (semver, no "wip")
- **Stage truth companion:** `<same-stem>.stage_truth.json` (required
  for references unless overridden)

## Capture workflow

1. Use a **pipeline config** YAML under
   [`config/profiles/freeze/`](../config/profiles/freeze/). Those profiles merge
   **`freeze/_defaults.yaml`** for placeholder RSS and paths; `freeze_profile.py`
   starts the **E2E mock RSS** server when the URL is the acceptance placeholder
   (same family as `config/acceptance/fragments/feeds_single.yaml`).
2. Close heavy background apps, use **AC power** on laptops, and run
   when the machine is idle.
3. From the repo root:

   ```bash
   make profile-freeze VERSION=v2.6.0-ml-dev \
     PIPELINE_CONFIG=config/profiles/freeze/ml_dev.yaml \
     DATASET_ID=e2e_podcast1_mtb_n2
   ```

4. Commit `data/profiles/<version>.yaml` and the companion
   `data/profiles/<version>.stage_truth.json` (trimmed metrics excerpt,
   per-stage walls, psutil attribution, sampling interval) with the
   release. Disable the JSON with `NO_STAGE_TRUTH=1` on
   `make profile-freeze` if needed.

## Promotion workflow

When a working profile is ready to become a release reference:

```bash
make profile-promote \
  SOURCE=data/profiles/v2.6-wip-openai.yaml \
  PROMOTED_ID=v2.6.0-openai \
  REASON="Release v2.6.0 OpenAI reference profile"
```

This copies the YAML (and stage_truth.json if present) into
`references/` and stamps a `promoted` metadata block into the YAML.

Preview first with `DRY_RUN=1`. Allow missing stage_truth with
`NO_STAGE_TRUTH_REQUIRED=1`.

## Comparing profiles

`make profile-diff FROM=v2.5.0 TO=v2.6.0` prints a terminal comparison
of two profiles.

For references, use the path relative to `data/profiles/`:

```bash
make profile-diff \
  FROM=references/v2.6.0-openai \
  TO=issue-477/issue477-staged-gpt4o
```

## Interpretation

- **Same hostname** -- profiles are directly comparable for regression
  spotting.
- **Different machines** -- useful for trends on that machine only; do
  not compare absolute RSS across different hardware.
- **`device: mps` (Apple Silicon)** -- RSS is a **lower bound**; GPU
  memory is not included in process RSS. See RFC-064 Measurement
  Methodology.

## Retroactive freeze

Check out a release tag, use a pipeline config appropriate for that era,
run `make profile-freeze`, then merge the YAML on `main`.

## Optional regression rules

If `data/profiles/regression_rules.yaml` exists, `diff_profiles.py` can
annotate large deltas (advisory only; see RFC-064 Phase 1).
