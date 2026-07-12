# Local corpus export/import — instance-to-instance portability without CI (#1175)

**Status:** implementation in progress
**Branch:** `feat/enterprise-hardening`
**Related:** RFC-084, ADR-092, existing SSOT guide `docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md`

## Problem

We can back up + restore a corpus via CI (`backup-corpus.yml` / `backup-corpus-prod.yml` push to `chipi/podcast_scraper-backup`, `make restore-corpus[-prod]` pulls from a release tag). We cannot yet:

1. Produce a portable `snapshot.tgz` from a live corpus **with one make command** — the tar step lives inline in `.github/workflows/backup-corpus.yml`.
2. Restore from a `snapshot.tgz` that arrived by any transport **other than a GitHub release** — `restore-corpus` always calls `gh release download`.

This blocks: laptop ↔ VPS moves, prod ↔ codespace transplants, airgapped restores, and any "here's a tarball, restore it" flow.

## Design principles

- **One format.** A locally-packed tarball is bit-format-identical to a CI-packed one. The CI-restore path can consume a local export; the local-import path can consume a CI backup. No format fork.
- **Reuse manifest primitives.** `emit_manifest.sh`, `finalize_backup_bundle.sh`, `validate_snapshot_manifest.sh`, `download_and_verify_snapshot.sh`'s post-download validation, and `lib.sh` all stay untouched. The two new scripts are wrappers.
- **No network in the local path.** Import must NOT depend on `gh` or GitHub. If the user has the tarball on disk, that's enough.
- **Same layout roots.** Codespace layout → archive root `.codespace_corpus/`; prod layout → archive root `corpus/`. Matches what `restore-corpus` / `restore-corpus-prod` already extract.
- **Producer identity still lands.** `git rev-parse HEAD` is the local fallback when `GITHUB_SHA` is unset.

## Interface

### Export

```
make export-corpus \
    CORPUS_DIR=<path to corpus root>        # required
    OUT=<path to output tarball>            # required, e.g. /tmp/snapshot.tgz
    LAYOUT=codespace|prod                   # optional, default codespace
```

- Copies the corpus tree into a temp workdir under the layout root.
- Packs with `tar -czf`.
- Calls `finalize_backup_bundle.sh` to inject manifest + write sibling manifest with `archive.sha256`.
- Produces two files: `$OUT` and `$(dirname $OUT)/snapshot.manifest.json`.

Sanity checks (mirrors `backup-corpus.yml`):

- corpus root must contain `feeds.spec.yaml`;
- corpus tree must contain ≥1 `*.gi.json`;
- final tarball size ≥ 1 KiB (override for tests via `CORPUS_SNAPSHOT_MIN_TARBALL_BYTES`).

### Import

```
make import-corpus \
    FILE=<path to snapshot.tgz>             # required
    WORKSPACE_DIR=<target parent dir>       # required
    LAYOUT=codespace|prod                   # optional, default codespace
```

- Prefers sibling `snapshot.manifest.json` (produced by finalize) if present next to `FILE`; else pulls the inner manifest from the archive root.
- Runs `validate_snapshot_manifest.sh` on it.
- Runs reader-range check against `config/corpus_snapshot_reader_support.json` (same as `download_and_verify_snapshot.sh`).
- If sibling manifest carries `archive.sha256`, verifies the tarball digest. Skip flag: `CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1`.
- Refuses layout mismatch: if the tarball root is `corpus/` but `LAYOUT=codespace` (or vice versa), error out and tell the operator which layout to pick.
- Extracts into `WORKSPACE_DIR`. Result: `WORKSPACE_DIR/.codespace_corpus/` or `WORKSPACE_DIR/corpus/`.

## New scripts

- `scripts/ops/corpus_snapshot/pack_corpus_local.sh`
- `scripts/ops/corpus_snapshot/import_local_snapshot.sh`

Both source `scripts/ops/corpus_snapshot/lib.sh`. Both `set -euo pipefail`. Both take long-form flags to match the existing script style.

## Compatibility with the CI flow

| Concern | Local export/import | Existing CI backup/restore |
| ------- | ------------------- | -------------------------- |
| Tarball layout | Same (layout root at archive root) | Same |
| Manifest schema | Same (via `emit_manifest.sh`) | Same |
| Reader-range check | Same helper | Same helper |
| Producer identity | `git rev-parse HEAD` fallback | `GITHUB_SHA` |
| Transport | User's choice | GH release |
| Auth | None | `gh` to backup repo |

**Round-trip contract:** an operator can `make export-corpus` on laptop A, transport the tarball however they like, and `make import-corpus` on host B, and the resulting corpus tree is byte-identical to what would land if the same corpus had round-tripped via the CI backup flow.

## Test plan

- Unit tests (`tests/unit/scripts/test_corpus_local_export_import.py`):
  - Argument parsing, layout switch, sanity refusals (empty corpus, missing `feeds.spec.yaml`, unknown layout), git-sha fallback.
  - Sibling-vs-inner manifest selection, sha256 mismatch refusal, layout-mismatch refusal, existing-target refusal, skip-sha env, missing manifest.
  - Round-trip parametrized over both layouts using `filecmp.dircmp`.

## Docs deliverables

- Extend `docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md` — add "Instance-to-instance transfer" section and a new matrix row.
- New `docs/guides/CORPUS_AIRGAP_RUNBOOK.md` — step-by-step ops runbook (pre-flight, transport, verification, rollback, failure-mode table).
- README ops section — one-bullet pointer to the new guide.
- AGENTS.md — brief entry so future agents pick these targets over CI backup when instance-to-instance is the actual need.
- mkdocs.yml — nav entry for the new runbook.
- `make help` — new lines under the corpus-snapshot section.

## Open questions

- Do we want a `--dry-run` on both targets? *Deferred until first operator use — easy to add later.*
- Should import verify the corpus is compatible with the current code (`assess_corpus_version_compat`)? *Yes — reuse `corpus-compat-check` semantics as a post-extract step, but only WARN, don't fail. Matches the current `restore-corpus` behavior.*
- Should we support importing from a URL (curl-then-import)? *No — that's a transport concern, not the wrapper's job.*

## Non-goals (restated for the record)

- Not replacing the CI backup path.
- Not adding delta snapshots.
- Not adding encryption at rest.
- Not adding a remote-to-remote transport.
