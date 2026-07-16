# Corpus airgap runbook — instance-to-instance corpus transfer

**Scope:** move a corpus between two instances **without** publishing to `chipi/podcast_scraper-backup`.
Covers laptop ↔ VPS, prod ↔ codespace, and airgapped restores.
**SSOT for corpus snapshot ops:** [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).
**Related:** [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) ·
[ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md) ·
[GitHub #1175](https://github.com/chipi/podcast_scraper/issues/1175).

This runbook covers the **hand-carried** path only. For the scheduled backup / restore flow that
publishes to the backup repo, use `backup-corpus.yml` / `backup-corpus-prod.yml` /
`prod-restore-corpus.yml` and their two dedicated runbooks
([Prod runbook](PROD_RUNBOOK.md), [DR drill runbook](DR_DRILL_RUNBOOK.md)).

## When to use this path

| Situation | Use this runbook | Use the CI path instead |
| --------- | ---------------- | ----------------------- |
| One-off transfer between two instances | ✅ | |
| Airgap: target has no `gh` / no GitHub access | ✅ | |
| Corpus surgery on a laptop, need to ship result to prod | ✅ | |
| Scheduled daily / weekly backups | | ✅ (`backup-corpus[-prod].yml`) |
| Prod → drill for a DR exercise | | ✅ (`drill-restore-corpus.yml`) |
| Publishing an artifact to the backup repo | | ✅ |

## Pre-flight checklist (source instance)

- [ ] Corpus root identified (the parent that contains `feeds.spec.yaml` — usually
      `.codespace_corpus/` or `/srv/podcast-scraper/corpus/`).
- [ ] At least one `*.gi.json` exists under the tree (`find <corpus_root> -name '*.gi.json' | head`).
      The pack step refuses an empty corpus (mirrors `backup-corpus.yml` sanity check; see #699).
- [ ] Free space at `OUT`'s parent is at least 2× corpus size (raw copy staging + tarball).
- [ ] `git`, `tar`, `jq` available (all standard on macOS/Linux).
- [ ] Producer identity: either `GITHUB_SHA` is set (CI environments), or `git rev-parse HEAD`
      resolves at the repo root (default fallback), or `GIT_SHA` is set explicitly.

## Pre-flight checklist (target instance)

- [ ] Target workspace parent chosen. **Do not** point at a directory that already contains
      `.codespace_corpus/` or `corpus/` — the import refuses to overwrite live data.
- [ ] Reader-range config `config/corpus_snapshot_reader_support.json` matches the code deployed on
      the target host (the import checks `corpus_format_version` against this range).
- [ ] `tar`, `jq`, and either `sha256sum` (Linux) or `shasum` (macOS) available.

## Recipe: laptop → VPS

Source: laptop with the corpus at `~/work/corpus/`. Target: prod VPS.

```bash
# 1. On the laptop — pack.
make export-corpus \
    CORPUS_DIR=$HOME/work/corpus \
    OUT=/tmp/corpus-$(date -u +%Y%m%d).tgz \
    LAYOUT=prod

# 2. Transport. Ship the tarball AND the sibling manifest together; the sibling
#    carries archive.sha256 which import uses for integrity.
scp /tmp/corpus-*.tgz /tmp/snapshot.manifest.json \
    deploy@prod-vps.example:/srv/staging/

# 3. On the VPS — import (as the deploy user).
ssh deploy@prod-vps.example
make -C /opt/podcast-scraper import-corpus \
    FILE=/srv/staging/corpus-YYYYMMDD.tgz \
    WORKSPACE_DIR=/srv/podcast-scraper \
    LAYOUT=prod

# 4. Sanity check + container recycle (mirrors prod-restore-corpus.yml host step).
make -C /opt/podcast-scraper corpus-compat-check \
    CORPUS_DIR=/srv/podcast-scraper/corpus
docker compose -f /opt/podcast-scraper/compose/docker-compose.prod.yml \
    up -d --force-recreate api viewer
```

## Recipe: prod VPS → codespace

Source: prod VPS. Target: local codespace (dev laptop).

```bash
# 1. On the VPS — pack with codespace layout.
sudo make -C /opt/podcast-scraper export-corpus \
    CORPUS_DIR=/srv/podcast-scraper/corpus \
    OUT=/tmp/prod-snapshot.tgz \
    LAYOUT=codespace

# 2. Pull down. Retrieve both files.
scp deploy@prod-vps.example:/tmp/prod-snapshot.tgz \
    deploy@prod-vps.example:/tmp/snapshot.manifest.json \
    ./

# 3. Import into the codespace bind-mount path.
make import-corpus \
    FILE=$PWD/prod-snapshot.tgz \
    WORKSPACE_DIR=$PWD \
    LAYOUT=codespace
# result: $PWD/.codespace_corpus/
```

## Recipe: airgap sneakernet (USB)

Source has no network. Target has no network. Transfer is a USB stick.

```bash
# 1. Source: pack.
make export-corpus \
    CORPUS_DIR=/data/corpus \
    OUT=/media/usb/snapshot.tgz \
    LAYOUT=prod
# result on the USB stick: snapshot.tgz + snapshot.manifest.json

# 2. Physically move the USB stick.

# 3. Target: import (offline; no `gh`, no network).
make import-corpus \
    FILE=/media/usb/snapshot.tgz \
    WORKSPACE_DIR=/srv/podcast-scraper \
    LAYOUT=prod
```

## Post-import: apply pending upgrade migrations

Importing a snapshot from an older source into a target running newer code will
land a corpus that needs one or more registered migrations (`m0002_two_tier_native_reindex`,
`m0003_gi_v3_typed_mentions`, and any future ones — see
[Corpus upgrade framework (#862)](CORPUS_UPGRADE.md)). **The import step does
not run these for you.** Run the upgrade path immediately after import so the
next reader / API call is served from a fully-migrated corpus:

```bash
CORPUS_DIR=<workspace>/<layout-root>

# Check what will run. Exits 2 when migrations are pending.
make upgrade-check CORPUS_DIR=$CORPUS_DIR

# Apply. Non-interactive; idempotent (already-recorded migrations skip).
make upgrade-corpus CORPUS_DIR=$CORPUS_DIR

# Confirm the ledger.
make upgrade-verify CORPUS_DIR=$CORPUS_DIR
```

On a prod VPS with the API stack running, do this BEFORE re-creating the api
container (the container needs to boot on the migrated corpus). The
`prod-restore-corpus.yml` GHA restore path runs this automatically via
`scripts/ops/restore_corpus_from_tarball_host.sh` (#1176); local airgap imports
need this step by hand.

## Verification

After every import:

```bash
# Manifest + reader-range are checked automatically by import.
# Additionally, run the corpus/code compat check to catch a mismatched deploy:
make corpus-compat-check CORPUS_DIR=<workspace>/<layout-root>

# Count what landed vs what the source reported.
find <workspace>/<layout-root>/transcripts -name '*.gi.json' | wc -l
```

Compare the count against what `find <corpus_root>/transcripts -name '*.gi.json' | wc -l` produced
on the source. They must match exactly — a mismatch means a partial transport (truncated scp,
corrupt USB) that slipped past the `archive.sha256` check somehow (should be impossible; investigate).

## Failure modes

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `feeds.spec.yaml missing` on pack | You pointed `CORPUS_DIR` at a subdirectory instead of the corpus root | Point at the parent containing `feeds.spec.yaml`. |
| `no *.gi.json artifacts` on pack | Corpus was never enriched, or wrong directory | Verify the source path; if truly empty, resolve at the source before packing. |
| `sha256 mismatch` on import | Sibling manifest was tampered, OR the tarball was truncated in transit | Re-copy the tarball. If persistent, re-pack from source. Only set `CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1` when you trust the transport (e.g. same-host cp). |
| `corpus_format_version outside reader range` | Producer version is newer than the target code understands | Upgrade the target code (`git pull` then rebuild), OR pack from an older source that matches the target's reader range. |
| `archive top-level does not contain 'X/'` on import | Layout mismatch — you packed with `LAYOUT=codespace` and are importing with `LAYOUT=prod` (or vice versa) | Retry the import with the correct layout. The tarball is unchanged; only the wrapper needs the matching flag. |
| `<workspace>/<layout-root> already exists` on import | Target already has a corpus | Move/rename the prior one first (`mv corpus corpus.pre-import-$(date -u +%Y%m%d%H%M)`) to preserve rollback. |
| Pack produced a tarball but no sibling manifest | `finalize_backup_bundle.sh` failed after tar; check its stderr | Delete the partial tarball and retry pack; `finalize` needs `jq` + `tar` + a sha256 tool. |

## Rollback

Before every import that overwrites a live corpus, keep the prior tree:

```bash
mv <workspace>/<layout-root> <workspace>/<layout-root>.pre-import-$(date -u +%Y%m%d%H%M)
```

If the import lands and later reveals a problem:

```bash
# Stop containers first (prod).
docker compose -f compose/docker-compose.prod.yml stop api viewer

mv <workspace>/<layout-root> <workspace>/<layout-root>.reverted
mv <workspace>/<layout-root>.pre-import-<ts> <workspace>/<layout-root>

docker compose -f compose/docker-compose.prod.yml up -d --force-recreate api viewer
```

Rollback should complete in under 5 minutes because the prior corpus stays on disk until you
explicitly remove it.

## Contract with the CI backup path

The tarball produced by `make export-corpus` is **bit-format identical** to what
`backup-corpus.yml` / `backup-corpus-prod.yml` produce:

- Same layout root (`.codespace_corpus/` or `corpus/`).
- Same inner + sibling `snapshot.manifest.json` (via `finalize_backup_bundle.sh`).
- Same `archive.sha256`.
- Same `schema_version` + `corpus_format_version` fields (via `emit_manifest.sh`).

Consequences:

- A **locally exported** tarball can be uploaded to a `chipi/podcast_scraper-backup` release
  manually (`gh release create ...`) and picked up by `make restore-corpus` on any other host.
- A **CI-produced** `snapshot.tgz` downloaded from a backup repo release is directly consumable
  by `make import-corpus` without any `gh` calls — useful when the restore host has no
  backup-repo credentials.

## See also

- [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) — SSOT for
  every corpus snapshot surface (this runbook is the airgap slice).
- [Prod runbook](PROD_RUNBOOK.md) — prod SSH, secrets, container recycle steps.
- [DR drill runbook](DR_DRILL_RUNBOOK.md) — drill-only confirms and destroy rules.
- [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) — manifest spec.
- [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md) —
  version-aware selection.
