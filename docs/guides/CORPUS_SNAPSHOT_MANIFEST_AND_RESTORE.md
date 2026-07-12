# Corpus snapshot manifest and restore (all surfaces)

**Normative design:** [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) · **Decision record:**
[ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md) · **Issue:**
[GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)

This page is the **single operator map**: how the same backup/restore contract is exercised **locally
via Make** versus **remotely via GitHub Actions** on production, pre-prod (Codespaces / pre-prod
paths), and DR drill. Detailed environment secrets, SSH users, and typed confirms stay in
[Prod runbook](PROD_RUNBOOK.md) and [DR drill runbook](DR_DRILL_RUNBOOK.md).

## Principles

1. **One behavior** — emit `snapshot.manifest.json`, validate, and **newest-compatible** selection
   are implemented in **`scripts/ops/corpus_snapshot/`**; **Make** and **workflows** call those
   entry points (RFC §5).
2. **Local loop** — use **`make`** on a laptop (with `gh` auth to the backup repo where needed) to
   validate manifests, rehearse tag selection, and run **`restore-corpus`** without opening prod or
   drill workflows.
3. **Remote execution** — **GitHub Actions** runs the same paths on the right project/host: prod
   restore/backup workflows for always-on VPS, drill restore for the drill stack, pre-prod/codespace
   backup via the pre-prod workflow family.

**Compatibility config:** `config/corpus_snapshot_reader_support.json` (reader min/max) and
`config/corpus_snapshot_format.json` (producer `corpus_format_version` on new backups). Bump both
when on-disk corpus layout changes in a breaking way ([ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)).

## Surface matrix

| Surface | Typical goal | Entry (target) | Detail |
| ------- | ------------ | -------------- | ------ |
| **Local dev** | Validate manifest JSON; test selection; download restore without GHA | **`make corpus-snapshot-manifest-validate`**, **`make corpus-snapshot-select-tag`**, **`make restore-corpus`** | Codespace layout (``.codespace_corpus/``). Requires network + `gh auth` for private backup repo when pulling releases. |
| **Pre-prod / Codespace** | Cloud backup tarball to backup repo; optional local pull | **`backup-corpus.yml`**; **`make codespace-backup-cloud`**; **`make restore-corpus`** | Codespace tags: `snapshot-YYYYMMDD`. |
| **Production VPS** | Corpus snapshot; controlled restore | **`backup-corpus-prod.yml`**, **`prod-restore-corpus.yml`**; on-host rehearsal: **`make restore-corpus-prod`** | Prod tags: `snapshot-prod-YYYYMMDD`; tarball top-level **`corpus/`**. |
| **DR drill** | Restore drill host from backup repo | **`drill-restore-corpus.yml`** | Uses **`snapshot-prod-*`** selection like prod. |
| **Instance-to-instance (local)** | Move a corpus laptop ↔ VPS, prod ↔ codespace, or across an airgap **without** going through the backup repo | **`make export-corpus`** + **`make import-corpus`** (#1175) | Same tarball format as `backup-corpus.yml`; no `gh` dependency; operator owns the transport (scp / USB / etc.). See [Instance-to-instance transfer](#instance-to-instance-transfer) below and the [Corpus airgap runbook](CORPUS_AIRGAP_RUNBOOK.md). |

**GitHub projects:** Workflows live on **`chipi/podcast_scraper`**. Published assets go to
**`chipi/podcast_scraper-backup`** (or `PODCAST_BACKUP_REPO`) unless overridden.

## Local testing (Make-first)

1. **Validate** — `make corpus-snapshot-manifest-validate FILE=path/to/snapshot.manifest.json`
2. **Select** — `make corpus-snapshot-select-tag` or **`make corpus-snapshot-select-tag-prod`**
   (optional `TAG_REGEX=…`, `PODCAST_BACKUP_TAG=…`, `PODCAST_BACKUP_REPO=…`); exits non-zero when
   no compatible release has a sibling manifest.
3. **Restore** — **`make restore-corpus`** (codespace layout; unset `PODCAST_BACKUP_TAG` → newest
   compatible `snapshot-YYYYMMDD`). On a prod VPS or local prod rehearsal: **`make restore-corpus-prod`**
   (`snapshot-prod-*`, top-level **`corpus/`**). Pin with `PODCAST_BACKUP_TAG=…`. See
   [VPS restore: Make vs GitHub Actions](#vps-restore-make-vs-github-actions).
4. **Selftest** — `make corpus-snapshot-selftest` (fixture validation + finalize smoke; no `gh`).

**Verify prod asset without restore:** `scripts/ops/verify_prod_backup_snapshot.sh` (manifest +
tarball inspection).

## GitHub Actions (remote hosts)

| Workflow | Surface | Role |
| -------- | ------- | ---- |
| `backup-corpus.yml` | Pre-prod / Codespace | Tarball + **`finalize_backup_bundle.sh`** → upload `snapshot.tgz` + `snapshot.manifest.json` when **`dry_run`** is false. |
| `backup-corpus-prod.yml` | Prod VPS | Same for `/srv/podcast-scraper/corpus` (default **`dry_run=true`** until operator sets false). |
| `prod-restore-corpus.yml` | Prod | **`resolve_latest_snapshot_prod_tag.sh`** then **`download_and_verify_snapshot.sh`**; host extract: **`restore_corpus_from_tarball_host.sh`**. |
| `drill-restore-corpus.yml` | DR drill | Same runner download/verify path; same host restore script on drill **`deploy@`**. |

See [WORKFLOWS.md](../ci/WORKFLOWS.md) and the two runbooks for confirms and secrets.

## VPS restore: Make vs GitHub Actions

Both paths share **`scripts/ops/corpus_snapshot/`** for tag selection and tarball verification
([ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)). They
diverge **after extract** on the VPS:

| Path | When | After extract |
| ---- | ---- | ------------- |
| **`make restore-corpus-prod`** | On-host rehearsal, migration SSH, laptop prod-layout rehearsal | Downloads and verifies assets, extracts top-level **`corpus/`** under **`WORKSPACE_DIR`** (default **`/srv/podcast-scraper`**). Does **not** recreate containers — recycle **`api`** + **`viewer`** per [Prod runbook](PROD_RUNBOOK.md) corpus migration. |
| **`prod-restore-corpus.yml`** / **`drill-restore-corpus.yml`** | Controlled GHA restore (typed confirm on manual runs) | Runner uploads tarball + **`restore_corpus_from_tarball_host.sh`**; host script backs up prior **`corpus/`**, extracts, asserts **`corpus/`**, **`compose up --force-recreate api viewer`**, in-container **`/api/health`** on **`api`** `:8000`. |

Codespace layout restore stays on **`make restore-corpus`** (``.codespace_corpus/`` in tarball).

## Instance-to-instance transfer

For moving a corpus between two instances **without** publishing to `chipi/podcast_scraper-backup`
(#1175): use `make export-corpus` on the source and `make import-corpus` on the target. The
resulting tarball is bit-format identical to what `backup-corpus.yml` produces, so it is
consumable by the CI restore path — and vice versa: a CI-produced `snapshot.tgz` sitting on
local disk works with `make import-corpus` without any `gh` calls.

### Export (source instance)

```bash
make export-corpus \
    CORPUS_DIR=/path/to/corpus_root \
    OUT=/tmp/snapshot.tgz \
    [LAYOUT=codespace|prod]        # default: codespace
```

- `CORPUS_DIR` must be the corpus root (the parent that contains `feeds.spec.yaml`).
- `LAYOUT=codespace` → archive root is `.codespace_corpus/`. `LAYOUT=prod` → `corpus/`.
- Producer identity (`git_sha`) falls back to `git rev-parse HEAD` when neither `GITHUB_SHA`
  nor `GIT_SHA` is set in the environment.
- Sanity checks refuse to pack: missing `feeds.spec.yaml`, no `*.gi.json` under the tree,
  or a suspiciously small tarball (< 1 KiB — override via
  `CORPUS_SNAPSHOT_MIN_TARBALL_BYTES` for tests only).
- Output: `OUT` (the tarball) plus a sibling `snapshot.manifest.json` next to it, carrying
  `archive.sha256`.

### Import (target instance)

```bash
make import-corpus \
    FILE=/tmp/snapshot.tgz \
    WORKSPACE_DIR=/path/to/target_parent \
    [LAYOUT=codespace|prod]        # default: codespace
```

- Prefers the sibling `snapshot.manifest.json` (has `archive.sha256`); falls back to the
  inner one at archive root when the sibling is missing.
- Runs the same `validate_snapshot_manifest.sh` + reader-range check
  (`config/corpus_snapshot_reader_support.json`) that the CI restore path runs after
  `gh release download`.
- Verifies `archive.sha256` when the sibling manifest carries one. Skip with
  `CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1` (only when you already trust the transport).
- Refuses to overwrite an existing `<workspace>/.codespace_corpus/` or `<workspace>/corpus/`
  — move or remove the prior tree first to avoid clobbering live data.
- No `gh` dependency. Works on a fully offline host as long as the tarball is on disk.

### Non-goals

- The pair does not add a network transport. Moving the tarball between hosts (scp, S3
  CLI, USB) is the operator's choice; see the runbook for worked recipes.
- It does not replace the CI backup path. `backup-corpus.yml` / `backup-corpus-prod.yml`
  stay authoritative for scheduled snapshots to `chipi/podcast_scraper-backup`.

Detailed operator recipes live in the [Corpus airgap runbook](CORPUS_AIRGAP_RUNBOOK.md).

## When newest-compatible default is wrong

Default restore (unset tag) picks the **newest** backup whose sibling **`snapshot.manifest.json`**
reports a **`corpus_format_version`** inside `config/corpus_snapshot_reader_support.json`. That is
**not** always the right choice:

- **Rollback** — deploy an older app build but need a corpus from an **older** compatible release:
  pin **`PODCAST_BACKUP_TAG`** / workflow **`backup_tag`**.
- **Format bump** — after a breaking corpus layout change, older releases may be skipped until
  manifests exist; if none match, selection **fails closed** (pin an explicit tag or publish a new
  backup).
- **Mixed-age hosts** — reader range in the repo may differ from what is deployed on a host; pin or
  align config before restore.

Pre-manifest releases (no sibling manifest) are skipped by default selection until a post-merge backup
exists.

## Related

- [Stack contract](STACK_CONTRACT.md) — cross-surface compose, health, and gate audit table
  ([ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md)).
- [Prod runbook](PROD_RUNBOOK.md) — prod SSH, secrets, **`prod-restore-corpus.yml`** (primary restore).
- [DR drill runbook](DR_DRILL_RUNBOOK.md) — drill-only confirms and destroy rules.
- [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md) — prod vs drill
  topology.
