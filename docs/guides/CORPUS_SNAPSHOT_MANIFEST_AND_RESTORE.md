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
   are implemented in **repo scripts**; **Make** and **workflows** call those entry points (RFC §5).
2. **Local loop** — use **`make`** on a laptop (with `gh` auth to the backup repo where needed) to
   dry-run selection, validate fixtures, and run **`restore-corpus`** without opening prod or drill
   workflows.
3. **Remote execution** — **GitHub Actions** runs the same paths on the right project/host: prod
   restore/backup workflows for always-on VPS, drill restore for the drill stack, pre-prod/codespace
   backup via the pre-prod workflow family.

Until scripts and workflow wiring land, behavior remains **tag/date** driven; after implementation,
this page stays the hub for **which surface runs which target**.

## Surface matrix

| Surface | Typical goal | Entry (target) | Detail |
| ------- | ------------ | -------------- | ------ |
| **Local dev** | Validate manifest JSON; test selection logic; download restore without GHA | **`make`** targets calling `scripts/ops/…` (names TBD in RFC rollout); existing **`make restore-corpus`** with `PODCAST_BACKUP_*` | Requires network + `gh auth` for private backup repo when pulling releases. |
| **Pre-prod / Codespace** | Cloud backup tarball to backup repo; optional local pull | **`.github/workflows/backup-corpus.yml`**; **`make codespace-backup-cloud`** (dispatch); **`make restore-corpus`** on the machine that holds corpus path | [Makefile restore-corpus](https://github.com/chipi/podcast_scraper/blob/main/Makefile); pre-prod layout in [RFC-081](../rfc/RFC-081-pre-prod-environment-and-control-plane.md). |
| **Production VPS** | Scheduled/manual corpus snapshot; controlled restore | **`backup-corpus-prod.yml`**, **`prod-restore-corpus.yml`** | [Prod runbook — backup / restore](PROD_RUNBOOK.md) (search **restore-corpus**, **backup-corpus-prod**). |
| **DR drill** | Restore drill host from backup repo; full cycle under drill confirms | **`drill-restore-corpus.yml`** (and orchestrator that includes restore) | [DR drill runbook](DR_DRILL_RUNBOOK.md); same backup repo conventions as prod unless runbook says otherwise. |

**GitHub projects:** All workflows above live on **`chipi/podcast_scraper`** (application repo).
Published assets go to **`chipi/podcast_scraper-backup`** (or `PODCAST_BACKUP_REPO`) unless you
override. DR drill **does not** change that contract by itself — it changes **which host** receives
the restore, not the manifest schema.

## Local testing (Make-first)

Use this order once implementation exists:

1. **Validate** — `make` target runs manifest schema checks on a file path (fixture or downloaded
   `snapshot.manifest.json`).
2. **Select (dry-run)** — `make` target lists candidate releases (newest first), prints which tag would
   be chosen for the **reader range** baked into the checkout (or `CORPUS_FORMAT_*` env), exits
   non-zero when nothing is compatible.
3. **Restore** — **`make restore-corpus`** with or without **`PODCAST_BACKUP_TAG`**; unset tag uses
   **newest-compatible** selection (post-#763); set tag keeps explicit pin (may still validate
   manifest when cheap).

Exact target names will be added to the **Makefile** and cross-linked here in the same change that
ships scripts.

**Today (pre-implementation):** `make restore-corpus` still follows existing behavior (e.g. latest
when tag unset — see Makefile comments). Treat **compatibility checks** as forthcoming until
[RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) rollout completes.

## GitHub Actions (remote hosts)

| Workflow | Surface | Role |
| -------- | ------- | ---- |
| `backup-corpus.yml` | Pre-prod / Codespace path | Build snapshot, **emit manifest** (post-763), publish release on backup repo. |
| `backup-corpus-prod.yml` | Prod VPS | Same for production corpus path on tailnet host. |
| `prod-restore-corpus.yml` | Prod | **Select** compatible snapshot (post-763) or honor pin; SSH as `deploy@` per prod runbook. |
| `drill-restore-corpus.yml` | DR drill | Same selection semantics on drill **`deploy@`** target. |

Workflow file names are stable in-repo; triggers and confirms are summarized in [WORKFLOWS.md](../ci/WORKFLOWS.md)
and the two runbooks.

## Related

- [Prod runbook](PROD_RUNBOOK.md) — prod SSH, secrets, **`make restore-corpus`** on VPS.
- [DR drill runbook](DR_DRILL_RUNBOOK.md) — drill-only confirms and destroy rules.
- [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md) — prod vs drill
  topology.
