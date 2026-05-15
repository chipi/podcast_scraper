# RFC-084: Corpus Snapshot Backup Manifest and Version-Aware Restore

- **Status**: Completed
- **Authors**: Podcast Scraper Team
- **Created**: 2026-05-12
- **Domain**: Infrastructure / backups / disaster recovery
- **Tracking**: [GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)
- **Related RFCs**:
  - [RFC-081](RFC-081-pre-prod-environment-and-control-plane.md) — Phase 1E corpus backup context
  - [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) — always-on host backup story
  - [RFC-063](RFC-063-multi-feed-corpus-append-resume.md) — **`corpus_manifest.json`** at corpus parent (distinct from **snapshot** manifest)
  - [RFC-004](RFC-004-filesystem-layout.md) — on-disk layout evolution
- **Related ADRs**:
  - [ADR-074](../adr/ADR-074-multi-feed-corpus-parent-layout-and-manifest.md) — operational corpus parent artifacts (not tarball metadata)
  - [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md) — accepted policy and contract for this RFC
- **Related Documents**:
  - `.github/workflows/backup-corpus-prod.yml`, `backup-corpus.yml`
  - `.github/workflows/drill-restore-corpus.yml`, `prod-restore-corpus.yml`
  - `docs/guides/DR_DRILL_RUNBOOK.md`, `docs/guides/PROD_RUNBOOK.md`
  - `docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md` — **operator hub**: local `make` vs GitHub Actions (prod, pre-prod, DR)

## Abstract

Corpus releases are tagged by **calendar date** (`snapshot-YYYYMMDD`, `snapshot-prod-YYYYMMDD`).
Restore automation often defaults to the **latest** matching tag. When **on-disk layout or JSON
schemas** change with a software release, operators cannot see whether a given **deployed image**
can safely read a given tarball without implicit knowledge.

This RFC defines a **small machine-readable manifest** shipped with each backup and a **default
restore selection policy**: choose the **newest backup whose `corpus_format_version` the deployed
reader supports**, otherwise **fail closed** and require an explicit operator-selected tag.

## Problem Statement

**Today**

1. Compatibility of backup versus reader is **implicit** (dates, release notes, tribal knowledge).
2. After **rollback** or during **schema migration**, “latest” may point at a corpus the rolled-back
   code cannot parse, or vice versa.
3. Under incident stress, wrong tarball choice wastes time or extends outage.

**Distinction**

- **`corpus_manifest.json`** ([ADR-074](../adr/ADR-074-multi-feed-corpus-parent-layout-and-manifest.md)):
  operational index inside a **live corpus parent** (multi-feed, discovery).
- **`snapshot.manifest.json`** (this RFC): **release artifact metadata** describing the **packed
  snapshot** and the **producer** that wrote it. They are complementary; do not overload one file
  for both roles.

## Goals

1. **Explicit format identity**: every published snapshot carries a **bumpon-breaking-change**
   `corpus_format_version` and **producer identity** (git SHA and/or image digest).
2. **Safe default restore**: workflows select **newest compatible** backup without operator input
   when possible.
3. **Fail closed**: if no candidate matches, restore **does not** silently pick “latest”; logs
   explain the mismatch and point to explicit `backup_tag` / pin inputs.
4. **Audit**: optional fields tie an artifact to **workflow run** or script version.

Non-goals for v1:

- **Exact producer match** as default (too strict for routine DR; optional strict mode later).
- **Encryption / compression** metadata beyond what the tarball already implies (optional later).

## Constraints and Assumptions

**Constraints**

- Manifest must be **small**, **JSON**, and easy to generate in GitHub Actions without new runtime
  services.
- Must not require reading the **multi-gigabyte** tarball to learn format version when a separate
  manifest asset is available (see placement).
- Drill and prod restore workflows already accept **explicit** tag overrides; default path must
  remain overridable.

**Assumptions**

- **Reader support** for format versions can be expressed as a **range** or table in code (e.g.
  min/max integer) owned by the application repo.
- **Breaking** layout or schema changes are rare and **review-gated**; bumping
  `corpus_format_version` is part of that change.

## Design

### 1. Artifact: `snapshot.manifest.json`

**MIME**: `application/json`  
**Filename**: `snapshot.manifest.json` (alongside `snapshot.tgz` or as sibling release asset).

**Required fields** (v1)

| Field | Type | Description |
| ----- | ---- | ----------- |
| `schema_version` | integer | Version of **this manifest schema** (start at `1`). |
| `corpus_format_version` | integer | Bumped only on **breaking** on-disk or schema changes **older readers cannot handle**. |
| `created_at` | string | RFC 3339 UTC timestamp when the manifest was written. |
| `producer` | object | Who produced the tarball (see below). |
| `archive` | object | What is inside the snapshot (see below). |

**`producer` object** (at least one of git or image identity)

| Field | Type | Description |
| ----- | ---- | ----------- |
| `git_sha` | string | Full **40-hex** commit SHA of the repo that wrote the corpus (preferred when backup runs from CI checkout). |
| `image_digest` | string | `sha256:…` of the **pipeline image** if backup runs from a published image (optional if `git_sha` suffices). |

**`archive` object**

| Field | Type | Description |
| ----- | ---- | ----------- |
| `relative_path` | string | Path of the tarball **relative to manifest** in the release layout (typically `snapshot.tgz`). |
| `sha256` | string | Hex digest of the tarball (optional but recommended for integrity checks). |

**Optional fields**

| Field | Type | Description |
| ----- | ---- | ----------- |
| `backup_workflow` | object | e.g. `{ "name": "backup-corpus-prod", "run_id": "…", "attempt": 1 }` |
| `notes` | string | Short human-readable line (not a substitute for release notes). |

**Example**

```json
{
  "schema_version": 1,
  "corpus_format_version": 1,
  "created_at": "2026-05-12T12:34:56Z",
  "producer": {
    "git_sha": "a1b2c3d4e5f6789012345678901234567890abcd"
  },
  "archive": {
    "relative_path": "snapshot.tgz",
    "sha256": "…"
  },
  "backup_workflow": {
    "name": "backup-corpus-prod",
    "run_id": "12345678901",
    "attempt": 1
  }
}
```

**Semantics**

- **`corpus_format_version`**: monotonic integer for **compatibility**, not feature richness. Patch
  releases that only fix bugs without changing on-disk contracts **do not** bump it. A release that
  changes layout so an old reader errors or mis-reads **must** bump it.
- **SemVer strings** were considered; **integer** keeps comparisons trivial in shell and Actions.

### 2. Placement

**Decision (see [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md))**

1. **Inside** the tarball: always place `snapshot.manifest.json` at a **well-known path** (e.g.
   tarball root) so an extracted tree is self-describing.
2. **Release / storage**: also upload **`snapshot.manifest.json`** as a **separate asset** next to
   `snapshot.tgz` so consumers can read compatibility **without** downloading the full archive.

### 3. Reader compatibility

The application (or restore script) **declares** which `corpus_format_version` values it supports,
for example:

- Constants: `CORPUS_FORMAT_MIN = 1`, `CORPUS_FORMAT_MAX = 2`, or
- A small table keyed by major release.

**Rules**

1. A backup is **compatible** if `corpus_format_version` is within the reader’s supported set
   (inclusive range unless documented otherwise).
2. **Newer** backup with **higher** format than the reader supports: **incompatible**.
3. **Older** backup within range: **compatible**.

Migration releases that **read old and write new** bump `corpus_format_version` only when **old
readers cannot open new dumps** (per ADR-092).

### 4. Restore selection algorithm (default)

Given deploy **reader** compatibility `[min,max]` and a list of backup candidates **newest first**
(e.g. tags or release assets sorted by date or semver tag):

1. For each candidate, fetch **`snapshot.manifest.json`** (prefer sibling asset; fallback: error if
   only tarball exists and operator did not opt in to download).
2. Select the **first** candidate whose `corpus_format_version` is compatible.
3. If **none** match: **fail** the workflow step with a clear message and require explicit
   `backup_tag` / URL override (existing workflow inputs).

**Logging**

- On success: print chosen tag, `corpus_format_version`, and `producer.git_sha` / `image_digest`.
- On skip: log incompatible candidate and reason (version too new / too old).

### 5. Shared implementation: scripts, Make, and GitHub Actions

**Goal:** define behavior once and reuse it from **local** `make`, **drill/prod** hosts, and
**GitHub Actions**, so restore checks do not drift from CI-only YAML.

| Layer | Responsibility |
| ----- | -------------- |
| **Repo scripts** | Canonical helpers under `scripts/ops/corpus_snapshot/`: **`emit_manifest.sh`**, **`finalize_backup_bundle.sh`**, **`validate_snapshot_manifest.sh`**, **`select_release_tag.sh`**, **`download_and_verify_snapshot.sh`**, **`restore_corpus_release.sh`**; prod workflows call **`resolve_latest_snapshot_prod_tag.sh`**. |
| **Makefile** | **`corpus-snapshot-manifest-validate`**, **`corpus-snapshot-select-tag`**, **`corpus-snapshot-select-tag-prod`**, **`corpus-snapshot-selftest`**, **`restore-corpus`** (codespace layout), **`restore-corpus-prod`** (VPS layout). Unset **`PODCAST_BACKUP_TAG`** → newest-compatible selection; set pin → skip scan but still validate when a sibling manifest exists. **`codespace-backup-cloud`** / **`backup-corpus.yml`** own cloud emit. |
| **Workflows** | **`backup-corpus-prod.yml`**, **`backup-corpus.yml`**: after tarball build, **`finalize_backup_bundle.sh`**, upload **`snapshot.tgz`** + sibling **`snapshot.manifest.json`** when **`dry_run`** is false. **`drill-restore-corpus.yml`**, **`prod-restore-corpus.yml`**: **`resolve_latest_snapshot_prod_tag.sh`** (or pin), runner **`download_and_verify_snapshot.sh`**, host restore via **`restore_corpus_from_tarball_host.sh`**. |
| **CI** | Optional: **`make`**-driven check job (or workflow step) that runs manifest validation on fixtures and/or tests the selection helper with golden JSON; optional path guard when “breaking” corpus layout paths change without a documented `corpus_format_version` bump (see §9). |

**Mapping to [GitHub #763](https://github.com/chipi/podcast_scraper/issues/763):** backup **writes** manifest + dual placement; restore **reads** manifests and **enforces** newest-compatible / fail closed; **scripts + Make + workflows** together cover the deliverable — not Actions-only logic with no local equivalent.

**Operator map (all surfaces):** [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) — **Make** for local validation/restore loop; **Actions** for prod, pre-prod/codespace backup, and drill restore.

### 6. Backup workflows

**Scope**: `.github/workflows/backup-corpus-prod.yml`, `backup-corpus.yml`, and any drill-specific
backup if present.

**Steps (conceptual)**

1. After tarball is built and before upload: compute `sha256` of `snapshot.tgz`.
2. Emit `snapshot.manifest.json` with fields above (`git_sha` from `${{ github.sha }}` or image
   digest from metadata) via the **shared** script or `make` target (§5).
3. Ensure manifest is **inside** the tarball at the chosen path, then upload tarball **and** sibling
   manifest; keep naming stable for downstream scripts.

**Sequencing** (from issue #763): implement after **dr-drill-exercise** E2E is stable if concurrent
risk is high; schema and RFC can land earlier.

### 7. Restore workflows and scripts

**Scope**: `.github/workflows/drill-restore-corpus.yml`, `prod-restore-corpus.yml`, shared restore
scripts called by them, and **`restore-corpus`** / **`restore-corpus-prod`** in the `Makefile`.

1. Resolve reader supported range (from env baked into image, or from a small repo file read by the
   script).
2. List candidates; fetch manifests; run **newest compatible** selection (shared implementation §5).
3. Preserve **operator override**: explicit tag/env wins over default.
4. Log chosen tag, `corpus_format_version`, and producer identity; on miss, fail with instruction to
   pin `backup_tag` / URL.

### 8. Documentation

- DR and prod runbooks: **when newest-compatible default is wrong** (rollback, format bump,
  mixed-age hosts) — see [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).
- Cross-link this RFC and [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md).
- Document **`restore-corpus`**, **`restore-corpus-prod`**, and env vars (`PODCAST_BACKUP_TAG`,
  `PODCAST_BACKUP_REPO`) vs default selection behavior after implementation.

### 9. Testing and CI (optional follow-up)

- Unit or script tests for selection given fixture manifests.
- Optional **lint**: if a PR changes documented “breaking” corpus layout paths or schemas, require
  a bump to `corpus_format_version` / changelog entry (lightweight guardrail).

## Alternatives Considered

1. **Calendar tags only, no manifest** — Rejected; fails under rollback and schema migration
   ([GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)).
2. **SemVer for `corpus_format_version`** — Rejected for v1; integer comparisons are simpler in
   Actions and shell.
3. **Only manifest inside tarball** — Rejected as sole option; forces full download to inspect
   compatibility. **Dual placement** keeps small-metadata fetches cheap.
4. **Reuse `corpus_manifest.json` as tarball metadata** — Rejected; different lifecycle and purpose
   ([ADR-074](../adr/ADR-074-multi-feed-corpus-parent-layout-and-manifest.md)).

## Rollout

1. **Schema + example** in repo; ADR ratified ([ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)).
2. **Shared scripts** + **`make`** targets for emit, validate, and select-newest-compatible (§5).
3. **Backup** workflows call shared emit path; upload tarball + sibling manifest.
4. **Restore** workflows and **`restore-corpus`** / **`restore-corpus-prod`** call shared selection
   and download paths; explicit pins unchanged.
5. **Docs** and runbooks updated; optional CI check (§9).

## Open Questions

Resolved in implementation:

1. **`snapshot.manifest.json` at tarball archive root** (alongside `corpus/` or `.codespace_corpus/`).
2. **Drill restore** uses **`snapshot-prod-*`** tags via `scripts/ops/resolve_latest_snapshot_prod_tag.sh` (same as prod).
3. **Reader range** in `config/corpus_snapshot_reader_support.json`; producer format in `config/corpus_snapshot_format.json`.

## References

- [GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)
- [ADR-092: Corpus snapshot backup manifest and newest-compatible restore default](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)
